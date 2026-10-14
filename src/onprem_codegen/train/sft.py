"""Supervised fine-tuning with QLoRA (TRL + PEFT), traced in W&B.

    python -m onprem_codegen.train.sft                                  # smoke, 1x 80 GB GPU
    python -m onprem_codegen.train.sft --config-name sft run_name=...   # real run on sft-dataset
    python -m onprem_codegen.train.sft model.tiny=true run_name=...     # local dry run (CPU / MPS)

Restartable: checkpoints are mirrored to W&B, so a fresh instance resumes from the last one.
"""

import math
import os
import subprocess
import tempfile
from pathlib import Path

import hydra
import torch
import transformers
import wandb
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs" / "train")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_tiny(name: str, out_dir: str) -> str:
    """Save a random 4-layer copy of the full multimodal architecture.

    Loading it exercises the same path as the real checkpoint (text weights under
    model.language_model.*, vision tower dropped), without downloading 54 GB.
    """
    cfg = AutoConfig.from_pretrained(name)
    t = cfg.text_config
    t.num_hidden_layers, t.layer_types = 4, t.layer_types[:4]
    t.hidden_size, t.intermediate_size = 128, 256
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 2, 1, 64
    t.linear_num_key_heads, t.linear_num_value_heads = 2, 2
    t.linear_key_head_dim, t.linear_value_head_dim = 32, 32
    v = cfg.vision_config
    v.depth, v.hidden_size, v.intermediate_size, v.num_heads, v.out_hidden_size = 1, 64, 128, 2, 128
    torch.manual_seed(0)  # same random weights on every run, so resume tests are exact
    getattr(transformers, cfg.architectures[0])._from_config(cfg).save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(name).save_pretrained(out_dir)
    return out_dir


def load_model(cfg: DictConfig, path: str):
    kwargs = {"dtype": torch.float32 if cfg.model.tiny else torch.bfloat16}
    if cfg.model.load_in_4bit and not cfg.model.tiny:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    # Text-only class: the vision encoder is never instantiated.
    model, info = AutoModelForCausalLM.from_pretrained(path, output_loading_info=True, **kwargs)
    if info["missing_keys"]:
        raise RuntimeError(f"Text weights not loaded: {sorted(info['missing_keys'])[:10]}")
    return model


def to_prompt_completion(tok, example: dict) -> dict:
    """Render with the chat template in non-thinking mode (empty <think> block)."""
    user = [{"role": "user", "content": example["prompt"]}]
    prompt = tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full = tok.apply_chat_template(
        user + [{"role": "assistant", "content": example["completion"]}], tokenize=False, enable_thinking=False
    )
    assert full.startswith(prompt), "chat template: prompt is not a prefix of the full conversation"
    return {"prompt": prompt, "completion": full[len(prompt):]}


class MirrorCheckpoints(TrainerCallback):
    """Instance disk is ephemeral: upload every checkpoint as a W&B artifact."""

    def __init__(self, name: str):
        self.name = name

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        art = wandb.Artifact(self.name, type="checkpoint", metadata={"step": state.global_step})
        art.add_dir(str(Path(args.output_dir) / f"checkpoint-{state.global_step}"))
        wandb.log_artifact(art, aliases=["latest", f"step-{state.global_step}"])


def find_checkpoint(cfg: DictConfig, run) -> str | None:
    if not cfg.train.resume:
        return None
    out = Path(cfg.train.output_dir)
    if out.is_dir() and (last := get_last_checkpoint(str(out))):
        return last
    try:
        art = run.use_artifact(f"{cfg.run_name}-ckpt:latest")
    except wandb.errors.CommError:
        return None
    path = out / f"checkpoint-{art.metadata['step']}"
    art.download(root=str(path))
    return str(path)


@hydra.main(config_path=CONFIG_DIR, config_name="sft_smoke", version_base=None)
def main(cfg: DictConfig) -> None:
    # Fixed run id so a restart on a fresh instance continues the same W&B run.
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "onprem-codegen"),
        id=cfg.run_name,
        name=cfg.run_name,
        resume="allow",
        job_type="sft",
        config={
            **OmegaConf.to_container(cfg, resolve=True),
            "git_commit": git_commit(),
            "vast_instance": os.environ.get("CONTAINER_ID") or os.environ.get("VAST_CONTAINERLABEL"),
        },
    )

    data_dir = run.use_artifact(cfg.data.artifact).download()

    model_path = cfg.model.name
    if cfg.model.tiny:
        model_path = build_tiny(cfg.model.name, tempfile.mkdtemp(prefix="tiny-"))
    tok = AutoTokenizer.from_pretrained(model_path)
    model = load_model(cfg, model_path)

    ds = load_dataset("json", data_files=str(Path(data_dir) / "sft.jsonl"), split="train")
    ds = ds.map(lambda ex: to_prompt_completion(tok, ex), remove_columns=ds.column_names)
    steps = cfg.train.max_steps if cfg.train.max_steps > 0 else cfg.train.num_train_epochs * math.ceil(
        len(ds) / (cfg.train.per_device_train_batch_size * cfg.train.gradient_accumulation_steps))

    args = SFTConfig(
        output_dir=cfg.train.output_dir,
        max_steps=cfg.train.max_steps,
        num_train_epochs=cfg.train.num_train_epochs,
        save_steps=cfg.train.save_steps,
        save_total_limit=2,
        logging_steps=cfg.train.logging_steps,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, steps // 10),
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        max_length=cfg.train.max_length,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        bf16=not cfg.model.tiny,
        report_to="wandb",
        run_name=cfg.run_name,
        seed=cfg.seed,
    )
    lora = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=lora,
        callbacks=[MirrorCheckpoints(f"{cfg.run_name}-ckpt")],
    )
    trainer.train(resume_from_checkpoint=find_checkpoint(cfg, run))

    adapter_dir = Path(cfg.train.output_dir) / "adapter"
    trainer.save_model(str(adapter_dir))
    adapter = wandb.Artifact(f"{cfg.run_name}-adapter", type="model", metadata={"base_model": cfg.model.name})
    adapter.add_dir(str(adapter_dir))
    run.log_artifact(adapter, aliases=["candidate"])

    # Sanity sample: one generation per smoke prompt, logged as a W&B Table.
    model = trainer.model.eval()
    table = wandb.Table(columns=["prompt", "generation"])
    for ex in ds.select(range(min(3, len(ds)))):
        ids = tok(ex["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=256, do_sample=False)
        table.add_data(ex["prompt"], tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True))
    run.log({"samples": table})
    run.finish()


if __name__ == "__main__":
    main()
