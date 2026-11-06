"""Supervised fine-tuning with QLoRA (TRL + PEFT), traced in W&B.

    python -m onprem_codegen.train.sft                                  # smoke, 1x 80 GB GPU
    python -m onprem_codegen.train.sft --config-name sft run_name=...   # real run on sft-dataset

Restartable: checkpoints are mirrored to W&B, so a fresh instance resumes from the last one.
"""

import math
import os
import random
import time
from pathlib import Path

import hydra
import torch
import wandb
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from onprem_codegen.utils import git_commit

CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs" / "train")


def load_model(cfg: DictConfig, path: str):
    kwargs = {"dtype": torch.bfloat16}
    if cfg.model.load_in_4bit:
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


def split_indices(task_ids: list, eval_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Hold out whole tasks (rows without task_id count as one task each), seeded."""
    groups: dict[tuple, list[int]] = {}
    for i, t in enumerate(task_ids):
        groups.setdefault((0, t) if t is not None else (1, i), []).append(i)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_eval = min(max(1, round(len(keys) * eval_fraction)), len(keys) - 1) if eval_fraction > 0 else 0
    ev = sorted(i for k in keys[:n_eval] for i in groups[k])
    held = set(ev)
    return [i for i in range(len(task_ids)) if i not in held], ev


def drop_overlength(ds, tok, max_length: int):
    """Drop (not truncate) rows whose prompt + completion exceed max_length tokens; returns (dataset, n_dropped)."""
    ids = tok([p + c for p, c in zip(ds["prompt"], ds["completion"])])["input_ids"]
    keep = [i for i, x in enumerate(ids) if len(x) <= max_length]
    return ds.select(keep), len(ds) - len(keep)


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


class StepStats(TrainerCallback):
    """Wall time per optimizer step and peak GPU memory (reserved, ~ nvidia-smi), for sizing real runs."""

    def on_step_begin(self, args, state, control, **kwargs):
        self.t = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            wandb.log({
                "perf/s_per_step": time.perf_counter() - self.t,
                "perf/peak_mem_gb": torch.cuda.max_memory_reserved() / 2**30,
                "train/global_step": state.global_step,
            })


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

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    model = load_model(cfg, cfg.model.name)

    ds = load_dataset("json", data_files=str(Path(data_dir) / "sft.jsonl"), split="train")
    # Split by task before rendering (rendering drops task_id), then drop over-length rows from both parts.
    task_ids = ds["task_id"] if "task_id" in ds.column_names else [None] * len(ds)
    train_idx, eval_idx = split_indices(task_ids, cfg.data.eval_fraction, cfg.seed)
    parts = [ds.select(idx) for idx in (train_idx, eval_idx)]
    parts = [p.map(lambda ex: to_prompt_completion(tok, ex), remove_columns=p.column_names) for p in parts]
    (ds, dropped_train), (eval_ds, dropped_eval) = (drop_overlength(p, tok, cfg.train.max_length) for p in parts)
    eval_ds = eval_ds if len(eval_ds) else None
    run.summary.update({"data/train_rows": len(ds), "data/eval_rows": len(eval_ds or []),
                        "data/dropped_overlength_train": dropped_train, "data/dropped_overlength_eval": dropped_eval})
    print(f"rows: train {len(ds)} (dropped over-length {dropped_train}), eval {len(eval_ds or [])} (dropped {dropped_eval})")
    steps = cfg.train.max_steps if cfg.train.max_steps > 0 else cfg.train.num_train_epochs * math.ceil(
        len(ds) / (cfg.train.per_device_train_batch_size * cfg.train.gradient_accumulation_steps))

    args = SFTConfig(
        output_dir=cfg.train.output_dir,
        max_steps=cfg.train.max_steps,
        num_train_epochs=cfg.train.num_train_epochs,
        save_steps=cfg.train.save_steps,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=cfg.train.save_steps,
        save_total_limit=2,
        logging_steps=cfg.train.logging_steps,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, steps // 10),
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        max_length=cfg.train.max_length,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        bf16=True,
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
        eval_dataset=eval_ds,
        processing_class=tok,
        peft_config=lora,
        callbacks=[MirrorCheckpoints(f"{cfg.run_name}-ckpt"), StepStats()],
    )
    trainer.model.print_trainable_parameters()
    run.summary["trainable_params"] = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
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
