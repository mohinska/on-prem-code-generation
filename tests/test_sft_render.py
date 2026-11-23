"""train/sft.py: chat-template rendering of SFT rows and the text-only model class, for Qwen3.8 and Gemma 4."""
from transformers import Gemma4Config, Qwen3_5Config

from onprem_codegen.train.sft import text_only_kwargs, to_prompt_completion

FENCED = "```cobol\n       IDENTIFICATION DIVISION.\n       PROGRAM-ID. X.\n           GOBACK.\n```"


class QwenLikeTok:
    """Renders like Qwen3.8 non-thinking: the full conversation continues the generation prompt; content is trimmed."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=True):
        out = "".join(f"<|im_start|>{m['role']}\n" + ("<think>\n\n</think>\n\n" if m["role"] == "assistant" else "")
                      + m["content"].strip() + "<|im_end|>\n" for m in messages)
        return out + ("<|im_start|>assistant\n<think>\n\n</think>\n\n" if add_generation_prompt else "")


class GemmaLikeTok:
    """Renders like Gemma 4 non-thinking: the generation prompt has an empty thought channel, the model turn does not."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=True):
        out = "<bos>" + "".join(f"<|turn>{'model' if m['role'] == 'assistant' else m['role']}\n"
                                + m["content"].strip() + "<turn|>\n" for m in messages)
        return out + ("<|turn>model\n<|channel>thought\n<channel|>" if add_generation_prompt else "")


def test_qwen_completion_is_the_rest_of_the_rendered_conversation():
    row = to_prompt_completion(QwenLikeTok(), {"prompt": "  Task", "completion": FENCED})
    assert row["prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert row["completion"] == FENCED + "<|im_end|>\n"


def test_gemma_prompt_is_the_generation_prompt_and_completion_the_answer_plus_end_of_turn():
    tok = GemmaLikeTok()
    row = to_prompt_completion(tok, {"prompt": "  Task", "completion": FENCED})
    eval_prompt = tok.apply_chat_template([{"role": "user", "content": "  Task"}], add_generation_prompt=True)
    assert row["prompt"] == eval_prompt                                   # what vLLM sends at eval time
    assert row["completion"] == FENCED + "<turn|>\n"


def test_gemma_keeps_the_indent_of_an_unfenced_answer():
    answer = "       IDENTIFICATION DIVISION.\n           GOBACK."
    row = to_prompt_completion(GemmaLikeTok(), {"prompt": "Task", "completion": answer})
    assert row["completion"] == answer + "<turn|>\n"                      # the template alone would drop 7 spaces


def test_gemma_loads_the_text_class_from_the_multimodal_checkpoint():
    kw = text_only_kwargs(Gemma4Config())
    assert kw["config"].model_type == "gemma4_text"
    assert kw["key_mapping"] == {r"^model\.language_model\.": "model."}


def test_qwen_auto_class_is_already_text_only():
    assert text_only_kwargs(Qwen3_5Config()) == {}
