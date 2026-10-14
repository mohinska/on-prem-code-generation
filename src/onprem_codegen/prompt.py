"""The one user message for every COBOL task (docs/contracts.md §3). SFT data and eval both build it here.

    from onprem_codegen.prompt import build_prompt, build_messages
    build_messages(task_text)   # -> OpenAI / apply_chat_template messages, no system prompt
"""

INSTRUCTION = (
    "Write a complete GnuCOBOL program in fixed format. Read input from standard input and write the\n"
    "result to standard output. Answer with a single ```cobol code block."
)


def build_prompt(task_text: str) -> str:
    """Raw user message, not chat-templated; the trainer / vLLM apply the model's own template."""
    task_text = task_text.strip()
    if not task_text:
        raise ValueError("empty task text")
    return f"{task_text}\n\n{INSTRUCTION}"


def build_messages(task_text: str) -> list[dict]:
    return [{"role": "user", "content": build_prompt(task_text)}]
