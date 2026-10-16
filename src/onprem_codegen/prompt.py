"""The one user message for every COBOL task (docs/contracts.md §3). SFT data and eval both build it here.

    from onprem_codegen.prompt import build_prompt, build_messages
    build_messages(task_text)   # -> OpenAI / apply_chat_template messages, no system prompt

Only the I/O sentence depends on the task: stdin/stdout for task-pool (SFT data, the default),
and one sentence per benchmark whose programs do I/O differently (IO_* below).
"""

IO_STDIN = "Read input from standard input and write the\nresult to standard output."
IO_COBOLEVAL = ("Complete the program above: keep its PROGRAM-ID and LINKAGE SECTION unchanged\n"
                "and store the answer in RESULT.")
IO_FILES = "Read input from the files named above and write the results to the output files named above."


def instruction(io: str = IO_STDIN) -> str:
    return f"Write a complete GnuCOBOL program in fixed format. {io} Answer with a single ```cobol code block."


def build_prompt(task_text: str, io: str = IO_STDIN) -> str:
    """Raw user message, not chat-templated; the trainer / vLLM apply the model's own template."""
    task_text = task_text.strip()
    if not task_text:
        raise ValueError("empty task text")
    return f"{task_text}\n\n{instruction(io)}"


def build_messages(task_text: str, io: str = IO_STDIN) -> list[dict]:
    return [{"role": "user", "content": build_prompt(task_text, io)}]
