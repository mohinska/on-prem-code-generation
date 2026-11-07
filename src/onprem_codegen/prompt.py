"""The one user message for every COBOL task (docs/contracts.md §3). SFT data and eval both build it here.

    from onprem_codegen.prompt import build_prompt, build_messages
    build_messages(task_text)   # -> OpenAI / apply_chat_template messages, no system prompt

Only the I/O sentence depends on the task: stdin/stdout for task-pool (SFT data, the default),
and one sentence per benchmark whose programs do I/O differently (IO_* below).
"""
import re

IO_STDIN = "Read input from standard input with ACCEPT and write the result to standard output with DISPLAY."
IO_COBOLEVAL = ("Complete the program above: keep its PROGRAM-ID and LINKAGE SECTION unchanged\n"
                "and store the answer in RESULT.")
IO_FILES = "Read input from the files named above and write the results to the output files named above."


def instruction(io: str = IO_STDIN) -> str:
    return f"Write a complete GnuCOBOL program in fixed format. {io} Answer with a single ```cobol code block."


def build_prompt(task_text: str, io: str = IO_STDIN) -> str:
    """Raw user message, not chat-templated; the trainer / vLLM apply the model's own template."""
    if not task_text.strip():
        raise ValueError("empty task text")
    # Fixed-format COBOL starts in column 8: drop blank lines and trailing space, keep the first line's indent.
    task_text = re.sub(r"\A(?:[ \t]*\n)+", "", task_text).rstrip()
    return f"{task_text}\n\n{instruction(io)}"


def build_messages(task_text: str, io: str = IO_STDIN) -> list[dict]:
    return [{"role": "user", "content": build_prompt(task_text, io)}]
