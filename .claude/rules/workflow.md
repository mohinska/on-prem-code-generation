# Workflow

## Communication
- Keep explanations simple and answers concise. No extra info beyond what was asked.
- Long jobs: always give a command to track progress; scripts print compact progress, never long unfiltered dumps.

## Stack and dependencies
- The stack is fixed: TRL + PEFT, transformers, vLLM, Hydra, W&B, datasketch, GnuCOBOL, pytest, ruff.
  Do not add a dependency without asking. Every dependency is pinned in `requirements.txt`.

## Reproducibility
- Seeds are explicit and fixed: training seeds 42 and 43, `data.split_seed` 42, bootstrap seed 0.
- Every stage is restartable and reads its inputs from W&B artifacts (see [wandb.md](wandb.md)).

## Commit messages
- One line: `type(scope): message` (e.g. `feat(data): add MinHash dedup`). No body, no Co-Authored-By trailer.
- When asked for a commit message, reply with ready-to-run commands: `git add <files>`, `git commit -m "..."`, `git push`.

## Project log
- Everything we do is documented in [docs/project-log.md](../../docs/project-log.md): decisions, code changes, runs,
  sweeps, evals, problems and **every result with exact numbers**. It is the source for the intermediate report,
  the final report and the paper.
- The log describes the project as it is now; new runs and results are added as entries under "Results" (what was run,
  config, exact numbers, W&B run/artifact link, commit hash). No dates in titles, and no "removed/dropped X" entries:
  update the description instead. Failed runs and negative results go in too.
- How to write it: [writing.md](writing.md).

## Keep README.md in sync
Whenever you add, move or remove a top-level directory or module, update the "Project structure" section of README.md
(and the layout in CLAUDE.md) in the same change. The same goes for "Key decisions" when a decision changes.
