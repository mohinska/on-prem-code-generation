"""Log the tiny hand-written COBOL SFT set as the `sft-smoke-data` W&B artifact.

    python -m onprem_codegen.data.upload_smoke
"""

import os
from pathlib import Path

import wandb

FIXTURE = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "sft_smoke.jsonl"


def main() -> None:
    with wandb.init(project=os.environ.get("WANDB_PROJECT", "onprem-codegen"), job_type="upload-data") as run:
        art = wandb.Artifact("sft-smoke-data", type="dataset", description="8 hand-written COBOL tasks for smoke tests")
        art.add_file(str(FIXTURE), name="sft.jsonl")
        run.log_artifact(art)


if __name__ == "__main__":
    main()
