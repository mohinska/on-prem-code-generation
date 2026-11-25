# W&B conventions

- Data artifact names/types are pinned in [configs/data/wandb.yaml](../../configs/data/wandb.yaml): `raw-corpus`
  (dataset, all sources + `manifest.parquet`, before filters), `clean-corpus` (dataset, after filters/dedup/decontam +
  stats), `decontam-report` (report, optional, what was removed and why).
- Every stage consumes its inputs **from W&B artifacts** and logs outputs as new artifact versions,
  so lineage from raw corpus to deployed model is automatic.
- Every run logs: git commit, config, input/output artifacts. GPU comes from W&B system metadata; note the vLLM version
  in the project log.
- Local smoke runs use `--no-wandb`; nothing from a `--no-wandb` run is reported as a result.
- Sweeps are small: learning rate × LoRA rank, few configs, short runs.
- Registry: aliases `candidate` → `production`; an adapter is promoted only when every preregistered guard is met.
