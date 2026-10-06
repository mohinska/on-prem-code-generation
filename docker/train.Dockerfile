# Training image: SFT / DPO / GRPO with TRL + PEFT (QLoRA), plus data-pipeline stages.
# Devel base is required: causal-conv1d and flash-linear-attention compile CUDA kernels.
ARG TORCH_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel
FROM ${TORCH_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    REPO_DIR=/workspace/on-prem-code-generation \
    PYTHONPATH=/workspace/on-prem-code-generation/src \
    # Limit kernel builds to the GPUs we actually rent (A100 = 8.0, H100 = 9.0).
    TORCH_CUDA_ARCH_LIST="8.0;9.0" \
    MAX_JOBS=8

RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs build-essential ninja-build openssh-client tmux \
    && rm -rf /var/lib/apt/lists/*

# Dependencies only. The code is not baked in: it is bind-mounted locally or git-cloned
# on the instance, so .git is present and every W&B run can log its commit.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install --no-build-isolation -r /tmp/requirements.txt

WORKDIR ${REPO_DIR}

# No API keys here: WANDB_API_KEY / HF_TOKEN are injected at runtime.
CMD ["bash"]
