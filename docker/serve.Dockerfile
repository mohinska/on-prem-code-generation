# Serving image: vLLM OpenAI-compatible API, base model + LoRA adapter.
# Pin VLLM_VERSION to a release that supports the Qwen3.8 hybrid architecture.
ARG VLLM_VERSION
FROM vllm/vllm-openai:${VLLM_VERSION}

ENV HF_HOME=/cache/huggingface \
    MODEL=Qwen/Qwen3.8-27B \
    SERVED_MODEL_NAME=base \
    ADAPTER_NAME=cobol \
    ADAPTER_PATH=/adapters/cobol \
    MAX_LORA_RANK=64 \
    MAX_MODEL_LEN=32768 \
    PORT=8000

COPY docker/serve-entrypoint.sh /usr/local/bin/serve-entrypoint.sh
RUN chmod +x /usr/local/bin/serve-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/serve-entrypoint.sh"]
