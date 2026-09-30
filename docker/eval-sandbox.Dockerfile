# Eval sandbox: compiles and runs model-generated COBOL with GnuCOBOL.
# Generated code is untrusted — run with --network none, read-only rootfs, and resource limits
# (see the eval-sandbox service in docker-compose.yml).
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends gnucobol \
    && rm -rf /var/lib/apt/lists/* \
    && cobc --version

RUN useradd --create-home --uid 10001 sandbox
USER sandbox
WORKDIR /sandbox

CMD ["bash"]
