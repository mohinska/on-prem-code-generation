# Eval sandbox: compiles and runs model-generated COBOL with GnuCOBOL.
# Generated code is untrusted — run with --network none, read-only rootfs, and resource limits
# (see the eval-sandbox service in docker-compose.yml and onprem_codegen/eval/sandbox.py).
#
# GnuCOBOL 3.2 is built from the upstream release: COBOLEval targets 3.2.0, and the Debian
# packages are either 3.1.2 (bookworm) or 3.2 linked against a downgraded libxml2 (trixie),
# which prints a version warning into every program's stderr. XML/JSON support is left out:
# no benchmark uses it. Bump the version and sha256 together (sha256 of the GPG-verified tarball).
ARG GNUCOBOL_VERSION=3.2
ARG GNUCOBOL_SHA256=3bb48af46ced4779facf41fdc2ee60e4ccb86eaa99d010b36685315df39c2ee2

FROM python:3.12-slim-bookworm AS build
ARG GNUCOBOL_VERSION
ARG GNUCOBOL_SHA256
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl libgmp-dev libdb-dev libncurses-dev \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL -o /tmp/gnucobol.tar.xz \
        "https://ftp.gnu.org/gnu/gnucobol/gnucobol-${GNUCOBOL_VERSION}.tar.xz" \
    && echo "${GNUCOBOL_SHA256}  /tmp/gnucobol.tar.xz" | sha256sum -c - \
    && tar -xJf /tmp/gnucobol.tar.xz -C /tmp \
    && cd "/tmp/gnucobol-${GNUCOBOL_VERSION}" \
    && ./configure --prefix=/opt/gnucobol --without-xml2 --without-json \
    && make -j"$(nproc)" \
    && make install

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PATH=/opt/gnucobol/bin:$PATH

# cobc generates C and calls gcc, so the compiler stays in the runtime image; the generated
# C includes gmp.h for decimal arithmetic, hence libgmp-dev rather than just libgmp10.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libc6-dev libgmp-dev libdb5.3 libncursesw6 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/gnucobol /opt/gnucobol
RUN echo /opt/gnucobol/lib > /etc/ld.so.conf.d/gnucobol.conf && ldconfig \
    && cobc --version | head -1

RUN useradd --create-home --uid 10001 sandbox
USER sandbox
WORKDIR /sandbox

CMD ["bash"]
