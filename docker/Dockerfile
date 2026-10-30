# atdev-serve container: CUDA 13.0.2 devel image (nvcc V13.0.88, the toolkit the published numbers were
# built with; needed for the first-start kernel JIT build), Python 3.12 (Ubuntu 24.04), vllm==0.31.0 from pip, atdev-serve from this checkout.
#
# build:  docker build -f docker/Dockerfile -t atdev-serve:0.1.0 .
# run:    docker run --gpus '"device=0"' --ipc=host -p 8100:8100 \
#           -v atdev-hf:/root/.cache/huggingface -v atdev-ext:/root/.cache/torch_extensions \
#           atdev-serve:0.1.0 c1            # or c32
# The two volumes keep the model downloads and the compiled kernels between runs.
ARG CUDA_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG VLLM_VERSION=0.31.0
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH} \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=8.0 \
    HF_HOME=/root/.cache/huggingface \
    TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions

# hadolint ignore=DL3008,DL3013
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev build-essential git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && python3.12 -m venv /opt/venv \
 && pip install --upgrade pip

# vLLM first (large, rarely changes -> own layer)
RUN pip install "vllm==${VLLM_VERSION}"

WORKDIR /opt/atdev-serve
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
# hadolint ignore=DL3013
RUN pip install ".[bench]" \
 && python -c "import atdev_serve, vllm; assert vllm.__version__ == '${VLLM_VERSION}'"

COPY serve_c1.sh serve_c32.sh ./
COPY bench ./bench
COPY docker/entrypoint.sh /usr/local/bin/atdev-entrypoint
RUN chmod +x serve_c1.sh serve_c32.sh bench/*.sh /usr/local/bin/atdev-entrypoint

EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=5s --start-period=3h --retries=3 \
  CMD ["sh", "-c", "curl -sf http://127.0.0.1:${PORT:-8100}/health || exit 1"]
ENTRYPOINT ["/usr/local/bin/atdev-entrypoint"]
CMD ["c1"]
