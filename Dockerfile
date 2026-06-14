FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
       python3 python3-pip python3-dev \
       build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ldconfig /usr/local/cuda-12.1/compat/ 2>/dev/null || true

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.0

COPY requirements.txt /requirements.txt
RUN python3 -m pip install --no-cache-dir -r /requirements.txt

COPY . .

ENV LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \
    MAX_MODEL_LEN=8192 \
    GPU_MEMORY_UTILIZATION=0.9 \
    TENSOR_PARALLEL_SIZE=1 \
    VLLM_USE_V1=0 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

CMD ["python3", "-u", "handler.py"]