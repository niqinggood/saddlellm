FROM python:3.11-slim-bookworm

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.6.0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/huggingface

WORKDIR /opt/saddle-src
COPY pyproject.toml README.md LICENSE ./
COPY saddlellm ./saddlellm
COPY saddle_llm ./saddle_llm
COPY deploy/constraints-serve.txt ./constraints-serve.txt

RUN python -m pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}"
RUN python -m pip install '.[serve]' -c constraints-serve.txt
RUN useradd --create-home --uid 10001 saddle

USER 10001:10001
EXPOSE 8000
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import json,urllib.request; r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/readyz',timeout=3)); raise SystemExit(0 if r.get('status')=='ready' else 1)"]

ENTRYPOINT ["saddlellm", "serve"]
CMD ["/models", "--host", "0.0.0.0", "--port", "8000", "--local-files-only", "--no-web-ui"]
