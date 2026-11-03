# Live demo (retrieval only) for a Hugging Face Docker Space. Build context: the repository root.
#
#   docker build -f demo/Dockerfile -t fedlex-demo .
#   docker run --rm -p 7860:7860 fedlex-demo
#
# This file has NOT been built or run by the author (no Docker on the machine it was written on).
# Read demo/README.md before using it.
#
# The image holds no article text in the repository: the build downloads the statutes from Fedlex
# (the pinned URLs and hashes in data/manifest.json) and turns them into data/processed/articles.parquet.

FROM python:3.12-slim

# uv, pinned. CPU only: nothing here installs torch (it lives in the "finetune" group).
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

# Hugging Face Spaces run the container as user 1000, so build as that user too.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/app/.venv/bin:/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860
WORKDIR /home/user/app

# 1. Dependencies first (cached unless pyproject.toml or uv.lock change).
COPY --chown=user pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --group serve --no-install-project

# 2. The project itself. Only what the demo needs; the corpus is built below, not copied.
COPY --chown=user src ./src
COPY --chown=user config.toml ./config.toml
COPY --chown=user data/manifest.json ./data/manifest.json
COPY --chown=user demo/examples.json ./demo/examples.json
RUN uv sync --frozen --no-dev --group serve

# 3. BUILD TIME: download the pinned int8 model (HUB_REPO / HUB_REVISION in serve/pipeline.py).
RUN python -c "from fedlex_bench.serve.pipeline import resolve_model_dir; print(resolve_model_dir())"

# 4. BUILD TIME: the corpus, with the project's own command (downloads raw XML from Fedlex using
#    data/manifest.json, writes data/processed/articles.parquet). Fails the build if Fedlex is down.
RUN python -m fedlex_bench.chunk

# 5. BUILD TIME: embed every passage once with the tuned model and keep the vectors in
#    data/processed/index/, so the first visitor does not wait. Then drop the raw XML we no longer need.
RUN python -m fedlex_bench.serve.web --warm && rm -rf data/raw

# Run time: no Hub access needed (the model is in the cache), no API key, no LLM.
ENV HF_HUB_OFFLINE=1
EXPOSE 7860
CMD ["python", "-m", "fedlex_bench.serve.web"]
