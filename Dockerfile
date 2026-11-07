FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/smartdocs/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system smartdocs \
    && useradd --system --gid smartdocs --create-home smartdocs \
    && mkdir -p /app/data/chroma /home/smartdocs/.cache/huggingface \
    && chown -R smartdocs:smartdocs /app /home/smartdocs

COPY requirements.txt requirements-dev.txt ./

# SmartDocs uses the CPU embedding model. Installing the CPU PyTorch wheel
# avoids pulling GPU-only libraries into the application image.
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch \
    && python -m pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

COPY --chown=smartdocs:smartdocs app ./app
COPY --chown=smartdocs:smartdocs tests ./tests
COPY --chown=smartdocs:smartdocs evaluation ./evaluation
COPY --chown=smartdocs:smartdocs README.md ./README.md

USER smartdocs

EXPOSE 8000 8501

CMD ["python", "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
