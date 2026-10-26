# Build context is the repository root, so that BAMBON itself can be installed
# from source alongside the integration package:
#
#   docker build -f examples/edc-dataspace-demo/Dockerfile .
#
# docker-compose.yml already sets `context: ../..` for you.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY examples/edc-dataspace-demo/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# BAMBON from this checkout, not from PyPI, so the demo tracks local changes.
COPY pyproject.toml setup.py README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY examples/edc-dataspace-demo/bambon_edc ./bambon_edc
COPY examples/edc-dataspace-demo/mock-backend ./mock-backend
COPY examples/edc-dataspace-demo/tests ./tests

CMD ["uvicorn", "bambon_edc.service:app", "--host", "0.0.0.0", "--port", "8100"]
