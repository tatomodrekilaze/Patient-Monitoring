# Cloud side only (dashboard + phones + AI assistant). The camera AI stays on the bedside laptop.
FROM python:3.12-slim
WORKDIR /app
COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt
COPY monitor/ monitor/
COPY web/ web/
COPY cloud.py .
ENV PYTHONUNBUFFERED=1
CMD ["python", "cloud.py"]
