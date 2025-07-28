# Dockerfile (root)
FROM --platform=linux/amd64 python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY models /app/models  # Offline model
COPY . .
CMD ["python", "src/main.py"]