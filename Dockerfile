FROM python:3.11-slim
WORKDIR /app
COPY requirements-headless.txt .
RUN pip install --no-cache-dir -r requirements-headless.txt
COPY . .
CMD ["python", "headless_bot.py", "--settings", "/app/settings.yaml"]
