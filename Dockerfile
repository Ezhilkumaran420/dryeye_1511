FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first to keep build fast and low memory usage
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision

# Copy requirements and install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (frontend, backend, models, assets)
COPY . .

# Render dynamically sets PORT via environment variable
ENV PORT=10000
EXPOSE 10000

CMD ["python", "backend/run_server.py"]
