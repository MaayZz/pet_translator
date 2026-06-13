FROM python:3.11-slim

# Install system dependencies required for llama-cpp-python compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY . .

# Set environment variable to suppress warnings
ENV PYTHONUNBUFFERED=1

# By default, run a simple test of the generator to prove the container works
CMD ["python", "-m", "llm.generator"]
