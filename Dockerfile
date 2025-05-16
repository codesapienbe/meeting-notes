FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install system dependencies needed for PyAudio, Redis, and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    portaudio19-dev \
    python3-dev \
    redis-server \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add redis to requirements
RUN pip install redis

# Install PyYAML separately first to avoid build issues
RUN pip install --no-build-isolation PyYAML==5.4.1

# Install the rest of the requirements
RUN pip install -r requirements.txt

COPY . .

# Create necessary directories
RUN mkdir -p data temp transcripts

# Set environment variables
ENV REDIS_HOST=localhost
ENV REDIS_PORT=6379
ENV REDIS_URL=redis://localhost:6379/0

# Expose the FastAPI port
EXPOSE 8000

# Start Redis server and the application
CMD service redis-server start && \
    python -m src.tasks & \
    uvicorn src.restapi:app --host 0.0.0.0 --port 8000
