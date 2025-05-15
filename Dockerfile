FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install system dependencies needed for PyAudio and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    portaudio19-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install PyYAML separately first to avoid build issues
RUN pip install --no-build-isolation PyYAML==5.4.1

# Install the rest of the requirements
RUN pip install -r requirements.txt

COPY . .

# Create a shell script to manage processes and keep container running
RUN echo '#!/bin/sh\n\
nohup python src/redissrv.py > /dev/null 2>&1 &\n\
nohup python src/restapi.py > /dev/null 2>&1 &\n\
nohup python src/tasks.py > /dev/null 2>&1 &\n\
# Keep container running by waiting for any process to exit\n\
wait' > /app/start.sh

RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
