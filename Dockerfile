# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /workspace


# Copy only requirements.txt and install dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Do not copy source code; it will be mounted by the devcontainer

# Default command
CMD ["python"]
