# Use a PyTorch-compatible base image
FROM docker.arvancloud.ir/python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --trusted-host https://mirror-pypi.runflare.com -i https://mirror-pypi.runflare.com/simple/ --no-cache-dir -r requirements.txt --timeout=100

# Copy the source code
COPY . .