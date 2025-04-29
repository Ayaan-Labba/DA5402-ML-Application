FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Copy model directory (will be mounted in production)
COPY model_deployment ./model_deployment

# Expose the port the app runs on
EXPOSE 8080
EXPOSE 8000

# Command to run the application
CMD ["python", "app.py"]