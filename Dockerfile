# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project files
COPY . .

# Expose ports for Jupyter Lab
EXPOSE 8888

# Default command to run the pipeline
CMD ["python", "src/pipeline_tweet_classifier.py"]