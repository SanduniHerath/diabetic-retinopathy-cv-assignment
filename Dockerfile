# Use official lightweight Python image
FROM python:3.11-slim

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, model checkpoint, and bundled samples
COPY . .

# Hugging Face Spaces default port
ENV PORT=7860
ENV HOST=0.0.0.0
EXPOSE 7860

# Launch NiceGUI Clinical App
CMD ["python", "app/main.py"]
