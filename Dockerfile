FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# 🚀 Install CPU-only torch FIRST
RUN pip install --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy project files
COPY . .
# Create necessary directories
RUN mkdir -p data chroma_db
# Expose Streamlit port
EXPOSE 8501
# Health check
HEALTHCHECK CMD curl --fail \
    http://localhost:8501/_stcore/health || exit 1
# Run Streamlit
CMD ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]