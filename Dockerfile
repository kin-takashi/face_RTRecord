FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install only necessary packages first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# Copy application code (data excluded by .dockerignore)
COPY . .

# Create dirs if not exist (from config.py logic)
RUN mkdir -p dataset embeddings logs predata videos

# Default flexible CMD - allows running any script: docker run ... python 04_recognize.py --loop
CMD ["python", "04_recognize.py"]

