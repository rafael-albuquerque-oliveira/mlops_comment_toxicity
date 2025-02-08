# Dockerfile
FROM python:3.10

# Set working directory inside the container
WORKDIR /app

# Copy necessary files
COPY src/ src/
COPY models/ models/
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install sentencepiece
RUN pip install torch torchvision torchaudio

# Run the API using Gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "src.api:app"]
