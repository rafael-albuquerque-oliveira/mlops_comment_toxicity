# Use official Python image
FROM python:3.10

# Set working directory inside the container
WORKDIR /app

# Copy necessary files
COPY requirements.txt .
COPY src/ src/
COPY models/ models/
COPY data/ data
COPY src/predict.py src/

# Install dependencies with a faster mirror
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir \
  --index-url https://pypi.org/simple \
  --timeout 300 \
  transformers==4.36.0 \
  huggingface_hub==0.20.3 \
  sentencepiece==0.2.0 \
  joblib==1.3.2 \
  scikit-learn==1.6.1 \
  gunicorn==20.1.0 \
  flask==3.0.0 \
  tensorflow==2.18.0 \
  pandas \
  torch==2.2.0 \
  tensorflow-cpu

# Install PyTorch CPU version separately
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

ENV PYTHONPATH="/app/src"

# Download and cache the Hugging Face model before deployment
RUN python -c "from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
  AutoModelForSeq2SeqLM.from_pretrained('unicamp-dl/translation-pt-en-t5', use_auth_token='$HF_TOKEN'); \
  AutoTokenizer.from_pretrained('unicamp-dl/translation-pt-en-t5', use_auth_token='$HF_TOKEN')"

# Expose the application port
EXPOSE 8080

# Ensure TensorFlow and PyTorch do NOT use GPU
ENV CUDA_VISIBLE_DEVICES="-1"
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_CPP_MIN_LOG_LEVEL="2"

# Run the API using Gunicorn
CMD exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 src.api:app
