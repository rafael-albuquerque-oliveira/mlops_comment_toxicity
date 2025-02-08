FROM python:3.10

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Start the application with Gunicorn on port 8080
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "src.api:app"]
