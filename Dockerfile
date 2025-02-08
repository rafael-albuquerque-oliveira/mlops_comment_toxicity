FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 3000

CMD ["bentoml", "serve", "src/api_service.py:svc"]
