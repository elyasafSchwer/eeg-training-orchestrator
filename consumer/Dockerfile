FROM python:3.13-slim

WORKDIR /app

RUN pip install uv
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

COPY visit_db.json .
COPY consumer.py .

CMD ["uv", "run", "python", "-u", "consumer.py"]