FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY hst_agent.py app.py ./

EXPOSE 3003

CMD ["streamlit","run","app.py","--server.port=3003"]



