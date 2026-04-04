FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY analysis_production.py .
COPY continuous\ dataset.csv .
CMD ["python", "analysis_production.py"]
