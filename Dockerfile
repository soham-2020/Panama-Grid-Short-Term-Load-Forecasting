# Panama Grid — Next-Hour Load Forecasting
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY analysis.py .

ENV MPLBACKEND=Agg

ENTRYPOINT ["python", "analysis.py"]
CMD ["data/continuous_dataset.csv"]