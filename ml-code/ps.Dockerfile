
FROM python:3.7-slim
WORKDIR /usr/src/app

COPY ./for_all/ ./
RUN pip install --no-cache-dir -r requirements.txt \
&& rm -rf /var/lib/apt/lists/*
        
CMD ["python3", "./distri_ml.py", "-j", "ps", "-t", "0", "-a", "workers.txt", "-s", "192.168.4.21:4000", "-m", "SGD"]