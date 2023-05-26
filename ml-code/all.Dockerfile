FROM python:3.7-slim
ENV JOB "ps"
ENV IDX "0"
ENV WORKERS "workers.txt"
ENV SERVER "192.168.4.21:4000"
ENV MODEL "CNN"
WORKDIR /usr/src/app
COPY ./cpy_from/ ./
RUN pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt --no-deps \
&& rm -rf /var/lib/apt/lists/*
CMD python3 ./distri_ml_ys.py -j ${JOB} -t ${IDX} -a ${WORKERS} -s ${SERVER} -m ${MODEL}
