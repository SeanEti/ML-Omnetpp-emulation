FROM python:3.9-slim
ENV JOB "ps"
ENV IDX "0"
ENV MODEL "CNN"
ENV EPOCHS "10"
ENV NUMOFWORKERS "8"
WORKDIR /usr/src/app
COPY ./cpy_from/ ./
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt \
&& rm -rf /var/lib/apt/lists/*
CMD python3 ./distri_ml_ys.py -j ${JOB} -t ${IDX} -e ${EPOCHS} -n ${NUMOFWORKERS} -m ${MODEL}
