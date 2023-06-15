FROM python:3.9-slim as base
COPY ./cpy_from/requirements.txt ./
RUN pip install --no-cache-dir --user torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cpu \
&& pip install --no-cache-dir --user -r requirements.txt \
&& rm -rf /var/lib/apt/lists/*

FROM python:3.9-slim
ENV JOB "ps"
ENV IDX "0"
ENV EPOCHS "10"
ENV NUMOFWORKERS "8"
WORKDIR /usr/src/app
COPY --from=base /root/.local /root/.local
COPY ./cpy_from/ ./
ENV PATH=/root/.local/bin:$PATH
CMD python3 ./distri_ml_ys.py -j ${JOB} -t ${IDX} -e ${EPOCHS} -n ${NUMOFWORKERS}
