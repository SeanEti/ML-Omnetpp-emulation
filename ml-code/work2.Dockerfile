FROM python:3.7-slim
WORKDIR /usr/src/app

COPY requirements.txt ./for_all/ ./correct_distr_worker2_ver_1.py ./
RUN pip install --no-cache-dir -r requirements.txt \
&& rm -rf /var/lib/apt/lists/*

# ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
CMD ["python3", "./correct_distr_worker2_ver_1.py"]
