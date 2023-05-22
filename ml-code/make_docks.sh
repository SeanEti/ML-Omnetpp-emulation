#!/bin/bash
echo "creating Dockerfiles for workers"

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--num-of-workers)
            NUMOFW="$2"
            shift
            shift
            ;;
        -s)
            PSIP="$2"
            shift
            shift
            ;;
        -w)
            WORKERS_PATH="$2"
            shift
            shift
            ;;
        -*) # Invalid option
            echo "ERRor: Invalid option"
            exit;;
    esac
done

## creating ps dockerfile
rm ps.Dockerfile

printf "
FROM python:3.7-slim
WORKDIR /usr/src/app

COPY ./for_all/ ./
RUN pip install --no-cache-dir -r requirements.txt \\
&& rm -rf /var/lib/apt/lists/*
        
CMD [\"python3\", \"./distri_ml.py\", \"-j\", \"ps\", \"-t\", \"0\", \"-a\", \"$WORKERS_PATH\", \"-s\", \"$PSIP\", \"-m\", \"SGD\"]" > ps.Dockerfile

echo "created ps"

## creating worker docker filess
for ((i=0; i < NUMOFW; i++));
do
    rm work$((i+1)).Dockerfile
    echo "making worker $i out of $NUMOFW"

    echo "FROM python:3.7-slim" >> work$((i+1)).Dockerfile
    echo "WORKDIR /usr/src/app" >> work$((i+1)).Dockerfile
        
    echo "COPY ./for_all/ ./" >> work$((i+1)).Dockerfile
    echo "RUN pip install --no-cache-dir -r requirements.txt \\" >> work$((i+1)).Dockerfile
    echo "&& rm -rf /var/lib/apt/lists/*" >> work$((i+1)).Dockerfile

    echo "CMD [\"python3\", \"./distri_ml.py\", \"-j\", \"worker\", \"-t\", \"$i\", \"-a\", \"$WORKERS_PATH\", \"-s\", \"$PSIP\", \"-m\", \"SGD\"]" >> work$((i+1)).Dockerfile
    echo "created $i"
done