#!/bin/bash

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--num-of-workers)
            NUMOFW="$2"
            shift
            shift
            ;;
        -*) # Invalid option
            echo "ERRor: Invalid option"
            exit;;
    esac
done

echo "Updating images!"
for ((i=0; i < NUMOFW; i++));
do
	docker rmi proj:work$((i+1))
	docker build -f work$((i+1)).Dockerfile -t proj:work$((i+1)) .
done

docker rmi proj:ps
docker build -f ps.Dockerfile -t proj:ps .

