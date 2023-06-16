#!/bin/bash
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--num-of-workers)
            NUMOFW="$2"
            shift
            shift
            ;;
        -e|--epochs)
            EPOCHNUM="$2"
            shift
            shift
            ;;
        -*) # Invalid option
            echo "ERRor: Invalid option."
            echo "Please use -n and number of workers"
            exit;;
    esac
done

echo "starting containers..."
sudo docker run -v /home/ubuntu/Desktop/project/dicts_to_text:/usr/src/app/logs --env EPOCHS=$EPOCHNUM --env NUMOFWORKERS=$NUMOFW --name ps --net ps-net --ip 192.168.0.21 -p 49152:49152/udp -dt proj:img
sleep 2s
for ((i=1; i <= NUMOFW; i++));
do
    sudo docker run -v /home/ubuntu/Desktop/project/dicts_to_text:/usr/src/app/logs \
    --name worker$i \
    --net work$i-net --ip 192.168.$i.21 \
    --env JOB=worker --env IDX=$((i-1)) --env EPOCHS=$EPOCHNUM --env NUMOFWORKERS=$NUMOFW \
    -dt proj:img
done

# save cpu utilization over time
mpstat 30 30 >> ../logs/cpu.txt &

