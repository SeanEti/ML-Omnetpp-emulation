while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--num-of-workers)
            NUMOFW="$2"
            shift
            shift
            ;;
        -*) # Invalid option
            echo "ERRor: Invalid option."
            echo "Please use -n and number of workers"
            exit;;
    esac
done

# start containers
sudo docker run --name ps --net ps-net --ip 192.168.0.21 -p 49152:49152 -dt proj:img
sleep 2s
for ((i=1; i <= NUMOFW; i++));
do
    sudo docker run --name worker$i --net work$i-net --ip 192.168.$i.21 -p $((i+49152)):$((i+49152)) --env JOB=worker --env IDX=$((i-1)) -dt proj:img
done

# start simulation
inet -u Cmdenv -f  omnetpp.ini

# kill child processes
trap 'kill $(jobs -pr)' SIGINT SIGTERM EXIT
