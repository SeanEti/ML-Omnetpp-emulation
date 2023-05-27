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

for ((i=0; i <= NUMOFW; i++));
do
    sudo ip tuntap del mode tap dev tap$i
    if [ $i = 0 ]; then
        sudo docker stop ps
        sudo docker logs ps > ../logs/ps.log
        sudo docker rm ps
        sudo docker network rm ps-net
    else
        sudo docker stop worker$i
        sudo docker logs worker$i > ../logs/worker$i.log
        sudo docker rm worker$i
        sudo docker network rm work$i-net
    fi
done
