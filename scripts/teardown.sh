#!/bin/bash
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
    sudo ip tuntap del mode tap dev tap$i       # delete TAP
    if [ $i = 0 ]; then
        sudo docker stop ps                     # stop parameter server container
        sudo docker logs ps > ../logs/ps.log    # save logs of server
        sudo docker rm ps                       # delete container
        sudo docker network rm ps-net           # delete container network
    else
        sudo docker stop worker$i                           # stop worker container
        sudo docker logs worker$i > ../logs/worker$i.log    # save worker logs 
        sudo docker rm worker$i                             # delete worker container
        sudo docker network rm work$i-net                   # delete his network
    fi
done

sudo iptables -F     # reset routing tables
