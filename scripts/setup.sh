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

# create TAP interfaces
echo "Setting up taps and networks..."

# sudo iptables -P INPUT DROP

for ((i=0; i <= NUMOFW; i++));
do
    sudo ip tuntap add mode tap dev tap$i
    sudo ip addr add 192.168.$i.20/24 dev tap$i   # assign IP addresses to interfaces
    sudo ip link set dev tap$i up                 # bring up interface
    # sudo iptables -t nat -A POSTROUTING -o tap$i -p udp --dport 49152 -j SNAT --to 192.168.$i.21:49152     # route packets from TAP to correlating docker(for bcast)
    # sudo iptables -A INPUT -i tap$i -p udp --dport 1234 -j ACCEPT
    # create macvlans to connect docker containers to omnet++ network
    if [ $i = 0 ];
    then
        sudo docker network  create -d macvlan --subnet=192.168.$i.0/24 --gateway=192.168.$i.99 -o parent=tap$i ps-net
    else
        sudo docker network  create -d macvlan --subnet=192.168.$i.0/24 --gateway=192.168.$i.99 -o parent=tap$i work$i-net
    fi
done

echo "Setup is complete!"