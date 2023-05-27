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

for ((i=0; i <= NUMOFW; i++));
do
    sudo ip tuntap add mode tap dev tap$i
    sudo ip addr add 192.168.$i.20/24 dev tap$i   # assign IP addresses to interfaces
    sudo ip link set dev tap$i up                 # bring up interface
    # create macvlans to connect docker containers to omnet++ network
    if [ $i = 0 ];
    then
        sudo docker network  create -d macvlan --subnet=192.168.$i.0/24 --gateway=192.168.$i.99 -o parent=tap$i ps-net
    else
        sudo docker network  create -d macvlan --subnet=192.168.$i.0/24 --gateway=192.168.$i.99 -o parent=tap$i work$i-net
        # writing ip of worker into file
        if [ $i = 1 ]; then
            printf 192.168.$i.21:$((49152+$i)) > ../ml-code/cpy_from/workers.txt
        else
            printf ,192.168.$i.21:$((49152+$i)) >> ../ml-code/cpy_from/workers.txt
        fi
    fi
done

# write xml file to configure interfaces of network
path_to_config="/home/ubuntu/Downloads/omnetpp-6.0.1/samples/inet4.5/showcases/emulation/proj/ifaces.xml"
printf "<config>\n" > $path_to_config
printf "<interface hosts='router[0]' names='eth0' address='192.168.0.99' netmask='255.255.255.0'/>\n" >> $path_to_config
printf "<interface hosts='router[0]' names='eth1' address='10.0.0.0' netmask='255.255.255.0'/>\n" >> $path_to_config
for ((i=1; i <= NUMOFW; i++));
do
     printf "<interface hosts='router[$i]' names='eth0' address='10.0.0.$i' netmask='255.255.255.0'/>\n" >> $path_to_config
     printf "<interface hosts='router[$i]' names='eth1' address='192.168.$((2*i)).99' netmask='255.255.255.0'/>\n" >> $path_to_config
     printf "<interface hosts='router[$i]' names='eth2' address='192.168.$((2*i-1)).99' netmask='255.255.255.0'/>\n" >> $path_to_config
done
printf "</config>" >> $path_to_config

echo "Setup is complete!"