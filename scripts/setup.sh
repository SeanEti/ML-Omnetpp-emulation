# create TAP interfaces
sudo ip tuntap add mode tap dev tapa
sudo ip tuntap add mode tap dev tapb
sudo ip tuntap add mode tap dev tapc

# assign IP addresses to interfaces
sudo ip addr add 192.168.2.20/24 dev tapa
sudo ip addr add 192.168.3.20/24 dev tapb
sudo ip addr add 192.168.4.20/24 dev tapc

# bring up all interfaces
sudo ip link set dev tapa up
sudo ip link set dev tapb up
sudo ip link set dev tapc up

# create macvlans to connect docker containers to omnet++ network
sudo docker network  create -d macvlan --subnet=192.168.4.0/24 --gateway=192.168.4.99 -o parent=tapc ps-net

sudo docker network  create -d macvlan --subnet=192.168.2.0/24 --gateway=192.168.2.99 -o parent=tapa work1-net

sudo docker network  create -d macvlan --subnet=192.168.3.0/24 --gateway=192.168.3.99 -o parent=tapb work2-net
