# create TAP interfaces
echo "Setting up taps..."
sudo ip tuntap add mode tap dev tapa
sudo ip tuntap add mode tap dev tapb
sudo ip tuntap add mode tap dev tapc
sudo ip tuntap add mode tap dev tapd
sudo ip tuntap add mode tap dev tape

# assign IP addresses to interfaces
sudo ip addr add 192.168.2.20/24 dev tapa
sudo ip addr add 192.168.3.20/24 dev tapb
sudo ip addr add 192.168.4.20/24 dev tape
sudo ip addr add 192.168.5.20/24 dev tapc
sudo ip addr add 192.168.6.20/24 dev tapd

# bring up all interfaces
sudo ip link set dev tapa up
sudo ip link set dev tapb up
sudo ip link set dev tapc up
sudo ip link set dev tapd up
sudo ip link set dev tape up

echo "Setting up docker networks..."
# create macvlans to connect docker containers to omnet++ network
sudo docker network  create -d macvlan --subnet=192.168.2.0/24 --gateway=192.168.2.99 -o parent=tapa work1-net

sudo docker network  create -d macvlan --subnet=192.168.3.0/24 --gateway=192.168.3.99 -o parent=tapb work2-net

sudo docker network  create -d macvlan --subnet=192.168.5.0/24 --gateway=192.168.5.99 -o parent=tapc work3-net

sudo docker network  create -d macvlan --subnet=192.168.6.0/24 --gateway=192.168.6.99 -o parent=tapd work4-net

sudo docker network  create -d macvlan --subnet=192.168.4.0/24 --gateway=192.168.4.99 -o parent=tape ps-net

echo "Setup is complete!"