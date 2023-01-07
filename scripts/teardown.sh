# destroy TAP interfaces
sudo ip tuntap del mode tap dev tapa
sudo ip tuntap del mode tap dev tapb
sudo ip tuntap del mode tap dev tapc

# stop running containers - they will delete right after stopping due to the -rm flag
sudo docker stop ps
sudo docker stop worker1
sudo docker stop worker2

# destroy vlan networks
sudo docker network rm ps-net
sudo docker network rm work1-net
sudo docker network rm work2-net

