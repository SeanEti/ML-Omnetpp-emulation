# destroy TAP interfaces
sudo ip tuntap del mode tap dev tapa
sudo ip tuntap del mode tap dev tapb
sudo ip tuntap del mode tap dev tapc
sudo ip tuntap del mode tap dev tapd
sudo ip tuntap del mode tap dev tape

# stop running containers
sudo docker stop ps
sudo docker stop worker1
sudo docker stop worker2
sudo docker stop worker3
sudo docker stop worker4

# save container logs
sudo docker logs ps > ../logs/ps.log
sudo docker logs worker1 > ../logs/worker1.log
sudo docker logs worker2 > ../logs/worker2.log
sudo docker logs worker3 > ../logs/worker3.log
sudo docker logs worker4 > ../logs/worker4.log

# Delete containers
sudo docker rm ps worker1 worker2 worker3 worker4

# destroy vlan networks
sudo docker network rm ps-net
sudo docker network rm work1-net
sudo docker network rm work2-net
sudo docker network rm work3-net
sudo docker network rm work4-net

