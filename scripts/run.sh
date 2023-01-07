# start containers
sudo docker run --rm --name ps --net ps-net --ip 192.168.4.21 -p 4000:4000 -dt proj:ps
sudo docker run --rm --name worker1 --net work1-net --ip 192.168.2.21 -p 4545:4545 -dt proj:worker1
sudo docker run --rm --name worker2 --net work2-net --ip 192.168.3.21 -p 4545:4545 -dt proj:worker2

# start simulation
inet -u Cmdenv -f /home/ubuntu/Downloads/omnetpp-6.0.1/samples/inet/showcases/emulation/videostreaming/omnetpp.ini
# inet -f omnetpp.ini

# kill child processes
trap 'kill $(jobs -pr)' SIGINT SIGTERM EXIT
