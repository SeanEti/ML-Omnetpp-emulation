# start containers
# sudo docker run --name ps --net ps-net --ip 192.168.4.21 -p 4000:4000 -dt proj:ps
# sleep 2s
# sudo docker run --name worker1 --net work1-net --ip 192.168.2.21 -p 4545:4545 -dt proj:work1
# sleep 2s
# sudo docker run --name worker2 --net work2-net --ip 192.168.3.21 -p 4444:4444 -dt proj:work2
# sleep 2s

# start simulation
inet -u Cmdenv -f  omnetpp.ini

# kill child processes
trap 'kill $(jobs -pr)' SIGINT SIGTERM EXIT
