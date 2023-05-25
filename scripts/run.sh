# start containers
sudo docker run --name ps --net ps-net --ip 192.168.4.21 -p 4000:4000 -dt proj:ps
sleep 2s
sudo docker run --name worker1 --net work1-net --ip 192.168.2.21 -p 4545:4545 --env JOB=worker --env IDX=0 -dt proj:work1
sleep 2s
sudo docker run --name worker2 --net work2-net --ip 192.168.3.21 -p 4444:4444 --env JOB=worker --env IDX=1 -dt proj:work2
sleep 2s
sudo docker run --name worker3 --net work3-net --ip 192.168.5.21 -p 4646:4646 --env JOB=worker --env IDX=2 -dt proj:work3
sleep 2s
sudo docker run --name worker4 --net work4-net --ip 192.168.6.21 -p 4747:4747 --env JOB=worker --env IDX=3 -dt proj:work4
sleep 2s

# start simulation
inet -u Cmdenv -f  omnetpp.ini

# kill child processes
trap 'kill $(jobs -pr)' SIGINT SIGTERM EXIT
