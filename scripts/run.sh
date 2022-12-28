# start containers

# start simulation
inet -u Cmdenv -f omnetpp.ini
# inet -f omnetpp.ini

# kill child processes
trap 'kill $(jobs -pr)' SIGINT SIGTERM EXIT
