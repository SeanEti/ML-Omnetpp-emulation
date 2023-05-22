to create new worker and ps dockerfiles/update them:
edit the make_docks.sh file and run it like that:

./make_docks.sh -n <number of workers here> -s <ps ip:ps port> -w <path to workers ips file>

and to build the images run:
sudo ./update_images.sh -n <number of workers>
