# Machine Learning Algorithm Emulation in Omnet++
Emulation of a machine learning algorithm running on a simulated network in Omnet++.
As a part of the development of modern datacenters networks, in-network computing via smart switches has created some buzz over the past years. Advanced switches with great capabilities of computing and aggregation with one big purpose, minimizing network utilization with a bounder in-network computing. 

Like every advanced device, its capabilities and special qualities come with a price. Therefore, to build an optimal network, the amount of these advanced switches should be taken into consideration – aiming to be as reduce traffic efficiently. 

This is where our project takes place, on our research we would like to simulate various applications of distributed machine learning methods using the capabilities of smart switches and their optimal placement compared to ordinary distributed networks.
That is because in distributed machine learning since workers share their results with other workers, the amount of data that is transferred between them across the network could overload the network, therefore we are trying to prove a better utilization using our environment to test the real performance of the smart switches in the distributed manner.
According to the studies (which will be mentioned later on), we expect our simulation to prove the smart switched network (with the optimal placement) is indeed a beneficial factor to increase our distributed network utilization.

<sub>Keywords: distributed machine learning, aggregations, smart switch, simulation, network utilization.</sub>


<img src="https://github.com/SeanEti/ML-Omnetpp-emulation/blob/master/Project_base_topology.png" width="797" height="727" class="center"/>

## Run Instructions
1) Download all the correct versions
2) clone the git depository
3) create a new OMNeT++ project
4)`right click Project > properties > Project references > tick the inet project`

5) build docker images from dockerfiles in 'ml-codes' folder with the next commands:
`
./update_image.sh
`

#### after downloading everything
5) Go to {OMNET folder}
6) Enter in terminal: 'source setenv'
7) Go to {OMNET folder}/samples/inet4.5/
8) Enter in terminal: 'source setenv'
9) Go to scripts folder in the downloaded depository
10) Run the setup.sh script by entering in terminal: 
`
./setup.sh -n <number of workers to run>
`
11) Go to the copied 'proj' folder in the {OMNET}/samples/inet/showcases/emulation/
12) Run the 'run.sh' script by entering in the terminal: 
`
./run.sh
`

#### after stopping the simulation
13) Run the 'teardown.sh' script in the scripts folder in the depository

##  Versions:
    Running on Ubuntu 20.04.4
    Omnet++ 6.0.1
    INET Framework 4.5
    Docker 20.10.12
    Python 3.9
    Pytorch 2.0
    

