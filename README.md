# Machine Learning Algorithm Emulation in Omnet++
Emulation of a machine learning algorithm running on a simulated network in Omnet++.
As a part of the development of modern datacenters networks, in-network computing via smart switches has created some buzz over the past years. Advanced switches with great capabilities of computing and aggregation with one big purpose, minimizing network utilization with a bounder in-network computing. 

Like every advanced device, its capabilities and special qualities come with a price. Therefore, to build an optimal network, the amount of these advanced switches should be taken into consideration – aiming to be as reduce traffic efficiently. 

This is where our project takes place, on our research we would like to simulate various applications of distributed machine learning methods using the capabilities of smart switches and their optimal placement compared to ordinary distributed networks.
That is because in distributed machine learning since workers share their results with other workers, the amount of data that is transferred between them across the network could overload the network, therefore we are trying to prove a better utilization using our environment to test the real performance of the smart switches in the distributed manner.
According to the studies (which will be mentioned later on), we expect our simulation to prove the smart switched network (with the optimal placement) is indeed a beneficial factor to increase our distributed network utilization.

<sub>Keywords: distributed machine learning, aggregations, smart switch, simulation, network utilization.</sub>


<img src="https://github.com/SeanEti/ML-Omnetpp-emulation/blob/master/topology.jpeg" width="780" height="1110" class="center"/>

## Run Instructions
1) Download all the correct versions
2) Download the git depository
3) Copy the 'proj' folder to {path to main omnet folder}/samples/inet/showcases/emulation/
4) build docker images from dockerfiles in 'ml-codes' folder with the next commands:
    sudo docker build -t proj:ps -f ps.Dockerfile .
    sudo docker build -t proj:work{worker_num} -f work{worker_num}.Dockerfile .

#### after downloading everything
5) Go to {OMNET folder}
6) Enter in terminal: 'source setenv'
7) Go to {OMNET folder}/samples/inet/
8) Enter in terminal: 'source setenv'
9) Go to scripts folder in the downloaded depository
10) Run the setup script
11) Go to the copied 'proj' folder in the {OMNET}/samples/inet/showcases/emulation/
12) Run the 'run.sh' script

#### after stopping the simulation
13) Run the 'teardown.sh' script in the scripts folder in the depository

##  Versions:
    Running on Ubuntu 20.04.4
    Omnet++ 6.0.1
    INET Framework 4.3.9
    Docker 20.10.12
    Python 3.7
    TensorFlow 1.15.0
    

