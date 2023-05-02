# Machine Learning Algorithm Emulation in Omnet++
Emulation of a machine learning algorithm running on a simulated network in Omnet++

![Topology of the network](https://github.com/SeanEti/ML-Omnetpp-emulation/tree/master/topology.jpeg "Topology")

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
    

