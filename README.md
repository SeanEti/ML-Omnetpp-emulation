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
#### Setup
1) Download all the correct versions
2) clone the git depository
3) build docker images from dockerfiles in 'ml-codes' folder with the next commands:
`
./update_images_all.sh
`
#### after downloading everything
4) Go to {OMNET folder}
5) Enter in terminal: 'source setenv'
6) Go to {OMNET folder}/samples/inet4.5/
7) Enter in terminal: 'source setenv'
8) Now enter 'omnetpp' to open the OMNeT++ GUI
9) create a new OMNeT++ project

10) `right click Project > properties > Project references > tick the inet project` and save
11) Now copy the project file from the 'proj' folder, in the cloned repository, into th enew project created, replacing everything nedded
12) build the project

<sub>To change the network size, loads, and placements of smat switches you need to go into the omnet.ini file in the new project</sub>
#### RUN
13) Go to scripts folder in the downloaded depository
14) Run the setup.sh script by entering in terminal: 
`
./setup.sh -n <number of workers to run>
`
15) Go to the src folder in the OMNeT++ project you created
16) Run the './run' script by entering in the terminal: 
`
./run -c <config-name>
`
17) and then back in the 'scripts' folder type:
`
./run.sh -n <number of workers> -e <number of epochs to run>
`
#### When wanting to close the simulation
18) stop all running scripts using CTRL+C
19) Run the './teardown.sh -n <number of workers>' script in the 'scripts' folder in the depository

#### Results and logs:
The results are stored in the 'dicts_to_text' folder of the repository, and logs are stored in the 'logs' folder
    
##  Versions:
    Running on Ubuntu 20.04.4
    Omnet++ 6.0.1
    INET Framework 4.5
    Docker 20.10.12
    Python 3.9
    Pytorch 2.0
    

