#include "SmartLayer.h"

namespace inet{

Define_Module(SmartLayer);

void SmartLayer::initialize()
{
    this->debug = bool(par("debug"));

    this->filtered = 0;
    this->num_of_forwarded_messages = 0;
    this->down = 0;
    this->up = 0;

    this->is_root = (getParentModule()->getIndex() == 0);
    this->is_smart = bool(par("is_smart"));
    this->num_of_messages_to_expect = int(par("num_of_messages_to_expect"));
    this->first_host_in_switch = int(par("first_host_in_switch"));
    this->got_msg_to_save = false;
}


void SmartLayer::handleMessage(cMessage* msg)
{
    std::ofstream debugfile;
    if(debug){
        debugfile.open("/home/ubuntu/Desktop/project/logs/debug.txt", std::ios_base::out | std::ios_base::app);
        debugfile << getParentModule()->getIndex() <<": Received a new message! <<<<<<<<<<<<\n";
    }

    if(msg->isSelfMessage()){   // check if this is a delayed message
        if(debug) debugfile << getParentModule()->getIndex() << ": Received a delayed message\n";
        StateDict *delayed_msg = check_and_cast<StateDict*>(msg);
        cChannel *channel = gate("gate$o", delayed_msg->getOutgate())->getTransmissionChannel();
        if(channel->isBusy()){
            scheduleAt(channel->getTransmissionFinishTime(), delayed_msg);
        }
        else{
            send(delayed_msg, "gate$o", delayed_msg->getOutgate());
        }
    }
    else{   // new message
        int gate_num = msg->getArrivalGate()->getIndex();
        StateDict *new_msg = dynamic_cast<StateDict*>(msg);

        if(!new_msg){   // check if the received message is not our custom message
            delete new_msg;
            // decapsulate received packet to check for src IP address
            physicallayer::EthernetSignal* es = check_and_cast<physicallayer::EthernetSignal*>(msg->dup());
            auto packet = check_and_cast<Packet *>(es->decapsulate());
            auto phyHeader = packet->popAtFront<physicallayer::EthernetPhyHeader>();
            const auto& frame = packet->popAtFront<EthernetMacHeader>();
            auto sourceAddress = frame->getSrc();

            std::stringstream ss;
            ss << sourceAddress;

            if(debug) debugfile << getParentModule()->getIndex() << ": received UDP from " << sourceAddress << "| from gate " << gate_num <<" \n";

            // check if sign from host to get the model
            if(std::strstr(msg->getName(), "UDP") && (ss.str() == "02-42-C0-A8-00-42")){
                if(debug) debugfile << getParentModule()->getIndex() << ": FILTERED FROM DOCKER : " << sourceAddress << "\n";
                if (!got_msg_to_save){  // receive first udp packet from host for signaling later
                    save_to_send = msg->dup();
                    got_msg_to_save = true;
                }
                else{
                    if(debug) debugfile << getParentModule()->getIndex() << ": Not first UDP" << "\n";
                    StateDict *sd = new StateDict("state dict");
                    sd->setByteLength(800000);
                    filtered++;

                    int sender_num;
                    if(gate_num != 0 && !is_root){  // if sender is a worker, need to read state dict and send it up
                        sender_num = first_host_in_switch + gate_num - 1;
                        if(debug) debugfile << "Recieved signal to send state dict from " << sender_num << "\n";

                        //  read state dict from file
                        std::string worker_path = "/home/ubuntu/Desktop/project/dicts_to_text/worker"+std::to_string(sender_num)+".txt";
                        if(debug) debugfile << getParentModule()->getIndex() << ": reading from " << worker_path << "\n";
                        std::vector<std::pair<std::string, std::vector<double>>> state_dict = readStateDictFromWorkerFile(worker_path);
                        if(debug) debugfile << getParentModule()->getIndex() << ": sending vector of tensors with size: " << state_dict.size() << "\n";
                        if(is_smart){   // check if leafd is smart and need to wait for more hosts
                            stored_dicts.push_back(state_dict);
                            if(num_of_messages_to_expect == stored_dicts.size()){ // received all messages from all the workers underneath
                                sd->setDict(aggregateDict());
                                stored_dicts.clear();
                                forward(sd, false);
                            }
                        }
                        else{   // not a smart leaf
                            sd->setDict(state_dict);
                            sd->setSrc(ss.str());
                            forward(sd, false);
                        }
                    }
                    else
                        forward(sd, is_root);
                }
            }
        }
        else{   // message is our custom dictionary

            if(debug) debugfile << getParentModule()->getIndex() << ": it is our custom msg| from gate: " << gate_num << "\n";

            // need to check if this switch is smart or not and handle the message accordingly
            if(gate_num == 0){  // message from server - need to broadcast to workers
                if(debug) debugfile << getParentModule()->getIndex() << ": message is from server\n";
                int height = par("height");
                if(getParentModule()->getIndex() >= std::pow(2, height)-1){    // is this a leaf switch?
                    // send UDP packets to hosts as signals that the file is ready to be read
                    if(debug) debugfile << getParentModule()->getIndex() << ": sending signal to hosts\n";
                    forward(save_to_send->dup(), true);
                }
                else{    // not leaf node
                    if(debug) debugfile << getParentModule()->getIndex() << ": not a leaf so broadcasting downstream\n";
                    StateDict *new_sd_msg = new StateDict("new state dict");
                    new_sd_msg->setByteLength(800000);
                    forward(new_sd_msg, true);
                }
            }
            else{ // sender is a worker
                if(debug) debugfile << getParentModule()->getIndex() << ": message is from worker| gate : " << gate_num << "\n";
                if(is_smart || is_root){   // this is a smart switch
                    // collect received state dictionary
                    if(debug) debugfile << getParentModule()->getIndex() << ": storing dictionary...\n";
                    stored_dicts.push_back(new_msg->getDict());
                    if(debug) debugfile << getParentModule()->getIndex() << ": received " << stored_dicts.size() << "/" << num_of_messages_to_expect << "\n";
                    if(num_of_messages_to_expect == stored_dicts.size()){ // received all messages from all the workers underneath
                        if(is_smart){
                            // Aggregation of all the stored state dictionaries
                            std::vector<std::pair<std::string, std::vector<double>>> new_state_dict = aggregateDict();
                            stored_dicts.clear();

                            if(!is_root){
                                StateDict *sd = new StateDict("state dict");
                                sd->setDict(new_state_dict);
                                sd->setByteLength(800000);
                                forward(sd, false);
                            }
                            else{   // a smart root switch
                                std::string root_switch_path = "/home/ubuntu/Desktop/project/dicts_to_text/root_switch/dict.txt";
                                writeStateDictToFile(root_switch_path, new_state_dict);

                                // send UDP packet to server as a signal that the file is ready to be sent
    //                            up++;
                                forward(save_to_send->dup(), false);
                            }
                        }
                        else{   // root but not smart
                            // write all received state dictionaries to text files
                            if(debug) debugfile << getParentModule()->getIndex() << ": writing to dict* files...\n";
                            int j = 0;
                            if(stored_dicts.size() > 0){
                                for(const auto& dict : stored_dicts){
                                    std::string root_switch_path = "/home/ubuntu/Desktop/project/dicts_to_text/root_switch/dict"+std::to_string(j)+".txt";
                                    if(debug) debugfile << getParentModule()->getIndex() << ": trying to write to " << root_switch_path << "| size: " << dict.size() << "\n";
                                    writeStateDictToFile(root_switch_path, dict);
                                    j++;
                                }
                                stored_dicts.clear();
                                if(debug) debugfile << getParentModule()->getIndex() << ": sending UDP signal..." << save_to_send << "\n";
                                up++;
                                forward(save_to_send->dup(), false);
                            }
                        }
                    }
                }
                else{   // regular switch should just forward the message up
                    if(debug) debugfile << getParentModule()->getIndex() << ": forwarding up\n";
                    forward(new_msg, false);
                }
            } // worker sent
        } // msg is statedict
    }


    if(debug) debugfile.close();
}


void SmartLayer::finish(){
    EV_INFO << "============ Switch-" << getParentModule()->getIndex() << " ============" << endl;
    EV_INFO << "Number of signal UDP packets received from the hosts: " << filtered << " UDPs caught from hosts" << endl;
    EV_INFO << "Number of messages going to the server: " << up << endl;
    EV_INFO << "Number of messages from server to workers: " << down << endl;
    EV_INFO << "Is this switch smart? " << is_smart << endl;
}

/* =========================================================================================================
 *                                           Helper Functions
 * =========================================================================================================*/
void SmartLayer::tokenize(std::string const &str, const char delim, std::vector<std::string> &out){
    /* This function gets a string, a char and a vector
     * splits the string by the delim and each item pushes to the vector
     */
    std::stringstream ss(str);
    std::string s;
    while(std::getline(ss, s, delim)){
        out.push_back(s);
    }
}


void SmartLayer::forward(cMessage *sd, bool to_bcast){
    cChannel* channel;
    if(to_bcast){
        for(int i=1; i < gateSize("gate"); i++){
            channel = gate("gate$o", i)->getTransmissionChannel();
            down++;
            if(channel->isBusy()){
                StateDict* temp_sd = check_and_cast<StateDict*>(sd);
                temp_sd->setOutgate(i);
                scheduleAt(channel->getTransmissionFinishTime(), temp_sd);
//                sendDelayed(sd->dup(), channel->getTransmissionFinishTime() - simTime(), "gate$o", i);
            }
            else send(sd->dup(), "gate$o", i);
        }
        delete sd;
    }
    else{
        channel = gate("gate$o", 0)->getTransmissionChannel();
        up++;
        if(channel->isBusy()){
            StateDict* temp_sd = check_and_cast<StateDict*>(sd);
            temp_sd->setOutgate(0);
            scheduleAt(channel->getTransmissionFinishTime(), sd);  // sendDelayed(sd, channel->getTransmissionFinishTime() - simTime(), "gate$o", 0);
        }
        else send(sd, "gate$o", 0);
    }
}


std::vector<std::pair<std::string, std::vector<double>>> SmartLayer::readStateDictFromWorkerFile(std::string filename){
    std::ifstream myFile(filename);
    if(!myFile){
        exit(-1);
    }
    std::string line;
    std::vector<std::pair<std::string, std::vector<double>>> state_dict;
    std::string param_name;

    while(getline(myFile, line, '\n')){    //read line by line
        if(line.substr(0, line.find(" ")) == "name"){
            std::vector<std::string> out;
            tokenize(line, ' ', out);
            param_name = out[1];    // save parameter name
        }
        else{   // read looong list of doubles (this is the flattened tensor)
            std::stringstream ss;
            std::vector<double> tensor_list;
            ss << line;
            double number;
            while (ss >> number)
                tensor_list.push_back(number);

            state_dict.push_back({param_name, tensor_list});   // insert parameter name and its value to the state dictionary
        }
    }
    myFile.close();

    return state_dict;
}


void SmartLayer::writeStateDictToFile(std::string filename, std::vector<std::pair<std::string, std::vector<double>>> state_dict){
    /*
     * receives a state dictionary and writes it to the given file path
     * the syntax of the file is:
     * for each parameter in the state dict there are two lines:
     * name <param name>
     * <flattened tensor(all the numbers are split by a space)>
     */
    std::ofstream fout(filename, std::ios_base::out);
    for(auto param : state_dict){
        fout << "name " << param.first << "\n";
        for(auto const& grad : param.second){
            fout << std::setprecision(20) << grad << " ";
        }
        fout << endl;
    }
}


std::vector<std::pair<std::string, std::vector<double>>> SmartLayer::aggregateDict(){
    /* returns a new dictionary where the values of the parameters are averaged between all the stored dictionaries */

    std::vector<std::pair<std::string, std::vector<double>>> aggregated_dict(stored_dicts[0].begin(), stored_dicts[0].end());

    for (int i = 1; i < stored_dicts.size(); i++){
        for(const auto& pair : stored_dicts[i]){ // iterate over the parameters in the current state dict
            std::string param_name = pair.first;
            const std::vector<double>& values = pair.second;
            auto it = std::find_if(aggregated_dict.begin(), aggregated_dict.end(), [&param_name](std::pair<std::string, std::vector<double>> i) { return i.first == param_name; });
            std::transform(it->second.begin(), it->second.end(), values.begin(), it->second.begin(), std::plus<double>());
        }
    }
    return aggregated_dict;  // return result
}


std::string SmartLayer::createMac(int i){
    std::stringstream ss;
    ss << std::uppercase << std::setfill('0') << std::setw(2) << std::hex << i;

    std::string mac = "02-42-C0-A8-" + ss.str() + "-15";
    return mac;
}

} //namespace
