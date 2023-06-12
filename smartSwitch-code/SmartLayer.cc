#include "SmartLayer.h"

namespace inet{

Define_Module(SmartLayer);

void SmartLayer::initialize()
{
    this->filtered = 0;
    this->num_of_forwarded_messages = 0;

    this->is_root = (getParentModule()->getIndex() == 0);
    this->is_smart = bool(par("is_smart"));
    this->num_of_messages_to_expect = int(par("num_of_messages_to_expect"));
    this->num_of_received_workers = 0;
    this->got_msg_to_save = false;
}

void SmartLayer::handleMessage(cMessage* msg)
{
    std::ofstream debugfile("/home/ubuntu/Desktop/project/logs/debug.txt", std::ios_base::out | std::ios_base::app);
    debugfile << getParentModule()->getIndex() <<": Received a new message! <<<<<<<<<<<<\n";
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
        int lower_son_host = 2*(getParentModule()->getIndex()) + 1 - (std::pow(2, int(par("height"))) - 1) + 1;

        debugfile << getParentModule()->getIndex() << ": received UDP from " << sourceAddress << "| from gate " << gate_num <<" \n";
//        debugfile << getParentModule()->getIndex() << ": " << msg->getName() << "\n";

        // check if sign from host to get the model
        if(std::strstr(msg->getName(), "UDP") && (ss.str() == "02-42-C0-A8-00-42")){
            debugfile << getParentModule()->getIndex() << ": FILTERED FROM DOCKER : " << sourceAddress << "\n";
            if (!got_msg_to_save){  // receive first udp packet from host for signaling later
                save_to_send = msg->dup();
                got_msg_to_save = true;
            }
            else{
                // TODO add a smart component for an edge switch
                debugfile << getParentModule()->getIndex() << ": Not first UDP" << "\n";
                StateDict *sd = new StateDict("state dict");
                filtered++;

                int sender_num;
                if(gate_num != 0 && !is_root){  // if sender is a worker, need to read state dict and send it up
                    sender_num = getParentModule()->getIndex() + 2 - std::pow(2, int(par("height")));
                    debugfile << "Recieved signal to send state dict from " << sender_num << "\n";

                    //  read state dict from file
                    std::string worker_path = "/home/ubuntu/Desktop/project/dicts_to_text/worker"+std::to_string(sender_num)+".txt";
                    debugfile << getParentModule()->getIndex() << ": reading from " << worker_path << "\n";
                    std::vector<std::pair<std::string, std::vector<double>>> state_dict = readStateDictFromWorkerFile(worker_path);
                    debugfile << getParentModule()->getIndex() << ": sending vector of tensors with size: " << state_dict.size() << "\n";
                    sd->setDict(state_dict);
                    sd->setSrc(ss.str());
                }

                forward(sd, is_root);
            }
        }
    }
    else{   // message is our custom dictionary

        debugfile << getParentModule()->getIndex() << ": it is our custom msg| from gate: " << gate_num << "\n";

        // need to check if this switch is smart or not and handle the message accordingly
        if(gate_num == 0){  // message from server - need to broadcast to workers
            debugfile << getParentModule()->getIndex() << ": message is from server\n";
            int height = par("height");
            if(getParentModule()->getIndex() >= std::pow(2, height)-1){    // is this a leaf switch?
                for(int i=1;i<gateSize("gate");i++){ // send UDP packets to hosts as signals that the file is ready to be read
                    debugfile << getParentModule()->getIndex() << ": sending signal to hosts\n";
                    num_of_forwarded_messages++;
                    send(save_to_send->dup(), "gate$o", i);
                }
            }
            else{    // not leaf node
                debugfile << getParentModule()->getIndex() << ": not a leaf so broadcasting downstream\n";
                StateDict *new_sd_msg = new StateDict("new state dict");
                forward(new_sd_msg, true);
            }
        }
        else{ // sender is a worker
            debugfile << getParentModule()->getIndex() << ": message is from worker| gate : " << gate_num << "\n";
            if(is_smart || is_root){   // this is a smart switch
//                // collect received state dictionary
                debugfile << getParentModule()->getIndex() << ": storing dictionary...\n";
                stored_dicts.push_back(new_msg->getDict());
                num_of_received_workers++;
                debugfile << getParentModule()->getIndex() << ": received " << num_of_received_workers << "/" << num_of_messages_to_expect << "\n";
                if(num_of_messages_to_expect == num_of_received_workers){ // received all messages from all the workers underneath
                    num_of_received_workers = 0;    // reset received messages

                    if(is_smart){
                        // Aggregation of all the stored state dictionaries
                        std::ofstream agg_times_file("/home/ubuntu/Desktop/project/logs/aggregation_times.txt", std::ios_base::out | std::ios_base::app);
                        auto start_agg_time = std::chrono::high_resolution_clock::now();
                        std::vector<std::pair<std::string, std::vector<double>>> new_state_dict = aggregateDict();
                        auto finish_agg_time = std::chrono::high_resolution_clock::now();
                        auto agg_time_s = std::chrono::duration_cast<std::chrono::milliseconds>(start_agg_time - finish_agg_time);
                        agg_times_file << "Switch-" << getParentModule()->getIndex() << ": aggregation time is " << agg_time_s.count() << "s\n";
                        agg_times_file.close();

                        stored_dicts.clear();

                        if(!is_root){
                            StateDict *sd = new StateDict("state dict");
                            sd->setDict(new_state_dict);
                            forward(sd, false);
                        }
                        else{   // a smart root switch
                            std::string root_switch_path = "/home/ubuntu/Desktop/project/dicts_to_text/root_switch/dict.txt";
                            writeStateDictToFile(root_switch_path, new_state_dict);

                            // send UDP packet to server as a signal that the file is ready to be sent
                            num_of_forwarded_messages++;
                            send(save_to_send->dup(), "gate$o", 0);
                        }
                    }
                    else{   // root but not smart
                        // write all received state dictionaries to text files
                        debugfile << getParentModule()->getIndex() << ": writing to dict* files...\n";
                        int j = 0;
                        if(stored_dicts.size() > 0){
                            for(const auto& dict : stored_dicts){
                                std::string root_switch_path = "/home/ubuntu/Desktop/project/dicts_to_text/root_switch/dict"+std::to_string(j)+".txt";
                                debugfile << getParentModule()->getIndex() << ": trying to write to " << root_switch_path << "| size: " << dict.size() << "\n";
                                writeStateDictToFile(root_switch_path, dict);
                                j++;
                            }
                            stored_dicts.clear();
                            debugfile << getParentModule()->getIndex() << ": sending UDP signal..." << save_to_send << "\n";
                            num_of_forwarded_messages++;
                            send(save_to_send->dup(), "gate$o", 0);
                        }
                    }
                }
            }
            else{   // regular switch should just forward the message up
                debugfile << getParentModule()->getIndex() << ": forwarding up\n";
                forward(new_msg, false);
            }
        } // worker sent
    } // msg is statedict
    debugfile.close();
}


void SmartLayer::finish(){
    EV_INFO << "============ Switch-" << getParentModule()->getIndex() << " ============" << endl;
    EV_INFO << "Number of signal UDP packets received from the hosts: " << filtered << " UDPs caught" << endl;
    EV_INFO << "number of forwarded messages: " << num_of_forwarded_messages << endl;
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


void SmartLayer::forward(StateDict *sd, bool to_bcast){
//    std::ofstream debugfile2("/home/ubuntu/Desktop/project/logs/debug.txt", std::ios_base::app);
//    debugfile2 << getParentModule()->getIndex() << ": Im in forward!\n";
    if(to_bcast){
        for(int i=1; i < gateSize("gate"); i++){
//            debugfile2 << getParentModule()->getIndex() << ": sending to gate " << i << "\n";
            num_of_forwarded_messages++;
            send(sd->dup(), "gate$o",i);
        }
        delete sd;
    }
    else{
//        debugfile2 << getParentModule()->getIndex() << ": sending up to gate 0\n";
        num_of_forwarded_messages++;
        send(sd, "gate$o", 0);
    }
//    debugfile2.close();
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

    // average the results
    double scaleFactor = 1.0 / stored_dicts.size();
    for (auto& pair : aggregated_dict){
        std::vector<double>& values = pair.second;
        for(double& value : values)
            value *= scaleFactor;
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
