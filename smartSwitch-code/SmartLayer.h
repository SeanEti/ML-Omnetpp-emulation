#ifndef __INET_SMARTLAYER_H_
#define __INET_SMARTLAYER_H_

#include "inet/linklayer/ethernet/base/EthernetMacBase.h"
#include "inet/linklayer/ethernet/basic/EthernetMac.h"

#include "inet/common/ProtocolTag_m.h"
#include "inet/linklayer/common/EtherType_m.h"
#include "inet/linklayer/common/InterfaceTag_m.h"
#include "inet/linklayer/common/MacAddressTag_m.h"
#include "inet/linklayer/ethernet/common/EthernetControlFrame_m.h"
#include "inet/linklayer/ethernet/common/EthernetMacHeader_m.h"
#include "inet/physicallayer/wired/ethernet/EthernetSignal_m.h"
#include "inet/networklayer/ipv4/Ipv4Header_m.h"
#include "inet/transportlayer/udp/UdpHeader_m.h"

#include "inet/physicallayer/wired/ethernet/EthernetPhy.h"

#include "inet/common/ProtocolTag_m.h"
#include "inet/common/packet/Packet.h"
#include "inet/physicallayer/wired/ethernet/EthernetPhyHeader_m.h"
#include "inet/physicallayer/wired/ethernet/EthernetSignal_m.h"
#include "inet/linklayer/ieee8021d/relay/Ieee8021dRelay.h"

#include "inet/common/IProtocolRegistrationListener.h"
#include "inet/common/ProtocolTag_m.h"
#include "inet/linklayer/common/EtherType_m.h"
#include "inet/linklayer/common/InterfaceTag_m.h"
#include "inet/linklayer/common/MacAddressTag_m.h"
#include "inet/linklayer/common/VlanTag_m.h"
#include "inet/linklayer/common/UserPriorityTag_m.h"
#include "inet/linklayer/configurator/Ieee8021dInterfaceData.h"

#include <omnetpp.h>
#include "inet/common/packet/Packet.h"
#include "inet/common/packet/recorder/PcapReader.h"
#include "inet/physicallayer/wired/ethernet/EthernetSignal_m.h"
#include "stateDict_m.h"
#include <string.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <utility>
#include <fstream>
#include <iomanip>
#include <map>
#include <cmath>
#include <chrono>

using namespace omnetpp;

namespace inet {

class SmartLayer : public cSimpleModule
{
  protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
    virtual void finish() override;
    void tokenize(std::string const &str, const char delim, std::vector<std::string> &out);
    void forward(cMessage *sd, bool is_root);
    std::vector<std::pair<std::string, std::vector<double>>> readStateDictFromWorkerFile(std::string filename);
    void writeStateDictToFile(std::string filename, std::vector<std::pair<std::string, std::vector<double>>> state_dict);
    std::vector<std::pair<std::string, std::vector<double>>> aggregateDict();
    std::string createMac(int i);

  public:
    bool debug;

    int filtered;
    int num_of_forwarded_messages;
    int down;
    int up;

    cMessage *save_to_send;
    bool got_msg_to_save;

    bool is_root;
    bool is_smart;
    int num_of_messages_to_expect;
    int first_host_in_switch;
    std::vector<std::vector<std::pair<std::string,std::vector<double>>>> stored_dicts;
};

} //namespace

#endif
