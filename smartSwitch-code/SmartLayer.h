#ifndef __INET_SMARTLAYER_H_
#define __INET_SMARTLAYER_H_

#include <omnetpp.h>
#include "inet/common/packet/printer/PacketPrinter.h"
#include "inet/common/packet/Packet.h"
#include "inet/common/packet/PacketFilter.h"
#include "inet/networklayer/ipv4/Ipv4.h"
#include <string.h>
#include <iostream>
#include <regex>

using namespace omnetpp;

namespace inet {

/**
 * TODO - Generated class
 */
class SmartLayer : public cSimpleModule
{
  protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
  public:
      int num_of_tensors;
      int down;
      Packet* first_tensor;
      PacketFilter filter;
      Ipv4Address to_check = Ipv4Address("10.0.0.2");

};

} //namespace

#endif
