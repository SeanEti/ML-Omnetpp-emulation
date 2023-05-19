#include "SmartLayer.h"

namespace inet{

Define_Module(SmartLayer);

void SmartLayer::initialize()
{
    // TODO - Generated method body
    this->num_of_tensors=0;
    this->down=0;
    this->first_tensor = NULL;
    this->filter.setPattern("tcpseg*");
}

void SmartLayer::handleMessage(cMessage* msg)
{
    // TODO - Generated method body
    Packet* packet=check_and_cast<Packet*>(msg);

    if (msg->arrivedOn("fromUpperL")) { // if the message is already processed then just forward it to send
            down++;

            // checking the destination IP of received packet!
            if(filter.matches(packet)){
                const auto& my_ipv4header = packet->peekAtFront<Ipv4Header>();
                auto destIP = my_ipv4header->getDestAddress();
                if(to_check == destIP)
                    EV_INFO << "FOUND FROM 10.0.0.1 TO 10.0.0.2!\n";
            }

            // Print received packet
            PacketPrinter printer;
            EV << printer.printPacketToString(packet) << endl;
            //
            send(msg, "toLowerL");
    } else {
        error("arrived on unknown gate");
    }

    // print some information above layer in simulation
    char buf[128];
    sprintf(buf, "tensors caught: %d, messages went through: %d", num_of_tensors, down);
    getDisplayString().setTagArg("t", 0, buf);
}

} //namespace
