#include "SmartLayer.h"

namespace inet {

Define_Module(SmartLayer);

void SmartLayer::initialize()
{
    // TODO - Generated method body
    this->num_of_tensors=0;
    this->down=0;
    this->first_tensor = NULL;
}

void SmartLayer::handleMessage(cPacket *msg)
{
    // TODO - Generated method body
    if (msg->arrivedOn("fromUpperL")) { // if the message is already processed then just forward it to send
            down++;
            PacketPrinter printer; // turns packets into human readable strings
            printer.printPacket(std::cout, msg); // print to standard output
            send(msg, "toLowerL");
    } else {
        error("arrived on unknown gate");
    }
    char buf[128];
    sprintf(buf, "tensors caught: %d, messages went through: %d", num_of_tensors, down);
    getDisplayString().setTagArg("t", 0, buf);
}

} //namespace
