#ifndef __INET_SMARTLAYER_H_
#define __INET_SMARTLAYER_H_

#include <omnetpp.h>
//#include "inet/common/MessageDispatcher.h"

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
      cPacket* first_tensor;
};

} //namespace

#endif
