package inet.smart;

import inet.smart.SmartLayer;

module SmartSwitch
{
    parameters:
        @networkNode();
        @labels(node,ethernet-node);
        @display("i=device/switch");
        int numEthInterfaces = default(0);  // minimum number of ethernet interfaces

    gates:
        inout ethg[numEthInterfaces] @labels(EtherFrame-conn);
    submodules:
        smartL: SmartLayer {
            @display("p=500,379");
        }
    connections:
        for i=0..sizeof(ethg)-1 {
            smartL.gate++ <--> { @display("m=s"); } <--> ethg[i];
        }
}
