package inet.smart;
import inet.common.MessageDispatcher;
import inet.common.lifecycle.NodeStatus;
import inet.common.packet.recorder.PcapRecorder;
import inet.linklayer.configurator.L2NodeConfigurator;
import inet.linklayer.contract.IEthernetInterface;
import inet.linklayer.contract.IMacAddressTable;
import inet.linklayer.contract.IMacRelayUnit;
import inet.linklayer.contract.IPolicyLayer;
import inet.linklayer.contract.ISpanningTree;
import inet.linklayer.ethernet.contract.IEthernetEncapsulation;
import inet.linklayer.ethernet.contract.IEthernetLayer;
import inet.linklayer.ieee8021q.Ieee8021qLayer;
import inet.networklayer.common.InterfaceTable;
import inet.smart.SmartLayer;

//
// Model of a Smart Ethernet switch.
//
// TWorks just as a regular switch but has a new layer that aggregates Tensors sent in the network
//
module SmartSwitch
{
    parameters:
        @networkNode();
        @labels(node,ethernet-node);
        @display("i=device/switch");
        int numPcapRecorders = default(0);
        bool hasStatus = default(false);
        bool hasStp = default(false);
        bool enableCutthrough = default(false);
        string fcsMode @enum("declared","computed") = default("declared");
        string spanningTreeProtocol = default("Stp");
        int numEthInterfaces = default(0);  // minimum number of ethernet interfaces
        eth[*].encap.typename = default("OmittedEthernetEncapsulation");
        *.fcsMode = this.fcsMode;
        **.interfaceTableModule = default(absPath(".interfaceTable"));
        **.macTableModule = default(absPath(".macTable"));

        relayUnit.hasStp = this.hasStp;
        encap.registerProtocol = true;
        eth[*].mac.promiscuous = default(true);
        ethernet.*.promiscuous = default(true);

    gates:
        inout ethg[numEthInterfaces] @labels(EtherFrame-conn);
    submodules:
        status: NodeStatus if hasStatus {
            @display("p=100,400;is=s");
        }
        interfaceTable: InterfaceTable {
            @display("p=100,200;is=s");
        }
        pcapRecorder[numPcapRecorders]: PcapRecorder {
            @display("p=100,500;is=s");
        }
        macTable: <default("MacAddressTable")> like IMacAddressTable {
            @display("p=100,100;is=s");
        }
        relayUnit: <default(firstAvailable("Ieee8021dRelay","MacRelayUnit"))> like IMacRelayUnit {
            @display("p=946,400;is=m");
        }
        li: MessageDispatcher {
            parameters:
                @display("p=800,527;b=1200,5");
        }
        eth[sizeof(ethg)]: <default(enableCutthrough ? "EthernetCutthroughInterface" : "EthernetInterface")> like IEthernetInterface {
            parameters:
                @display("p=250,700,row,150;q=txQueue");
        }
        smartLayer: SmartLayer {

            @display("p=649,400");
        }
    connections allowunconnected:
        relayUnit.ifOut --> smartLayer.fromUpperL;
        smartLayer.toLowerL --> li.in++;
        li.out++ --> relayUnit.ifIn;

        for i=0..sizeof(ethg)-1 {
            li.out++ --> eth[i].upperLayerIn;
            li.in++ <-- eth[i].upperLayerOut;
            eth[i].phys <--> { @display("m=s"); } <--> ethg[i];
        }

}
