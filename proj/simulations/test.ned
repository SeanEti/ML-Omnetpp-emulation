package test2.simulations;

import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
//import inet.visualizer.common.IntegratedVisualizer;
import inet.node.inet.StandardHost;
import inet.node.ethernet.Eth100M;
import inet.smart.SmartSwitch;

network testnet
{
    parameters:
        int height @prompt("Height of tree") = default(3);
        @display("bgb=738.85394,827.18");
    submodules:
        configurator: Ipv4NetworkConfigurator {
            parameters:
                @display("p=100,100;is=s");
        }
        sw[2^(height+1)-1]: SmartSwitch;
        host[2^(height)+1]: StandardHost;
    connections:
        sw[0].ethg++ <--> Eth100M <--> host[0].ethg++;
        for i=0..2^(height)-2 {
            // build binary tree of switches with given height
            sw[i].ethg++ <--> Eth100M <--> sw[2*i+1].ethg++;
            sw[i].ethg++ <--> Eth100M <--> sw[2*i+2].ethg++;
        }
        for i=1..2^(height) {
            // connect leave switches to hosts
            sw[2^(height)-1+i-1].ethg++ <--> Eth100M <--> host[i].ethg++;
        }
}

