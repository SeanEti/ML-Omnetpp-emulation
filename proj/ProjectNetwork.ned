package inet.showcases.emulation.proj;

import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.ethernet.Eth100M;
import inet.node.ethernet.EthernetSwitch;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;
import inet.visualizer.common.IntegratedVisualizer;


network ProjectNetwork
{
    parameters:
        @display("bgb=1000,400");
    submodules:
        configurator: Ipv4NetworkConfigurator {
            parameters:
                @display("p=100,100;is=s");
        }
        visualizer: IntegratedVisualizer {
            parameters:
                @display("p=100,200;is=s");
        }
        router: Router {
            parameters:
                @display("p=425.44498,336.34");
        }
        router2: Router {
            parameters:
                @display("p=553.45496,272.335");
        }
        router3: Router {
            parameters:
                @display("p=553.45496,272.335");
        }
        host1: StandardHost {
            parameters:
                @display("p=100,300");
        }
        host2: StandardHost {
            parameters:
                @display("p=900,300");
        }
        host3: StandardHost {
            parameters:
                @display("p=500,400");
        }
        host4: StandardHost {
            parameters:
                @display("p=500,400");
        }
        host5: StandardHost {
            parameters:
                @display("p=500,400");
        }
    connections:
        host1.ethg++ <--> Eth100M <--> router.ethg++;
        router.ethg++ <--> Eth100M <--> router2.ethg++;
        router.ethg++ <--> Eth100M <--> router3.ethg++;
        router2.ethg++ <--> Eth100M <--> host2.ethg++;
        router2.ethg++ <--> Eth100M <--> host3.ethg++;
        router3.ethg++ <--> Eth100M <--> host4.ethg++;
        router3.ethg++ <--> Eth100M <--> host5.ethg++;
}
