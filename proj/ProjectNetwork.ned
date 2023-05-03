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
        @display("bgb=998.2425,648.795");
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
                @display("p=474.69748,165.33");
        }
        router2: Router {
            parameters:
                @display("p=405.81,453.405");
        }
        router3: Router {
            parameters:
                @display("p=543.585,452.1525");
        }
        sw1: EthernetSwitch {
            parameters:
                @display("p=474.69748,268.035");
        }
        sw2: EthernetSwitch {
            parameters:
                @display("p=405.81,341.9325");
        }
        sw3: EthernetSwitch {
            parameters:
                @display("p=542.3325,341.9325");
        }
        host1: StandardHost {
            parameters:
                @display("p=474.69748,60.12");
        }
        host2: StandardHost {
            parameters:
                @display("p=278.055,582.4125");
        }
        host3: StandardHost {
            parameters:
                @display("p=425.85,582.4125");
        }
        host4: StandardHost {
            parameters:
                @display("p=523.545,582.4125");
        }
        host5: StandardHost {
            parameters:
                @display("p=670.08746,582.4125");
        }
    connections:
        host1.ethg++ <--> Eth100M <--> router.ethg++;
        router.ethg++ <--> Eth100M <--> sw1.ethg++;
        sw1.ethg++ <--> Eth100M <--> sw2.ethg++;
        sw1.ethg++ <--> Eth100M <--> sw3.ethg++;
        sw2.ethg++ <--> Eth100M <--> router2.ethg++;
        sw3.ethg++ <--> Eth100M <--> router3.ethg++;
        router2.ethg++ <--> Eth100M <--> host2.ethg++;
        router2.ethg++ <--> Eth100M <--> host3.ethg++;
        router3.ethg++ <--> Eth100M <--> host4.ethg++;
        router3.ethg++ <--> Eth100M <--> host5.ethg++;
}
