package test2.simulations;

import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.inet.StandardHost;
import ned.DatarateChannel;
import inet.smart.SmartSwitch;

channel sslink extends DatarateChannel{
    datarate = 0.33Mbps;
    delay = 1s;
}

channel hslink extends DatarateChannel{
    datarate = 100Mbps;
    delay = 1s;
}

network testnet
{
    parameters:
        int height @prompt("Height of tree") = default(3);
        @display("bgb=1003.848,915.762");
    submodules:
        configurator: Ipv4NetworkConfigurator {
            parameters:
                @display("p=100,100;is=s");
        }
        sw[2^(height+1)-1]: SmartSwitch;
        host[32]: StandardHost;
    connections:
        sw[0].ethg++ <--> hslink <--> host[0].ethg++;
        for i=0..2^(height)-2 {
            // build binary tree of switches with given height
            sw[i].ethg++ <--> sslink <--> sw[2*i+1].ethg++;
            sw[i].ethg++ <--> sslink <--> sw[2*i+2].ethg++;
        }
        
        // s7
        sw[7].ethg++ <--> hslink <--> host[1].ethg++;
        sw[7].ethg++ <--> hslink <--> host[2].ethg++;
        // s8
        sw[8].ethg++ <--> hslink <--> host[3].ethg++;
        sw[8].ethg++ <--> hslink <--> host[4].ethg++;
        sw[8].ethg++ <--> hslink <--> host[5].ethg++;
        sw[8].ethg++ <--> hslink <--> host[6].ethg++;
        sw[8].ethg++ <--> hslink <--> host[7].ethg++;
        // s9
        sw[9].ethg++ <--> hslink <--> host[8].ethg++;
        sw[9].ethg++ <--> hslink <--> host[9].ethg++;
        // s10
        sw[10].ethg++ <--> hslink <--> host[10].ethg++;
        // s11
        sw[11].ethg++ <--> hslink <--> host[11].ethg++;
        sw[11].ethg++ <--> hslink <--> host[12].ethg++;
        sw[11].ethg++ <--> hslink <--> host[13].ethg++;
        sw[11].ethg++ <--> hslink <--> host[14].ethg++;
        sw[11].ethg++ <--> hslink <--> host[15].ethg++;
        // s12
        sw[12].ethg++ <--> hslink <--> host[16].ethg++;
        sw[12].ethg++ <--> hslink <--> host[17].ethg++;
        sw[12].ethg++ <--> hslink <--> host[18].ethg++;
        sw[12].ethg++ <--> hslink <--> host[19].ethg++;
        sw[12].ethg++ <--> hslink <--> host[20].ethg++;
        sw[12].ethg++ <--> hslink <--> host[21].ethg++;
        sw[12].ethg++ <--> hslink <--> host[22].ethg++;
        sw[12].ethg++ <--> hslink <--> host[23].ethg++;
        // s13
        sw[13].ethg++ <--> hslink <--> host[24].ethg++;
        sw[13].ethg++ <--> hslink <--> host[25].ethg++;
        // s14
        sw[14].ethg++ <--> hslink <--> host[26].ethg++;
        sw[14].ethg++ <--> hslink <--> host[27].ethg++;
        sw[14].ethg++ <--> hslink <--> host[28].ethg++;
        sw[14].ethg++ <--> hslink <--> host[29].ethg++;
        sw[14].ethg++ <--> hslink <--> host[30].ethg++;
        sw[14].ethg++ <--> hslink <--> host[31].ethg++;
        
//        // for one host in switch
//        for i=1..2^(height) {
//            // connect leave switches to hosts
//            sw[2^(height)-1+i-1].ethg++ <--> hslink <--> host[i].ethg++;
//        }
}

