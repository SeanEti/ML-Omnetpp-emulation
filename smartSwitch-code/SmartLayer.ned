package inet.smart;

simple SmartLayer
{
    parameters:
        bool debug = default(false);
        bool is_smart = default(false);
        int num_of_messages_to_expect = default(2);
        int first_host_in_switch = default(0);
        int height = default(3);
    @display("i=block/layer");

    gates:
        inout gate[];
}