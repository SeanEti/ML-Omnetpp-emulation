package inet.smart;

simple SmartLayer
{
    parameters:
        bool is_smart = default(false);
        int num_of_messages_to_expect = default(2);
        int height = default(3);
    @display("i=block/layer");

    gates:
        inout gate[];
}