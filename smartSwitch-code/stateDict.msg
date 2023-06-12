namespace inet;
cplusplus{{
    #include <vector>
    #include <map>
    #include <string>
    
    typedef std::vector<std::pair<std::string, std::vector<double>>> stateDic;
    typedef std::string regString;
}};

class stateDic { @existingClass; };
class regString { @existingClass; };

message StateDict extends cMessage{
    stateDic dict;
    regString src;
}
