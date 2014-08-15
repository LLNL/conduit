///
/// file: Core.cpp
///

#include "Core.h"
#include "Node.h"

namespace conduit
{

///============================================
std::string
about()
{
    Node n;
    about(n);
    return n.to_json(true,2);
}

///============================================
void
about(Node &n)
{
    n.reset();
    n["version"] = "{alpha}";
    
    // TODO: include compiler info, license info, etc
}


}