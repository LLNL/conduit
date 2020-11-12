// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_log.cpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_log.hpp"

using namespace conduit;

typedef bool (*VerifyFun)(const Node&);

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::utils --
//-----------------------------------------------------------------------------
namespace utils
{

//-----------------------------------------------------------------------------
// -- begin conduit::utils::log --
//-----------------------------------------------------------------------------
namespace log
{

//-----------------------------------------------------------------------------
void
info(Node &info,
     const std::string &proto_name,
     const std::string &msg)
{
    info["info"].append().set(proto_name + ": " + msg);
}

//-----------------------------------------------------------------------------
void
optional(Node &info,
         const std::string &proto_name,
         const std::string &msg)
{
    info["optional"].append().set(proto_name + ": " + msg);
}

//-----------------------------------------------------------------------------
void
error(Node &info,
      const std::string &proto_name,
      const std::string &msg)
{
    info["errors"].append().set(proto_name + ": " + msg);
}


//-----------------------------------------------------------------------------
void
validation(Node &info,
           bool res)
{
    // NOTE: if the given node already has a "valid" child, it's updated to
    // be the logical and of its current value and the new value so that
    // invalid nodes aren't changed to valid by later updates
    bool info_res = info.has_child("valid") ? info["valid"].as_string() == "true" : true;
    info["valid"].set(res && info_res ? "true" : "false");
}


//-----------------------------------------------------------------------------
bool
remove_tree(Node &info, const VerifyFun should_remove_fun)
{
    if(info.dtype().is_object() || info.dtype().is_list())
    {
        std::vector<index_t> removal_subtrees;

        NodeIterator info_itr = info.children();
        while(info_itr.has_next())
        {
            conduit::Node &info_child = info_itr.next();
            if(remove_tree(info_child, should_remove_fun))
            {
                removal_subtrees.push_back(info_itr.index());
            }
        }

        for(index_t ci = removal_subtrees.size(); ci-- > 0;)
        {
            info.remove(removal_subtrees[ci]);
        }

        // FIXME: This part of the solution makes it imperfect from a recursive
        // standpoint, but it makes it well-suited to accomodate the child-informed
        // removal criteria used for the 'log::remove_*' functions.
        if(should_remove_fun(info))
        {
            info.set(DataType::empty());
        }
    }

    return should_remove_fun(info);
}


//-----------------------------------------------------------------------------
bool is_valid(const Node &n)
{
    return n.dtype().is_empty() || (n.has_child("valid") && n["valid"].dtype().is_string() && n["valid"].as_string() == "true");
};

//-----------------------------------------------------------------------------
void
remove_valid(Node &info)
{
    remove_tree(info, is_valid);
}


//-----------------------------------------------------------------------------
bool is_invalid(const Node &n)
{
    return n.dtype().is_empty() || (n.has_child("valid") && n["valid"].dtype().is_string() && n["valid"].as_string() == "false");
};

//-----------------------------------------------------------------------------
void
remove_invalid(Node &info)
{
    remove_tree(info, is_invalid);
}


//-----------------------------------------------------------------------------
bool is_optional(const Node &n)
{
    return n.dtype().is_empty() || (n.name() == "optional");
};

//-----------------------------------------------------------------------------
void
remove_optional(Node &info)
{
    remove_tree(info, is_optional);
}


//-----------------------------------------------------------------------------
std::string
quote(const std::string &str,
      bool pad_before)
{
    std::ostringstream oss;
    oss << (pad_before ? " " : "") << "'" << str << "'" << (pad_before ? "" : " ");
    return (str != "") ? oss.str() : "";
}

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::utils::log --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::utils --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

