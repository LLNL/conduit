// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_cpp_to_c.cpp
///
//-----------------------------------------------------------------------------
#include "conduit.h"
#include "conduit.hpp"

#include "conduit_cpp_to_c.hpp"

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{


//---------------------------------------------------------------------------//
Node *
cpp_node(conduit_node *cnode)
{
    return static_cast<Node*>(cnode);
}

//---------------------------------------------------------------------------//
conduit_node *
c_node(Node *node)
{
    return (void*)node;
}


//---------------------------------------------------------------------------//
const Node *
cpp_node(const conduit_node *cnode)
{
    return static_cast<const Node*>(cnode);
}

//---------------------------------------------------------------------------//
const conduit_node *
c_node(const Node *node)
{
    return (void*)node;
}

//---------------------------------------------------------------------------//
Node &
cpp_node_ref(conduit_node *cnode)
{
    return *static_cast<Node*>(cnode);
}

//---------------------------------------------------------------------------//
const Node &
cpp_node_ref(const conduit_node *cnode)
{
    return *static_cast<const Node*>(cnode);
}

//---------------------------------------------------------------------------//
DataType *
cpp_datatype(conduit_datatype *cdatatype)
{
    return static_cast<DataType*>(cdatatype);
}

//---------------------------------------------------------------------------//
conduit_datatype *
c_datatype(DataType *datatype)
{
    return (void*)datatype;
}


//---------------------------------------------------------------------------//
const DataType *
cpp_datatype(const conduit_datatype *cdatatype)
{
    return static_cast<const DataType*>(cdatatype);
}

//---------------------------------------------------------------------------//
const conduit_datatype *
c_datatype(const DataType *datatype)
{
    return (void*)datatype;
}

//---------------------------------------------------------------------------//
DataType &
cpp_datatype_ref(conduit_datatype *cdatatype)
{
    return *static_cast<DataType*>(cdatatype);
}

//---------------------------------------------------------------------------//
const DataType &
cpp_datatype_ref(const conduit_datatype *cdatatype)
{
    return *static_cast<const DataType*>(cdatatype);
}

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------


