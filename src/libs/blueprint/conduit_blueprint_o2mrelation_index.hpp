// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_o2mrelation_index.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_O2MRELATION_INDEX_HPP
#define CONDUIT_BLUEPRINT_O2MRELATION_INDEX_HPP

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "conduit_blueprint_exports.h"


//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint --
//-----------------------------------------------------------------------------
namespace blueprint
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::o2mrelation --
//-----------------------------------------------------------------------------
namespace o2mrelation
{

//-----------------------------------------------------------------------------
// -- begin conduit::O2Mindex --
//-----------------------------------------------------------------------------
///
/// class: conduit::O2MIndex
///
/// description:
///  General purpose index for 'o2mrelation' Nodes.
///
//-----------------------------------------------------------------------------
class CONDUIT_BLUEPRINT_API O2MIndex
{
public:
//-----------------------------------------------------------------------------
//
// -- conduit::O2MIndex public members --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// O2MIndex Construction and Destruction
//-----------------------------------------------------------------------------
    /// Copy constructor.
    O2MIndex(const O2MIndex &idx);

    /// Primary index constructor.
    O2MIndex(const Node *node);

    /// Primary index constructor.
    /// this will use the pointer to the passed Node ref.
    O2MIndex(const Node &node);

    /// Destructor
    ~O2MIndex();

    /// Assignment operator.
    O2MIndex &operator=(const O2MIndex &itr);

//-----------------------------------------------------------------------------
/// Retrieve a flat-index.
//-----------------------------------------------------------------------------
    index_t     index(index_t one_index, index_t many_index) const;

//-----------------------------------------------------------------------------
/// Get info on the "ones" of this index.
///
/// size() returns the number of "ones".  size(i) returns the number of "many"
/// associated with item i.  offset(i) returns the position in the data array
/// where the data for item i starts.
//-----------------------------------------------------------------------------
    index_t     size(index_t one_index = -1) const;
    index_t     offset(index_t one_index) const;

//-----------------------------------------------------------------------------
/// Human readable info about this iterator
//-----------------------------------------------------------------------------
    void        info(Node &res) const;

private:

//-----------------------------------------------------------------------------
//
// -- conduit::O2MIndex private members --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Index state/fields.
//-----------------------------------------------------------------------------
    /// The following 3 members are used so we can look them up once and then
    /// use the existence of the pointer instead of expensive calls that look
    /// up the child in m_node while we use the iterator.

    /// pointer to an internal sizes Node for the 'o2mrelation'
    const Node *m_sizes_node;
    /// pointer to an internal indices Node for the 'o2mrelation'
    const Node *m_indices_node;
    /// pointer to an internal offsets Node for the 'o2mrelation'
    const Node *m_offsets_node;

    /// Accessors for sizes, indices, and offsets
    index_t_accessor m_sizes_acc;
    index_t_accessor m_indices_acc;
    index_t_accessor m_offsets_acc;
};
//-----------------------------------------------------------------------------
// -- end conduit::O2MIndex --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::o2mrelation --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------


#endif
