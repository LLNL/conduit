// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_o2miterator.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_O2MITERATOR_HPP
#define CONDUIT_BLUEPRINT_O2MITERATOR_HPP

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

typedef enum
{
    DATA = 0,     // data array index
    ONE  = 1,     // one group (outer) index
    MANY = 2      // many item (inner) index
} IndexType;

//-----------------------------------------------------------------------------
// -- begin conduit::O2MIterator --
//-----------------------------------------------------------------------------
///
/// class: conduit::O2MIterator
///
/// description:
///  General purpose iterator for 'o2mrelation' Nodes.
///
//-----------------------------------------------------------------------------
class CONDUIT_BLUEPRINT_API O2MIterator
{
public:
//-----------------------------------------------------------------------------
//
// -- conduit::O2MIterator public members --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// O2MIterator Construction and Destruction
//-----------------------------------------------------------------------------
    /// Default constructor.
    O2MIterator();
    /// Copy constructor.
    O2MIterator(const O2MIterator &itr);

    /// Primary iterator constructor.
    O2MIterator(const Node *node);

    /// Primary iterator constructor.
    /// this will use the pointer to the passed Node ref.
    O2MIterator(const Node &node);

    /// Destructor
    ~O2MIterator();

    /// Assignment operator.
    O2MIterator &operator=(const O2MIterator &itr);

//-----------------------------------------------------------------------------
/// Iterator value and property access.
//-----------------------------------------------------------------------------
    index_t     index(IndexType itype = DATA) const;
    index_t     elements(IndexType itype = DATA) const;

//-----------------------------------------------------------------------------
/// Iterator forward control.
//-----------------------------------------------------------------------------
    bool        has_next(IndexType itype = DATA) const;
    index_t     next(IndexType itype = DATA);
    index_t     peek_next(IndexType itype = DATA) const;
    void        to_front(IndexType itype = DATA);

//-----------------------------------------------------------------------------
/// Iterator reverse control.
//-----------------------------------------------------------------------------
    bool        has_previous(IndexType itype = DATA) const;
    index_t     previous(IndexType itype = DATA);
    index_t     peek_previous(IndexType itype = DATA) const;
    void        to_back(IndexType itype = DATA);

//-----------------------------------------------------------------------------
/// Human readable info about this iterator
//-----------------------------------------------------------------------------
    void        info(Node &res) const;

private:

//-----------------------------------------------------------------------------
//
// -- conduit::O2MIterator private members --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Iterator property helper functions.
//-----------------------------------------------------------------------------
    index_t index(index_t one_index, index_t many_index, IndexType itype) const;
    index_t elements(index_t one_index, IndexType itype) const;

//-----------------------------------------------------------------------------
/// Iterator state/fields.
//-----------------------------------------------------------------------------
    /// pointer to the Node wrapped by this iterator
    const Node *m_node;
    /// pointer to an internal data Node for the 'o2mrelation'
    const Node *m_data_node;

    /// current 'one' index in 'o2mrelation' space
    index_t  m_one_index;
    /// current 'many' index in 'one' space
    index_t  m_many_index;

    // /// current 'one' count for 'o2mrelation' (constant)
    // index_t m_num_ones;
    // /// current 'many' count for 'o2mrelation' (depends on 'one')
    // index_t m_num_manys;
};
//-----------------------------------------------------------------------------
// -- end conduit::O2MIterator --
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
