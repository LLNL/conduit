//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see: http://software.llnl.gov/conduit/.
// 
// Please also read conduit/LICENSE
// 
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, 
//   this list of conditions and the disclaimer below.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
// 
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
// 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

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
class CONDUIT_API O2MIterator
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
