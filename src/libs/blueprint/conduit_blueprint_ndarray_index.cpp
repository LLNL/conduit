// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_ndarray_index.cpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_ndarray_index.hpp"

//-----------------------------------------------------------------------------
// -- begin conduit --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint --
//-----------------------------------------------------------------------------
namespace blueprint
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::ndarray --
//-----------------------------------------------------------------------------
namespace ndarray
{

//-----------------------------------------------------------------------------
NDIndex::NDIndex()
: m_ndim(0)
{

}

//-----------------------------------------------------------------------------
NDIndex::NDIndex(const Node& idx)
: NDIndex(&idx)
{
    
}

//-----------------------------------------------------------------------------
NDIndex::NDIndex(const Node* idx)
: m_ndim(0),
  m_shape_acc(),
  m_offset_acc(),
  m_stride_acc()
{
    // TODO: error if idx has no child "shape"
    // TODO: error if shape, offset, and stride differ in length

    if (idx->has_child("shape"))
    {
        m_shape_acc = (*idx)["shape"].as_index_t_accessor();
        m_ndim = m_shape_acc.number_of_elements();
    }

    if (idx->has_child("offset"))
    {
        m_offset_acc = (*idx)["offset"].as_index_t_accessor();
    }

    if (idx->has_child("stride"))
    {
        m_stride_acc = (*idx)["stride"].as_index_t_accessor();
    }
}

//---------------------------------------------------------------------------//
NDIndex::NDIndex(const index_t ndim,
                 const index_t* shape,
                 const index_t* offset,
                 const index_t* stride)
: m_ndim(ndim)
{
    // TODO Error if dim < 1
    // TODO Error if shape is NULL

    m_shape_acc = index_t_accessor(shape, DataType::index_t(ndim));

    if (offset)
    {
        m_offset_acc = index_t_accessor(offset, DataType::index_t(ndim));
    }

    if (stride)
    {
        m_stride_acc = index_t_accessor(stride, DataType::index_t(ndim));
    }
}

//---------------------------------------------------------------------------//
NDIndex::NDIndex(const NDIndex& idx)
: m_ndim(idx.m_ndim),
  m_shape_acc(idx.m_shape_acc),
  m_offset_acc(idx.m_offset_acc),
  m_stride_acc(idx.m_stride_acc)
{

}

//---------------------------------------------------------------------------//
NDIndex&
NDIndex::operator=(const NDIndex& idx)
{
    if (this != &idx)
    {
        this->m_ndim = idx.m_ndim;
        this->m_shape_acc = idx.m_shape_acc;
        this->m_offset_acc = idx.m_offset_acc;
        this->m_stride_acc = idx.m_stride_acc;
    }
    return *this;
}

//---------------------------------------------------------------------------//
void
NDIndex::info(Node& res) const
{
    index_t dim = ndims();
    res["shape"].set(DataType::index_t(dim));
    res["offset"].set(DataType::index_t(dim));
    res["stride"].set(DataType::index_t(dim));

    index_t* p_shape = res["shape"].as_index_t_ptr();
    index_t* p_offset = res["offset"].as_index_t_ptr();
    index_t* p_stride = res["stride"].as_index_t_ptr();
    for (index_t d = 0; d < dim; ++d)
    {
        p_shape[d] = shape(d);
        p_offset[d] = offset(d);
        p_stride[d] = stride(d);
    }

    // shape is always required, so we don't report user_provided/shape

    if (m_offset_acc.number_of_elements() < 1)
    {
        res["user_provided/offset"] = "false";
    }
    else
    {
        res["user_provided/offset"] = "true";
    }

    if (m_stride_acc.number_of_elements() < 1)
    {
        res["user_provided/stride"] = "false";
    }
    else
    {
        res["user_provided/stride"] = "true";
    }
}

//-----------------------------------------------------------------------------
// -- end conduit::blueprint::ndarray::NDIndex
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::ndarray
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

