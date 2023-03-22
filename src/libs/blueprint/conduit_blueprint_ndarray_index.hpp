// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_ndarray_index.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_NDARRAY_INDEX_HPP
#define CONDUIT_BLUEPRINT_NDARRAY_INDEX_HPP

//-----------------------------------------------------------------------------
// std includes
//-----------------------------------------------------------------------------
#include <map>
#include <set>
#include <string>
#include <vector>

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "conduit_blueprint_exports.h"

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
///
/// class: conduit::blueprint::ndarray::NDIndex
///
/// description:
///  General purpose index for NDArrays
///
//-----------------------------------------------------------------------------
class CONDUIT_BLUEPRINT_API NDIndex
{
public:
    //-----------------------------------------------------------------------------
    //
    // -- NDIndex public members --
    //
    //-----------------------------------------------------------------------------

    //-----------------------------------------------------------------------------
    /// NDIndex Construction and Destruction
    //-----------------------------------------------------------------------------
    NDIndex();

    // Copy constructor.
    NDIndex(const NDIndex& idx);

    /// Primary index constructor.
    NDIndex(const Node* node);

    /// Primary index constructor.  The argument node should contain numeric
    /// children, all of length equal to the array's dimensionality:
    ///     - shape (required) specifies the shape of the array to index
    ///     - offset (optional) specifies where the data starts in the array
    ///     - stride (optional) specifies the extent of the array storing data
    ///
    /// The shape is required.  It is an error if shape is omitted.
    /// If offset is not specified, it defaults to 0 in each dimension.
    /// If stride is not specified, it defaults to
    /// \code
    /// stride[0] = 1
    /// stride[i] = stride[i-1] * (offset[i-1] + shape[i-1])
    /// \endcode
    /// Node that offset is specified in terms of logical index, not
    /// flatindex, and stride is specified in terms of flatindex.  Also note
    /// that the default stride holds an assumption that the data is laid
    /// out in C-style, with fastest-varying dimension left-most.  Users may
    /// specify a custom stride to index Fortran-style arrays, where the
    /// fastest-varying index is right-most.
    ///
    /// Here are a few examples:
    ///
    /// - A 6x4 array
    ///   \code
    ///   shape: [6, 4]
    ///   \endcode
    /// - A 6x4 array with two extra elements at the end of each row
    ///   \code
    ///   shape: [6, 4]
    ///   stride: [1, 8]
    ///   \endcode
    /// - A 6x4 array with two elements of padding on the low end of a
    ///   dimension and one element of padding on the high end
    ///   \code
    ///   shape: [6, 4]
    ///   offset: [2, 2]
    ///   stride: [1, 9]
    ///   \endcode
    /// - A 6x4x5 array with two elements of padding on the low end of each
    ///   dimension and one element of padding on the high end (adds third
    ///   dimension to previous)
    ///   \code
    ///   shape: [6, 4, 5]
    ///   offset: [2, 2, 2]
    ///   stride: [1, 9, 63]
    ///   \endcode
    /// - A Fortran 6x4x5 array with two elements of padding on the low
    ///   end of each dimension and one element of padding on the high end
    ///   (previous example changed to column-major)
    ///   \code
    ///   shape: [6, 4, 5]
    ///   offset: [2, 2, 2]
    ///   stride: [63, 7, 1]
    ///   \endcode
    NDIndex(const Node& node);

    /// Array constructor
    NDIndex(const index_t  ndim,
            const index_t* shape,
            const index_t* offset = NULL,
            const index_t* stride = NULL);

    /// Destructor
    ~NDIndex() { };

    /// Assignment operator.
    NDIndex& operator=(const NDIndex& itr);

    //-----------------------------------------------------------------------------
    /// Retrieve a flat-index: public interface.
    //-----------------------------------------------------------------------------
    template<typename T, typename... Ts>
    index_t     index(T idx, Ts... idxs) const;
    template<typename T>
    index_t     index(T idx) const;

    /// Returns the number of dimensions
    index_t     ndims() const;

    /// Returns the extent of this NDIndex for dimension dim.
    index_t     shape(index_t dim) const;

    /// Returns the logical index in dimension dim where the data starts.
    index_t     offset(index_t dim) const;

    /// Returns the stride along dimension dim.
    index_t     stride(index_t dim) const;

    //-----------------------------------------------------------------------------
    /// Human readable info about this iterator
    //-----------------------------------------------------------------------------
    void        info(Node& res) const;

private:

    //-----------------------------------------------------------------------------
    //
    // -- conduit::blueprint::mesh::utils::NDIndex private members --
    //
    //-----------------------------------------------------------------------------

    /// Dimension (length of shape, offset, and stride nodes)
    index_t m_ndim;

    /// Accessors for shape, offset, and stride
    index_t_accessor m_shape_acc;
    index_t_accessor m_offset_acc;
    index_t_accessor m_stride_acc;

};

//-----------------------------------------------------------------------------
template<typename T, typename... Ts>
index_t
NDIndex::index(T idx, Ts... idxs) const
{
    index_t depth = m_ndim - sizeof...(idxs) - 1;
    index_t component = (offset(depth) + idx) * stride(depth);
    return component + index(idxs...);
}

//-----------------------------------------------------------------------------
template<typename T>
index_t
NDIndex::index(T idx) const
{
    index_t depth = m_ndim - 1;
    index_t component = (offset(depth) + idx) * stride(depth);
    return component;
}

//-----------------------------------------------------------------------------
inline
index_t
NDIndex::shape(index_t dim) const
{
    return m_shape_acc[dim];
}

//-----------------------------------------------------------------------------
inline
index_t
NDIndex::offset(index_t dim) const
{
    if (m_offset_acc.number_of_elements() < 1)
    {
        return 0;
    }
    return m_offset_acc[dim];
}

//-----------------------------------------------------------------------------
inline
index_t
NDIndex::stride(index_t dim) const
{
    if (m_stride_acc.number_of_elements() < 1)
    {
        index_t acc = 1;
        for (int d = 0; d < dim && d < m_ndim; ++d)
        {
            acc = acc * (m_shape_acc[d] + offset(d));
        }
        return acc;
    }
    return m_stride_acc[dim];
}

//-----------------------------------------------------------------------------
inline
index_t
NDIndex::ndims() const
{
    return m_ndim;
}


//-----------------------------------------------------------------------------
// -- end conduit::blueprint::ndarray::NDIndex --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::ndarray --
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
