// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_vector_view.hpp
///
//-----------------------------------------------------------------------------
#ifndef CONDUIT_VECTOR_VIEW_HPP
#define CONDUIT_VECTOR_VIEW_HPP

#include <vector>

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

/**
 @brief This class provides a std::vector-like interface for an external buffer.

 @note An alternative to this is conduit::data_array but this class is simpler,
       does not contain complicated indexing support, and looks more like vector.
       Another alternative would be C++20 views.
 */
template <typename T>
class vector_view
{
public:
    vector_view() = default;

    vector_view(T* _data, size_t n) : data(_data), num_elements(n)
    {
    }

    vector_view(const vector_view &obj) : data(obj.data), num_elements(obj.num_elements)
    {
    }

    void operator =(const vector_view &obj)
    {
        data = obj.data;
        num_elements = obj.num_elements;
    }

    void resize(size_t n)
    {
       num_elements = n;
    }

    void clear()
    {
       num_elements = 0;
    }

    size_t size() const
    {
        return num_elements;
    }

    bool empty() const
    {
        return num_elements == 0;
    }

    T &operator[](size_t index)
    {
        return data[index];
    }

    const T &operator[](size_t index) const
    {
        return data[index];
    }

    T &at(size_t index)
    {
        return data[index];
    }

    const T &at(size_t index) const
    {
        return data[index];
    }

    // Use pointers for iterators
    using const_iterator = const T*;
    using iterator = T*;

    iterator begin() const
    {
        return data;
    }

    iterator end() const
    {
        return data + num_elements;
    }

    const_iterator cbegin() const
    {
        return data;
    }

    const_iterator cend() const
    {
        return data + num_elements;
    }

private:
    T     *data{nullptr};
    size_t num_elements{0};
};

//---------------------------------------------------------------------------
template <typename T>
bool
operator == (const std::vector<T> &lhs, const vector_view<T> &rhs)
{
    bool same = lhs.size() == rhs.size();
    size_t n = lhs.size();
    for(size_t i = 0; i < n && same; i++)
        same &= lhs[i] == rhs[i];
    return same;
}

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

#endif
