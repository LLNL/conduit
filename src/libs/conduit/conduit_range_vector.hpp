// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_range_vector.hpp
///
//-----------------------------------------------------------------------------
#ifndef CONDUIT_RANGE_VECTOR_HPP
#define CONDUIT_RANGE_VECTOR_HPP

#include <vector>

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

/**
 @brief This class provides a std::vector-like interface for representing an
        implicit range of numbers. This makes it look like a vector without
        having to actually store the data.
 */
template <typename T>
class range_vector
{
public:
    range_vector(T _start, T _incr, size_t n) : start(_start), incr(_incr),
        num_elements(n)
    {
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

    T operator[](size_t index) const
    {
        return start + incr * index;
    }

    T at(size_t index) const
    {
        return start + incr * index;
    }

    class iterator
    {
    public:
        iterator() : parent(nullptr), index(0)
        {
        }

        iterator(const iterator &obj) : parent(obj.parent), index(obj.index)
        {
        }

        iterator(const range_vector<T> *p, size_t i) : parent(p), index(i)
        {
        }

        iterator operator = (const iterator &obj)
        {
            parent = obj.parent;
            index = obj.index;
            return *this;
        }

        T operator *() const
        {
            return parent->operator[](index);
        }

        T operator ->() const
        {
            return parent->operator[](index);
        }

        void operator ++()
        {
            index++;
        }
        void operator ++(int)
        {
            index++;
        }
        void operator --()
        {
            index--;
        }
        void operator --(int)
        {
            index--;
        }

        iterator operator +(size_t n) const
        {
            return iterator(parent, index + n);
        }

        iterator operator +=(size_t n)
        {
            index += n;
            return *this;
        }

        bool operator == (const iterator &rhs) const
        {
            return parent == rhs.parent && index == rhs.index;
        }

        bool operator < (const iterator &rhs) const
        {
            return parent == rhs.parent && index < rhs.index;
        }

        bool operator > (const iterator &rhs) const
        {
            return parent == rhs.parent && index > rhs.index;
        }

        bool operator != (const iterator &rhs) const
        {
            return parent != rhs.parent || index != rhs.index;
        }
    private:
        const range_vector<T> *parent{nullptr};
        size_t index{0};
    };

    using const_iterator = iterator;

    iterator begin() const
    {
        return iterator(this,0);
    }

    iterator end() const
    {
        return iterator(this,num_elements);
    }

    const_iterator cbegin() const
    {
        return const_iterator(this,0);
    }

    const_iterator cend() const
    {
        return const_iterator(this,num_elements);
    }

private:
    T start{};
    T incr{};
    size_t num_elements{0};
};

//---------------------------------------------------------------------------
template <typename T>
std::vector<T>
to_vector(const range_vector<T> &input)
{
    std::vector<T> output(input.size());
    for(size_t i = 0; i < input.size(); i++)
        output[i] = input[i];
    return std::move(output);
}

//---------------------------------------------------------------------------
template <typename T>
bool
operator == (const std::vector<T> &lhs, const range_vector<T> &rhs)
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
