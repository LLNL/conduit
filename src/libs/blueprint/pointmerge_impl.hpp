#ifndef KDTREE_HPP
#define KDTREE_HPP

#include <cstddef>
#include <cmath>
#include <utility>
#include <algorithm>
#include <array>
#include <vector>
#include <limits>

#ifdef DEBUG_KDTREE
#include <iostream>
#endif

namespace conduit
{

namespace blueprint
{

template<typename T, size_t Size>
struct vector
{
    using this_type = vector<T, Size>;
    using data_type = std::array<T, Size>;
    using value_type = T;
private:
    // Access the vector data as x, y, z
    template<size_t Index>
    struct accessor
    {
        data_type data;

        constexpr operator T() const
        {
            static_assert(Index < Size, "Invalid access into data.");
            return data[Index];
        }

        T operator=(T v)
        {
            static_assert(Index < Size, "Invalid access into data.");
            return data[Index] = v;
        }
    };

public:
    union 
    {
        data_type    v;
        accessor<0>  x;
        accessor<1>  y;
        accessor<2>  z;
    };

    constexpr size_t size() const 
    {
        return Size;
    }

    T operator[](size_t index) const
    {
        return v[index];
    }

    T &operator[](size_t index) 
    {
        return v[index];
    }

    void zero() 
    {
        set_all(0);
    }

    void set_all(T val)
    {
        for(size_t i = 0u; i < size(); i++)
        {
            v[i] = val;
        }
    }

    // NOTE: Defining operator= makes this non-POD type
    // this_type operator=(const this_type &other)
    // {
    //     for(size_t i = 0u; i < size(); i++)
    //     {
    //         v[i] = other[i];
    //     }
    //     return *this;
    // }

    void copy(const this_type &other)
    {
        for(auto i = 0u; i < size(); i++)
            other.v[i] = v[i];
    }

    bool operator<=(const this_type &other) const
    {
        bool retval = true;
        for(size_t i = 0u; i < size(); i++)
            retval &= v[i] <= other[i];
        return retval;
    }

    bool operator>=(const this_type &other) const
    {
        bool retval = true;
        for(size_t i = 0u; i < size(); i++)
            retval &= v[i] >= other[i];
        return retval;
    }

    this_type operator+(T scalar) const
    {
        this_type retval;
        for(size_t i = 0u; i < size(); i++)
        {
            retval[i] = v[i] + scalar;
        }
        return retval;
    }

    this_type operator-(T scalar) const
    {
        this_type retval;
        for(size_t i = 0u; i < size(); i++)
        {
            retval[i] = v[i] - scalar;
        }
        return retval;
    }

    double distance2(const this_type &other) const
    {
        double d2 = 0.;
        for(size_t i = 0u; i < size(); i++)
        {
            const auto diff = other[i] - v[i];
            d2 += (diff*diff);
        }
        return d2;
    }

    double distance(const this_type &other) const
    {
        return std::sqrt(distance2(other));
    }
};

template<typename VectorType>
struct bounding_box
{
    using value_type = typename VectorType::value_type;
    using data_type = typename VectorType::data_type;
    using T = value_type;
    VectorType min;
    VectorType max;

    bool contains(const VectorType &point) const
    {
        return (point >= min && point <= max);
    }

    bool contains(const VectorType &point, double tolerance) const
    {
        return (point >= (min - tolerance) && point <= (max + tolerance));
    }

    void expand(const VectorType &point)
    {
        for(size_t i = 0u; i < min.size(); i++)
        {
            min[i] = std::min(min[i], point[i]);
            max[i] = std::max(max[i], point[i]);
        }
    }
};

using vec2f = vector<float,2>;
using vec3f = vector<float,3>;
using vec2  = vector<double,2>;
using vec3  = vector<double,3>;

template<typename VectorType, typename DataType>
class kdtree
{
private:
    using Float = typename VectorType::value_type;
    // using IndexType = conduit_index_t;
public:
    constexpr static auto dimension = std::tuple_size<typename VectorType::data_type>::value;
    using vector_type = VectorType;
    using data_type = DataType;
    using IndexType = size_t;

    template<typename Precision, size_t D>
    struct kdnode
    {
        using node = kdnode<Precision, D>;
        std::vector<VectorType> points;
        std::vector<DataType> data;
        bounding_box<VectorType> bb;
        node *left{nullptr};
        node *right{nullptr};
        Float split{0.0};
        unsigned int dim;
        bool has_split{false};
    };
    using node = kdnode<Float, dimension>;
    using pair = std::pair<std::pair<node*,DataType&>, bool>;

    kdtree() = default;
    ~kdtree()
    { 
        const auto lambda = [](node *node, unsigned int)
        {
            delete node;
        };
        if(root) { traverse_lrn(lambda, root); }
    }

    /**
    @brief Searches the tree for the given point using
            tolerance as an acceptable distance to merge like-points.
    */
    DataType *find_point(const VectorType &point, Float tolerance)
    {
        DataType *retval = nullptr;
        if(!root)
        {
            retval = nullptr;
        }
        else if(root->bb.contains(point, tolerance))
        {
            retval = find_point(root, 0, point, tolerance);
        }
        return retval;
    }

    void insert(const VectorType &point, const DataType &r)
    {
        scratch.reserve(point_vec_size*2);
        if(!root)
        {
            root = create_node(point, r);
            npoints++;
        }
        else
        {
            insert(root, 0, point, r);
        }
    }

    IndexType size()  const { return npoints; }
    IndexType nodes() const { return nnodes; }
    IndexType depth() const { return tree_depth; }

    void set_bucket_size(IndexType n) { point_vec_size = n; }
    IndexType get_bucket_size() const { return point_vec_size; }

    template<typename Func>
    void iterate_points(Func &&func)
    {
        IndexType point_id = 0u;
        const auto lambda = [&](node *n, unsigned int) {
            for(IndexType i = 0u; i < n->points.size(); i++)
            {
                func(point_id, n->points[i], n->data[i]);
                point_id++;
            }
        };
        if(root) { traverse_lnr(lambda, root); }
    }

    template<typename Func>
    void traverse_nodes(Func &&func) 
    {
        if(root) { traverse_lnr(func, root); }
    }

private:
    // Create an empty node
    node *create_node()
    {
        node *newnode = new node;
        newnode->points.reserve(point_vec_size);
        newnode->data.reserve(point_vec_size);
        newnode->bb.min[0] = std::numeric_limits<Float>::max();
        newnode->bb.min[1] = std::numeric_limits<Float>::max();
        newnode->bb.min[2] = std::numeric_limits<Float>::max();
        newnode->bb.max[0] = std::numeric_limits<Float>::lowest();
        newnode->bb.max[1] = std::numeric_limits<Float>::lowest();
        newnode->bb.max[2] = std::numeric_limits<Float>::lowest();
        newnode->left = nullptr;
        newnode->right = nullptr;
        newnode->split = 0;
        newnode->dim = 0;
        newnode->has_split = false;
        nnodes++;
        return newnode;
    }

    // Create a node with initial values inserted
    node *create_node(VectorType loc, const DataType &r)
    {
        node *newnode = create_node();
        node_add_data(newnode, loc, r);
        return newnode;
    }

    static void node_add_data(node *n, const VectorType &p, const DataType &d)
    {
        n->bb.expand(p);
        n->points.push_back(p);
        n->data.push_back(d);
    }

    /**
    @brief Splits the given node and inserts point/data into the proper child
    */
    void node_split(node *n, const VectorType &point, const DataType &data)
    {
        // Determine which dim to split on
        IndexType dim = 0;
        {
            Float longest_dim = std::numeric_limits<Float>::lowest();
            for(IndexType i = 0; i < n->bb.min.size(); i++)
            {
                const Float dim_len = n->bb.max[i] - n->bb.min[i];
                if(longest_dim < dim_len)
                {
                    dim = i;
                    longest_dim = dim_len;
                }
            }
            n->dim = dim;
        }

        // Determine what value on the dim to split on
        {
            scratch.clear();
            for(IndexType i = 0; i < point_vec_size; i++)
            {
                scratch.push_back(i);
            }
            std::sort(scratch.begin(), scratch.end(), [=](IndexType i0, IndexType i1) {
                return n->points[i0][dim] < n->points[i1][dim];
            });

            // If the index stored in scratch is point_vec_size
            const IndexType scratch_idx = scratch.size() / 2;
            const IndexType median_idx = scratch[scratch_idx];
            Float median = n->points[median_idx][dim];
            // Check if the new point is our actual median
            if(point[dim] > n->points[scratch[scratch_idx-1]][dim] && point[dim] < median)
            {
                median = point[dim];
            }

            n->split = median;
            n->left = create_node();
            n->right = create_node();
            n->has_split = true;

            for(IndexType i = 0; i < point_vec_size; i++)
            {
                const Float temp = n->points[i][dim];
                if(temp < median)
                {
                    node_add_data(n->left, n->points[i], n->data[i]);
                }
                else
                {
                    node_add_data(n->right, n->points[i], n->data[i]);
                }
            }

            if(point[dim] < median)
            {
                node_add_data(n->left, point, data);
            }
            else
            {
                node_add_data(n->right, point, data);
            }

            // Clear the data from the parent node
            std::vector<VectorType>{}.swap(n->points);
            std::vector<DataType>{}.swap(n->data);
        }
    }

    DataType *find_point(node *current, unsigned int depth, const VectorType &point, Float tolerance)
    {
        // If we got here we know that the point was in this node's bounding box
        DataType *retval = nullptr;

        // This node has children
        if(current->has_split)
        {
            const bool left_contains = current->left->bb.contains(point, tolerance);
            const bool right_contains = current->right->bb.contains(point, tolerance);
            if(!left_contains && !right_contains)
            {
                // ERROR! This shouldn't happen, the tree must've been built improperly
                retval = nullptr;
            }
            else if(left_contains)
            {
                // Traverse left
                retval = find_point(current->left, depth+1, point, tolerance);
            }
            else if(right_contains)
            {
                // Traverse right
                retval = find_point(current->right, depth+1, point, tolerance);
            }
            else // (left_contains && right_contains)
            {
                // Rare, but possible due to tolerance.
                // Check if the left side has the point without tolerance
                const bool pref_left = current->left->bb.contains(point);
                retval = (pref_left)
                    ? find_point(current->left, depth+1, point, tolerance)
                    : find_point(current->right, depth+1, point, tolerance);
                // We tried the preferred side but it didn't contain the point
                if(retval == nullptr)
                {
                    retval = (pref_left)
                        ? find_point(current->right, depth+1, point, tolerance)
                        : find_point(current->left, depth+1, point, tolerance);
                }
            }
        }
        else
        {
            // This is a leaf node.
            const auto t2 = tolerance * tolerance;
            const IndexType N = current->points.size();
            IndexType idx = 0;
            for(idx = 0; idx < N; idx++)
            {
                const auto &p = current->points[idx];
                const auto dist2 = point.distance2(p);
                if(dist2 <= t2)
                {
                    break;
                }
            }

            // Did not find point
            if(idx == N)
            {
                retval = nullptr;
            }
            else
            {
                retval = &current->data[idx];
            }
        }
        return retval;
    }

    void insert(node *current, unsigned int depth, const VectorType &loc, const DataType &r)
    {
        // No matter what we need to add this point to the current bounding box
        current->bb.expand(loc);
        
        // This node has children
        if(current->has_split)
        {
            const auto dim = current->dim;
            if(loc[dim] < current->split)
            {
                // Go left
                insert(current->left, depth+1, loc, r);
            }
            else // (loc[dim] >= current->split)
            {
                // Go right
                insert(current->right, depth+1, loc, r);
            }
        }
        else
        {
            // This is a leaf node
            // Determine if the node needs to be split
            if((current->points.size()) == point_vec_size)
            {
                // This will add the point and data to the correct child
                node_split(current, loc, r);
                tree_depth = std::max(tree_depth, (IndexType)depth+1);
                npoints++;
            }
            else
            {
                // This node does not need to be split
                node_add_data(current, loc, r);
                npoints++;
            }
        }
    }

    template<typename Func>
    void traverse_lnr(Func &&func, node *node, unsigned int depth = 0)
    {
        if(node->left) { traverse_lnr(func, node->left, depth + 1); }
        func(node, depth);
        if(node->right) { traverse_lnr(func, node->right, depth + 1); }
    }

    template<typename Func>
    void traverse_lrn(Func &&func, node *node, unsigned int depth = 0)
    {
        if(node->left) { traverse_lrn(func, node->left, depth + 1); }
        if(node->right) { traverse_lrn(func, node->right, depth + 1); }
        func(node, depth);
    }

    // Keep track of tree performance
    IndexType npoints{0u};
    IndexType nnodes{0u};
    IndexType tree_depth{0u};

    node *root{nullptr};
    IndexType point_vec_size{32};
    std::vector<IndexType> scratch;
};

}

}

#endif
