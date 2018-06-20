//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2018, Lawrence Livermore National Security, LLC.
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
/// file: adios_test_utils.hpp
///
//-----------------------------------------------------------------------------

#ifndef ADIOS_TEST_UTILS_HPP
#define ADIOS_TEST_UTILS_HPP

// Some utility functions for ADIOS tests.

//-----------------------------------------------------------------------------
template <typename T>
bool
compare_arrays(const T *a, const T *b, int len)
{
    if(a == NULL || b == NULL)
        return false;
    for(int i = 0; i < len; ++i)
    {
        if(a[i] != b[i])
        {
//std::cout << "a[" << i << "]=" << (long)a[i] << ", b[" << i << "]=" << (long)b[i] << std::endl;
            return false;
        }
    }
    return true;
}

//-----------------------------------------------------------------------------
template <typename T>
bool
compare_arrays_tol(const T *a, const T *b, int len, T tolerance)
{
    if(a == NULL || b == NULL)
        return false;
    for(int i = 0; i < len; ++i)
    {
        T delta = a[i] - b[i];
        if(delta < 0)
            delta = -delta;
        if(delta > tolerance)
        {
//std::cout << "a[" << i << "]=" << (long)a[i] << ", b[" << i << "]=" << (long)b[i] << std::endl;
            return false;
        }
    }
    return true;
}

//-----------------------------------------------------------------------------
bool
compare_nodes(const conduit::Node &out_root, const Node &in_root, const Node &node,
    bool exact = true, double tolerance = 0.)
{
    bool equal = false;

    if(node.number_of_children() == 0)
    {
        if(in_root.has_path(node.path()))
        {
            const Node &in_node = in_root[node.path()];
            // Check that types were preserved.
            if(node.dtype().name() == in_node.dtype().name())
            {
                if(node.dtype().number_of_elements() ==
                   in_node.dtype().number_of_elements())
                {
                    if(node.dtype().number_of_elements() > 1 && !node.dtype().is_string())
                    {
                        int n = node.dtype().number_of_elements();
                        // arrays
                        if(node.dtype().is_int8())
                            equal = compare_arrays(node.as_int8_ptr(), in_node.as_int8_ptr(), n);
                        else if(node.dtype().is_int16())
                            equal = compare_arrays(node.as_int16_ptr(), in_node.as_int16_ptr(), n);
                        else if(node.dtype().is_int32())
                            equal = compare_arrays(node.as_int32_ptr(), in_node.as_int32_ptr(), n);
                        else if(node.dtype().is_int64())
                            equal = compare_arrays(node.as_int64_ptr(), in_node.as_int64_ptr(), n);
                        else if(node.dtype().is_uint8())
                            equal = compare_arrays(node.as_uint8_ptr(), in_node.as_uint8_ptr(), n);
                        else if(node.dtype().is_uint16())
                            equal = compare_arrays(node.as_uint16_ptr(), in_node.as_uint16_ptr(), n);
                        else if(node.dtype().is_uint32())
                            equal = compare_arrays(node.as_uint32_ptr(), in_node.as_uint32_ptr(), n);
                        else if(node.dtype().is_uint64())
                            equal = compare_arrays(node.as_uint64_ptr(), in_node.as_uint64_ptr(), n);
                        else if(node.dtype().is_float32())
                        {
                            if(exact)
                                equal = compare_arrays(node.as_float32_ptr(), in_node.as_float32_ptr(), n);
                            else
                                equal = compare_arrays_tol(node.as_float32_ptr(), in_node.as_float32_ptr(), n, (float32)tolerance);
                        }
                        else if(node.dtype().is_float64())
                        {
                            if(exact)
                                equal = compare_arrays(node.as_float64_ptr(), in_node.as_float64_ptr(), n);
                            else
                                equal = compare_arrays_tol(node.as_float64_ptr(), in_node.as_float64_ptr(), n, (float64)tolerance);
                        }
                        else
                        {
                            CONDUIT_INFO(node.path() << " unsupported array type: "
                                         << node.dtype().name());
                        }
                    }
                    else
                    {
                        // scalars
                        if(node.dtype().is_int8())
                            equal = (node.as_int8() == in_node.as_int8());
                        else if(node.dtype().is_int16())
                            equal = (node.as_int16() == in_node.as_int16());
                        else if(node.dtype().is_int32())
                            equal = (node.as_int32() == in_node.as_int32());
                        else if(node.dtype().is_int64())
                            equal = (node.as_int64() == in_node.as_int64());
                        else if(node.dtype().is_uint8())
                            equal = (node.as_uint8() == in_node.as_uint8());
                        else if(node.dtype().is_uint16())
                            equal = (node.as_uint16() == in_node.as_uint16());
                        else if(node.dtype().is_uint32())
                            equal = (node.as_uint32() == in_node.as_uint32());
                        else if(node.dtype().is_uint64())
                            equal = (node.as_uint64() == in_node.as_uint64());
                        else if(node.dtype().is_float32())
                            equal = (node.as_float32() == in_node.as_float32());
                        else if(node.dtype().is_float64())
                            equal = (node.as_float64() == in_node.as_float64());
                        else if(node.dtype().is_string())
                            equal = (node.as_string() == in_node.as_string());
                        else
                        {
                            CONDUIT_INFO(node.path() << " unsupported type: "
                                         << node.dtype().name());
                        }
                    }
                }
                else
                {
                    CONDUIT_INFO(node.path() << " differing number_of_elements: "
                                 << node.dtype().number_of_elements() << " != "
                                 << in_node.dtype().number_of_elements());
                }
//                if(equal)
//                    std::cout << node.path() << " is equal" << std::endl;
            }
            else
            {
                CONDUIT_INFO(node.path() << " types not equal ("
                             << node.dtype().name() << "!=" 
                             << in_node.dtype().name() << ")");
            }
        }
        else
        {
            CONDUIT_INFO(node.path() << " not found.");
        }
    }
    else
    {
        equal = true;
        for(conduit::index_t i = 0; i < node.number_of_children(); ++i)
        {
            if(!compare_nodes(out_root, in_root, node.child(i), exact, tolerance))
            {
                equal = false;
                break;
            }
        }
    }

    return equal;
}

//-----------------------------------------------------------------------------
void add_rectilinear_mesh(Node &n, float64 origin[3], float64 size[3], int dims[3])
{
    std::vector<float64> coords[3],radius;
    for(int c = 0; c < 3; ++c)
    {
        coords[c].reserve(dims[c]);
        for(int i = 0; i < dims[c]; ++i)
        {
            float64 t = float64(i) / float64(dims[c] - 1);
            coords[c].push_back(origin[c] + t * size[c]);
        }
    }

    radius.reserve(dims[0]*dims[1]*dims[2]);
    for(int k = 0; k < dims[2]; ++k)
    for(int j = 0; j < dims[1]; ++j)
    for(int i = 0; i < dims[0]; ++i)
    {
        float64 x = coords[0][i];
        float64 y = coords[1][j];
        float64 z = coords[2][k];
        radius.push_back(sqrt(x*x + y*y + z*z));
    }   

    n["coordsets/coords/type"] = "rectilinear";
    n["coordsets/coords/values/x"] = coords[0];
    n["coordsets/coords/values/y"] = coords[1];
    n["coordsets/coords/values/z"] = coords[2];
    n["topologies/mesh/coordset"] = "coords";
    n["topologies/mesh/type"] = "rectilinear";
    n["topologies/mesh/elements/origin/i0"] = origin[0];
    n["topologies/mesh/elements/origin/j0"] = origin[1];
    n["topologies/mesh/elements/origin/k0"] = origin[2];

    n["fields/radius/association"] = "vertex";
    n["fields/radius/type"] = "scalar";
    n["fields/radius/topology"] = "mesh";
    n["fields/radius/values"] = radius;
}

#endif
