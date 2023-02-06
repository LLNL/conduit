// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh_examples_poly_star.cpp
///
//-----------------------------------------------------------------------------

#if defined(CONDUIT_PLATFORM_WINDOWS)
#define NOMINMAX
#undef min
#undef max
#include "windows.h"
#endif

//-----------------------------------------------------------------------------
// std lib includes
//-----------------------------------------------------------------------------
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <limits>
#include <algorithm>
#include <cassert>
#include <map>
#include <set>
#include <vector>
#include <queue>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mesh_examples.hpp"
#include "conduit_blueprint_mesh.hpp"


//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{


//-----------------------------------------------------------------------------
// -- begin conduit::blueprint:: --
//-----------------------------------------------------------------------------
namespace blueprint
{


//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh --
//-----------------------------------------------------------------------------
namespace mesh
{


//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::examples --
//-----------------------------------------------------------------------------
namespace examples
{

//---------------------------------------------------------------------------//
void polystar(Node &res)
{
    res.reset();
    // create an explicit coordset

    res["coordsets/coords/type"] = "explicit";
    Node &n_vals_x = res["coordsets/coords/values/x"];
    Node &n_vals_y = res["coordsets/coords/values/y"];

    // center line:
    // 0:  0,0
    // 1:  2,0
    // 2:  3,0
    // 3:  4,0
    // 4:  5,0
    // 5:  7,0

    // top line:
    // 6:  2,1
    // 7:  3,1
    // 8:  4,1
    // 9:  5,1

    // bottom line:
    // 10:  2,-1
    // 11:  3,-1
    // 12:  4,-1
    // 13:  5,-1
    
    // star top
    // 14:  3.5,5

    // star bottom
    // 14:  3.5,-5

    n_vals_x = {0.0, 2.0, 3.0, 4.0, 5.0, 7.0,
                2.0, 3.0, 4.0, 5.0,
                2.0, 3.0, 4.0, 5.0,
                3.5,3.5};

    n_vals_y = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 1.0,
                -1.0, -1.0, -1.0, -1.0,
                5.0,-5.0};


    // create a points topology
    res["topologies/points/type"] = "points";
    res["topologies/points/coordset"] = "coords";

    // create our polyhedral topo
    res["topologies/topo/type"] = "unstructured";
    res["topologies/topo/coordset"] = "coords";
    
    res["topologies/topo/coordset"] = "coords";

    res["topologies/topo/elements/shape"] = "polygonal";
        
        
    Node &n_conn = res["topologies/topo/elements/connectivity"];
    Node &n_szes = res["topologies/topo/elements/sizes"];
    Node &n_offs = res["topologies/topo/elements/offsets"];
    
    
    // left triangle:
    // 0, 10, 1, 6

    // left top sq:
    // 1, 2, 7, 6

    // left bot sq:
    // 10, 11, 1, 0

    // mid quad (one hanging node on left):
    // 11, 12, 3, 7, 6, 

    // right bot sq:
    // 12, 13, 4, 3

    // right top sq:
    // 3, 4, 9, 8

    // right triangle:
    // 13, 5, 9, 4

    // top triangle: (2 hanging nodes on bottom)
    // 6, 9, 14

    // bottom triangle:
    // 10, 14, 13, 12, 11


    n_conn = {  0, 10,  1,  6, // l tri
                1,  2,  7,  6, // l t sq
               10, 11,  2,  1, // l b sq
               11, 12,  3,  8, 7, // mq
               12, 13,  4,  3, // r b sq
                3,  4,  9,  8, // r t sq
               13,  5,  9,  4, // r tri
                6,  9, 14, // t tri
               10, 15, 13, 12, 11 // b tri
              };

    n_szes = { 4,
               4,
               4, 
               5,
               4,
               4,
               4,
               3,
               5
             };
 
    n_offs = {  0,
                4,
                8,
               12,
               17,
               21,
               25,
               29,
               32
             };

    // generate explicit sides and corners for this mesh
    Node s2dmap, d2smap;

    conduit::blueprint::mesh::topology::unstructured::generate_sides(res["topologies/topo"],
                                                                     res["topologies/sides"],
                                                                     res["coordsets/sides_points"],
                                                                     s2dmap,
                                                                     d2smap);


    conduit::blueprint::mesh::topology::unstructured::generate_corners(res["topologies/topo"],
                                                                       res["topologies/corners"],
                                                                       res["coordsets/corners_points"],
                                                                       s2dmap,
                                                                       d2smap);

    // create the fields

    //---
    // verts per ele fields
    //---

    // actual number of vertices per element
    res["fields/nverts/association"] = "element";
    res["fields/nverts/topology"] = "topo";

    // hypothetical max number of vertices per element
    res["fields/nverts_max/association"] = "element";
    res["fields/nverts_max/topology"] = "topo";

    // diff between actual and hypothetical
    res["fields/nverts_diff/association"] = "element";
    res["fields/nverts_diff/topology"] = "topo";

    Node &n_nv      = res["fields/nverts/values"];
    Node &n_nv_max  = res["fields/nverts_max/values"];
    Node &n_nv_diff = res["fields/nverts_diff/values"];
    
    n_nv      = {4,4,4,5,4,4,4,3,5};
    n_nv_max  = {4,4,4,6,4,4,4,5,5};
    n_nv_diff = {0,0,0,1,0,0,0,2,0};

    //---
    // eles per vert fields
    //---

    // actual number of elements per vertex
    res["fields/neles/association"] = "vertex";
    res["fields/neles/topology"] = "topo";

    // hypothetical max number of elements per vertex
    res["fields/neles_max/association"] = "vertex";
    res["fields/neles_max/topology"] = "topo";

    // diff between actual and hypothetical
    res["fields/neles_diff/association"] = "vertex";
    res["fields/neles_diff/topology"] = "topo";


    Node &n_ne      = res["fields/neles/values"];
    Node &n_ne_max  = res["fields/neles_max/values"];
    Node &n_ne_diff = res["fields/neles_diff/values"];

    n_ne      = {1,3,2,3,3,1,3,2,2,3,3,3,3,3,1,1};
    n_ne_max  = {1,3,3,3,3,1,3,3,3,3,3,3,3,3,1,1};
    n_ne_diff = {0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0};
    
}


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::examples --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint:: --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
