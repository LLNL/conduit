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
/// file: conduit_blueprint_mesh_examples.cpp
///
//-----------------------------------------------------------------------------

#if defined(CONDUIT_PLATFORM_WINDOWS)
#define NOMINMAX
#undef min
#undef max
#include "Windows.h"
#endif

//-----------------------------------------------------------------------------
// std lib includes
//-----------------------------------------------------------------------------
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <cassert>
#include <map>
#include <set>
#include <vector>

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
const float64 PI_VALUE = 3.14159265359;


//---------------------------------------------------------------------------//
struct point
{
    point(float64 px = 0.0, float64 py = 0.0, float64 pz = 0.0) : x(px), y(py), z(pz) {};
    point(float64* ps) : x(ps[0]), y(ps[1]), z(ps[2]) {};

    bool operator<(const point& other) const
    {
        float64 mvals[3] = {this->x, this->y, this->z};
        float64 ovals[3] = {other.x, other.y, other.z};

        for(index_t i = 0; i < 3; i++)
        {
            if(fabs(mvals[i] - ovals[i]) >= 1e-6)
            {
                return mvals[i] < ovals[i];
            }
        }
        return false;
    };

    float64 x, y, z;
};


//---------------------------------------------------------------------------//
void basic_init_example_element_scalar_field(index_t nele_x,
                                             index_t nele_y,
                                             index_t nele_z,
                                             Node &res,
                                             index_t prims_per_ele=1)
{
    index_t nele = nele_x*nele_y;

    if(nele_z > 0)
    {
        nele = nele * nele_z;
    }

    res["association"] = "element";
    res["topology"] = "mesh";
    res["volume_dependent"] = "false";
    res["values"].set(DataType::float64(nele*prims_per_ele));

    float64 *vals = res["values"].value();
    for(index_t i = 0; i < nele*prims_per_ele; i++)
    {
        vals[i] = i + 0.0;
    }
}


//---------------------------------------------------------------------------//
void braid_init_example_state(Node &res)
{
    res["state/time"]   = (float64)3.1415;
    res["state/cycle"]  = (uint64) 100;
}


//---------------------------------------------------------------------------//
void braid_init_example_point_scalar_field(index_t npts_x,
                                           index_t npts_y,
                                           index_t npts_z,
                                           Node &res)
{

    if(npts_z < 1) 
    {
        npts_z = 1;
    }

    index_t npts = npts_x * npts_y * npts_z;
    
    res["association"] = "vertex";
    res["type"] = "scalar";
    res["topology"] = "mesh";
    res["values"].set(DataType::float64(npts));
    
    float64 *vals = res["values"].value();

    float64 dx = (float) (4.0 * PI_VALUE) / float64(npts_x - 1);
    float64 dy = (float) (2.0 * PI_VALUE) / float64(npts_y-1);
    float64 dz = (float) (3.0 * PI_VALUE) / float64(npts_z-1);
    
    index_t idx = 0;

    for(index_t k = 0; k < npts_z ; k++)
    {
        float64 cz =  (k * dz) - (1.5 * PI_VALUE);

        for(index_t j = 0; j < npts_y ; j++)
        {
            float64 cy =  (j * dy) - ( PI_VALUE);
            for(index_t i = 0; i < npts_x ; i++)
            {
            
                float64 cx =  (i * dx) + (2.0 * PI_VALUE);
                
                float64 cv =  sin( cx ) + 
                              sin( cy ) + 
                              2 * cos(sqrt( (cx*cx)/2.0 +cy*cy) / .75) +
                              4 * cos( cx*cy / 4.0);
                                  
                if(npts_z > 1)
                {
                    cv += sin( cz ) + 
                          1.5 * cos(sqrt(cx*cx + cy*cy + cz*cz) / .75);
                }
                
                vals[idx] = cv;
                idx++;
            }
        }
    }
}

//---------------------------------------------------------------------------//
void braid_init_example_point_vector_field(index_t npts_x,
                                           index_t npts_y,
                                           index_t npts_z,
                                           Node &res)
{
    index_t npts = npts_x * npts_y * npts_z;
    
    res["association"] = "vertex";
    res["type"] = "vector";
    res["topology"] = "mesh";
    res["values/u"].set(DataType::float64(npts));
    res["values/v"].set(DataType::float64(npts));

    float64 *u_vals = res["values/u"].value();
    float64 *v_vals = res["values/v"].value();
    float64 *w_vals = NULL;
    
    if(npts_z > 1)
    {
        res["values/w"].set(DataType::float64(npts));
        w_vals = res["values/w"].value();
    }

    // this logic is from the explicit coord set setup function
    // we are using the coords (distance from origin)
    // to create an example vector field
    
    float64 dx = 20.0 / float64(npts_x - 1);
    float64 dy =  20.0 / float64(npts_y-1);
    float64 dz = 0.0;

    if(npts_z > 1)
    {
        dz = 20.0 / float64(npts_z-1);
    }

    index_t idx = 0;
    for(index_t k = 0; k < npts_z ; k++)
    {
        float64 cz = -10.0 + k * dz;
        
        for(index_t j = 0; j < npts_y ; j++)
        {
            float64 cy =  -10.0 + j * dy;
            
            for(index_t i = 0; i < npts_x ; i++)
            {
                float64 cx =  -10.0 + i * dx;

                u_vals[idx] = cx;
                v_vals[idx] = cy;

                if(npts_z > 1)
                {
                    w_vals[idx] = cz;
                }
                
                idx++;
            }
        
        }
    }
}


//---------------------------------------------------------------------------//
void braid_init_example_element_scalar_field(index_t nele_x,
                                             index_t nele_y,
                                             index_t nele_z,
                                             Node &res,
                                             index_t prims_per_ele=1)
{
    index_t nele = nele_x * nele_y;
    
    if(nele_z > 0)
    {
        nele = nele * nele_z;
    }

    res["association"] = "element";
    res["type"] = "scalar";
    res["topology"] = "mesh";
    
    index_t vals_size = nele * prims_per_ele;
    
    res["values"].set(DataType::float64(vals_size));

    float64 *vals = res["values"].value();

    float64 dx = 20.0 / float64(nele_x);
    float64 dy = 20.0 / float64(nele_y);
    float64 dz = 0.0f;
    
    if(nele_z > 0 )
    {
        dz = 20.0 / float64(nele_z);
    }
    
    index_t idx = 0;
    for(index_t k = 0; (idx == 0 || k < nele_z); k++)
    {
        float64 cz =  (k * dz) + -10.0;

        for(index_t j = 0; (idx == 0 || j < nele_y) ; j++)
        {
            float64 cy =  (j * dy) + -10.0;
            
            for(index_t i = 0; (idx == 0 || i < nele_x) ; i++)
            {
                float64 cx =  (i * dx) + -10.0;

                float64 cv = 10.0 * sqrt( cx*cx + cy*cy );
                
                if(nele_z != 0)
                {
                    cv = 10.0 * sqrt( cx*cx + cy*cy +cz*cz );
                }

                for(index_t ppe = 0; ppe < prims_per_ele; ppe++ )
                {
                    vals[idx] = cv;
                    idx++;
                }
            }
        }
    }
}


//---------------------------------------------------------------------------//
void braid_init_example_matset(index_t nele_x,
                               index_t nele_y,
                               index_t nele_z,
                               Node &res)
{
    index_t nele = nele_x * nele_y * ((nele_z > 0) ? nele_z : 1);

    res["topology"] = "mesh";

    Node &vfs = res["volume_fractions"];
    vfs["mat1"].set(DataType::float64(nele));
    vfs["mat2"].set(DataType::float64(nele));

    float64 *mat1_vals = vfs["mat1"].value();
    float64 *mat2_vals = vfs["mat2"].value();

    for(index_t k = 0, idx = 0; (idx == 0 || k < nele_z); k++)
    {
        for(index_t j = 0; (idx == 0 || j < nele_y) ; j++)
        {
            for(index_t i = 0; (idx == 0 || i < nele_x) ; i++, idx++)
            {
                float64 mv = (nele_x == 1) ? 0.5 : i / (nele_x - 1.0);

                mat1_vals[idx] = mv;
                mat2_vals[idx] = 1.0 - mv;
            }
        }
    }
}


//---------------------------------------------------------------------------//
void braid_init_example_specset(index_t nele_x,
                                index_t nele_y,
                                index_t nele_z,
                                Node &res)
{
    index_t nele = nele_x * nele_y * ((nele_z > 0) ? nele_z : 1);

    res["matset"] = "mesh";
    res["volume_dependent"] = "false";

    Node &mfs = res["matset_values"];
    mfs["mat1/spec1"].set(DataType::float64(nele));
    mfs["mat1/spec2"].set(DataType::float64(nele));
    mfs["mat2/spec1"].set(DataType::float64(nele));
    mfs["mat2/spec2"].set(DataType::float64(nele));

    float64 *spec1_vals[2] = {mfs["mat1/spec1"].value(), mfs["mat2/spec1"].value()};
    float64 *spec2_vals[2] = {mfs["mat1/spec2"].value(), mfs["mat2/spec2"].value()};

    for(index_t k = 0, idx = 0; (idx == 0 || k < nele_z); k++)
    {
        for(index_t j = 0; (idx == 0 || j < nele_y) ; j++)
        {
            for(index_t i = 0; (idx == 0 || i < nele_x) ; i++, idx++)
            {
                float64 mv = (nele_y == 1) ? 0.5 : i / (nele_y - 1.0);
                for(index_t s = 0; s < 2; s++)
                {
                    spec1_vals[s][idx] = mv;
                    spec2_vals[s][idx] = 1.0 - mv;
                }
            }
        }
    }
}


//---------------------------------------------------------------------------//
void braid_init_uniform_coordset(index_t npts_x,
                                 index_t npts_y,
                                 index_t npts_z,
                                 Node &coords)
{
    coords["type"] = "uniform";
    Node &dims = coords["dims"];
    dims["i"] = npts_x;
    dims["j"] = npts_y;

    if(npts_z > 1)
    {
        dims["k"] = npts_z;
    }
        
    // -10 to 10 in each dim, 
    Node &origin = coords["origin"];
    origin["x"] = -10.0;
    origin["y"] = -10.0;
    
    if(npts_z > 1)
    {
        origin["z"] = -10.0;
    }
    
    Node &spacing = coords["spacing"];
    spacing["dx"] = 20.0 / (float64)(npts_x-1);
    spacing["dy"] = 20.0 / (float64)(npts_y-1);

    if(npts_z > 1 )
    {
        spacing["dz"] = 20.0 / (float64)(npts_z-1);
    }
}


//---------------------------------------------------------------------------//
void braid_init_rectilinear_coordset(index_t npts_x,
                                     index_t npts_y,
                                     index_t npts_z,
                                     Node &coords)
{
    coords["type"] = "rectilinear";
    Node &coord_vals = coords["values"];
    coord_vals["x"].set(DataType::float64(npts_x));
    coord_vals["y"].set(DataType::float64(npts_y));
    
    if(npts_z > 1)
    {
        coord_vals["z"].set(DataType::float64(npts_z));
    }

    float64 *x_vals = coord_vals["x"].value();
    float64 *y_vals = coord_vals["y"].value();
    float64 *z_vals = NULL;

    if(npts_z > 1)
    {
        z_vals = coord_vals["z"].value();
    }


    float64 dx = 20.0 / (float64)(npts_x-1);
    float64 dy = 20.0 / (float64)(npts_y-1);
    float64 dz = 0.0;
    
    if(npts_z > 1)
    {
        dz = 20.0 / (float64)(npts_z-1);
    }

    for(int i=0; i < npts_x; i++)
    {
        x_vals[i] = -10.0 + i * dx;
    }
    
    for(int j=0; j < npts_y; j++)
    {
        y_vals[j] = -10.0 + j * dy;
    }
    
    if(npts_z > 1)
    {
        for(int k=0; k < npts_z; k++)
        {
            z_vals[k] = -10.0 + k * dz;
        }
    }
}

//---------------------------------------------------------------------------//
void
braid_init_explicit_coordset(index_t npts_x,
                             index_t npts_y,
                             index_t npts_z,
                             Node &coords)
{
    coords["type"] = "explicit";
    
    index_t npts = npts_x * npts_y;

    if(npts_z > 1)
    {
        npts *= npts_z;
    }

    // also support interleaved
    Node &coord_vals = coords["values"];
    coord_vals["x"].set(DataType::float64(npts));
    coord_vals["y"].set(DataType::float64(npts));

    if(npts_z > 1)
    {
        coord_vals["z"].set(DataType::float64(npts));
    }

    float64 *x_vals = coord_vals["x"].value();
    float64 *y_vals = coord_vals["y"].value();
    float64 *z_vals = NULL;
    
    if(npts_z > 1)
    {
        z_vals = coord_vals["z"].value();
    }

    float64 dx = 20.0 / float64(npts_x-1);
    float64 dy = 20.0 / float64(npts_y-1);

    float64 dz = 0.0;

    if(npts_z > 1)
    {
        dz = 20.0 / float64(npts_z-1);
    }

    index_t idx = 0;
    for(index_t k = 0; k < npts_z ; k++)
    {
        float64 cz = -10.0 + k * dz;
        
        for(index_t j = 0; j < npts_y ; j++)
        {
            float64 cy =  -10.0 + j * dy;
            
            for(index_t i = 0; i < npts_x ; i++)
            {
                x_vals[idx] = -10.0 + i * dx;
                y_vals[idx] = cy;
                
                if(npts_z > 1)
                {
                    z_vals[idx] = cz;
                }
                
                idx++;
            }
        
        }
    }
}


//---------------------------------------------------------------------------//
void braid_init_example_adjset(Node &mesh)
{
    typedef std::map< point, std::map<index_t, index_t> > point_doms_map;
    typedef std::map<std::set<index_t>, std::vector<std::vector<index_t> > > group_idx_map;

    const std::string dim_names[3] = {"x", "y", "z"};
    const index_t dim_count = blueprint::mesh::coordset::dims(
        mesh.child(0).fetch("coordsets").child(0));

    // From mesh data, create a map from domain combination tuple to point list.
    // These domain combination tuples represent groups and the point lists contain
    // the points that lie on the shared boundary between these domains.
    point_doms_map mesh_point_doms_map;
    conduit::NodeConstIterator doms_it = mesh.children();
    while(doms_it.has_next())
    {
        doms_it.next();
        const conduit::Node& dom_node = doms_it.node();
        const conduit::Node& dom_coords = dom_node["coordsets/coords/values"];
        const index_t dom_id = dom_node["state/domain_id"].to_uint64();

        conduit::float64_array dom_dim_coords[3];
        for(index_t d = 0; d < dim_count; d++)
        {
            dom_dim_coords[d] = dom_coords[dim_names[d]].as_float64_array();
        }

        for(index_t i = 0; i < dom_dim_coords[0].number_of_elements(); i++)
        {
            float64 cvals[3] = {0.0, 0.0, 0.0};
            for(index_t d = 0; d < dim_count; d++)
            {
                cvals[d] = dom_dim_coords[d][i];
            }
            mesh_point_doms_map[point(&cvals[0])][dom_id] = i;
        }
    }

    group_idx_map groups_map;
    point_doms_map::const_iterator pm_itr;
    for(pm_itr = mesh_point_doms_map.begin();
        pm_itr != mesh_point_doms_map.end(); ++pm_itr)
    {
        const std::map<index_t, index_t>& point_dom_idx_map = pm_itr->second;
        if(point_dom_idx_map.size() > 1)
        {
            std::set<index_t> point_group;

            std::map<index_t, index_t>::const_iterator pg_itr;
            for(pg_itr = point_dom_idx_map.begin();
                pg_itr != point_dom_idx_map.end(); ++pg_itr)
            {
                point_group.insert(pg_itr->first);
            }

            std::vector<std::vector<index_t> >& group_indices = groups_map[point_group];
            if(group_indices.empty())
            {
                group_indices.resize(point_group.size());
            }

            std::set<index_t>::const_iterator gd_itr;
            std::set<index_t>::size_type gi = 0;
            for(gd_itr = point_group.begin();
                gd_itr != point_group.end(); ++gd_itr, ++gi)
            {
                index_t g_idx = static_cast<index_t>(point_dom_idx_map.find(*gd_itr)->second);
                group_indices[gi].push_back(g_idx);
            }
        }
    }

    group_idx_map::const_iterator gm_itr;
    index_t gid = 0;
    for(gm_itr = groups_map.begin();
        gm_itr != groups_map.end(); ++gm_itr, ++gid)
    {
        const std::set<index_t>& group_doms = gm_itr->first;
        const std::vector<std::vector<index_t> >& group_indices = gm_itr->second;

        std::ostringstream oss;
        oss << "group" << gid;
        const std::string group_name = oss.str();

        std::set<index_t>::const_iterator dg_itr;
        std::set<index_t>::size_type d = 0;
        for(dg_itr = group_doms.begin();
            dg_itr != group_doms.end(); ++dg_itr, ++d)
        {
          const index_t& dom_id = *dg_itr;
          const std::vector<index_t>& dom_idxs = group_indices[d];

          oss.str("");
          oss << "domain" << dom_id;
          const std::string dom_name = oss.str();

          std::vector<index_t> dom_neighbors(group_doms.begin(), group_doms.end());
          dom_neighbors.erase(dom_neighbors.begin()+d);

          conduit::Node& dom_node = mesh[dom_name]["adjsets"]["mesh_adj"];
          dom_node["association"].set("vertex");
          dom_node["topology"].set("mesh");
          dom_node["groups"][group_name]["neighbors"].set(
            const_cast<index_t*>(dom_neighbors.data()), dom_neighbors.size());
          dom_node["groups"][group_name]["values"].set(
            const_cast<index_t*>(dom_idxs.data()), dom_idxs.size());
        }
    }
}


//---------------------------------------------------------------------------//
void braid_init_example_nestset(Node &mesh)
{
    typedef std::map<point, index_t> point_id_map;
    typedef std::pair<index_t, index_t> window;

    // TODO(JRC): Extend this function to support input domains with cylindrical
    // and spherical coordinates as well.
    const std::string cartesian_dims[3] = {"x", "y", "z"};
    const std::string logical_dims[3] = {"i", "j", "k"};
    const index_t dim_count = blueprint::mesh::coordset::dims(
        mesh.child(0).fetch("coordsets").child(0));

    // initialize data to easily index domains by id/level //

    std::map<index_t, const Node*> mesh_id_map;
	index_t max_dom_id = 0, max_level_id = 0;
	{
        conduit::NodeConstIterator doms_it = mesh.children();
        while(doms_it.has_next())
        {
            const conduit::Node& dom_node = doms_it.next();
            const index_t dom_id = dom_node["state/domain_id"].to_uint64();
            mesh_id_map[dom_id] = &dom_node;
            max_dom_id = std::max(dom_id, max_dom_id);

            const index_t dom_level = dom_node["state/level_id"].to_uint64();
            max_level_id = std::max(dom_level, max_level_id);
        }
    }

    // transform rectilinear input data into unstructured data //

    std::vector<point_id_map> mesh_point_maps(max_dom_id + 1);
    std::vector< std::vector<const Node*> > mesh_level_map(max_level_id + 1);
    {
        conduit::NodeConstIterator doms_it = mesh.children();
        while(doms_it.has_next())
        {
            const conduit::Node &dom_node = doms_it.next();
            const index_t dom_id = dom_node["state/domain_id"].to_uint64();
            const index_t level_id = dom_node["state/level_id"].to_uint64();
            const conduit::Node &dom_coordset = dom_node["coordsets"].child(0);

            conduit::Node dom_coordset_explicit;
            if(dom_coordset["type"].as_string() == "uniform")
            {
                blueprint::mesh::coordset::uniform::to_explicit(
                    dom_coordset, dom_coordset_explicit);
            }
            else if(dom_coordset["type"].as_string() == "rectilinear")
            {
                blueprint::mesh::coordset::rectilinear::to_explicit(
                    dom_coordset, dom_coordset_explicit);
            }
            else
            {
                dom_coordset_explicit.set_external(dom_coordset);
            }
            const index_t num_points = dom_coordset_explicit["values"].
                child(0).dtype().number_of_elements();

            point_id_map &dom_point_map = mesh_point_maps[dom_id];
            {
                for(index_t i = 0; i < num_points; i++)
                {
                    float64 dom_point_vals[3] = {0.0, 0.0, 0.0};
                    for(index_t d = 0; d < dim_count; d++)
                    {
                        conduit::Node &dim_coords =
                            dom_coordset_explicit["values"][cartesian_dims[d]];
                        conduit::Node dim_cval(
                            conduit::DataType(dim_coords.dtype().id(), 1),
                            dim_coords.element_ptr(i), true);
                        dom_point_vals[d] = dim_cval.to_float64();
                    }
                    dom_point_map[point(&dom_point_vals[0])] = i;
                }
            }

            mesh_level_map[level_id].push_back(&dom_node);
        }
    }

    // NOTE(JRC): 'mesh_window_maps' maps a given domain ID to all of the windows
    // for that domain, which are returned via a map from other domain ID to
    // the indices of the extents in the primary domain.
    std::map< index_t, std::map<index_t, window> > mesh_window_maps;
    {
        for(index_t l = 0; l < (index_t)mesh_level_map.size() - 1; l++)
        {
            const std::vector<const Node*> &hi_nodes = mesh_level_map[l];
            const std::vector<const Node*> &lo_nodes = mesh_level_map[l+1];
            for(index_t hi = 0; hi < (index_t)hi_nodes.size(); hi++)
            {
                for(index_t lo = 0; lo < (index_t)lo_nodes.size(); lo++)
                {
                    const Node &hi_node = *hi_nodes[hi];
                    const Node &lo_node = *lo_nodes[lo];

                    const index_t hi_dom_id = hi_node["state/domain_id"].to_uint64();
                    const index_t lo_dom_id = lo_node["state/domain_id"].to_uint64();
                    const point_id_map &hi_point_map = mesh_point_maps[hi_dom_id];
                    const point_id_map &lo_point_map = mesh_point_maps[lo_dom_id];

                    std::vector<point> point_intx_list;
                    point_id_map::const_iterator hi_pt_itr = hi_point_map.begin();
                    point_id_map::const_iterator lo_pt_itr = lo_point_map.begin();
                    while(hi_pt_itr != hi_point_map.end() && lo_pt_itr != lo_point_map.end())
                    {
                        if(hi_pt_itr->first < lo_pt_itr->first)
                        {
                            ++hi_pt_itr;
                        }
                        else if(lo_pt_itr->first < hi_pt_itr->first)
                        {
                            ++lo_pt_itr;
                        }
                        else
                        {
                            point_intx_list.push_back(hi_pt_itr->first);
                            ++hi_pt_itr;
                            ++lo_pt_itr;
                        }
                    }
                    // TODO(JRC): Handle cases wherein the low mesh doesn't
                    // have a sensible window with the high mesh.
                    const point min_intx_point = point_intx_list.front();
                    const point max_intx_point = point_intx_list.back();

                    window &hi_window = mesh_window_maps[hi_dom_id][lo_dom_id];
                    hi_window.first = hi_point_map.at(min_intx_point);
                    hi_window.second = hi_point_map.at(max_intx_point);

                    window &lo_window = mesh_window_maps[lo_dom_id][hi_dom_id];
                    lo_window.first = lo_point_map.at(min_intx_point);
                    lo_window.second = lo_point_map.at(max_intx_point);
                }
            }
        }
    }

    conduit::NodeIterator doms_it = mesh.children();
    while(doms_it.has_next())
    {
        conduit::Node &dom_node = doms_it.next();
        index_t dom_id = dom_node["state/domain_id"].to_uint64();
        index_t dom_level = dom_node["state/level_id"].to_uint64();

        index_t dom_dims[3] = {0, 0, 0}; // needed for 1d to 3d xform per domain
        {
            const conduit::Node &dom_coords = dom_node["coordsets/coords/values"];
            for(index_t d = 0; d < 3; d++)
            {
                dom_dims[d] = !dom_coords.has_child(cartesian_dims[d]) ? 1 :
                    dom_coords[cartesian_dims[d]].dtype().number_of_elements();
            }
        }

        conduit::Node &dom_nestset = dom_node["nestsets/mesh_nest"];
        dom_nestset["association"].set("element");
        dom_nestset["topology"].set("mesh");

        std::map<index_t, window>::const_iterator dom_window_itr;
        for(dom_window_itr = mesh_window_maps[dom_id].begin();
            dom_window_itr != mesh_window_maps[dom_id].end(); ++dom_window_itr)
        {
            index_t odom_id = dom_window_itr->first;
            const conduit::Node &odom_node = *mesh_id_map[odom_id];
            index_t odom_level = odom_node["state/level_id"].to_uint64();

            window window_extrema = dom_window_itr->second;
            std::string window_name;
            {
                std::ostringstream oss;
                // window_{min_dom_id}_{max_dom_id}
                oss << "window_" << std::min(dom_id, odom_id) 
                                 << "_"
                                 << std::max(dom_id, odom_id);
                window_name = oss.str();
            }

            conduit::Node &dom_window = dom_nestset["windows"][window_name];
            dom_window["domain_id"].set(odom_id);
            dom_window["domain_type"].set(dom_level < odom_level ? "child" : "parent");

            index_t window_extents[2][3] = {{0, 0, 0}, {0, 0, 0}};
            for(index_t e = 0; e < 2; e++)
            {
                index_t window_extreme = e == 0 ? window_extrema.first : window_extrema.second;
                index_t *window_extent = &window_extents[e][0];

                index_t dim_remainder = window_extreme;
                for(index_t d = 3; d-- > 0;)
                {
                    index_t dim_stride = 1;
                    for(index_t dd = 0; dd < d; dd++)
                    {
                        dim_stride *= dom_dims[dd];
                    }

                    window_extent[d] = dim_remainder / dim_stride;
                    dim_remainder = dim_remainder % dim_stride;
                }
            }

            for(index_t d = 0; d < dim_count; d++)
            {
                // NOTE(JRC): These values may seem incorrect since they're relative
                // to point space, but they actually work out to calculate the proper
                // values because the coordinate indices for an element will always
                // match its minimum point indices and h-l points is number of elements.
                dom_window["origin"][logical_dims[d]].set(window_extents[0][d]);
                dom_window["dims"][logical_dims[d]].set(
                    window_extents[1][d] - window_extents[0][d]);
            }
        }
    }

    doms_it = mesh.children();
    while(doms_it.has_next())
    {
        conduit::Node &dom_node = doms_it.next();
        conduit::Node &dom_windows_node = dom_node["nestsets/mesh_nest/windows"];
        conduit::NodeIterator windows_it = dom_windows_node.children();
        while(windows_it.has_next())
        {
            conduit::Node &dom_window_node = windows_it.next();
            const std::string dom_window_name = windows_it.name();
            index_t odom_id = dom_window_node["domain_id"].to_uint64();

            const conduit::Node &odom_node = *mesh_id_map[odom_id];
            const conduit::Node &odom_window_node =
                odom_node["nestsets/mesh_nest/windows"][dom_window_name];

            const conduit::Node *parent_window_node, *child_window_node;
            if(dom_window_node["domain_type"].as_string() == "child")
            {
                parent_window_node = &dom_window_node;
                child_window_node = &odom_window_node;
            }
            else
            {
                parent_window_node = &odom_window_node;
                child_window_node = &dom_window_node;
            }

            for(index_t d = 0; d < dim_count; d++)
            {
                dom_window_node["ratio"][logical_dims[d]].set(
                    (*child_window_node)["dims"][logical_dims[d]].to_uint64() /
                    (*parent_window_node)["dims"][logical_dims[d]].to_uint64());
            }
        }
    }
}


//---------------------------------------------------------------------------//
void
braid_uniform(index_t npts_x,
              index_t npts_y,
              index_t npts_z,
              Node &res)
{
    res.reset();
    
    index_t nele_x = npts_x -1;
    index_t nele_y = npts_y -1;
    index_t nele_z = npts_z -1;
    
    braid_init_example_state(res);
    braid_init_uniform_coordset(npts_x,
                                npts_y,
                                npts_z,
                                res["coordsets/coords"]);

    res["topologies/mesh/type"] = "uniform";
    res["topologies/mesh/coordset"] = "coords"; 
    
    Node &fields = res["fields"];


    braid_init_example_point_scalar_field(npts_x,
                                          npts_y,
                                          npts_z,
                                          fields["braid"]);

    braid_init_example_element_scalar_field(nele_x,
                                            nele_y,
                                            nele_z,
                                            fields["radial"]);

    braid_init_example_point_vector_field(npts_x,
                                          npts_y,
                                          npts_z,
                                          fields["vel"]);

}



//---------------------------------------------------------------------------//
void
braid_rectilinear(index_t npts_x,
                  index_t npts_y,
                  index_t npts_z,
                  Node &res)
{
    res.reset();
    
    index_t nele_x = npts_x -1;
    index_t nele_y = npts_y -1;
    index_t nele_z = npts_z -1;
    
    braid_init_example_state(res);
    braid_init_rectilinear_coordset(npts_x,
                                    npts_y,
                                    npts_z,
                                    res["coordsets/coords"]);
    
    res["topologies/mesh/type"] = "rectilinear";
    res["topologies/mesh/coordset"] = "coords"; 
    
    Node &fields = res["fields"];

    braid_init_example_point_scalar_field(npts_x,
                                          npts_y,
                                          npts_z,
                                          fields["braid"]);

    braid_init_example_element_scalar_field(nele_x,
                                            nele_y,
                                            nele_z,
                                            fields["radial"]);

    braid_init_example_point_vector_field(npts_x,
                                          npts_y,
                                          npts_z,
                                          fields["vel"]);

}

//---------------------------------------------------------------------------//
void
braid_structured(index_t npts_x,
                 index_t npts_y,
                 index_t npts_z,
                 Node &res)
{
    res.reset();
    
    index_t nele_x = npts_x -1;
    index_t nele_y = npts_y -1;
    index_t nele_z = npts_z -1;
    
    braid_init_example_state(res);
    braid_init_explicit_coordset(npts_x,
                                 npts_y,
                                 npts_z,
                                 res["coordsets/coords"]);
  
    res["topologies/mesh/type"] = "structured";
    res["topologies/mesh/coordset"] = "coords";
    res["topologies/mesh/elements/dims/i"] = (int32)nele_x;
    res["topologies/mesh/elements/dims/j"] = (int32)nele_y;
    
    if(nele_z > 0)
    {
        res["topologies/mesh/elements/dims/k"] = (int32)nele_z; 
    }

    Node &fields = res["fields"];

    braid_init_example_point_scalar_field(npts_x,
                                          npts_y,
                                          npts_z,
                                          fields["braid"]);
                                          
    braid_init_example_element_scalar_field(nele_x,
                                            nele_y, 
                                            nele_z,
                                            fields["radial"]);

    braid_init_example_point_vector_field(npts_x,
                                          npts_y,
                                          npts_z,
                                          fields["vel"]);

}


//---------------------------------------------------------------------------//
void
braid_points_explicit(index_t npts_x,
                      index_t npts_y,
                      index_t npts_z,
                      Node &res)
{
    res.reset();
    index_t npts_total = npts_x * npts_y * npts_z;
    
    braid_init_example_state(res);
    braid_init_explicit_coordset(npts_x,
                                 npts_y,
                                 npts_z,
                                 res["coordsets/coords"]);
    
    res["topologies/mesh/type"] = "unstructured";
    res["topologies/mesh/coordset"] = "coords";
    res["topologies/mesh/elements/shape"] = "point";
    res["topologies/mesh/elements/connectivity"].set(DataType::int32(npts_total));
    int32 *conn = res["topologies/mesh/elements/connectivity"].value();

    for(int32 i = 0; i < (int32)npts_total ; i++)
    {
        conn[i] = i;
    }

    Node &fields = res["fields"];
    
    braid_init_example_point_scalar_field(npts_x,
                                          npts_y,
                                          npts_z,
                                          fields["braid"]);
    
    braid_init_example_element_scalar_field(npts_x,
                                            npts_y, 
                                            npts_z,
                                            fields["radial"]);

    braid_init_example_point_vector_field(npts_x,
                                          npts_y,
                                          npts_z,
                                          fields["vel"]);

}


//---------------------------------------------------------------------------//
void
braid_points_implicit(index_t npts_x,
                      index_t npts_y,
                      index_t npts_z,
                      Node &res)
{
    res.reset();
    
    braid_init_example_state(res);
    braid_init_explicit_coordset(npts_x,
                                 npts_y,
                                 npts_z,
                                 res["coordsets/coords"]);
    
    res["topologies/mesh/type"] = "points";
    res["topologies/mesh/coordset"] = "coords";

    Node &fields = res["fields"];
    
    braid_init_example_point_scalar_field(npts_x,
                                          npts_y,
                                          npts_z,
                                          fields["braid"]);
    
    braid_init_example_element_scalar_field(npts_x,
                                            npts_y, 
                                            npts_z,
                                            fields["radial"]);

    braid_init_example_point_vector_field(npts_x,
                                          npts_y,
                                          npts_z,
                                          fields["vel"]);

}


//---------------------------------------------------------------------------//
void
braid_quads(index_t npts_x,
            index_t npts_y,
            Node &res)
{
    res.reset();
    
    int32 nele_x = (int32)(npts_x - 1);
    int32 nele_y = (int32)(npts_y - 1);
    int32 nele = nele_x * nele_y;
    
    braid_init_example_state(res);
    braid_init_explicit_coordset(npts_x,
                                 npts_y,
                                 1,
                                 res["coordsets/coords"]);
  
    res["topologies/mesh/type"] = "unstructured";
    res["topologies/mesh/coordset"] = "coords";
    res["topologies/mesh/elements/shape"] = "quad";
    res["topologies/mesh/elements/connectivity"].set(DataType::int32(nele*4));
    int32 *conn = res["topologies/mesh/elements/connectivity"].value();

    int32 idx = 0;
    for(int32 j = 0; j < nele_x ; j++)
    {
        int32 yoff = j * (nele_x+1);
        for(int32 i = 0; i < nele_y; i++)
        {
            conn[idx+0] = yoff + i;
            conn[idx+1] = yoff + i + (nele_x+1);
            conn[idx+2] = yoff + i + 1 + (nele_x+1);
            conn[idx+3] = yoff + i + 1;

            idx+=4;
        }
    }


    Node &fields = res["fields"];

    braid_init_example_point_scalar_field(npts_x,
                                          npts_y,
                                          1,
                                          fields["braid"]);

    braid_init_example_element_scalar_field(nele_x,
                                            nele_y,
                                            0,
                                            fields["radial"]);

    braid_init_example_point_vector_field(npts_x,
                                          npts_y,
                                          1,
                                          fields["vel"]);
}

//---------------------------------------------------------------------------//
void
braid_quads_and_tris(index_t npts_x,
            index_t npts_y,
            Node &res)
{
    res.reset();
    
    int32 nele_x = (int32)(npts_x - 1);
    int32 nele_y = (int32)(npts_y - 1);
    
    braid_init_example_state(res);
    braid_init_explicit_coordset(npts_x,
                                 npts_y,
                                 1,
                                 res["coordsets/coords"]);
  
    res["topologies/mesh/type"] = "unstructured";
    res["topologies/mesh/coordset"] = "coords";

    Node &elems = res["topologies/mesh/elements"];
    elems["element_types/quads/stream_id"] = 9; // VTK_QUAD
    elems["element_types/quads/shape"]     = "quad";
    elems["element_types/tris/stream_id"]  = 5; // VTK_TRIANGLE
    elems["element_types/tris/shape"]      = "tri";

    // Fill in stream IDs and calculate size of the connectivity array
    int32 count   = 0;
    int32 ielem   = 0;
    std::vector< int32 > stream_ids_buffer;
    std::vector< int32 > stream_lengths;

    for(int32 j = 0; j < nele_x ; j++)
    {
        for(int32 i = 0; i < nele_y; i++)
        {
             if ( ielem % 2 == 0 )
             {
                 // QUAD
                 stream_ids_buffer.push_back( 9 );
                 stream_lengths.push_back( 1 );
                 count += 4;
             }
             else
             {
                 // TRIANGLE
                 stream_ids_buffer.push_back( 5 );
                 count += 6;
                 stream_lengths.push_back( 2 );
             }

             ++ielem;

        } // END for all i

    } // END for all j


    elems["element_index/stream_ids"].set(stream_ids_buffer);
    elems["element_index/element_counts"].set(stream_lengths);

    // Allocate connectivity array
    elems["stream"].set(DataType::int32(count));
    int32* conn = elems["stream"].value();

    // Fill in connectivity array
    int32 idx = 0;
    int32 elem  = 0;
    for(int32 j = 0; j < nele_x ; j++)
    {
        int32 yoff = j * (nele_x+1);

        for(int32 i = 0; i < nele_y; i++)
        {
            int32 n1 = yoff + i;
            int32 n2 = n1 + (nele_x+1);
            int32 n3 = n1 + 1 + (nele_x+1);
            int32 n4 = n1 + 1;

            if ( elem % 2 == 0 )
            {
                conn[idx  ] = n1;
                conn[idx+1] = n2;
                conn[idx+2] = n3;
                conn[idx+3] = n4;
                idx+=4;
            }
            else
            {
               conn[idx   ] = n1;
               conn[idx+1 ] = n2;
               conn[idx+2 ] = n4;
               idx+=3;

               conn[idx   ] = n2;
               conn[idx+1 ] = n3;
               conn[idx+2 ] = n4;
               idx+=3;
            }

            ++elem;

        } // END for all i

    } // END for all j


    Node &fields = res["fields"];

    braid_init_example_point_scalar_field(npts_x,
                                          npts_y,
                                          1,
                                          fields["braid"]);

    braid_init_example_point_vector_field(npts_x,
                                          npts_y,
                                          1,
                                          fields["vel"]);

    // braid_init_example_element_scalar_field(nele_x,
    //                                         nele_y,
    //                                         0,
    //                                         fields["radial"]);
}

//---------------------------------------------------------------------------//
void
braid_quads_and_tris_offsets(index_t npts_x,
                             index_t npts_y,
                             Node &res)
{

    res.reset();

    int32 nele_x = (int32)(npts_x - 1);
    int32 nele_y = (int32)(npts_y - 1);

    braid_init_example_state(res);
    braid_init_explicit_coordset(npts_x,
                                 npts_y,
                                 1,
                                 res["coordsets/coords"]);

    res["topologies/mesh/type"] = "unstructured";
    res["topologies/mesh/coordset"] = "coords";
    
    Node &elems = res["topologies/mesh/elements"];
    elems["element_types/quads/stream_id"] = 9; // VTK_QUAD
    elems["element_types/quads/shape"]     = "quad";
    elems["element_types/tris/stream_id"]  = 5; // VTK_TRIANGLE
    elems["element_types/tris/shape"]      = "tri";

    // Fill in stream IDs and calculate size of the connectivity array
    int32 count   = 0;
    int32 ielem   = 0;
    std::vector< int32 > stream_ids;
    std::vector< int32 > stream_offsets;
    stream_offsets.push_back( 0 );

    for(int32 j = 0; j < nele_x ; j++)
    {
        for(int32 i = 0; i < nele_y; i++)
        {
            int32 next = stream_offsets.back();

             if ( ielem % 2 == 0 )
             {
                 // QUAD
                 stream_offsets.push_back( next+4 );
                 stream_ids.push_back( 9 );
                 count += 4;
             }
             else
             {
                 // TRIANGLE
                 stream_offsets.push_back( next+3 );
                 stream_offsets.push_back( next+6 );
                 stream_ids.push_back( 5 );
                 stream_ids.push_back( 5 );
                 count += 6;
             }

             ++ielem;

        } // END for all i

    } // END for all j


    elems["element_index/stream_ids"].set(stream_ids);
    elems["element_index/offsets"].set(stream_offsets);

    // Allocate connectivity array
    elems["stream"].set(DataType::int32(count));
    int32* conn = elems["stream"].value();

    // Fill in connectivity array
    int32 idx = 0;
    int32 elem  = 0;
    for(int32 j = 0; j < nele_x ; j++)
    {
        int32 yoff = j * (nele_x+1);

        for(int32 i = 0; i < nele_y; i++)
        {
            int32 n1 = yoff + i;
            int32 n2 = n1 + (nele_x+1);
            int32 n3 = n1 + 1 + (nele_x+1);
            int32 n4 = n1 + 1;

            if ( elem % 2 == 0 )
            {
                conn[idx  ] = n1;
                conn[idx+1] = n2;
                conn[idx+2] = n3;
                conn[idx+3] = n4;
                idx+=4;
            }
            else
            {
               conn[idx   ] = n1;
               conn[idx+1 ] = n2;
               conn[idx+2 ] = n4;
               idx+=3;

               conn[idx   ] = n2;
               conn[idx+1 ] = n3;
               conn[idx+2 ] = n4;
               idx+=3;
            }

            ++elem;

        } // END for all i

    } // END for all j


    Node &fields = res["fields"];

    braid_init_example_point_scalar_field(npts_x,
                                          npts_y,
                                          1,
                                          fields["braid"]);

    braid_init_example_point_vector_field(npts_x,
                                          npts_y,
                                          1,
                                          fields["vel"]);
}


//---------------------------------------------------------------------------//
void
braid_lines_2d(index_t npts_x,
               index_t npts_y,
               Node &res)
{
    res.reset();
    
    // require npts_x > 0 && npts_y > 0

    int32 nele_quads_x = (int32)(npts_x-1);
    int32 nele_quads_y = (int32)(npts_y-1);
    int32 nele_quads = nele_quads_x * nele_quads_y;
        
    braid_init_example_state(res);
    braid_init_explicit_coordset(npts_x,
                                 npts_y,
                                 1,
                                 res["coordsets/coords"]);
  
    res["topologies/mesh/type"] = "unstructured";
    res["topologies/mesh/coordset"] = "coords";
    res["topologies/mesh/elements/shape"] = "line";
    res["topologies/mesh/elements/connectivity"].set(DataType::int32(nele_quads*4*2));
    int32 *conn = res["topologies/mesh/elements/connectivity"].value();

    int32 idx = 0;
    for(int32 j = 0; j < nele_quads_y ; j++)
    {
        int32 yoff = j * (nele_quads_x+1);
        
        for(int32 i = 0; i < nele_quads_x; i++)
        {
            // 4 lines per quad.

            // Note: this pattern allows for simple per-quad construction,
            // but it creates spatially overlapping lines

            conn[idx++] = yoff + i;
            conn[idx++] = yoff + i + (nele_quads_x+1);

            conn[idx++] = yoff + i + (nele_quads_x+1);
            conn[idx++] = yoff + i + 1 + (nele_quads_x+1);

            conn[idx++] = yoff + i;
            conn[idx++] = yoff + i + 1;

            conn[idx++] = yoff + i + 1;
            conn[idx++] = yoff + i + 1 + (nele_quads_x+1);
        }
    }


    Node &fields = res["fields"];

    braid_init_example_point_scalar_field(npts_x,
                                          npts_y,
                                          1,
                                          fields["braid"]);

    braid_init_example_element_scalar_field(nele_quads_x,
                                            nele_quads_y,
                                            0,
                                            fields["radial"],4);

    braid_init_example_point_vector_field(npts_x,
                                          npts_y,
                                          1,
                                          fields["vel"]);

}

//---------------------------------------------------------------------------//
void
braid_tris(index_t npts_x,
           index_t npts_y,
           Node &res)
{
    res.reset();
    
    // require npts_x > 0 && npts_y > 0

    int32 nele_quads_x = (int32) npts_x-1;
    int32 nele_quads_y = (int32) npts_y-1;
    int32 nele_quads = nele_quads_x * nele_quads_y;
        
    braid_init_example_state(res);
    braid_init_explicit_coordset(npts_x,
                                 npts_y,
                                 1,
                                 res["coordsets/coords"]);
  
    res["topologies/mesh/type"] = "unstructured";
    res["topologies/mesh/coordset"] = "coords";
    res["topologies/mesh/elements/shape"] = "tri";
    res["topologies/mesh/elements/connectivity"].set(DataType::int32(nele_quads*6));
    int32 *conn = res["topologies/mesh/elements/connectivity"].value();

    int32 idx = 0;
    for(int32 j = 0; j < nele_quads_y ; j++)
    {
        int32 yoff = j * (nele_quads_x+1);
        
        for(int32 i = 0; i < nele_quads_x; i++)
        {
            // two tris per quad. 
            conn[idx+0] = yoff + i;
            conn[idx+1] = yoff + i + (nele_quads_x+1);
            conn[idx+2] = yoff + i + 1 + (nele_quads_x+1);

            conn[idx+3] = yoff + i;
            conn[idx+4] = yoff + i + 1;
            conn[idx+5] = yoff + i + 1 + (nele_quads_x+1);
            
            idx+=6;
        }
    }


    Node &fields = res["fields"];

    braid_init_example_point_scalar_field(npts_x,
                                          npts_y,
                                          1,
                                          fields["braid"]);

    braid_init_example_element_scalar_field(nele_quads_x,
                                            nele_quads_y,
                                            0,
                                            fields["radial"],2);

    braid_init_example_point_vector_field(npts_x,
                                          npts_y,
                                          1,
                                          fields["vel"]);

}


//---------------------------------------------------------------------------//
void
braid_hexs(index_t npts_x,
           index_t npts_y,
           index_t npts_z,
           Node &res)
{
    res.reset();
    
    int32 nele_x = (int32)(npts_x - 1);
    int32 nele_y = (int32)(npts_y - 1);
    int32 nele_z = (int32)(npts_z - 1);
    int32 nele = nele_x * nele_y * nele_z;
    
    braid_init_example_state(res);
    braid_init_explicit_coordset(npts_x,
                                 npts_y,
                                 npts_z,
                                 res["coordsets/coords"]);
  
    res["topologies/mesh/type"] = "unstructured";
    res["topologies/mesh/coordset"] = "coords";
    res["topologies/mesh/elements/shape"] = "hex";
    res["topologies/mesh/elements/connectivity"].set(DataType::int32(nele*8));
    int32 *conn = res["topologies/mesh/elements/connectivity"].value();

    int32 idx = 0;
    for(int32 k = 0; k < nele_z ; k++)
    {
        int32 zoff = k * (nele_x+1)*(nele_y+1);
        int32 zoff_n = (k+1) * (nele_x+1)*(nele_y+1);
        
        for(int32 j = 0; j < nele_y ; j++)
        {
            int32 yoff = j * (nele_x+1);
            int32 yoff_n = (j+1) * (nele_x+1);


            for(int32 i = 0; i < nele_x; i++)
            {
                // ordering is same as VTK_HEXAHEDRON

                conn[idx+0] = zoff + yoff + i;
                conn[idx+1] = zoff + yoff + i + 1;
                conn[idx+2] = zoff + yoff_n + i + 1;
                conn[idx+3] = zoff + yoff_n + i;

                conn[idx+4] = zoff_n + yoff + i;
                conn[idx+5] = zoff_n + yoff + i + 1;
                conn[idx+6] = zoff_n + yoff_n + i + 1;
                conn[idx+7] = zoff_n + yoff_n + i;

                idx+=8;
            }
        }
    }

    Node &fields = res["fields"];

    braid_init_example_point_scalar_field(npts_x,
                                          npts_y,
                                          npts_z,
                                          fields["braid"]);

    braid_init_example_element_scalar_field(nele_x,
                                            nele_y,
                                            nele_z,
                                            fields["radial"]);

    braid_init_example_point_vector_field(npts_x,
                                          npts_y,
                                          npts_z,
                                          fields["vel"]);
}

//---------------------------------------------------------------------------//
void
braid_tets(index_t npts_x,
           index_t npts_y,
           index_t npts_z,
           Node &res)
{
    res.reset();
    
    int32 nele_hexs_x = (int32) (npts_x - 1);
    int32 nele_hexs_y = (int32) (npts_y - 1);
    int32 nele_hexs_z = (int32) (npts_z - 1);
    int32 nele_hexs = nele_hexs_x * nele_hexs_y * nele_hexs_z;
    
    int32 tets_per_hex = 6;
    int32 verts_per_tet = 4;
    int32 n_tets_verts = nele_hexs * tets_per_hex * verts_per_tet;

    braid_init_example_state(res);
    braid_init_explicit_coordset(npts_x,
                                 npts_y,
                                 npts_z,
                                 res["coordsets/coords"]);
  

    res["topologies/mesh/type"] = "unstructured";
    res["topologies/mesh/coordset"] = "coords";
    res["topologies/mesh/elements/shape"] = "tet";
    res["topologies/mesh/elements/connectivity"].set(DataType::int32(n_tets_verts));
    int32 *conn = res["topologies/mesh/elements/connectivity"].value();


    int32 idx = 0;
    for(int32 k = 0; k < nele_hexs_z ; k++)
    {
        int32 zoff = k * (nele_hexs_x+1)*(nele_hexs_y+1);
        int32 zoff_n = (k+1) * (nele_hexs_x+1)*(nele_hexs_y+1);
        
        for(int32 j = 0; j < nele_hexs_y ; j++)
        {
            int32 yoff = j * (nele_hexs_x+1);
            int32 yoff_n = (j+1) * (nele_hexs_x+1);


            for(int32 i = 0; i < nele_hexs_z; i++)
            {
                // Create a local array of the vertex indices
                // ordering is same as VTK_HEXAHEDRON
                int32 vidx[8] =   {zoff + yoff + i
                                  ,zoff + yoff + i + 1
                                  ,zoff + yoff_n + i + 1
                                  ,zoff + yoff_n + i
                                  ,zoff_n + yoff + i
                                  ,zoff_n + yoff + i + 1
                                  ,zoff_n + yoff_n + i + 1
                                  ,zoff_n + yoff_n + i};

                // Create six tets all sharing diagonal from vertex 0 to 6
                // Uses SILO convention for vertex order (normals point in)
                conn[idx++] = vidx[0];
                conn[idx++] = vidx[2];
                conn[idx++] = vidx[1];
                conn[idx++] = vidx[6];

                conn[idx++] = vidx[0];
                conn[idx++] = vidx[3];
                conn[idx++] = vidx[2];
                conn[idx++] = vidx[6];

                conn[idx++] = vidx[0];
                conn[idx++] = vidx[7];
                conn[idx++] = vidx[3];
                conn[idx++] = vidx[6];

                conn[idx++] = vidx[0];
                conn[idx++] = vidx[4];
                conn[idx++] = vidx[7];
                conn[idx++] = vidx[6];

                conn[idx++] = vidx[0];
                conn[idx++] = vidx[5];
                conn[idx++] = vidx[4];
                conn[idx++] = vidx[6];

                conn[idx++] = vidx[0];
                conn[idx++] = vidx[1];
                conn[idx++] = vidx[5];
                conn[idx++] = vidx[6];

            }
        }
    }

    Node &fields = res["fields"];

    braid_init_example_point_scalar_field(npts_x,
                                          npts_y,
                                          npts_z,
                                          fields["braid"]);

    braid_init_example_element_scalar_field(nele_hexs_x,
                                            nele_hexs_y,
                                            nele_hexs_z,
                                            fields["radial"],
                                            tets_per_hex);

    braid_init_example_point_vector_field(npts_x,
                                          npts_y,
                                          npts_z,
                                          fields["vel"]);

}


//---------------------------------------------------------------------------//
void
braid_lines_3d(index_t npts_x,
               index_t npts_y,
               index_t npts_z,
               Node &res)
{
    res.reset();
    
    int32 nele_hexs_x = (int32)(npts_x - 1);
    int32 nele_hexs_y = (int32)(npts_y - 1);
    int32 nele_hexs_z = (int32)(npts_z - 1);
    int32 nele_hexs = nele_hexs_x * nele_hexs_y * nele_hexs_z;
    

    braid_init_example_state(res);
    braid_init_explicit_coordset(npts_x,
                                 npts_y,
                                 npts_z,
                                 res["coordsets/coords"]);

    res["topologies/mesh/type"] = "unstructured";
    res["topologies/mesh/coordset"] = "coords";
    res["topologies/mesh/elements/shape"] = "line";
    res["topologies/mesh/elements/connectivity"].set(DataType::int32(nele_hexs * 12 * 2));
    int32 *conn = res["topologies/mesh/elements/connectivity"].value();

    int32 idx = 0;
    for(int32 k = 0; k < nele_hexs_z ; k++)
    {
        int32 zoff = k * (nele_hexs_x+1)*(nele_hexs_y+1);
        int32 zoff_n = (k+1) * (nele_hexs_x+1)*(nele_hexs_y+1);
        
        for(int32 j = 0; j < nele_hexs_y ; j++)
        {
            int32 yoff = j * (nele_hexs_x+1);
            int32 yoff_n = (j+1) * (nele_hexs_x+1);


            for(int32 i = 0; i < nele_hexs_z; i++)
            {
                // 12 lines per hex 
                // Note: this pattern allows for simple per-hex construction,
                // but it creates spatially overlapping lines

                // front face
                conn[idx++] = zoff + yoff + i;
                conn[idx++] = zoff + yoff + i +1;

                conn[idx++] = zoff + yoff + i + 1;
                conn[idx++] = zoff + yoff_n + i + 1;
                
                conn[idx++] = zoff + yoff_n + i + 1;
                conn[idx++] = zoff + yoff_n + i;
                
                conn[idx++] = zoff + yoff_n + i;
                conn[idx++] = zoff + yoff + i;

                // back face
                conn[idx++] = zoff_n + yoff + i;
                conn[idx++] = zoff_n + yoff + i +1;

                conn[idx++] = zoff_n + yoff + i + 1;
                conn[idx++] = zoff_n + yoff_n + i + 1;
                
                conn[idx++] = zoff_n + yoff_n + i + 1;
                conn[idx++] = zoff_n + yoff_n + i;
                
                conn[idx++] = zoff_n + yoff_n + i;
                conn[idx++] = zoff_n + yoff + i;

                // sides 
                conn[idx++] = zoff   + yoff + i;
                conn[idx++] = zoff_n + yoff + i;

                conn[idx++] = zoff   + yoff + i + 1;
                conn[idx++] = zoff_n + yoff + i + 1;
                
                conn[idx++] = zoff   + yoff_n + i + 1;
                conn[idx++] = zoff_n + yoff_n + i + 1;
                
                conn[idx++] = zoff   + yoff_n + i;
                conn[idx++] = zoff_n + yoff_n + i;

            }
        }
    }

    Node &fields = res["fields"];

    braid_init_example_point_scalar_field(npts_x,
                                          npts_y,
                                          npts_z,
                                          fields["braid"]);

    braid_init_example_element_scalar_field(nele_hexs_x,
                                            nele_hexs_y,
                                            nele_hexs_z,
                                            fields["radial"],
                                            12);

    braid_init_example_point_vector_field(npts_x,
                                          npts_y,
                                          npts_z,
                                          fields["vel"]);

}



//---------------------------------------------------------------------------//
void
braid_hexs_and_tets(index_t npts_x,
                    index_t npts_y,
                    index_t npts_z,
                    Node &res)
{

    // WARNING -- The code below is UNTESTED.
    //            The SILO writer is missing an implementation for
    //            unstructured indexed_stream meshes in 3D.

    res.reset();

    int32 nele_hexs_x = (int32)(npts_x - 1);
    int32 nele_hexs_y = (int32)(npts_y - 1);
    int32 nele_hexs_z = (int32)(npts_z - 1);
    int32 nele_hexs = nele_hexs_x * nele_hexs_y * nele_hexs_z;


    // Set the number of voxels containing hexs and tets
    int32 n_hex_hexs = (nele_hexs > 1)? nele_hexs / 2 : nele_hexs;
    int32 n_hex_tets = nele_hexs - n_hex_hexs;

    // Compute the sizes of the connectivity array for each element type
    int32 hexs_per_hex = 1;
    int32 verts_per_hex = 8;
    int32 n_hexs_verts = n_hex_hexs * hexs_per_hex * verts_per_hex;

    int32 tets_per_hex = 6;
    int32 verts_per_tet = 4;
    int32 n_tets_verts = n_hex_tets * tets_per_hex * verts_per_tet;


    braid_init_example_state(res);
    braid_init_explicit_coordset(npts_x,
                                 npts_y,
                                 npts_z,
                                 res["coordsets/coords"]);

    // Setup mesh as unstructured indexed_stream mesh of hexs and tets
    res["topologies/mesh/type"] = "unstructured";
    res["topologies/mesh/coordset"] = "coords";

    res["topologies/mesh/elements/element_types/hexs/stream_id"] = 0;
    res["topologies/mesh/elements/element_types/hexs/shape"] = "hex";

    res["topologies/mesh/elements/element_types/tets/stream_id"] = 1;
    res["topologies/mesh/elements/element_types/tets/shape"] = "tet";

    res["topologies/mesh/elements/element_index/stream_ids"].set(DataType::int32(4));
    res["topologies/mesh/elements/element_index/element_counts"].set(DataType::int32(4));

    int32* sidx_ids = res["topologies/mesh/elements/element_index/stream_ids"].value();
    int32* sidx_lengths = res["topologies/mesh/elements/element_index/element_counts"].value();

    // There are four groups -- alternating between hexs and tets
    sidx_ids[0] = 0;
    sidx_ids[1] = 1;
    sidx_ids[2] = 0;
    sidx_ids[3] = 1;

    // Set the lengths of the groups
    // The first two groups have at most length 1
    // The last two groups have the balance of the elements
    switch(nele_hexs)
    {
    case 0:
        sidx_lengths[0] = 0;  sidx_lengths[1] = 0;
        sidx_lengths[2] = 0;  sidx_lengths[3] = 0;
        break;
    case 1:
        sidx_lengths[0] = 1;  sidx_lengths[1] = 0;
        sidx_lengths[2] = 0;  sidx_lengths[3] = 0;
        break;
    case 2:
        sidx_lengths[0] = 1;  sidx_lengths[1] = 1;
        sidx_lengths[2] = 0;  sidx_lengths[3] = 0;
        break;
    case 3:
        sidx_lengths[0] = 1;  sidx_lengths[1] = 1;
        sidx_lengths[2] = 1;  sidx_lengths[3] = 0;
        break;
    default:
        sidx_lengths[0] = 1;
        sidx_lengths[1] = 1;
        sidx_lengths[2] = n_hex_hexs-1;
        sidx_lengths[3] = n_hex_tets-1;
        break;
    }

    res["topologies/mesh/elements/stream"].set( DataType::int32(n_hexs_verts + n_tets_verts) );
    int32* conn = res["topologies/mesh/elements/stream"].value();

    int32 idx = 0;
    int32 elem_count = 0;
    for(int32 k = 0; k < nele_hexs_z ; k++)
    {
        int32 zoff = k * (nele_hexs_x+1)*(nele_hexs_y+1);
        int32 zoff_n = (k+1) * (nele_hexs_x+1)*(nele_hexs_y+1);

        for(int32 j = 0; j < nele_hexs_y ; j++)
        {
            int32 yoff = j * (nele_hexs_x+1);
            int32 yoff_n = (j+1) * (nele_hexs_x+1);


            for(int32 i = 0; i < nele_hexs_z; i++)
            {
                // Create a local array of the vertex indices
                // ordering is same as VTK_HEXAHEDRON
                int32 vidx[8] = {zoff + yoff + i
                                  ,zoff + yoff + i + 1
                                  ,zoff + yoff_n + i + 1
                                  ,zoff + yoff_n + i
                                  ,zoff_n + yoff + i
                                  ,zoff_n + yoff + i + 1
                                  ,zoff_n + yoff_n + i + 1
                                  ,zoff_n + yoff_n + i};

                bool isHex = (elem_count == 0)
                          || (elem_count > 1 && elem_count <= n_hex_hexs);


                if(isHex)
                {
                    conn[idx++] = vidx[0];
                    conn[idx++] = vidx[1];
                    conn[idx++] = vidx[2];
                    conn[idx++] = vidx[3];

                    conn[idx++] = vidx[4];
                    conn[idx++] = vidx[5];
                    conn[idx++] = vidx[6];
                    conn[idx++] = vidx[7];
                }
                else // it is a tet
                {
                    // Create six tets all sharing diagonal from vertex 0 to 6
                    // Uses SILO convention for vertex order (normals point in)
                    conn[idx++] = vidx[0];
                    conn[idx++] = vidx[2];
                    conn[idx++] = vidx[1];
                    conn[idx++] = vidx[6];

                    conn[idx++] = vidx[0];
                    conn[idx++] = vidx[3];
                    conn[idx++] = vidx[2];
                    conn[idx++] = vidx[6];

                    conn[idx++] = vidx[0];
                    conn[idx++] = vidx[7];
                    conn[idx++] = vidx[3];
                    conn[idx++] = vidx[6];

                    conn[idx++] = vidx[0];
                    conn[idx++] = vidx[4];
                    conn[idx++] = vidx[7];
                    conn[idx++] = vidx[6];

                    conn[idx++] = vidx[0];
                    conn[idx++] = vidx[5];
                    conn[idx++] = vidx[4];
                    conn[idx++] = vidx[6];

                    conn[idx++] = vidx[0];
                    conn[idx++] = vidx[1];
                    conn[idx++] = vidx[5];
                    conn[idx++] = vidx[6];
                }

                elem_count++;
            }
        }
    }

    Node &fields = res["fields"];

    braid_init_example_point_scalar_field(npts_x,
                                          npts_y,
                                          npts_z,
                                          fields["braid"]);

    braid_init_example_point_vector_field(npts_x,
                                          npts_y,
                                          npts_z,
                                          fields["vel"]);

//    // Omit for now -- the function assumes a uniform element type
//    braid_init_example_element_scalar_field(nele_hexs_x,
//                                            nele_hexs_y,
//                                            nele_hexs_z,
//                                            fields["radial"],
//                                            tets_per_hex);
}


//---------------------------------------------------------------------------//
void
braid_to_poly(Node &res)
{
    const index_t topo_count = res["topologies"].number_of_children();
    std::vector<Node> poly_topos(topo_count);
    std::vector<std::string> topo_names(topo_count);

    conduit::NodeConstIterator topos_it = res["topologies"].children();
    while(topos_it.has_next())
    {
        const conduit::Node &topo_node = topos_it.next();
        const std::string topo_name = topos_it.name();
        const index_t topo_index = topos_it.index();

        conduit::Node &poly_node = poly_topos[topo_index];
        blueprint::mesh::topology::unstructured::to_polygonal(topo_node, poly_node);
        blueprint::mesh::topology::unstructured::generate_offsets(poly_node, poly_node["elements/offsets"]);
        topo_names[topo_index] = topo_name;
    }

    res["topologies"].reset();
    for(index_t ti = 0; ti < topo_count; ti++)
    {
        res["topologies"][topo_names[ti]].set(poly_topos[ti]);
    }
}


//---------------------------------------------------------------------------//
void
basic(const std::string &mesh_type,
      index_t npts_x, // number of points in x
      index_t npts_y, // number of points in y
      index_t npts_z, // number of points in z
      Node &res)
{
    // NOTE(JRC): The basic mesh example only supports simple, homogenous
    // element types that can be spanned by zone-centered fields.
    const std::string mesh_types[] = {
        "uniform", "rectilinear", "structured",
        "tris", "quads", "polygons",
        "tets", "hexs", "polyhedra"};
    const std::string braid_types[] = {
        "uniform", "rectilinear", "structured",
        "tris", "quads", "quads_poly",
        "tets", "hexs", "hexs_poly"};
    const index_t mesh_types_subelems_per_elem[] = {
        1, 1, 1,
        2, 1, 1,
        6, 1, 1};

    const index_t num_mesh_types = sizeof(mesh_types) / sizeof(std::string);

    index_t mesh_type_index = -1;
    for(index_t i = 0; i < num_mesh_types; i++)
    {
        if(mesh_type == mesh_types[i])
        {
            mesh_type_index = i;
        }
    }
    if(mesh_type_index < 0 || mesh_type_index >= num_mesh_types)
    {
        CONDUIT_ERROR("unknown mesh_type = " << mesh_type);
    }

    braid(braid_types[mesh_type_index], npts_x, npts_y, npts_z, res);
    res.remove("fields");
    res.remove("state");

    // TODO(JRC): Consider removing this code if the extra complexity of having
    // the "offsets" array in the basic examples is decided to be a non-issue.
    Node &topo = res["topologies"].child(0);
    if(topo.has_child("elements") && topo["elements"].has_child("offsets"))
    {
        topo["elements"].remove("offsets");
    }

    basic_init_example_element_scalar_field(npts_x-1, npts_y-1, npts_z-1,
        res["fields/field"], mesh_types_subelems_per_elem[mesh_type_index]);
}


//---------------------------------------------------------------------------//
void
braid(const std::string &mesh_type,
      index_t npts_x, // number of points in x
      index_t npts_y, // number of points in y
      index_t npts_z, // number of points in z
      Node &res)
{
    index_t nele_x = npts_x -1;
    index_t nele_y = npts_y -1;

    if( (nele_x == 0 || nele_y == 0) && 
        (mesh_type != "points" && mesh_type != "points_implicit") )
    {
        // error, not enough points to create the topo
        CONDUIT_ERROR("braid with non-points topology requires"
                      " npts_x > 1 and npts_y > 1");
    }

    if(mesh_type == "uniform")
    {
        braid_uniform(npts_x,npts_y,npts_z,res);
    }
    else if(mesh_type == "rectilinear")
    {
        braid_rectilinear(npts_x,npts_y,npts_z,res);
    }
    else if(mesh_type == "structured")
    {
        braid_structured(npts_x,npts_y,npts_z,res);
    }
    else if(mesh_type == "lines")
    {
        if( npts_z <= 1 )
            braid_lines_2d(npts_x,npts_y,res);
        else
            braid_lines_3d(npts_x,npts_y,npts_x,res);
    }
    else if(mesh_type == "tris")
    {
        braid_tris(npts_x,npts_y,res);
    }
    else if(mesh_type == "quads")
    {
        braid_quads(npts_x,npts_y,res);
    }
    else if(mesh_type == "quads_poly")
    {
        braid_quads(npts_x,npts_y,res);
        braid_to_poly(res);
    }
    else if(mesh_type == "quads_and_tris")
    {
        braid_quads_and_tris(npts_x,npts_y,res);
    }
    else if(mesh_type == "quads_and_tris_offsets")
    {
        braid_quads_and_tris_offsets(npts_x,npts_y,res);
    }
    else if(mesh_type == "tets")
    {
        braid_tets(npts_x,npts_y,npts_z,res);
    }
    else if(mesh_type == "hexs")
    {
        braid_hexs(npts_x,npts_y,npts_z,res);
    }
    else if(mesh_type == "hexs_poly")
    {
        braid_hexs(npts_x,npts_y,npts_z,res);
        braid_to_poly(res);
    }
    else if(mesh_type == "hexs_and_tets")
    {
        braid_hexs_and_tets(npts_x,npts_y,npts_z,res);
    }
    else if(mesh_type == "points")
    {
        braid_points_explicit(npts_x,npts_y,npts_z,res);
    }
    else if(mesh_type == "points_implicit")
    {
        braid_points_implicit(npts_x,npts_y,npts_z,res);
    }
    else
    {
        CONDUIT_ERROR("unknown mesh_type = " << mesh_type);
    }
}


//---------------------------------------------------------------------------//
void julia_fill_values(index_t nx,
                       index_t ny,
                       float64 x_min,
                       float64 x_max,
                       float64 y_min,
                       float64 y_max,
                       float64 c_re,
                       float64 c_im,
                       int32_array &out)
{
    index_t idx = 0;
    for(index_t j = 0; j < ny; j++)
    {
        for(index_t i = 0; i < nx; i++)
        {
            float64 zx = float64(i) / float64(nx-1);
            float64 zy = float64(j) / float64(ny-1);
            
            zx = x_min + (x_max - x_min) * zx;
            zy = y_min + (y_max - y_min) * zy;

            int32 iter = 0;
            int32 max_iter = 1000;
            
            while( (zx * zx) + (zy * zy ) < 4.0 && iter < max_iter)
            {
                float64 x_temp = zx*zx - zy*zy;
                zy = 2*zx*zy  + c_im;
                zx = x_temp    + c_re;
                iter++;
            }
            if(iter == max_iter)
            {
                out[idx] = 0;
            }
            else
            {
                out[idx] = iter;
            }
    
            idx++;
        }
    }
}


//---------------------------------------------------------------------------//
void julia(index_t nx,
           index_t ny,
           float64 x_min,
           float64 x_max,
           float64 y_min,
           float64 y_max,
           float64 c_re,
           float64 c_im,
           Node &res)
{
    res.reset();
    // create a rectilinear coordset 
    res["coordsets/coords/type"] = "rectilinear";
    res["coordsets/coords/values/x"] = DataType::float64(nx+1);
    res["coordsets/coords/values/y"] = DataType::float64(ny+1);
    
    float64_array x_coords = res["coordsets/coords/values/x"].value();
    float64_array y_coords = res["coordsets/coords/values/y"].value(); 
    
    float64 dx = (x_max - x_min) / float64(nx);
    float64 dy = (y_max - y_min) / float64(ny);

    float64 vx = x_min;
    for(index_t i =0; i< nx+1; i++)
    {
        x_coords[i] = vx;
        vx+=dx;
    }
    
    float64 vy = x_min;
    for(index_t i =0; i< ny+1; i++)
    {
        y_coords[i] = vy;
        vy+=dy;
    }
    
    // create the topology
    
    res["topologies/topo/type"] = "rectilinear";
    res["topologies/topo/coordset"] = "coords";
    
    
    // create the fields

    res["fields/iters/association"] = "element";
    res["fields/iters/topology"] = "topo";
    res["fields/iters/values"] = DataType::int32(nx * ny);
    
    int32_array out = res["fields/iters/values"].value();
    
    julia_fill_values(nx,ny,
                      x_min, x_max,
                      y_min, y_max,
                      c_re, c_im,
                      out);
}


//---------------------------------------------------------------------------//
void spiral(index_t ndoms,
            Node &res)
{
    res.reset();
    
    int f_1 = 1;
    int f = 1;
    
    float64 x = 0.0;
    float64 y = 0.0;

    float64 loc_xo = x + f;
    float64 loc_yo = y + f;

    int rot_case = 0;

    for(int d=0; d < ndoms; d++)
    {
        // create the current domain
        std::ostringstream oss;
        oss << "domain_" << std::setw(6) << std::setfill('0') << d;
        std::string domain_name = oss.str();

        Node &dom = res[domain_name];

        // create a rectilinear coordset 
        dom["coordsets/coords/type"] = "rectilinear";
        dom["coordsets/coords/values/x"] = DataType::float64(f+1);
        dom["coordsets/coords/values/y"] = DataType::float64(f+1);
    
        float64_array x_coords = dom["coordsets/coords/values/x"].value();
        float64_array y_coords = dom["coordsets/coords/values/y"].value(); 

        float64 xv = x;
        float64 yv = y;

        for(int i=0; i < f+1; i++)
        {
            x_coords[i] = xv;
            y_coords[i] = yv;
            xv+=1;
            yv+=1;
        }

        // create the topology
        dom["topologies/topo/type"] = "rectilinear";
        dom["topologies/topo/coordset"] = "coords";
        // todo, add topo logical origin
    
    
        // create the fields
        dom["fields/dist/association"] = "vertex";
        dom["fields/dist/topology"] = "topo";
        dom["fields/dist/values"] = DataType::float64((f+1) * (f+1));
    
        float64_array dist_vals = dom["fields/dist/values"].value();

        index_t idx = 0;
        // fill the scalar with approx dist to spiral 
        yv = y;

        for(int j=0; j < f+1; j++)
        {
            xv = x;
            for(int i=0; i < f+1; i++)
            {
                float64 l_x = xv - loc_xo;
                float64 l_y = yv - loc_yo;
                dist_vals[idx] = sqrt( l_x * l_x + l_y * l_y) - f;
                xv+=1;
                idx++;
            }
            yv+=1;
        }
    
        // setup for next domain using one of 4 rotation cases 
        switch(rot_case)
        {
            case 0:
            {
                x += f;
                // next loc orig == top left
                loc_xo = x;
                if (f <= 1)
                    loc_yo = y + f;
                else
                    loc_yo = y + f + f_1;
                break;
            }
            case 1:
            {
                y += f;
                x -= f_1;
                // next loc orig == bottom left
                loc_xo = x;
                loc_yo = y;
                break;
            }
            case 2:
            {
                x -= (f + f_1);
                y -= f_1;
                // next loc orig == bottom right
                loc_xo = x + (f + f_1);
                loc_yo = y;
                break;
            }
            case 3:
            {
                y -= (f + f_1);
                // next loc orig == top right
                loc_xo = x + (f + f_1);
                loc_yo = y + (f + f_1);
                break;
            }
        }
        // update the rotate case
        rot_case =  (rot_case +1) % 4;
            
        // calc next fib #
        // domain id is one less than the fib #
        if( (d+1) > 1)
        {
            int f_prev = f;
            f = f + f_1;
            f_1 = f_prev;
        }
    }
}


//---------------------------------------------------------------------------//
point
polytess_calc_polygon_center(const std::vector<index_t> polygon,
                             std::map< point, index_t > &/*point_map*/,
                             std::map< index_t, point > &point_rmap)
{
    point polygon_center(0.0, 0.0);

    for(index_t pi = 0; pi < (index_t)polygon.size(); pi++)
    {
        const point &polygon_point = point_rmap[polygon[pi]];
        polygon_center.x += polygon_point.x;
        polygon_center.y += polygon_point.y;
    }

    polygon_center.x /= (index_t)polygon.size();
    polygon_center.y /= (index_t)polygon.size();

    return polygon_center;
}


//---------------------------------------------------------------------------//
point
polytess_displace_point(const point &start_point,
                        index_t displace_dir,
                        float64 displace_mag)
{
    const bool is_dir_x = displace_dir % 2 == 0;
    const bool is_dir_pos = displace_dir > 1;
    return point(
        start_point.x + (is_dir_pos ? 1 : -1) * (is_dir_x ? 1.0 : 0.0) * displace_mag,
        start_point.y + (is_dir_pos ? 1 : -1) * (is_dir_x ? 0.0 : 1.0) * displace_mag);
}


//---------------------------------------------------------------------------//
std::vector<point>
polytess_make_polygon(point poly_center,
                      float64 side_length,
                      index_t ncorners)
{
    const float64 poly_radius = side_length / (2.0 * sin(PI_VALUE / ncorners));

    std::vector<point> poly_points;
    for(index_t c = 0; c < ncorners; c++)
    {
        point cpoint = poly_center;
        float64 cangle = PI_VALUE + (c + 0.5) * (2.0 * PI_VALUE / ncorners);
        cpoint.x += poly_radius * cos(cangle);
        cpoint.y += poly_radius * sin(cangle);
        poly_points.push_back(cpoint);
    }

    return poly_points;
}


//---------------------------------------------------------------------------//
bool
polytess_add_polygon(const std::vector<point> &polygon_points,
                     const index_t polygon_level,
                     std::map< point, index_t > &point_map,
                     std::map< index_t, point > &point_rmap,
                     std::vector< std::vector<index_t> > &polygons,
                     std::vector< index_t > &levels)
{
    std::vector<index_t> polygon_indices(polygon_points.size());

    bool is_polygon_duplicate = true;
    for(index_t pi = 0; pi < (index_t)polygon_points.size(); pi++)
    {
        const point &polygon_point = polygon_points[pi];
        index_t &point_index = polygon_indices[pi];

        if(point_map.find(polygon_point) != point_map.end())
        {
            point_index = point_map.find(polygon_point)->second;
        }
        else
        {
            point_index = point_map.size();
            point_map[polygon_point] = point_index;
            point_rmap[point_index] = polygon_point;
            is_polygon_duplicate = false;
        }
    }

    if(!is_polygon_duplicate)
    {
        polygons.push_back(polygon_indices);
        levels.push_back(polygon_level);
    }

    return !is_polygon_duplicate;
}


//---------------------------------------------------------------------------//
void polytess_recursive(index_t nlevels,
                        std::map< point, index_t > &point_map,
                        std::map< index_t, point > &point_rmap,
                        std::vector< std::vector<index_t> > &polygons,
                        std::vector< index_t > &levels)
{
    const float64 side_length = 1.0;
    const float64 octogon_to_center = side_length / (2.0 * tan(PI_VALUE / 8.0));
    const float64 adj_poly_distance = octogon_to_center + (side_length / 2.0);

    // base case
    if(nlevels <= 1)
    {
        std::vector<point> center_polygon_points = polytess_make_polygon(
            point(0.0, 0.0), side_length, 8);
        polytess_add_polygon(center_polygon_points, nlevels,
            point_map, point_rmap, polygons, levels);
    }
    // recursive case
    else // if(nlevels > 1)
    {
        polytess_recursive(nlevels - 1, point_map, point_rmap, polygons, levels);

        for(index_t o = polygons.size() - 1; o >= 0 && levels[o] == nlevels - 1; o--)
        {
            if(polygons[o].size() != 8) { continue; }

            const std::vector<index_t> &octogon = polygons[o];
            const point octogon_center = polytess_calc_polygon_center(octogon, point_map, point_rmap);
            for(index_t d = 0; d < 4; d++)
            {
                const point dir_square_center = polytess_displace_point(
                    octogon_center, d, adj_poly_distance);

                std::vector<point> dir_square_points = polytess_make_polygon(
                    dir_square_center, side_length, 4);

                if(polytess_add_polygon(dir_square_points, nlevels,
                    point_map, point_rmap, polygons, levels))
                {
                    const point square_octogon_center = polytess_displace_point(
                        dir_square_center, (d + 1) % 4, adj_poly_distance);

                    std::vector<point> square_octogon_points = polytess_make_polygon(
                        square_octogon_center, side_length, 8);

                    polytess_add_polygon(square_octogon_points, nlevels,
                        point_map, point_rmap, polygons, levels);
                }
            }
        }
    }
}


//---------------------------------------------------------------------------//
void polytess(index_t nlevels,
              Node &res)
{
    std::map< point, index_t > point_map;
    std::map< index_t, point > point_rmap;
    std::vector< std::vector<index_t> > polygons;
    std::vector< index_t > levels;

    polytess_recursive(nlevels, point_map, point_rmap, polygons, levels);

    index_t conn_size = polygons.size();
    for(index_t p = 0; p < (index_t)polygons.size(); p++)
    {
        conn_size += polygons[p].size();
    }

    // Populate Coordinates //

    Node &coordset = res["coordsets/coords"];
    coordset["type"].set("explicit");
    coordset["values/x"].set(DataType::float64(point_map.size()));
    coordset["values/y"].set(DataType::float64(point_map.size()));

    float64_array x_coords = coordset["values/x"].value();
    float64_array y_coords = coordset["values/y"].value();
    for(index_t pi = 0; pi < (index_t)point_map.size(); pi++)
    {
        const point &p = point_rmap[pi];
        x_coords[pi] = p.x;
        y_coords[pi] = p.y;
    }

    // Populate Topology //

    Node &topology = res["topologies/topo"];
    topology["coordset"].set("coords");
    topology["type"].set("unstructured");
    topology["elements/shape"].set("polygonal");
    topology["elements/connectivity"].set(DataType::uint64(conn_size));

    uint64_array conn_array = topology["elements/connectivity"].value();
    for(index_t pi = 0, ci = 0; pi < (index_t)polygons.size(); pi++)
    {
        const std::vector<index_t> &p = polygons[pi];

        conn_array[ci++] = p.size();
        for(index_t ii = 0; ii < (index_t)p.size(); ii++)
        {
            conn_array[ci++] = p[ii];
        }
    }

    blueprint::mesh::topology::unstructured::generate_offsets(
        topology, topology["elements/offsets"]);

    // Populate Field //

    Node &field =  res["fields/level"];
    field["topology"].set("topo");
    field["association"].set("element");
    field["volume_dependent"].set("false");
    // TODO: should we try to use index_t as the data type here?
    field["values"].set(DataType::uint32(polygons.size()));

    uint32_array level_array = field["values"].value();
    for(index_t pi = 0; pi < (index_t)polygons.size(); pi++)
    {
        level_array[pi] = (uint32) levels[pi];
    }
}


//---------------------------------------------------------------------------//
void
misc(const std::string &mesh_type,
     index_t npts_x, // number of points in x
     index_t npts_y, // number of points in y
     index_t /*npts_z*/, // number of points in z
     Node &res)
{
    // TODO(JRC): Improve these examples so that they use different example
    // geometry than is used in the "braid" examples.
    if(mesh_type == "matsets")
    {
        braid_quads(npts_x,npts_y,res);
        braid_init_example_matset(npts_x-1,npts_y-1,0,res["matsets/mesh"]);
    }
    else if(mesh_type == "specsets")
    {
        braid_quads(npts_x,npts_y,res);
        braid_init_example_matset(npts_x-1,npts_y-1,0,res["matsets/mesh"]);
        braid_init_example_specset(npts_x-1,npts_y-1,0,res["specsets/mesh"]);
    }
    else if(mesh_type == "adjsets")
    {
        for(index_t j = 0; j < 2; j++)
        {
            for(index_t i = 0; i < 2; i++)
            {
                const index_t domain_id = j * 2 + i;

                std::ostringstream oss;
                oss << "domain" << domain_id;
                const std::string domain_name = oss.str();

                Node &domain_node = res[domain_name];
                braid_quads(npts_x,npts_y,domain_node);
                domain_node["state/domain_id"].set(domain_id);

                Node &domain_coords = domain_node["coordsets/coords/values"];
                float64_array domain_coords_x = domain_coords["x"].as_float64_array();
                for(index_t x = 0; x < domain_coords_x.number_of_elements(); x++)
                {
                    domain_coords_x[x] += i * 20.0;
                }
                float64_array domain_coords_y = domain_coords["y"].as_float64_array();
                for(index_t y = 0; y < domain_coords_y.number_of_elements(); y++)
                {
                    domain_coords_y[y] += j * 20.0;
                }
            }
        }

        braid_init_example_adjset(res);
    }
    else if(mesh_type == "nestsets")
    {
        braid_rectilinear(npts_x,npts_y,1,res["domain0"]);
        res["domain0/state/domain_id"].set(0);
        res["domain0/state/level_id"].set(0);

        for(index_t j = 0; j < 2; j++)
        {
            for(index_t i = 0; i < 2; i++)
            {
                const index_t domain_id = j * 2 + i + 1;

                std::ostringstream oss;
                oss << "domain" << domain_id;
                const std::string domain_name = oss.str();

                Node &domain_node = res[domain_name];
                braid_rectilinear(npts_x,npts_y,1,domain_node);
                domain_node["state/domain_id"].set(domain_id);
                domain_node["state/level_id"].set(1);

                Node &domain_coords = domain_node["coordsets/coords/values"];
                float64_array domain_coords_x = domain_coords["x"].as_float64_array();
                for(index_t x = 0; x < domain_coords_x.number_of_elements(); x++)
                {
                    domain_coords_x[x] = ( domain_coords_x[x] / 2.0 ) - 5.0 + i * 10.0;
                }
                float64_array domain_coords_y = domain_coords["y"].as_float64_array();
                for(index_t y = 0; y < domain_coords_y.number_of_elements(); y++)
                {
                    domain_coords_y[y] = ( domain_coords_y[y] / 2.0 ) - 5.0 + j * 10.0;
                }
            }
        }

        braid_init_example_nestset(res);
    }
    else
    {
        CONDUIT_ERROR("unknown mesh_type = " << mesh_type);
    }
}

//-----------------------------------------------------------------------------
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
