// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh_examples.cpp
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
#include "conduit_blueprint_mesh_utils.hpp"


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

// NOTE(CYRUSH) 
// Using npts_z != 0 for 2d only shape types in our examples is deprecated
// this bool controls if we throw an exception when npts_z != 0 for 2D
// only examples. In a future release move to throw error and remove
// this and related CONDUIT_INFO logic
const bool STRICT_NPTS_Z_FOR_2D = false;

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
    index_t nele = nele_x;

    if(nele_y > 0)
    {
        nele = nele * nele_y;
    }

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
bool
braid_2d_only_shape_type(const std::string& mesh_type)
{
    if ( mesh_type == "tris"  ||
         mesh_type == "quads" ||
         mesh_type == "quads_poly" ||
         mesh_type == "quads_and_tris" ||
         mesh_type == "quads_and_tris_offsets")
    {
        return true;
    }
    else
    {
        return false;
    }
}

//---------------------------------------------------------------------------//
bool
braid_3d_only_shape_type(const std::string& mesh_type)
{
    if( mesh_type == "tets" ||
        mesh_type == "hexs" ||
        mesh_type == "hexs_poly" ||
        mesh_type == "hexs_and_tets")
    {
        return true;
    }
    else
    {
        return false;
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

    if(npts_y < 1)
    {
        npts_y = 1;
    }

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
   index_t npts = npts_x;

    if(npts_y > 0)
    {
        npts *= npts_y;
    }

    if(npts_z > 0)
    {
        npts *= npts_z;
    }

    res["association"] = "vertex";
    res["type"] = "vector";
    res["topology"] = "mesh";

    res["values/u"].set(DataType::float64(npts));
    float64 *u_vals = res["values/u"].value();

    float64 * v_vals = NULL;
    if (npts_y > 1)
    {
        res["values/v"].set(DataType::float64(npts));
        v_vals = res["values/v"].value();
    }

    float64 *w_vals = NULL;
    if(npts_z > 1)
    {
        res["values/w"].set(DataType::float64(npts));
        w_vals = res["values/w"].value();
    }

    // this logic is from the explicit coord set setup function
    // we are using the coords (distance from origin)
    // to create an example vector field

    float64 dx = 0.0;
    float64 dy = 0.0;
    
    if(npts_x > 1)
    {
        dx = 20.0  / float64(npts_x - 1);
    }

    // AGC is this right?  Note change to variable being tested
    // and used in divisor.
    if(npts_y > 1)
    {
        dy = 20.0  / float64(npts_y - 1);
    }

    
    float64 dz = 0.0;

    if(npts_z > 1)
    {
        dz = 20.0 / float64(npts_z-1);
    }

    // make sure outerloop exex
    if(npts_y < 1)
    {
        npts_y = 1;
    }

    if(npts_z < 1)
    {
        npts_z = 1;
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

                if(dy > 0.0)
                {
                    v_vals[idx] = cy;
                }

                if(dz > 0.0)
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
    index_t nele = nele_x;

    if(nele_y > 0)
    {
        nele = nele * nele_y;
    }

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

    float64 dy = 0.0f;
    if(nele_y > 0 )
    {
        dy = 20.0 / float64(nele_y);
    }

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
                float64 cx = (i * dx) + -10.0;

                float64 cv = 10.0 * sqrt( cx*cx );

                if(nele_y != 0)
                {
                    cv = 10.0 * sqrt( cx*cx + cy*cy );
                }

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
    if(npts_y > 1)
    {
        dims["j"] = npts_y;
    }

    if(npts_z > 1)
    {
        dims["k"] = npts_z;
    }

    // -10 to 10 in each dim,
    Node &origin = coords["origin"];
    origin["x"] = -10.0;

    if(npts_y > 1)
    {
        origin["y"] = -10.0;
    }

    if(npts_z > 1)
    {
        origin["z"] = -10.0;
    }

    Node &spacing = coords["spacing"];
    spacing["dx"] = 20.0 / (float64)(npts_x-1);
    if(npts_y > 1)
    {
        spacing["dy"] = 20.0 / (float64)(npts_y-1);
    }

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
    // default to one loop iteration (2d case)
    index_t outer = 1;
    // expand loop iteration for 3d case
    if(npts_z > 1)
    {
        outer = npts_z;
    }

    for(index_t k = 0; k < outer; k++)
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
void
braid_init_example_adjset(Node &mesh)
{
    typedef std::map< point, std::map<index_t, index_t> > point_doms_map;
    typedef std::map<std::set<index_t>, std::vector<std::vector<index_t> > > group_idx_map;

    const std::string dim_names[3] = {"x", "y", "z"};
    const index_t dim_count = blueprint::mesh::coordset::dims(
        mesh.child(0).fetch("coordsets").child(0));

    // FIXME(JRC): Update this code to use the utilities provided in
    // 'conduit_blueprint_mesh_utils.hpp'.

    // From mesh data, create a map from domain combination tuple to point list.
    // These domain combination tuples represent groups and the point lists contain
    // the points that lie on the shared boundary between these domains.
    point_doms_map mesh_point_doms_map;
    NodeConstIterator doms_it = mesh.children();
    while(doms_it.has_next())
    {
        const Node& dom_node = doms_it.next();
        const Node& dom_cset = dom_node["coordsets"].child(0);
        const std::string dom_type = dom_cset["type"].as_string();
        const index_t dom_id = dom_node["state/domain_id"].to_index_t();

        Node dom_coords;
        if(dom_type == "uniform")
        {
            blueprint::mesh::coordset::uniform::to_explicit(dom_cset, dom_coords);
        }
        else if(dom_type == "rectilinear")
        {
            blueprint::mesh::coordset::rectilinear::to_explicit(dom_cset, dom_coords);
        }
        else // if(dom_type == "explicit")
        {
            dom_coords.set_external(dom_cset);
        }

        float64_array dom_dim_coords[3];
        for(index_t d = 0; d < dim_count; d++)
        {
            dom_dim_coords[d] = dom_coords["values"][dim_names[d]].as_float64_array();
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

          Node& dom_node = mesh[dom_name]["adjsets/mesh_adj"];
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

    braid_init_example_state(res);

    braid_init_explicit_coordset(npts_x,
                                 npts_y,
                                 npts_z,
                                 res["coordsets/coords"]);

    res["topologies/mesh/type"] = "unstructured";
    res["topologies/mesh/coordset"] = "coords";
    res["topologies/mesh/elements/shape"] = "point";

    if(npts_z <= 0)
    {
        npts_z = 1;
    }

    index_t npts_total = npts_x * npts_y * npts_z;

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
    for(int32 j = 0; j < nele_y ; j++)
    {
        int32 yoff = j * (nele_x+1);
        for(int32 i = 0; i < nele_x; i++)
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


            for(int32 i = 0; i < nele_hexs_x; i++)
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

    res["topologies/mesh/elements/stream"].set( DataType::int32(n_hexs_verts + n_tets_verts) );
    int32* conn = res["topologies/mesh/elements/stream"].value();

    std::vector<int32> ele_counts;
    std::vector<int32> stream_ids;
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


            for(int32 i = 0; i < nele_hexs_x; i++)
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
                    ele_counts.push_back(1);
                    stream_ids.push_back(0);
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
                    ele_counts.push_back(6);
                    stream_ids.push_back(1);
                }

                elem_count++;
            }
        }
    }
    res["topologies/mesh/elements/element_index/element_counts"].set(ele_counts);
    res["topologies/mesh/elements/element_index/stream_ids"].set(stream_ids);

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
    const index_t mesh_types_dims[] = {
        2, 2, 2,
        2, 2, 2,
        3, 3, 3};
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
        CONDUIT_ERROR("blueprint::mesh::examples::basic unknown mesh_type = "
                      << mesh_type);
    }

    const bool npts_x_ok = npts_x > 1;
    const bool npts_y_ok = npts_y > 1;
    bool npts_z_ok = mesh_types_dims[mesh_type_index] == 2 || npts_z > 1;
    
    
    if( npts_z != 0 &&
        braid_2d_only_shape_type(mesh_type) )
    {
        if(STRICT_NPTS_Z_FOR_2D)
        {
            npts_z_ok = false;
        }
        else
        {
            CONDUIT_INFO("DEPRECATED:"
                     " Detected npts_z != 0 for example with 2D shape type."
                     " This will throw a conduit::Error in a future release.");
        }
    }

    // don't let de-morgan get you ...
    if( ! (npts_x_ok && npts_y_ok && npts_z_ok) )
    {
        // error, not enough points to create the topo
        CONDUIT_ERROR("blueprint::mesh::examples::basic requires: " << std::endl <<
                      "For 2D only topologies"
                      " ( mesh_type={\"tris\", \"quads\", or \"polygons\"} )"
                      " npts_x > 1 and npts_y > 1 and npts_z == 0" 
                      << std::endl <<
                      "For 3D only topologies"
                      " ( mesh_type={\"tets\", \"hexs\", or \"polyhedra\"} )"
                      "npts_x > 1 and npts_y > 1 and "
                      " npts_z > 1"
                      << std::endl <<
                      "values provided:" << std::endl << 
                      " mesh_type: " << mesh_type << std::endl <<
                      " npts_x: " << npts_x << std::endl <<
                      " npts_y: " << npts_y << std::endl <<
                      " npts_z: " << npts_z << std::endl);
    }

    braid(braid_types[mesh_type_index], npts_x, npts_y, npts_z, res);
    res.remove("fields");
    res.remove("state");

    basic_init_example_element_scalar_field(npts_x-1, npts_y-1, npts_z-1,
        res["fields/field"], mesh_types_subelems_per_elem[mesh_type_index]);
}


//---------------------------------------------------------------------------//
void
basic1D(const std::string &mesh_type,
        index_t npts_x, // number of points in x
        Node &res)
{
    // NOTE(JRC): The basic mesh example only supports simple, homogenous
    // element types that can be spanned by zone-centered fields.
    const std::string mesh_types[] = {
        "uniform", "structured"};
    const std::string braid_types[] = {
        "uniform", "structured"};
    const index_t mesh_types_dims[] = {
        1, 1};
    const index_t mesh_types_subelems_per_elem[] = {
        1, 1};

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
        CONDUIT_ERROR("blueprint::mesh::examples::basic1D unknown mesh_type = "
                      << mesh_type);
    }

    const bool npts_x_ok = npts_x > 1;

    // don't let de-morgan get you ...
    if( ! (npts_x_ok) )
    {
        // error, not enough points to create the topo
        CONDUIT_ERROR("blueprint::mesh::examples::basic1D requires npts_x > 1" << std::endl <<
                      std::endl <<
                      "values provided:" << std::endl <<
                      " mesh_type: " << mesh_type << std::endl <<
                      " npts_x: " << npts_x << std::endl);
    }

    braid(braid_types[mesh_type_index], npts_x, 0, 0, res);
    res.remove("fields");
    res.remove("state");

    basic_init_example_element_scalar_field(npts_x-1, 0, 0,
        res["fields/field"], mesh_types_subelems_per_elem[mesh_type_index]);
}


//---------------------------------------------------------------------------//
void
grid(const std::string &mesh_type,
     index_t npts_x, // number of per-domain points in x
     index_t npts_y, // number of per-domain points in y
     index_t npts_z, // number of per-domain points in z
     index_t ndoms_x, // number of domains in x
     index_t ndoms_y, // number of domains in y
     index_t ndoms_z, // number of domains in z
     Node &res)
{
    const bool ndoms_x_ok = ndoms_x > 0;
    const bool ndoms_y_ok = ndoms_y > 0;
    const bool ndoms_z_ok = ndoms_z > 0;

    if( ! (ndoms_x_ok && ndoms_y_ok && ndoms_z_ok) )
    {
        // error, not enough points to create the topo
        CONDUIT_ERROR("blueprint::mesh::examples::grid requires "
                      "ndoms_x > 1 and ndoms_y > 1 and ndoms_z > 1 " << std::endl <<
                      "values provided:" << std::endl <<
                      " mesh_type: " << mesh_type << std::endl <<
                      " ndoms_x: " << ndoms_x << std::endl <<
                      " ndoms_y: " << ndoms_y << std::endl <<
                      " ndoms_z: " << ndoms_z << std::endl);
    }

    // validate braid input here for nicer exception loc
    bool npts_ok = (npts_x > 1 && npts_y > 1);

    if( npts_z != 0 &&
        braid_2d_only_shape_type(mesh_type) )
    {
        if(STRICT_NPTS_Z_FOR_2D)
        {
            npts_ok = false;
        }
        else
        {
            CONDUIT_INFO("DEPRECATED:"
                    " Detected npts_z != 0 for example with 2D shape type."
                    " This will throw a conduit::Error in a future release.");
        }
    }

    if( braid_3d_only_shape_type(mesh_type) )
    {
        // z must be valid for these cases
        if(npts_z < 2)
        {
            npts_ok = false;
        }
    }

    if(!npts_ok)
    {
        CONDUIT_ERROR("blueprint::mesh::examples::grid requires: " << std::endl <<
                      "For 2D only topologies"
                      " npts_x > 1 and npts_y > 1 and npts_z == 0"
                      << std::endl <<
                      "For 3D only topologies"
                      "npts_x > 1 and npts_y > 1 and "
                      " npts_z > 1"
                      << std::endl <<
                      "values provided:" << std::endl <<
                      " mesh_type: " << mesh_type << std::endl <<
                      " npts_x: " << npts_x << std::endl <<
                      " npts_y: " << npts_y << std::endl <<
                      " npts_z: " << npts_z << std::endl);
    }


    for(index_t dz = 0, domain_id = 0; dz < ndoms_z; dz++)
    {
        for(index_t dy = 0; dy < ndoms_y; dy++)
        {
            for(index_t dx = 0; dx < ndoms_x; dx++, domain_id++)
            {
                Node &domain_node = res["domain" + std::to_string(domain_id)];
                braid(mesh_type, npts_x, npts_y, npts_z, domain_node);
                domain_node["state/domain_id"].set(domain_id);

                Node &domain_coords_node = domain_node["coordsets/coords"];
                const std::string domain_coords_path =
                    (domain_coords_node["type"].as_string() == "uniform") ? "origin" : "values";
                const std::vector<std::string> domain_axes =
                    conduit::blueprint::mesh::utils::coordset::axes(domain_coords_node);

                for(const std::string &domain_axis : domain_axes)
                {
                    const index_t domain_axis_offset = 20.0 * (
                        (domain_axis == "x") ? dx : (
                        (domain_axis == "y") ? dy : (
                        (domain_axis == "z") ? dz : 0)));

                    float64_array domain_axis_coords =
                        domain_coords_node[domain_coords_path][domain_axis].as_float64_array();
                    for(index_t dai = 0; dai < domain_axis_coords.number_of_elements(); dai++)
                    {
                        domain_axis_coords[dai] += domain_axis_offset;
                    }
                }
            }
        }
    }

    braid_init_example_adjset(res);
}


//---------------------------------------------------------------------------//
void
braid(const std::string &mesh_type,
      index_t npts_x, // number of points in x
      index_t npts_y, // number of points in y
      index_t npts_z, // number of points in z
      Node &res)
{
    bool npts_x_ok = true;
    bool npts_y_ok = true;
    bool npts_z_ok = true;

    if( mesh_type == "points" || mesh_type == "points_implicit" )
    {
        if( npts_x < 1 )
        {
            npts_x_ok = false;
        }
        if( npts_y < 1 )
        {
            npts_y_ok = false;
        }
        if( npts_z < 0 )
        {
            npts_z_ok = false;
        }
    }
    else
    {
        if( npts_x < 2 )
        {
            npts_x_ok = false;
        }

        if( npts_y < 2 )
        {
            npts_y_ok = false;
        }

        // check 2d cases which require npts z = 0
        

        if ( npts_z != 0 &&
             braid_2d_only_shape_type(mesh_type) )
        {
            if(STRICT_NPTS_Z_FOR_2D)
            {
                npts_z_ok = false;
            }
            else
            {
                CONDUIT_INFO("DEPRECATED:"
                    " Detected npts_z != 0 for example with 2D shape type."
                    " This will throw a conduit::Error in a future release.");
            }
        }

        // check 3d cases which require z
        if( braid_3d_only_shape_type(mesh_type) )
        {
            // z must be valid for these cases
            if(npts_z < 2)
            {
                npts_z_ok = false;
            }
        }
    }


    if( ! (npts_x_ok && npts_y_ok && npts_z_ok) )
    {
        if( mesh_type == "points" || mesh_type == "points_implicit" )
        {
            // error, not enough points to create the topo
            CONDUIT_ERROR("braid with points-based topology requires"
                          "npts_x > 0,  npts_y > 0  and npts_z >= 0 "
                          "values provided:" << std::endl << 
                          " mesh_type: " << mesh_type << std::endl  << 
                          " npts_x: " << npts_x << std::endl <<
                          " npts_y: " << npts_y << std::endl <<
                          " npts_z: " << npts_z << std::endl);
        }
        else if( npts_z != 0 && braid_2d_only_shape_type(mesh_type) )
        {
            // we won't pass z on, so error if z is non zero.
            CONDUIT_ERROR("braid with 2D topology requires "
                          "npts_x > 1 and npts_y > 1 "
                          " and npts_z == 0 " << std::endl << 
                          "values provided:" << std::endl << 
                          " mesh_type: " << mesh_type << std::endl <<
                          " npts_x: " << npts_x << std::endl <<
                          " npts_y: " << npts_y << std::endl <<
                          " npts_z: " << npts_z << std::endl);
        }
        else
        {
            // error, not enough points to create the topo
            CONDUIT_ERROR("braid with non-points topology requires "
                          "npts_x > 1 and npts_y > 1 "
                          " and for mesh_type={\"tets\", \"hexs\", "
                          " \"hexs_poly\", or \"hexs_and_tets\"} "
                          " npts_z must be > 1" << std::endl << 
                          "values provided:" << std::endl << 
                          " mesh_type: " << mesh_type << std::endl <<
                          " npts_x: " << npts_x << std::endl <<
                          " npts_y: " << npts_y << std::endl <<
                          " npts_z: " << npts_z << std::endl);
        }
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
        // set cycle and domain id
        dom["state/cycle"] = 0;
        dom["state/domain_id"] = d;

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
              index_t nz,
              Node &res)
{
    if (nz == 1)
    {
        std::map< point, index_t > point_map;
        std::map< index_t, point > point_rmap;
        std::vector< std::vector<index_t> > polygons;
        std::vector< index_t > levels;

        polytess_recursive(nlevels, point_map, point_rmap, polygons, levels);

        index_t conn_size = 0;
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
        topology["elements/sizes"].set(DataType::uint64(polygons.size()));

        uint64_array conn_array = topology["elements/connectivity"].value();
        uint64_array size_array = topology["elements/sizes"].value();  
        for(index_t pi = 0, ci = 0; pi < (index_t)polygons.size(); pi++)
        {
            const std::vector<index_t> &p = polygons[pi];

            size_array[pi] = p.size();
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
    else if (nz > 1)
    {
        // Our goal here is to take the original polytess and extend it
        // into 3 dimensions. The way we will accomplish this is by
        // placing the original polytess into the z = 0 plane, placing a 
        // copy of it into the z = 1 plane, another into the z = 2 plane, 
        // etc., and constructing "walls" between the top and bottom edges
        // of each polytessalation. Then we will specify polyhedra that use
        // all the faces at our disposal.
        Node poly;
        polytess(nlevels, 1, poly);

        Node &res_coords = res["coordsets/coords"];
        Node &res_topo = res["topologies/topo"];
        Node &res_fields = res["fields"];

        // SET UP COORDINATES

        res_coords["type"] = poly["coordsets/coords/type"];

        int num_orig_points = poly["coordsets/coords/values/x"].dtype().number_of_elements();
        int num_points = nz * num_orig_points;

        res_coords["values/x"].set(conduit::DataType::float64(num_points));
        res_coords["values/y"].set(conduit::DataType::float64(num_points));
        res_coords["values/z"].set(conduit::DataType::float64(num_points));

        float64 *poly_x_vals = poly["coordsets/coords/values/x"].value();
        float64 *poly_y_vals = poly["coordsets/coords/values/y"].value();

        float64 *x_vals = res_coords["values/x"].value();
        float64 *y_vals = res_coords["values/y"].value();
        float64 *z_vals = res_coords["values/z"].value();

        // all the original points are added nz times, the first time with a z-value of 0,
        // and the second time with a z-value of 1, etc.
        for (int i = 0; i < num_points; i ++)
        {
            int i_mod_num_orig_points = i % num_orig_points;
            x_vals[i] = poly_x_vals[i_mod_num_orig_points];
            y_vals[i] = poly_y_vals[i_mod_num_orig_points];
            z_vals[i] = i / num_orig_points;
        }

        res_topo["type"] = poly["topologies/topo/type"];
        res_topo["coordset"] = poly["topologies/topo/coordset"];

        // SUBELEMENTS

            // In the nz = 2 case, 
            // if we take our polytess and reflect it, we have two polytessalations on top of one another,
            // with a distance of 1 unit in between. If we go through each polygon, and for each one, 
            // select every pair of adjacent points and add a new polygon that uses the points from that pair
            // as well as the points directly above in the reflected polytess, then we will have duplicate
            // polygons, simply because the polygons share vertices with one another, so multiple polygons
            // will have the same "walls". This section accounts for that. The nz > 2 cases are similar, 
            // except this situation is repeated for each level of the stacked polytess layers.

            res_topo["subelements/shape"] = poly["topologies/topo/elements/shape"];

            // CALCULATE THE NUMBER OF NEW POLYGONS
            int num_duplicate_polygons = 0;
            for (int n = 1; n < nlevels; n ++)
            {
                // this formula is the sum of two other formulas:
                // 1) the number of outward edges for an (n - 1) polytess, and
                // 2) the number of edges that squares in the polytess share with 
                //    neighboring octagons in their level.
                // This sum is again summed for each level, which will produce the 
                // number of duplicates we would have gotten had we simply made a 
                // polygon for each pair of adjacent points in each polygon, 
                // as described above.
                num_duplicate_polygons += 8 * (3 * n - 1);
            }
            int num_new_polygons = -1 * num_duplicate_polygons;
            int sizeof_poly_sizes = poly["topologies/topo/elements/sizes"].dtype().number_of_elements();
            uint64 *poly_sizes = poly["topologies/topo/elements/sizes"].value();
            for (int i = 0; i < sizeof_poly_sizes; i ++)
            {
                num_new_polygons += poly_sizes[i];
            }
            // we need this number of polygons for each level we add
            num_new_polygons *= nz - 1;

            // SET UP SIZES
            const int points_per_quad = 4;
            // the sizes must have space for the original sizes array, nz copies of it extending upwards, 
            // and all the walls; hence the addition of the num_new_polygons.
            int length_of_new_sizes = sizeof_poly_sizes * nz + num_new_polygons;
            res_topo["subelements/sizes"].set(conduit::DataType::uint64(length_of_new_sizes));
            uint64 *sizes = res_topo["subelements/sizes"].value();
            for (int i = 0; i < length_of_new_sizes; i ++)
            {
                // the original and reflected polytess sizes
                if (i < sizeof_poly_sizes * nz)
                {
                    sizes[i] = poly_sizes[i % sizeof_poly_sizes];
                }
                // all the new polygons are quads
                else
                {
                    sizes[i] = points_per_quad;
                }
            }

            // SET UP OFFSETS
            res_topo["subelements/offsets"].set(conduit::DataType::uint64(length_of_new_sizes));
            uint64 *offsets = res_topo["subelements/offsets"].value();
            offsets[0] = 0;
            for (int i = 1; i < length_of_new_sizes; i ++)
            {
                offsets[i] = sizes[i - 1] + offsets[i - 1];
            }

            // SET UP CONNECTIVITY
            const int sizeof_poly_connec = poly["topologies/topo/elements/connectivity"].dtype().number_of_elements();
            const int sizeof_sub_connec = sizeof_poly_connec * nz + num_new_polygons * points_per_quad;
            res_topo["subelements/connectivity"].set(conduit::DataType::uint64(sizeof_sub_connec));
            uint64 *connec = res_topo["subelements/connectivity"].value();
            uint64 *poly_connec = poly["topologies/topo/elements/connectivity"].value();

            // first, copy the original connectivity, then the reflected connectivities, which luckily
            // is as simple as adding an offset to the original.
            for (int i = 0; i < sizeof_poly_connec * nz; i ++)
            {
                connec[i] = poly_connec[i % sizeof_poly_connec] + (i / sizeof_poly_connec) * num_orig_points;
            }

            // now the tricky part, where we want to add new faces for the quads that make 
            // up the walls, and, most importantly, keep track of them.
            // To do this, we use a map. Put simply, it maps quad faces to polyhedra that 
            // will use them.

            // map a quad (a set of 4 ints) to a pair,
            // where car is the index of the quad in connec (an int)
            // and cdr is the list of associated polyhedra (a vector of integers)
            std::map<std::set<int>, std::pair<int, std::vector<int>>> quad_map;

            int k = 0;
            for (int i = 0; i < sizeof_poly_sizes * (nz - 1); i ++)
            {
                for (uint64 j = 0; j < sizes[i]; j ++)
                {
                    int curr = connec[offsets[i] + j];
                    int next = connec[(j + 1) % sizes[i] + offsets[i]];

                    // adding num_orig_points will give us the points directly above (one level up) in our mesh
                    std::set<int> quad = {curr, next, next + num_orig_points, curr + num_orig_points};
                    std::vector<int> associated_polyhedra{i};
                    int currpos = sizeof_poly_sizes * nz + k / points_per_quad;

                    if (quad_map.insert(std::make_pair(quad, std::make_pair(currpos, associated_polyhedra))).second)
                    {
                        connec[sizeof_poly_connec * nz + k] = curr;
                        k ++;
                        connec[sizeof_poly_connec * nz + k] = next;
                        k ++;
                        connec[sizeof_poly_connec * nz + k] = next + num_orig_points;
                        k ++;
                        connec[sizeof_poly_connec * nz + k] = curr + num_orig_points;
                        k ++;
                    }
                    else
                    {
                        quad_map[quad].second.push_back(i);
                    }
                }
            }

        // ELEMENTS

            res_topo["elements/shape"] = "polyhedral";

            const int num_polyhedra = sizeof_poly_sizes * (nz - 1);

            // SET UP SIZES
            const int points_per_octagon = 8;
            const int faces_per_octaprism = 10;
            const int faces_per_hex = 6;
            res_topo["elements/sizes"].set(conduit::DataType::uint64(num_polyhedra));
            uint64 *elements_sizes = res_topo["elements/sizes"].value();
            for (int i = 0; i < num_polyhedra; i ++)
            {
                // this ensures that each original octagon is associated with an octagonal prism,
                // and each original square gets a cube.
                elements_sizes[i] = (poly_sizes[i % sizeof_poly_sizes] == points_per_octagon) ? faces_per_octaprism : faces_per_hex;
            }

            // SET UP OFFSETS
            res_topo["elements/offsets"].set(conduit::DataType::uint64(num_polyhedra));
            uint64 *elements_offsets = res_topo["elements/offsets"].value();
            elements_offsets[0] = 0;
            for (int i = 1; i < num_polyhedra; i ++)
            {
                elements_offsets[i] = elements_sizes[i - 1] + elements_offsets[i - 1];
            }

            // SET UP CONNECTIVITY
            std::map<int, std::vector<int>> polyhedra_to_quads;
            std::map<std::set<int>, std::pair<int, std::vector<int>>>::iterator itr;
            // the one-to-many map we set up before must be reversed. Before, we mapped
            // quads to polyhedra that used them, now, we wish to map polyhedra to
            // quads they use.
            for (itr = quad_map.begin(); itr != quad_map.end(); itr ++)
            {
                int pos = itr->second.first;
                std::vector<int> curr = itr->second.second;
                for (uint64 i = 0; i < curr.size(); i ++)
                {
                    std::vector<int> quads{pos};
                    if (!polyhedra_to_quads.insert(std::make_pair(curr[i], quads)).second)
                    {
                        polyhedra_to_quads[curr[i]].push_back(pos);
                    }
                }
            }

            int sizeof_elements_connec = 0;
            for (int i = 0; i < num_polyhedra; i ++)
            {
                sizeof_elements_connec += elements_sizes[i];
            }

            res_topo["elements/connectivity"].set(conduit::DataType::uint64(sizeof_elements_connec));
            uint64 *elements_connec = res_topo["elements/connectivity"].value();
            int l = 0;
            for (int i = 0; i < num_polyhedra; i ++)
            {
                // to the polyhedral connectivity array, for each polyhedron, we first add 
                // the polygon from the original polytess, then the polygon directly above,
                // then each of the vertical faces, thanks to the maps we set up earlier
                elements_connec[l] = i;
                l ++;
                elements_connec[l] = i + sizeof_poly_sizes;
                l ++;
                std::vector<int> quads = polyhedra_to_quads[i];
                for (uint64 j = 0; j < quads.size(); j ++)
                {
                    elements_connec[l] = quads[j];
                    l ++;
                }
            }

        // SET UP FIELDS
        res_fields["level/topology"] = poly["fields/level/topology"];
        res_fields["level/association"] = poly["fields/level/association"];
        res_fields["level/volume_dependent"] = poly["fields/level/volume_dependent"];

        const int sizeof_poly_field_values = poly["fields/level/values"].dtype().number_of_elements();
        res_fields["level/values"].set(conduit::DataType::uint32(num_polyhedra));

        uint32 *values = res_fields["level/values"].value();
        uint32 *poly_values = poly["fields/level/values"].value();

        // because for each original polygon we have made nz new polyhedra, 
        // setting up the field is a simple matter of just copying over our 
        // original field data a few times.
        for (int i = 0; i < num_polyhedra; i ++)
        {
            values[i] = poly_values[i % sizeof_poly_field_values];
        }
    }
    else
    {
        CONDUIT_ERROR("polytess: nz must be an integer greater than 0.")
    }
}


//-----------------------------------------------------------------------------
void 
polychain(const index_t length, // how long the chain ought to be
          Node &res)
{
    res.reset();

    Node &chain_coords = res["coordsets/coords"];
    Node &chain_topo = res["topologies/topo"];
    Node &chain_fields = res["fields"];

    chain_coords["type"] = "explicit";

    const index_t num_verts_per_hex = 8;
    const index_t num_verts_per_triprism = 6;
    const index_t num_verts_per_chain_pair = num_verts_per_hex + 2 * num_verts_per_triprism;

    chain_coords["values/x"].set(conduit::DataType::int64(length * num_verts_per_chain_pair));
    chain_coords["values/y"].set(conduit::DataType::int64(length * num_verts_per_chain_pair));
    chain_coords["values/z"].set(conduit::DataType::int64(length * num_verts_per_chain_pair));

    int64 *values_x = chain_coords["values/x"].value();
    int64 *values_y = chain_coords["values/y"].value();
    int64 *values_z = chain_coords["values/z"].value();

    for (int i = 0; i < length; i ++)
    {
        // points for the cubes
        values_x[i * num_verts_per_chain_pair] = 1 + i * 2;           values_y[i * num_verts_per_chain_pair] = 1;       values_z[i * num_verts_per_chain_pair] = 1 + i * 2;
        values_x[i * num_verts_per_chain_pair + 1] = 1 + i * 2;       values_y[i * num_verts_per_chain_pair + 1] = 1;   values_z[i * num_verts_per_chain_pair + 1] = -1 + i * 2;
        values_x[i * num_verts_per_chain_pair + 2] = -1 + i * 2;      values_y[i * num_verts_per_chain_pair + 2] = 1;   values_z[i * num_verts_per_chain_pair + 2] = -1 + i * 2;
        values_x[i * num_verts_per_chain_pair + 3] = -1 + i * 2;      values_y[i * num_verts_per_chain_pair + 3] = 1;   values_z[i * num_verts_per_chain_pair + 3] = 1 + i * 2;
        values_x[i * num_verts_per_chain_pair + 4] = 1 + i * 2;       values_y[i * num_verts_per_chain_pair + 4] = -1;  values_z[i * num_verts_per_chain_pair + 4] = 1 + i * 2;
        values_x[i * num_verts_per_chain_pair + 5] = 1 + i * 2;       values_y[i * num_verts_per_chain_pair + 5] = -1;  values_z[i * num_verts_per_chain_pair + 5] = -1 + i * 2;
        values_x[i * num_verts_per_chain_pair + 6] = -1 + i * 2;      values_y[i * num_verts_per_chain_pair + 6] = -1;  values_z[i * num_verts_per_chain_pair + 6] = -1 + i * 2;
        values_x[i * num_verts_per_chain_pair + 7] = -1 + i * 2;      values_y[i * num_verts_per_chain_pair + 7] = -1;  values_z[i * num_verts_per_chain_pair + 7] = 1 + i * 2;

        // points for half the triangular prisms
        values_x[i * num_verts_per_chain_pair + 8] = 1 + i * 2;       values_y[i * num_verts_per_chain_pair + 8] = 1;   values_z[i * num_verts_per_chain_pair + 8] = -1 + i * 2 + 2;
        values_x[i * num_verts_per_chain_pair + 9] = -1 + i * 2;      values_y[i * num_verts_per_chain_pair + 9] = 1;   values_z[i * num_verts_per_chain_pair + 9] = -1 + i * 2 + 2;
        values_x[i * num_verts_per_chain_pair + 10] = -1 + i * 2 + 2; values_y[i * num_verts_per_chain_pair + 10] = 1;  values_z[i * num_verts_per_chain_pair + 10] = 1 + i * 2 + 2;
        values_x[i * num_verts_per_chain_pair + 11] = 1 + i * 2;      values_y[i * num_verts_per_chain_pair + 11] = -1; values_z[i * num_verts_per_chain_pair + 11] = -1 + i * 2 + 2;
        values_x[i * num_verts_per_chain_pair + 12] = -1 + i * 2;     values_y[i * num_verts_per_chain_pair + 12] = -1; values_z[i * num_verts_per_chain_pair + 12] = -1 + i * 2 + 2;
        values_x[i * num_verts_per_chain_pair + 13] = -1 + i * 2 + 2; values_y[i * num_verts_per_chain_pair + 13] = -1; values_z[i * num_verts_per_chain_pair + 13] = 1 + i * 2 + 2;

        // points for the other half
        values_x[i * num_verts_per_chain_pair + 14] = 1 + i * 2;      values_y[i * num_verts_per_chain_pair + 14] = 1;  values_z[i * num_verts_per_chain_pair + 14] = -1 + i * 2 + 2;
        values_x[i * num_verts_per_chain_pair + 15] = -1 + i * 2 + 2; values_y[i * num_verts_per_chain_pair + 15] = 1;  values_z[i * num_verts_per_chain_pair + 15] = -1 + i * 2;
        values_x[i * num_verts_per_chain_pair + 16] = -1 + i * 2 + 4; values_y[i * num_verts_per_chain_pair + 16] = 1;  values_z[i * num_verts_per_chain_pair + 16] = 1 + i * 2;
        values_x[i * num_verts_per_chain_pair + 17] = 1 + i * 2;      values_y[i * num_verts_per_chain_pair + 17] = -1; values_z[i * num_verts_per_chain_pair + 17] = -1 + i * 2 + 2;
        values_x[i * num_verts_per_chain_pair + 18] = -1 + i * 2 + 2; values_y[i * num_verts_per_chain_pair + 18] = -1; values_z[i * num_verts_per_chain_pair + 18] = -1 + i * 2;
        values_x[i * num_verts_per_chain_pair + 19] = -1 + i * 2 + 4; values_y[i * num_verts_per_chain_pair + 19] = -1; values_z[i * num_verts_per_chain_pair + 19] = 1 + i * 2;
    }

    chain_topo["type"] = "unstructured";
    chain_topo["coordset"] = "coords";
    chain_topo["elements/shape"] = "polyhedral";

    const index_t num_faces_per_hex = 6;
    const index_t num_faces_per_triprism = 5;
    const index_t num_faces_per_chain_pair = num_faces_per_hex + 2 * num_faces_per_triprism;

    chain_topo["elements/connectivity"].set(conduit::DataType::int64(length * num_faces_per_chain_pair));
    int64 *connec = chain_topo["elements/connectivity"].value();
    for (int i = 0; i < length * num_faces_per_chain_pair; i ++)
    {
        // our faces are specified in order and no faces are reused
        connec[i] = i;
    }

    // this is 3 because every time length is increased by 1, 3 more polyhedra are added,
    const index_t num_polyhedra_per_chain_pair = 3;

    // the cube and two prisms
    chain_topo["elements/sizes"].set(conduit::DataType::int64(length * num_polyhedra_per_chain_pair));
    int64 *sizes = chain_topo["elements/sizes"].value();
    for (int i = 0; i < length * num_polyhedra_per_chain_pair; i ++)
    {
        // this ensures that sizes will be of the form {6,5,5, 6,5,5, 6,5,5, ..., 6,5,5}
        sizes[i] = ((i % num_polyhedra_per_chain_pair) > 0) ? num_faces_per_triprism : num_faces_per_hex;
    }

    chain_topo["subelements/shape"] = "polygonal";

    const index_t num_points_per_quad_face = 4;
    const index_t num_points_per_tri_face = 3;
    const index_t num_tri_faces_in_triprism = 2;
    const index_t num_quad_faces_in_triprism = 3;
    const index_t sizeof_hex_connectivity = num_faces_per_hex * num_points_per_quad_face;
    const index_t sizeof_triprism_connectivity = num_tri_faces_in_triprism * num_points_per_tri_face +
                                                 num_quad_faces_in_triprism * num_points_per_quad_face;
    const index_t sizeof_chainpair_connec = sizeof_hex_connectivity + sizeof_triprism_connectivity * 2;

    chain_topo["subelements/connectivity"].set(conduit::DataType::int64(length * sizeof_chainpair_connec));
    int64 *sub_connec = chain_topo["subelements/connectivity"].value();
    for (int i = 0; i < length; i ++)
    {
        // CUBE
        // top                                                          // bottom
        sub_connec[i * sizeof_chainpair_connec] = 0 + i * 20;        sub_connec[i * sizeof_chainpair_connec + 4] = 4 + i * 20;
        sub_connec[i * sizeof_chainpair_connec + 1] = 1 + i * 20;    sub_connec[i * sizeof_chainpair_connec + 5] = 5 + i * 20;
        sub_connec[i * sizeof_chainpair_connec + 2] = 2 + i * 20;    sub_connec[i * sizeof_chainpair_connec + 6] = 6 + i * 20;
        sub_connec[i * sizeof_chainpair_connec + 3] = 3 + i * 20;    sub_connec[i * sizeof_chainpair_connec + 7] = 7 + i * 20;
        
        // side where x = 1                                             // side where x = -1
        sub_connec[i * sizeof_chainpair_connec + 8] = 0 + i * 20;    sub_connec[i * sizeof_chainpair_connec + 12] = 2 + i * 20;
        sub_connec[i * sizeof_chainpair_connec + 9] = 1 + i * 20;    sub_connec[i * sizeof_chainpair_connec + 13] = 3 + i * 20;
        sub_connec[i * sizeof_chainpair_connec + 10] = 5 + i * 20;   sub_connec[i * sizeof_chainpair_connec + 14] = 7 + i * 20;
        sub_connec[i * sizeof_chainpair_connec + 11] = 4 + i * 20;   sub_connec[i * sizeof_chainpair_connec + 15] = 6 + i * 20;
        
        // side where z = 1                                             // side where z = -1
        sub_connec[i * sizeof_chainpair_connec + 16] = 0 + i * 20;   sub_connec[i * sizeof_chainpair_connec + 20] = 1 + i * 20;
        sub_connec[i * sizeof_chainpair_connec + 17] = 3 + i * 20;   sub_connec[i * sizeof_chainpair_connec + 21] = 2 + i * 20;
        sub_connec[i * sizeof_chainpair_connec + 18] = 7 + i * 20;   sub_connec[i * sizeof_chainpair_connec + 22] = 6 + i * 20;
        sub_connec[i * sizeof_chainpair_connec + 19] = 4 + i * 20;   sub_connec[i * sizeof_chainpair_connec + 23] = 5 + i * 20;

        // PRISM 1
        sub_connec[i * sizeof_chainpair_connec + 24] = 9 + i * 20;  sub_connec[i * sizeof_chainpair_connec + 28] = 8 + i * 20;  sub_connec[i * sizeof_chainpair_connec + 32] = 8 + i * 20;
        sub_connec[i * sizeof_chainpair_connec + 25] = 10 + i * 20; sub_connec[i * sizeof_chainpair_connec + 29] = 9 + i * 20;  sub_connec[i * sizeof_chainpair_connec + 33] = 10 + i * 20;
        sub_connec[i * sizeof_chainpair_connec + 26] = 13 + i * 20; sub_connec[i * sizeof_chainpair_connec + 30] = 12 + i * 20; sub_connec[i * sizeof_chainpair_connec + 34] = 13 + i * 20;
        sub_connec[i * sizeof_chainpair_connec + 27] = 12 + i * 20; sub_connec[i * sizeof_chainpair_connec + 31] = 11 + i * 20; sub_connec[i * sizeof_chainpair_connec + 35] = 11 + i * 20;

        sub_connec[i * sizeof_chainpair_connec + 36] = 8 + i * 20;   sub_connec[i * sizeof_chainpair_connec + 39] = 11 + i * 20;
        sub_connec[i * sizeof_chainpair_connec + 37] = 9 + i * 20;   sub_connec[i * sizeof_chainpair_connec + 40] = 12 + i * 20;
        sub_connec[i * sizeof_chainpair_connec + 38] = 10 + i * 20;  sub_connec[i * sizeof_chainpair_connec + 41] = 13 + i * 20;

        // PRISM 2
        sub_connec[i * sizeof_chainpair_connec + 42] = 15 + i * 20; sub_connec[i * sizeof_chainpair_connec + 46] = 14 + i * 20; sub_connec[i * sizeof_chainpair_connec + 50] = 14 + i * 20;
        sub_connec[i * sizeof_chainpair_connec + 43] = 16 + i * 20; sub_connec[i * sizeof_chainpair_connec + 47] = 15 + i * 20; sub_connec[i * sizeof_chainpair_connec + 51] = 16 + i * 20;
        sub_connec[i * sizeof_chainpair_connec + 44] = 19 + i * 20; sub_connec[i * sizeof_chainpair_connec + 48] = 18 + i * 20; sub_connec[i * sizeof_chainpair_connec + 52] = 19 + i * 20;
        sub_connec[i * sizeof_chainpair_connec + 45] = 18 + i * 20; sub_connec[i * sizeof_chainpair_connec + 49] = 17 + i * 20; sub_connec[i * sizeof_chainpair_connec + 53] = 17 + i * 20;

        sub_connec[i * sizeof_chainpair_connec + 54] = 14 + i * 20; sub_connec[i * sizeof_chainpair_connec + 57] = 17 + i * 20;
        sub_connec[i * sizeof_chainpair_connec + 55] = 15 + i * 20; sub_connec[i * sizeof_chainpair_connec + 58] = 18 + i * 20;
        sub_connec[i * sizeof_chainpair_connec + 56] = 16 + i * 20; sub_connec[i * sizeof_chainpair_connec + 59] = 19 + i * 20;
    }

    chain_topo["subelements/sizes"].set(conduit::DataType::int64(length * num_faces_per_chain_pair));

    int64 *sub_sizes = chain_topo["subelements/sizes"].value();

    for (int i = 0; i < length * num_faces_per_chain_pair; i ++)
    {
        // this ensures sizes will be of the form {4,4,4,4,4,4,4,4,4,3,3,4,4,4,3,3, 4,4,4,4,4,4,4,4,4,3,3,4,4,4,3,3, ...}
        int imodfaces = i % num_faces_per_chain_pair;
        sub_sizes[i] = ((imodfaces < 9) || ((imodfaces > 10) && (imodfaces < 14))) ? num_points_per_quad_face : num_points_per_tri_face;
    }

    blueprint::mesh::topology::unstructured::generate_offsets(chain_topo,
                                                              chain_topo["elements/offsets"]);
    
    chain_fields["chain/topology"] = "topo";
    chain_fields["chain/association"] = "element";
    chain_fields["chain/volume_dependent"] = "false";
    chain_fields["chain/values"].set(conduit::DataType::int64(length * num_polyhedra_per_chain_pair));
    int64 *field_values = chain_fields["chain/values"].value();

    for (int i = 0; i < length * num_polyhedra_per_chain_pair; i ++)
    {
        // ensures that the field is of the form {0,1,1, 0,1,1, ..., 0,1,1}
        field_values[i] = (i % num_polyhedra_per_chain_pair) == 0 ? 0 : 1;
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
void
adjset_uniform(Node &res)
{
    for(int32 i = 0; i < 8; i++)
    {
        std::ostringstream oss;
        oss << "domain_" << std::setfill('0') << std::setw(6) << i;
        const std::string domain_name = oss.str();

        Node &domain_node = res[domain_name];

        domain_node["state/domain_id"].set(i);
        Node &domain_coords = domain_node["coordsets/coords"];
        domain_coords["type"].set_string("uniform");
        domain_coords["dims/i"].set_int32(21);
        domain_coords["dims/j"].set_int32(21);
        domain_coords["spacing/dx"].set_float64(0.0125);
        domain_coords["spacing/dy"].set_float64(0.025);
        domain_coords["origin/x"].set_float64(0.25*(i/2));
        domain_coords["origin/y"].set_float64(0.5*(i%2));

        Node &domain_topo = domain_node["topologies/topo"];
        domain_topo["elements/origin/i"].set_int32(20*(i/2));
        domain_topo["elements/origin/j"].set_int32(20*(i%2));
        domain_topo["type"].set_string("uniform");
        domain_topo["coordset"].set_string("coords");

        // add a simple field that has the domain id
        Node &domain_field = domain_node["fields/id"];
        domain_field["association"] = "element";
        domain_field["topology"] = "topo";
        domain_field["values"] = DataType::int32(20 * 20);

        int32_array vals =  domain_field["values"].value();
        for(int j=0;j<20*20;j++)
        {
            vals[j] = i;
        }

        Node &domain_adjsets = domain_node["adjsets/adjset"];
        domain_adjsets["association"].set_string("vertex");
        domain_adjsets["topology"].set_string("topo");
        Node &adjset_groups = domain_adjsets["groups"];

        if (i == 0)
        {
            adjset_groups["group_000001/neighbors"] = DataType::int32(2);
            adjset_groups["group_000002/neighbors"] = DataType::int32(2);
            adjset_groups["group_000003/neighbors"] = DataType::int32(2);
            adjset_groups["group_000001/neighbors"].as_int_array()[0] = 0;
            adjset_groups["group_000002/neighbors"].as_int_array()[0] = 0;
            adjset_groups["group_000003/neighbors"].as_int_array()[0] = 0;
            adjset_groups["group_000001/neighbors"].as_int_array()[1] = 1;
            adjset_groups["group_000002/neighbors"].as_int_array()[1] = 2;
            adjset_groups["group_000003/neighbors"].as_int_array()[1] = 3;
            Node &windows_node1 = adjset_groups["group_000001/windows"];
            for(index_t w = 0; w <= 1; w++)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node1[window_name]["origin/i"] = 0;
                windows_node1[window_name]["origin/j"] = 20;
                windows_node1[window_name]["dims/i"] = 21;
                windows_node1[window_name]["dims/j"] = 1;
                windows_node1[window_name]["ratio/i"] = 1;
                windows_node1[window_name]["ratio/j"] = 1;
            }
            Node &windows_node2 = adjset_groups["group_000002/windows"];
            for(index_t w = 0; w <= 2; w += 2)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node2[window_name]["origin/i"] = 0;
                windows_node2[window_name]["origin/j"] = 20;
                windows_node2[window_name]["dims/i"] = 21;
                windows_node2[window_name]["dims/j"] = 1;
                windows_node2[window_name]["ratio/i"] = 1;
                windows_node2[window_name]["ratio/j"] = 1;
            }
            Node &windows_node3 = adjset_groups["group_000003/windows"];
            for(index_t w = 0; w <= 3; w += 3)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node3[window_name]["origin/i"] = 20;
                windows_node3[window_name]["origin/j"] = 20;
                windows_node3[window_name]["dims/i"] = 1;
                windows_node3[window_name]["dims/j"] = 1;
                windows_node3[window_name]["ratio/i"] = 1;
                windows_node3[window_name]["ratio/j"] = 1;
            }
        }
        else if (i == 1)
        {
            adjset_groups["group_000000/neighbors"] = DataType::int32(2);
            adjset_groups["group_000002/neighbors"] = DataType::int32(2);
            adjset_groups["group_000003/neighbors"] = DataType::int32(2);
            adjset_groups["group_000000/neighbors"].as_int_array()[0] = 1;
            adjset_groups["group_000002/neighbors"].as_int_array()[0] = 1;
            adjset_groups["group_000003/neighbors"].as_int_array()[0] = 1;
            adjset_groups["group_000000/neighbors"].as_int_array()[1] = 0;
            adjset_groups["group_000002/neighbors"].as_int_array()[1] = 2;
            adjset_groups["group_000003/neighbors"].as_int_array()[1] = 3;
            Node &windows_node0 = adjset_groups["group_000000/windows"];
            for(index_t w = 0; w <= 1; w++)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node0[window_name]["origin/i"] = 0;
                windows_node0[window_name]["origin/j"] = 20;
                windows_node0[window_name]["dims/i"] = 21;
                windows_node0[window_name]["dims/j"] = 1;
                windows_node0[window_name]["ratio/i"] = 1;
                windows_node0[window_name]["ratio/j"] = 1;
            }
            Node &windows_node2 = adjset_groups["group_000002/windows"];
            for(index_t w = 1; w <= 2; w++)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node2[window_name]["origin/i"] = 20;
                windows_node2[window_name]["origin/j"] = 20;
                windows_node2[window_name]["dims/i"] = 1;
                windows_node2[window_name]["dims/j"] = 1;
                windows_node2[window_name]["ratio/i"] = 1;
                windows_node2[window_name]["ratio/j"] = 1;
            }
            Node &windows_node3 = adjset_groups["group_000003/windows"];
            for(index_t w = 1; w <= 3; w += 2)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node3[window_name]["origin/i"] = 20;
                windows_node3[window_name]["origin/j"] = 20;
                windows_node3[window_name]["dims/i"] = 1;
                windows_node3[window_name]["dims/j"] = 21;
                windows_node3[window_name]["ratio/i"] = 1;
                windows_node3[window_name]["ratio/j"] = 1;
            }
        }
        else if (i == 2)
        {
            adjset_groups["group_000000/neighbors"] = DataType::int32(2);
            adjset_groups["group_000001/neighbors"] = DataType::int32(2);
            adjset_groups["group_000003/neighbors"] = DataType::int32(2);
            adjset_groups["group_000004/neighbors"] = DataType::int32(2);
            adjset_groups["group_000005/neighbors"] = DataType::int32(2);
            adjset_groups["group_000000/neighbors"].as_int_array()[0] = 2;
            adjset_groups["group_000001/neighbors"].as_int_array()[0] = 2;
            adjset_groups["group_000003/neighbors"].as_int_array()[0] = 2;
            adjset_groups["group_000004/neighbors"].as_int_array()[0] = 2;
            adjset_groups["group_000005/neighbors"].as_int_array()[0] = 2;
            adjset_groups["group_000000/neighbors"].as_int_array()[1] = 0;
            adjset_groups["group_000001/neighbors"].as_int_array()[1] = 1;
            adjset_groups["group_000003/neighbors"].as_int_array()[1] = 3;
            adjset_groups["group_000004/neighbors"].as_int_array()[1] = 4;
            adjset_groups["group_000005/neighbors"].as_int_array()[1] = 5;
            Node &windows_node0 = adjset_groups["group_000000/windows"];
            for(index_t w = 0; w <= 2; w += 2)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node0[window_name]["origin/i"] = 20;
                windows_node0[window_name]["origin/j"] = 0;
                windows_node0[window_name]["dims/i"] = 1;
                windows_node0[window_name]["dims/j"] = 21;
                windows_node0[window_name]["ratio/i"] = 1;
                windows_node0[window_name]["ratio/j"] = 1;
            }
            Node &windows_node1 = adjset_groups["group_000001/windows"];
            for(index_t w = 1; w <= 2; w++)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node1[window_name]["origin/i"] = 20;
                windows_node1[window_name]["origin/j"] = 20;
                windows_node1[window_name]["dims/i"] = 1;
                windows_node1[window_name]["dims/j"] = 1;
                windows_node1[window_name]["ratio/i"] = 1;
                windows_node1[window_name]["ratio/j"] = 1;
            }
            Node &windows_node3 = adjset_groups["group_000003/windows"];
            for(index_t w = 2; w <= 3; w++)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node3[window_name]["origin/i"] = 20;
                windows_node3[window_name]["origin/j"] = 20;
                windows_node3[window_name]["dims/i"] = 21;
                windows_node3[window_name]["dims/j"] = 1;
                windows_node3[window_name]["ratio/i"] = 1;
                windows_node3[window_name]["ratio/j"] = 1;
            }
            Node &windows_node4 = adjset_groups["group_000004/windows"];
            for(index_t w = 2; w <= 4; w += 2)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node4[window_name]["origin/i"] = 40;
                windows_node4[window_name]["origin/j"] = 0;
                windows_node4[window_name]["dims/i"] = 1;
                windows_node4[window_name]["dims/j"] = 21;
                windows_node4[window_name]["ratio/i"] = 1;
                windows_node4[window_name]["ratio/j"] = 1;
            }
            Node &windows_node5 = adjset_groups["group_000005/windows"];
            for(index_t w = 2; w <= 5; w += 3)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node5[window_name]["origin/i"] = 40;
                windows_node5[window_name]["origin/j"] = 20;
                windows_node5[window_name]["dims/i"] = 1;
                windows_node5[window_name]["dims/j"] = 1;
                windows_node5[window_name]["ratio/i"] = 1;
                windows_node5[window_name]["ratio/j"] = 1;
            }
        }
        else if (i == 3)
        {
            adjset_groups["group_000000/neighbors"] = DataType::int32(2);
            adjset_groups["group_000001/neighbors"] = DataType::int32(2);
            adjset_groups["group_000002/neighbors"] = DataType::int32(2);
            adjset_groups["group_000004/neighbors"] = DataType::int32(2);
            adjset_groups["group_000005/neighbors"] = DataType::int32(2);
            adjset_groups["group_000000/neighbors"].as_int_array()[0] = 3;
            adjset_groups["group_000001/neighbors"].as_int_array()[0] = 3;
            adjset_groups["group_000002/neighbors"].as_int_array()[0] = 3;
            adjset_groups["group_000004/neighbors"].as_int_array()[0] = 3;
            adjset_groups["group_000005/neighbors"].as_int_array()[0] = 3;
            adjset_groups["group_000000/neighbors"].as_int_array()[1] = 0;
            adjset_groups["group_000001/neighbors"].as_int_array()[1] = 1;
            adjset_groups["group_000002/neighbors"].as_int_array()[1] = 2;
            adjset_groups["group_000004/neighbors"].as_int_array()[1] = 4;
            adjset_groups["group_000005/neighbors"].as_int_array()[1] = 5;
            Node &windows_node0 = adjset_groups["group_000000/windows"];
            for(index_t w = 0; w <= 3; w += 3)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node0[window_name]["origin/i"] = 20;
                windows_node0[window_name]["origin/j"] = 20;
                windows_node0[window_name]["dims/i"] = 1;
                windows_node0[window_name]["dims/j"] = 1;
                windows_node0[window_name]["ratio/i"] = 1;
                windows_node0[window_name]["ratio/j"] = 1;
            }
            Node &windows_node1 = adjset_groups["group_000001/windows"];
            for(index_t w = 1; w <= 3; w += 2)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node1[window_name]["origin/i"] = 20;
                windows_node1[window_name]["origin/j"] = 20;
                windows_node1[window_name]["dims/i"] = 1;
                windows_node1[window_name]["dims/j"] = 21;
                windows_node1[window_name]["ratio/i"] = 1;
                windows_node1[window_name]["ratio/j"] = 1;
            }
            Node &windows_node2 = adjset_groups["group_000002/windows"];
            for(index_t w = 2; w <= 3; w++)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node2[window_name]["origin/i"] = 20;
                windows_node2[window_name]["origin/j"] = 20;
                windows_node2[window_name]["dims/i"] = 21;
                windows_node2[window_name]["dims/j"] = 1;
                windows_node2[window_name]["ratio/i"] = 1;
                windows_node2[window_name]["ratio/j"] = 1;
            }
            Node &windows_node4 = adjset_groups["group_000004/windows"];
            for(index_t w = 3; w <= 4; w++)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node4[window_name]["origin/i"] = 40;
                windows_node4[window_name]["origin/j"] = 20;
                windows_node4[window_name]["dims/i"] = 1;
                windows_node4[window_name]["dims/j"] = 1;
                windows_node4[window_name]["ratio/i"] = 1;
                windows_node4[window_name]["ratio/j"] = 1;
            }
            Node &windows_node5 = adjset_groups["group_000005/windows"];
            for(index_t w = 3; w <= 4; w += 2)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node5[window_name]["origin/i"] = 40;
                windows_node5[window_name]["origin/j"] = 20;
                windows_node5[window_name]["dims/i"] = 1;
                windows_node5[window_name]["dims/j"] = 21;
                windows_node5[window_name]["ratio/i"] = 1;
                windows_node5[window_name]["ratio/j"] = 1;
            }
        }
        else if (i == 4)
        {
            adjset_groups["group_000002/neighbors"] = DataType::int32(2);
            adjset_groups["group_000003/neighbors"] = DataType::int32(2);
            adjset_groups["group_000005/neighbors"] = DataType::int32(2);
            adjset_groups["group_000006/neighbors"] = DataType::int32(2);
            adjset_groups["group_000007/neighbors"] = DataType::int32(2);
            adjset_groups["group_000002/neighbors"].as_int_array()[0] = 4;
            adjset_groups["group_000003/neighbors"].as_int_array()[0] = 4;
            adjset_groups["group_000005/neighbors"].as_int_array()[0] = 4;
            adjset_groups["group_000006/neighbors"].as_int_array()[0] = 4;
            adjset_groups["group_000007/neighbors"].as_int_array()[0] = 4;
            adjset_groups["group_000002/neighbors"].as_int_array()[1] = 2;
            adjset_groups["group_000003/neighbors"].as_int_array()[1] = 3;
            adjset_groups["group_000005/neighbors"].as_int_array()[1] = 5;
            adjset_groups["group_000006/neighbors"].as_int_array()[1] = 6;
            adjset_groups["group_000007/neighbors"].as_int_array()[1] = 7;
            Node &windows_node2 = adjset_groups["group_000002/windows"];
            for(index_t w = 2; w <= 4; w += 2)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node2[window_name]["origin/i"] = 40;
                windows_node2[window_name]["origin/j"] = 0;
                windows_node2[window_name]["dims/i"] = 1;
                windows_node2[window_name]["dims/j"] = 21;
                windows_node2[window_name]["ratio/i"] = 1;
                windows_node2[window_name]["ratio/j"] = 1;
            }
            Node &windows_node3 = adjset_groups["group_000003/windows"];
            for(index_t w = 3; w <= 4; w++)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node3[window_name]["origin/i"] = 40;
                windows_node3[window_name]["origin/j"] = 20;
                windows_node3[window_name]["dims/i"] = 1;
                windows_node3[window_name]["dims/j"] = 1;
                windows_node3[window_name]["ratio/i"] = 1;
                windows_node3[window_name]["ratio/j"] = 1;
            }
            Node &windows_node5 = adjset_groups["group_000005/windows"];
            for(index_t w = 4; w <= 5; w++)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node5[window_name]["origin/i"] = 40;
                windows_node5[window_name]["origin/j"] = 20;
                windows_node5[window_name]["dims/i"] = 21;
                windows_node5[window_name]["dims/j"] = 1;
                windows_node5[window_name]["ratio/i"] = 1;
                windows_node5[window_name]["ratio/j"] = 1;
            }
            Node &windows_node6 = adjset_groups["group_000006/windows"];
            for(index_t w = 4; w <= 6; w += 2)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node6[window_name]["origin/i"] = 60;
                windows_node6[window_name]["origin/j"] = 0;
                windows_node6[window_name]["dims/i"] = 1;
                windows_node6[window_name]["dims/j"] = 21;
                windows_node6[window_name]["ratio/i"] = 1;
                windows_node6[window_name]["ratio/j"] = 1;
            }
            Node &windows_node7 = adjset_groups["group_000007/windows"];
            for(index_t w = 4; w <= 7; w += 3)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node7[window_name]["origin/i"] = 60;
                windows_node7[window_name]["origin/j"] = 20;
                windows_node7[window_name]["dims/i"] = 1;
                windows_node7[window_name]["dims/j"] = 1;
                windows_node7[window_name]["ratio/i"] = 1;
                windows_node7[window_name]["ratio/j"] = 1;
            }
        }
        else if (i == 5)
        {
            adjset_groups["group_000002/neighbors"] = DataType::int32(2);
            adjset_groups["group_000003/neighbors"] = DataType::int32(2);
            adjset_groups["group_000004/neighbors"] = DataType::int32(2);
            adjset_groups["group_000006/neighbors"] = DataType::int32(2);
            adjset_groups["group_000007/neighbors"] = DataType::int32(2);
            adjset_groups["group_000002/neighbors"].as_int_array()[0] = 5;
            adjset_groups["group_000003/neighbors"].as_int_array()[0] = 5;
            adjset_groups["group_000004/neighbors"].as_int_array()[0] = 5;
            adjset_groups["group_000006/neighbors"].as_int_array()[0] = 5;
            adjset_groups["group_000007/neighbors"].as_int_array()[0] = 5;
            adjset_groups["group_000002/neighbors"].as_int_array()[1] = 2;
            adjset_groups["group_000003/neighbors"].as_int_array()[1] = 3;
            adjset_groups["group_000004/neighbors"].as_int_array()[1] = 4;
            adjset_groups["group_000006/neighbors"].as_int_array()[1] = 6;
            adjset_groups["group_000007/neighbors"].as_int_array()[1] = 7;
            Node &windows_node2 = adjset_groups["group_000002/windows"];
            for(index_t w = 2; w <= 5; w += 3)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node2[window_name]["origin/i"] = 40;
                windows_node2[window_name]["origin/j"] = 20;
                windows_node2[window_name]["dims/i"] = 1;
                windows_node2[window_name]["dims/j"] = 1;
                windows_node2[window_name]["ratio/i"] = 1;
                windows_node2[window_name]["ratio/j"] = 1;
            }
            Node &windows_node3 = adjset_groups["group_000003/windows"];
            for(index_t w = 3; w <= 5; w += 2)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node3[window_name]["origin/i"] = 40;
                windows_node3[window_name]["origin/j"] = 20;
                windows_node3[window_name]["dims/i"] = 1;
                windows_node3[window_name]["dims/j"] = 21;
                windows_node3[window_name]["ratio/i"] = 1;
                windows_node3[window_name]["ratio/j"] = 1;
            }
            Node &windows_node4 = adjset_groups["group_000004/windows"];
            for(index_t w = 4; w <= 5; w++)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node4[window_name]["origin/i"] = 40;
                windows_node4[window_name]["origin/j"] = 20;
                windows_node4[window_name]["dims/i"] = 1;
                windows_node4[window_name]["dims/j"] = 21;
                windows_node4[window_name]["ratio/i"] = 1;
                windows_node4[window_name]["ratio/j"] = 1;
            }
            Node &windows_node6 = adjset_groups["group_000006/windows"];
            for(index_t w = 5; w <= 6; w++)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node6[window_name]["origin/i"] = 60;
                windows_node6[window_name]["origin/j"] = 20;
                windows_node6[window_name]["dims/i"] = 1;
                windows_node6[window_name]["dims/j"] = 1;
                windows_node6[window_name]["ratio/i"] = 1;
                windows_node6[window_name]["ratio/j"] = 1;
            }
            Node &windows_node7 = adjset_groups["group_000007/windows"];
            for(index_t w = 5; w <= 7; w++)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node7[window_name]["origin/i"] = 60;
                windows_node7[window_name]["origin/j"] = 20;
                windows_node7[window_name]["dims/i"] = 1;
                windows_node7[window_name]["dims/j"] = 21;
                windows_node7[window_name]["ratio/i"] = 1;
                windows_node7[window_name]["ratio/j"] = 1;
            }
        }
        else if (i == 6)
        {
            adjset_groups["group_000004/neighbors"] = DataType::int32(2);
            adjset_groups["group_000005/neighbors"] = DataType::int32(2);
            adjset_groups["group_000007/neighbors"] = DataType::int32(2);
            adjset_groups["group_000004/neighbors"].as_int_array()[0] = 6;
            adjset_groups["group_000005/neighbors"].as_int_array()[0] = 6;
            adjset_groups["group_000007/neighbors"].as_int_array()[0] = 6;
            adjset_groups["group_000004/neighbors"].as_int_array()[1] = 4;
            adjset_groups["group_000005/neighbors"].as_int_array()[1] = 5;
            adjset_groups["group_000007/neighbors"].as_int_array()[1] = 7;
            Node &windows_node4 = adjset_groups["group_000004/windows"];
            for(index_t w = 4; w <= 6; w += 2)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node4[window_name]["origin/i"] = 60;
                windows_node4[window_name]["origin/j"] = 0;
                windows_node4[window_name]["dims/i"] = 1;
                windows_node4[window_name]["dims/j"] = 21;
                windows_node4[window_name]["ratio/i"] = 1;
                windows_node4[window_name]["ratio/j"] = 1;
            }
            Node &windows_node5 = adjset_groups["group_000005/windows"];
            for(index_t w = 5; w <= 6; w++)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node5[window_name]["origin/i"] = 60;
                windows_node5[window_name]["origin/j"] = 20;
                windows_node5[window_name]["dims/i"] = 1;
                windows_node5[window_name]["dims/j"] = 1;
                windows_node5[window_name]["ratio/i"] = 1;
                windows_node5[window_name]["ratio/j"] = 1;
            }
            Node &windows_node7 = adjset_groups["group_000007/windows"];
            for(index_t w = 6; w <= 7; w++)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node7[window_name]["origin/i"] = 60;
                windows_node7[window_name]["origin/j"] = 20;
                windows_node7[window_name]["dims/i"] = 21;
                windows_node7[window_name]["dims/j"] = 1;
                windows_node7[window_name]["ratio/i"] = 1;
                windows_node7[window_name]["ratio/j"] = 1;
            }
        }
        else if (i == 7)
        {
            adjset_groups["group_000004/neighbors"] = DataType::int32(2);
            adjset_groups["group_000005/neighbors"] = DataType::int32(2);
            adjset_groups["group_000006/neighbors"] = DataType::int32(2);
            adjset_groups["group_000004/neighbors"].as_int_array()[0] = 7;
            adjset_groups["group_000005/neighbors"].as_int_array()[0] = 7;
            adjset_groups["group_000006/neighbors"].as_int_array()[0] = 7;
            adjset_groups["group_000004/neighbors"].as_int_array()[1] = 4;
            adjset_groups["group_000005/neighbors"].as_int_array()[1] = 5;
            adjset_groups["group_000006/neighbors"].as_int_array()[1] = 6;
            Node &windows_node4 = adjset_groups["group_000004/windows"];
            for(index_t w = 4; w <= 7; w += 3)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node4[window_name]["origin/i"] = 60;
                windows_node4[window_name]["origin/j"] = 20;
                windows_node4[window_name]["dims/i"] = 1;
                windows_node4[window_name]["dims/j"] = 1;
                windows_node4[window_name]["ratio/i"] = 1;
                windows_node4[window_name]["ratio/j"] = 1;
            }
            Node &windows_node5 = adjset_groups["group_000005/windows"];
            for(index_t w = 5; w <= 7; w += 2)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node5[window_name]["origin/i"] = 60;
                windows_node5[window_name]["origin/j"] = 20;
                windows_node5[window_name]["dims/i"] = 1;
                windows_node5[window_name]["dims/j"] = 21;
                windows_node5[window_name]["ratio/i"] = 1;
                windows_node5[window_name]["ratio/j"] = 1;
            }
            Node &windows_node6 = adjset_groups["group_000006/windows"];
            for(index_t w = 6; w <= 7; w++)
            {
                std::ostringstream w_oss;
                w_oss << "window_00000" << w;
                const std::string window_name = w_oss.str();
                windows_node6[window_name]["origin/i"] = 60;
                windows_node6[window_name]["origin/j"] = 20;
                windows_node6[window_name]["dims/i"] = 21;
                windows_node6[window_name]["dims/j"] = 1;
                windows_node6[window_name]["ratio/i"] = 1;
                windows_node6[window_name]["ratio/j"] = 1;
            }
        }
    }
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
