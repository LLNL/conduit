//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
/// file: relay_silo.cpp
///
//-----------------------------------------------------------------------------

#include "relay_silo.hpp"

//-----------------------------------------------------------------------------
// standard lib includes
//-----------------------------------------------------------------------------
#include <iostream>

//-----------------------------------------------------------------------------
// external lib includes
//-----------------------------------------------------------------------------
#include <silo.h>


//-----------------------------------------------------------------------------
//
/// The CONDUIT_CHECK_SILO_ERROR macro is used to check error codes from silo.
//-----------------------------------------------------------------------------
#define CONDUIT_CHECK_SILO_ERROR( silo_err, msg )                   \
{                                                                   \
    if( silo_err != 0)                                              \
    {                                                               \
        std::ostringstream silo_err_oss;                            \
        silo_err_oss << "Silo Error code"                           \
            << silo_err                                             \
            << " " << msg;                                          \
        CONDUIT_ERROR( silo_err_oss.str());                         \
    }                                                               \
}                                                                   \


//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::relay --
//-----------------------------------------------------------------------------
namespace relay
{

//-----------------------------------------------------------------------------
// -- begin conduit::relay:io --
//-----------------------------------------------------------------------------
namespace io
{

//---------------------------------------------------------------------------//
void 
silo_write(const  Node &node,
           const std::string &path)
{
    // check for ":" split
    std::string file_path;
    std::string silo_obj_base;
    conduit::utils::split_string(path,
                                 std::string(":"),
                                 file_path,
                                 silo_obj_base);

    /// If silo_obj_base is empty, we have a problem ... 
    if(silo_obj_base.size() == 0)
    {
        CONDUIT_ERROR("Invalid path for save: " << path);
    }

    silo_write(node,file_path,silo_obj_base);
}

//---------------------------------------------------------------------------//
void
silo_read(const std::string &path,
          Node &node)
{
    // check for ":" split    
    std::string file_path;
    std::string silo_obj_base;
    conduit::utils::split_string(path,
                                 std::string(":"),
                                 file_path,
                                 silo_obj_base);

    /// If silo_obj_base is empty, we have a problem ... 
    if(silo_obj_base.size() == 0)
    {
        CONDUIT_ERROR("Invalid path for load: " << path);
    }

    silo_read(file_path,silo_obj_base,node);
}


//---------------------------------------------------------------------------//
void silo_write(const  Node &node,
                const std::string &file_path,
                const std::string &silo_obj_path)
{
    DBfile *dbfile = DBCreate(file_path.c_str(),
                              DB_CLOBBER,
                              DB_LOCAL,
                              NULL,
                              DB_HDF5);

    if(dbfile)
    {
        silo_write(node,dbfile,silo_obj_path);
    }
    else 
    {
        CONDUIT_ERROR("Error opening Silo file for writing: " << file_path );
        return;
    }
    
    if(DBClose(dbfile) != 0)
    {
        CONDUIT_ERROR("Error closing Silo file: " << file_path);
    }
}

//---------------------------------------------------------------------------//
void silo_read(const std::string &file_path,
               const std::string &silo_obj_path,
               Node &n)
{
    DBfile *dbfile = DBOpen(file_path.c_str(), DB_HDF5, DB_READ);

    if(dbfile)
    {
        silo_read(dbfile,silo_obj_path,n);
    }
    else 
    {
        CONDUIT_ERROR("Error opening Silo file for reading: " << file_path );
    }
    
    if(DBClose(dbfile) != 0)
    {
        CONDUIT_ERROR("Error closing Silo file: " << file_path );
    }
}


//---------------------------------------------------------------------------//
void silo_write(const  Node &node,
                DBfile *dbfile,
                const std::string &silo_obj_path)
{
    Schema schema_c;
    node.schema().compact_to(schema_c);
    std::string schema = schema_c.to_json();
    int schema_len = schema.length() + 1;

    std::vector<uint8> data;
    node.serialize(data);
    int data_len = data.size();

    // use path to construct dest silo obj paths
    
    std::string dest_json = silo_obj_path +  "_conduit_json";
    std::string dest_data = silo_obj_path +  "_conduit_bin";

    int silo_error = 0;
    silo_error += DBWrite(dbfile,
                          dest_json.c_str(),
                          schema.c_str(),
                          &schema_len,
                          1,
                          DB_CHAR);
    silo_error += DBWrite(dbfile,
                          dest_data.c_str(),
                          &data[0],
                          &data_len,
                          1,
                          DB_CHAR);

    CONDUIT_CHECK_SILO_ERROR(silo_error,
                             "Error writing conduit Node to Silo file");
}


//---------------------------------------------------------------------------//
void silo_read(DBfile *dbfile,
               const std::string &silo_obj_path,
               Node &node)
{
    std::string src_json = silo_obj_path +  "_conduit_json";
    std::string src_data = silo_obj_path +  "_conduit_bin";

    int schema_len = DBGetVarLength(dbfile, src_json.c_str());
    int data_len   = DBGetVarLength(dbfile, src_data.c_str());

    char *schema = new char[schema_len];
    char *data   = new char[data_len];



    DBReadVar(dbfile, src_json.c_str(), schema);
    DBReadVar(dbfile, src_data.c_str(), data);

    if (schema == NULL || data == NULL) 
    {
        CONDUIT_ERROR("Error extracting data conduit Node from Silo file");
    }

    Generator node_gen(schema, data);
    /// gen copy 
    node_gen.walk(node);
    
    delete [] schema;
    delete [] data;
}


//---------------------------------------------------------------------------//
DBoptlist * 
silo_generate_state_optlist(Node &n)
{
    DBoptlist *res = NULL;
    
    if (n.has_path("state"))
    {
        int silo_error = 0;
        Node &n_state = n["state"];
        res = DBMakeOptlist(2);
        
        if(n.has_path("cycle"))
        {
            int cyc_value = n_state["cycle"].to_int();
            silo_error += DBAddOption(res,
                                      DBOPT_CYCLE,
                                      &cyc_value);
        }
        
        if(n.has_path("time"))
        {
            double time_value =  n_state["time"].to_double();
            silo_error += DBAddOption(res,
                                      DBOPT_DTIME,
                                      &time_value);
        }

        CONDUIT_CHECK_SILO_ERROR(silo_error,
                                 " creating state optlist (time, cycle) ");
    }

    return res;
}


//---------------------------------------------------------------------------//
void 
silo_write_field(DBfile *dbfile, 
                 const std::string &var_name,
                 Node &n_var,
                 Node &n_mesh_info)
{


    if (!n_var.has_path("topologies"))
    {
        CONDUIT_ERROR( "Missing linked topology! " 
                        << "fields/"
                        << var_name 
                        << "/topologies");
    }

    NodeIterator fld_topos_itr = n_var["topologies"].children();

    while(fld_topos_itr.has_next())
    {
        std::string topo_name = fld_topos_itr.next().as_string();

        if(!n_mesh_info.has_path(topo_name))
        {
            CONDUIT_ERROR( "Invalid linked topology! " 
                            << "fields/" 
                            << var_name << "/topologies: "
                            << topo_name);
        }

        std::string mesh_type = n_mesh_info[topo_name]["type"].as_string();
        int num_elems = n_mesh_info[topo_name]["num_elems"].to_value();
        int num_pts   = n_mesh_info[topo_name]["num_pts"].to_value();;


        int centering  = 0;
        int num_values = 0;

        if (!n_var.has_path("association"))
        {
            CONDUIT_ERROR( "Missing association! "
                           << "fields/" 
                           << var_name << "/association");
        }

        if (n_var["association"].as_string() == "element")
        {
            centering  = DB_ZONECENT;
            num_values = num_elems;
        }

        if (n_var["association"].as_string() == "point")
        {
            centering  = DB_NODECENT;
            num_values = num_pts;
        }

        if (!n_var.has_path("values"))
        {
            CONDUIT_ERROR( "Missing field data ! " 
                           << "fields/" 
                           << var_name << "/values");
        }

        // we compact to support a strided array cases
        Node n_values;
        n_var["values"].compact_to(n_values);

        // create a name
        int vals_type = 0;
        void *vals_ptr = NULL;

        DataType dtype = n_var["values"].dtype();

        if( dtype.is_float() )
        {
            vals_type = DB_FLOAT;
            vals_ptr = (void*)n_values.as_float_ptr();
        }
        else  if( dtype.is_double() )

        {
            vals_type = DB_DOUBLE;
            vals_ptr = (void*)n_values.as_double_ptr();
        }
        else
        {
            // skip the field if we don't support its type
            CONDUIT_INFO( "skipping field " 
                           << var_name 
                           << ", since its type is not implemented, found " 
                           << dtype.name() );
            continue;
        }

        int silo_error = 0;
    
        if(mesh_type == "unstructured")
        {
            silo_error = DBPutUcdvar1(dbfile, 
                                      var_name.c_str(), 
                                      topo_name.c_str(), 
                                      vals_ptr,
                                      num_values, 
                                      NULL, 
                                      0, 
                                      vals_type, 
                                      centering,
                                      NULL);
        }
        else if(mesh_type == "rectilinear" ||
                mesh_type == "uniform" ||
                mesh_type == "structured")
        {
            int ele_dims[3] = {0,0,0};
            int pts_dims[3] = {1,1,1};
            
            int num_dims = 2;
            
            ele_dims[0] = n_mesh_info[topo_name]["elements/i"].value();
            ele_dims[1] = n_mesh_info[topo_name]["elements/j"].value();
            
            pts_dims[0] = ele_dims[0] + 1;
            pts_dims[1] = ele_dims[1] + 1;

            if(n_mesh_info[topo_name]["elements"].has_path("k"))
            {
                num_dims = 3;
                ele_dims[2] = n_mesh_info[topo_name]["elements/k"].value();
                pts_dims[2] = ele_dims[2] + 1;
            }
            

            int *dims = ele_dims;
            if(centering == DB_NODECENT)
            {
                dims = pts_dims;
            }

            silo_error = DBPutQuadvar1(dbfile, 
                                       var_name.c_str(), 
                                       topo_name.c_str(), 
                                       vals_ptr,
                                       dims,
                                       num_dims, 
                                       NULL, 
                                       0, 
                                       vals_type, 
                                       centering,
                                       NULL);
        }
        else if( mesh_type == "points")
        {
            
            silo_error = DBPutPointvar1(dbfile,            // dbfile Database file pointer.
                                        var_name.c_str(),  // variable name
                                        topo_name.c_str(), // mesh name 
                                        vals_ptr,          // data values 
                                        num_pts,           // Number of elements (points).
                                        vals_type,         // Datatype of the variable.
                                        NULL);
        }
        else
        {
            CONDUIT_ERROR( "only DBPutQuadvar1 + DBPutUcdvar1 var are supported");
        }

        CONDUIT_CHECK_SILO_ERROR(silo_error,
                                 " after creating field " << var_name);
    }// end field topologies itr while
}

//---------------------------------------------------------------------------//
void
silo_write_pointmesh(DBfile *dbfile, 
                     const std::string &topo_name,
                     Node &n_coords,
                     DBoptlist *state_optlist,
                     Node &n_mesh_info)
{
    // expects explicit coords
    Node &n_coord_vals = n_coords["values"];
    
    int num_dims = 2;    
    if(!n_coord_vals.has_path("x"))
    {
        CONDUIT_ERROR( "mesh coordset missing: x");
    }
    if(!n_coord_vals.has_path("y"))
    {
        CONDUIT_ERROR( "mesh coordset missing: y");
    }
    if(n_coord_vals.has_path("z"))
    {
        num_dims = 3;
    }

    Node n_coords_compact;
    // compaction is necessary to support ragged arrays
    n_coord_vals.compact_to(n_coords_compact);

    int num_pts = n_coords_compact["x"].dtype().number_of_elements();
    
    n_mesh_info[topo_name]["num_pts"].set(num_pts);
    n_mesh_info[topo_name]["num_elems"].set(num_pts);

    int coords_dtype = 0;
    void *coords_ptrs[3] = {NULL, NULL, NULL};

    // assume x,y,z are all the same type
    DataType dtype = n_coords_compact["x"].dtype();

    if( dtype.is_float() )
    {
        coords_dtype = DB_FLOAT;
        coords_ptrs[0] = (void*)n_coords_compact["x"].as_float_ptr();
        coords_ptrs[1] = (void*)n_coords_compact["y"].as_float_ptr();
        if(num_dims == 3)
        {
            coords_ptrs[2] = (void*)n_coords_compact["z"].as_float_ptr();
        }
    }
    else if( dtype.is_double() )

    {
        coords_dtype = DB_DOUBLE;
        coords_ptrs[0] = (void*)n_coords_compact["x"].as_double_ptr();
        coords_ptrs[1] = (void*)n_coords_compact["y"].as_double_ptr();
        if(num_dims == 3)
        {
            coords_ptrs[2] = (void*)n_coords_compact["z"].as_double_ptr();
        }
    }
    else
    {
        // n_coords["x"].to_double_array(n_convert["x"]);
        // n_coords["y"].to_double_array(n_convert["y"]);
        CONDUIT_ERROR("coords data type not implemented, found " << 
                      dtype.name());
    }
    

    int silo_error = DBPutPointmesh(dbfile, // silo file ptr
                                    topo_name.c_str(), // mesh name
                                    num_dims, // num_dims
                                    coords_ptrs, // coords values
                                    num_pts, // num eles = num pts
                                    coords_dtype, // type of data array
                                    state_optlist); // opt list

    CONDUIT_CHECK_SILO_ERROR(silo_error,
                             " after saving DBPutPointmesh");
}

//---------------------------------------------------------------------------//
void 
silo_write_ucd_zonelist(DBfile *dbfile, 
                        const std::string &topo_name,
                        Node &n_topo,
                        Node &n_mesh_info)
{
    Node ucd_zlist;
    
    index_t num_shapes = 0;
    ucd_zlist["shapetype"].set(DataType::c_int(1));
    ucd_zlist["shapesize"].set(DataType::c_int(1));
    ucd_zlist["shapecnt"].set(DataType::c_int(1));
    
    Node &n_elements = n_topo["elements"];
    std::string coordset_name = n_topo["coordset"].as_string();
    
    bool shape_list = true;
    
    if(n_elements.dtype().is_object())
    {
        // simple path case
        num_shapes = 1;
    }
    else if(n_elements.dtype().is_list())
    {
        shape_list = false;
        num_shapes = n_elements.number_of_children();
    }
    else
    {
        CONDUIT_ERROR("Invalid elements for 'unstructured' case");
    }
    
    
    int *shapetype = ucd_zlist["shapetype"].value();
    int *shapesize = ucd_zlist["shapesize"].value();
    int *shapecnt  = ucd_zlist["shapecnt"].value();
    
    int total_conn_len  = 0;
    int total_num_elems = 0;
    
    
    
    Node *shape_block = &n_elements;
    Node n_conn;
    
    for(index_t i=0;i < num_shapes;i++)
    {
        if(shape_list)
        {
            Node *shape_block = n_elements.child_ptr(i);
        }
       
        std::string topo_shape = shape_block->fetch("shape").as_string();

        Node &n_mesh_conn = shape_block->fetch("connectivity");

        // convert to compact ints ... 
        if(shape_list)
        {
            n_mesh_conn.compact_to(n_conn.append());
        }
        else
        {
            n_mesh_conn.compact_to(n_conn);
        }


        if( topo_shape == "quads")
        {
            // TODO: check for explicit # of elems
            int num_elems    = n_mesh_conn.dtype().number_of_elements() / 4;
            shapetype[i] = DB_ZONETYPE_QUAD;
            shapesize[i] = 4;
            shapecnt[i]  = num_elems;
            total_num_elems  += num_elems;

        }
        else  if( topo_shape == "tris")
        {
            // TODO: check for explicit # of elems
            int num_elems  = n_mesh_conn.dtype().number_of_elements() / 3;
            shapetype[i]   = DB_ZONETYPE_TRIANGLE;
            shapesize[i]   = 3;
            shapecnt[i]    = num_elems;
            total_num_elems += num_elems;
        }
        else if( topo_shape == "hexs")
        {
            // TODO: check for explicit # of elems
            int num_elems    = n_mesh_conn.dtype().number_of_elements() / 8;
            shapetype[i] = DB_ZONETYPE_HEX;
            shapesize[i] = 8;
            shapecnt[i]  = num_elems;
            total_num_elems  += num_elems;

        }
        else  if( topo_shape == "tets")
        {
            // TODO: check for explicit # of elems
            int num_elems  = n_mesh_conn.dtype().number_of_elements() / 4;
            shapetype[i]   = DB_ZONETYPE_TET;
            shapesize[i]   = 4;
            shapecnt[i]    = num_elems;
            total_num_elems += num_elems;
        }
        else  if( topo_shape == "lines")
        {
            // TODO: check for explicit # of elems
            int num_elems  = n_mesh_conn.dtype().number_of_elements() / 2;
            shapetype[i]   = DB_ZONETYPE_BEAM;
            shapesize[i]   = 2;
            shapecnt[i]    = num_elems;
            total_num_elems += num_elems;
        }
        
    }
    
    // Final Compaction
    Node n_conn_final;
    n_conn.compact_to(n_conn_final);

    int  conn_len = n_conn_final.total_bytes() / sizeof(int);
    int *conn_ptr = (int*) n_conn_final.data_ptr();


    n_mesh_info[topo_name]["num_elems"].set(total_num_elems);

    std::string zlist_name = topo_name + "_connectivity";

    int silo_error = DBPutZonelist2(dbfile,  // silo file
                                    zlist_name.c_str() ,  // silo obj name
                                    total_num_elems,  // number of elements
                                    2,  // spatial dims
                                    conn_ptr,  // connectivity array 
                                    conn_len, // len of connectivity array
                                    0,  // base offset
                                    0,  // # ghosts low
                                    0,  // # ghosts high
                                    shapetype, // list of shapes ids
                                    shapesize, // number of points per shape id
                                    shapecnt,  // number of elements each shape id is used for
                                    num_shapes,  // number of shapes ids
                                    NULL); // optlist

    CONDUIT_CHECK_SILO_ERROR(silo_error,
                             " after saving ucd quad topology");


}


//---------------------------------------------------------------------------//
void 
silo_write_ucd_mesh(DBfile *dbfile, 
                    const std::string &topo_name,
                    Node &n_coords,
                    DBoptlist *state_optlist,
                    Node &n_mesh_info)
{
    // also support interleaved:
    // xy, xyz 
    // convert these to separate coord arrays for silo 
    Node &n_coord_vals = n_coords["values"];

    // check if we are 2d or 3d
    int num_coords = 2;    
    if(!n_coord_vals.has_path("x"))
    {
        CONDUIT_ERROR( "mesh coordset missing: x");
    }
    if(!n_coord_vals.has_path("y"))
    {
        CONDUIT_ERROR( "mesh coordset missing: y");
    }
    if(n_coord_vals.has_path("z"))
    {
        num_coords = 3;
    }

    const char* coordnames[3] = {"x", "y", "z"};

    Node n_coords_compact;
    // compaction is necessary to support ragged arrays
    n_coord_vals.compact_to(n_coords_compact);

    int num_pts = n_coords_compact["x"].dtype().number_of_elements();
    // TODO: check that y & z have the same number of points

    n_mesh_info[topo_name]["num_pts"].set(num_pts);

    int coords_type = 0;
    void *coords_ptrs[3] = {NULL, NULL, NULL};

    // assume x,y,z are all the same type
    DataType dtype = n_coords_compact["x"].dtype();

    if( dtype.is_float() )
    {
        coords_type = DB_FLOAT;
        coords_ptrs[0] = (void*)n_coords_compact["x"].as_float_ptr();
        coords_ptrs[1] = (void*)n_coords_compact["y"].as_float_ptr();
        if(num_coords == 3)
        {
            coords_ptrs[2] = (void*)n_coords_compact["z"].as_float_ptr();
        }
    }
    else if( dtype.is_double() )

    {
        coords_type = DB_DOUBLE;
        coords_ptrs[0] = (void*)n_coords_compact["x"].as_double_ptr();
        coords_ptrs[1] = (void*)n_coords_compact["y"].as_double_ptr();
        if(num_coords == 3)
        {
            coords_ptrs[2] = (void*)n_coords_compact["z"].as_double_ptr();
        }
    }
    else
    {
        // n_coords["x"].to_double_array(n_convert["x"]);
        // n_coords["y"].to_double_array(n_convert["y"]);
        CONDUIT_ERROR("coords data type not implemented, found " << 
                      dtype.name());
    }

    int num_elems = n_mesh_info[topo_name]["num_elems"].value();
    
    std::string zlist_name =  topo_name + "_connectivity";

    int silo_error = DBPutUcdmesh(dbfile, // silo file ptr
                                  topo_name.c_str(), // mesh name
                                  num_coords, // number of dims
                                  (char**)&coordnames[0], // coord names
                                  coords_ptrs, // coords values
                                  num_pts,  // number of points
                                  num_elems,  // number of elements
                                  zlist_name.c_str(), // zone list name
                                  0, // face list names
                                  coords_type, // type of data array
                                  state_optlist); // opt list

    CONDUIT_CHECK_SILO_ERROR(silo_error,
                             " DBPutUcdmesh");


}

//---------------------------------------------------------------------------//
void 
silo_write_quad_rect_mesh(DBfile *dbfile, 
                          const std::string &topo_name,
                          Node &n_coords,
                          DBoptlist *state_optlist,
                          Node &n_mesh_info)
{
    // TODO: also support interleaved:
    // xy, xyz 
    // convert these to separate coord arrays for silo 
    Node &n_coord_vals = n_coords["values"];

    // check if we are 2d or 3d
    int num_coords = 2;    
    if(!n_coord_vals.has_path("x"))
    {
        CONDUIT_ERROR( "mesh coordset missing: x");
    }
    if(!n_coord_vals.has_path("y"))
    {
        CONDUIT_ERROR( "mesh coordset missing: y");
    }
    if(n_coord_vals.has_path("z"))
    {
        num_coords = 3;
    }

    const char* coordnames[3] = {"x", "y", "z"};


    Node n_coords_compact;
    // compaction is necessary to support ragged arrays
    n_coord_vals.compact_to(n_coords_compact);


    int pts_dims[3];
    pts_dims[0] = n_coords_compact["x"].dtype().number_of_elements();
    pts_dims[1] = n_coords_compact["y"].dtype().number_of_elements();
    pts_dims[2] = 1;
    
    int num_pts = pts_dims[0]* pts_dims[1];
    int num_elems = (pts_dims[0]-1)*(pts_dims[1]-1);
    if(num_coords == 3)
    {
        pts_dims[2] = n_coords_compact["z"].dtype().number_of_elements();
        num_pts   = num_pts * pts_dims[2];
        num_elems = num_elems * (pts_dims[2]-1);
    }
    
    n_mesh_info[topo_name]["num_pts"].set(num_pts);
    n_mesh_info[topo_name]["num_elems"].set(num_elems);
    n_mesh_info[topo_name]["elements/i"] = pts_dims[0]-1;
    n_mesh_info[topo_name]["elements/j"] = pts_dims[1]-1;
    if(num_coords == 3)
    {
        n_mesh_info[topo_name]["elements/k"] = pts_dims[2]-1;
    }
    
    int coords_dtype = 0;
    void *coords_ptrs[3] = {NULL, NULL, NULL};

    // assume x,y,z are all the same type
    DataType dtype = n_coords_compact["x"].dtype();

    if( dtype.is_float() )
    {
        coords_dtype = DB_FLOAT;
        coords_ptrs[0] = (void*)n_coords_compact["x"].as_float_ptr();
        coords_ptrs[1] = (void*)n_coords_compact["y"].as_float_ptr();
        if(num_coords == 3)
        {
            coords_ptrs[2] = (void*)n_coords_compact["z"].as_float_ptr();
        }
    }
    else if( dtype.is_double() )
    {
        coords_dtype = DB_DOUBLE;
        coords_ptrs[0] = (void*)n_coords_compact["x"].as_double_ptr();
        coords_ptrs[1] = (void*)n_coords_compact["y"].as_double_ptr();
        if(num_coords == 3)
        {
            coords_ptrs[2] = (void*)n_coords_compact["z"].as_double_ptr();
        }
    }
    else
    {
        // n_coords["x"].to_double_array(n_convert["x"]);
        // n_coords["y"].to_double_array(n_convert["y"]);
        CONDUIT_ERROR("coords data type not implemented, found " << 
                      dtype.name());
    }

    int silo_error = DBPutQuadmesh(dbfile, // silo file ptr
                                   topo_name.c_str(), // mesh name
                                   (char**)&coordnames[0], // coord names
                                   coords_ptrs, // coords values
                                   pts_dims, //dims vals
                                   num_coords, // number of dims
                                   coords_dtype, // type of data array
                                   DB_COLLINEAR, // DB_COLLINEAR or DB_NONCOLLINEAR
                                   state_optlist); // opt list

    CONDUIT_CHECK_SILO_ERROR(silo_error,
                             " DBPutUcdmesh");
}

//---------------------------------------------------------------------------//
void 
silo_write_quad_uniform_mesh(DBfile *dbfile, 
                             const std::string &topo_name,
                             Node &n_coords,
                             DBoptlist *state_optlist,
                             Node &n_mesh_info)
{
    // TODO: USE XFORM expand uniform coords to rect-style

    // silo doesn't have a direct path for a uniform mesh
    // we need to convert its implicit uniform coords to 
    // implicit rectilinear coords 
    
    index_t npts_x = 0;
    index_t npts_y = 0;
    index_t npts_z = 0;
    
    float64 x0 = 0.0;
    float64 y0 = 0.0;
    float64 z0 = 0.0;
    
    float64 dx =1;
    float64 dy =1;
    float64 dz =1;
    
    if(!n_coords.has_path("dims"))
    {
        CONDUIT_ERROR("uniform mesh missing 'dims'")
    }
    
    Node &n_dims = n_coords["dims"];
    
    if( n_dims.has_path("i") )
    {
        npts_x = n_dims["i"].to_value();
    }

    if( n_dims.has_path("j") )
    {
        npts_y = n_dims["j"].to_value();
    }
    
    if( n_dims.has_path("k") )
    {
        npts_z = n_dims["k"].to_value();
    }
    

    if(n_coords.has_path("origin"))
    {
        Node &n_origin = n_coords["origin"];
        
        if( n_origin.has_path("x") )
        {
            x0 = n_origin["x"].to_value();
        }

        if( n_origin.has_path("y") )
        {
            y0 = n_origin["y"].to_value();
        }
        
        if( n_origin.has_path("z") )
        {
            z0 = n_origin["z"].to_value();
        }
    }
    
    if(n_coords.has_path("spacing"))
    {
        Node &n_spacing = n_coords["spacing"];
        
        if( n_spacing.has_path("dx") )
        {
            dx = n_spacing["dx"].to_value();
        }

        if( n_spacing.has_path("dy") )
        {
            dy = n_spacing["dy"].to_value();
        }
        
        if( n_spacing.has_path("dz") )
        {
            dz = n_spacing["dz"].to_value();
        }
    }
    
    Node n_rect_coords;
    

    n_rect_coords["type"] = "rectilinear";
    Node &n_rect_coord_vals = n_rect_coords["values"];
    n_rect_coord_vals["x"].set(DataType::float64(npts_x));
    n_rect_coord_vals["y"].set(DataType::float64(npts_y));
    
    if(npts_z > 1)
    {
        n_rect_coord_vals["z"].set(DataType::float64(npts_z));
    }
    
    
    float64 *x_coords_ptr = n_rect_coord_vals["x"].value();
    float64 *y_coords_ptr = n_rect_coord_vals["y"].value();
    float64 *z_coords_ptr = NULL;
    
    if(npts_z > 1)
    {
        z_coords_ptr = n_rect_coord_vals["z"].value();
    }
    
    float64 cv = x0;
    for(index_t i=0; i < npts_x; i++)
    {
        x_coords_ptr[i] = cv;
        cv += dx;
    }
        
    cv = y0;
    for(index_t i=0; i < npts_y; i++)
    {
        y_coords_ptr[i] = cv;
        cv += dy;
    }

    if(npts_z > 1)
    {
        cv = z0;
        for(index_t i=0; i < npts_z; i++)
        {
            z_coords_ptr[i] = cv;
            cv += dz;
        }
    }
        
    
    silo_write_quad_rect_mesh(dbfile,
                              topo_name,
                              n_rect_coords,
                              state_optlist,
                              n_mesh_info);

}


//---------------------------------------------------------------------------//
void 
silo_write_structured_mesh(DBfile *dbfile, 
                           const std::string &topo_name,
                           Node &n_topo,
                           Node &n_coords,
                           DBoptlist *state_optlist,
                           Node &n_mesh_info)
{
    // also support interleaved:
    // xy, xyz 
    // convert these to separate coord arrays for silo 
    Node& n_coords_vals = n_coords["values"];

    // check if we are 2d or 3d
    int num_coords = 2;    
    if(!n_coords_vals.has_path("x"))
    {
        CONDUIT_ERROR( "mesh coordset missing: x");
    }
    if(!n_coords_vals.has_path("y"))
    {
        CONDUIT_ERROR( "mesh coordset missing: y");
    }
    if(n_coords_vals.has_path("z"))
    {
        num_coords = 3;
    }

    // blueprint::transmute(n_coord_vals,,n_coords_compact)

    const char* coordnames[3] = {"x", "y", "z"};

    Node n_coords_compact;
    // compaction is necessary to support ragged arrays
    n_coords_vals.compact_to(n_coords_compact);

    int num_pts = n_coords_compact["x"].dtype().number_of_elements();
    // TODO: check that y & z have the same number of points

    n_mesh_info[topo_name]["num_pts"].set(num_pts);

    int coords_dtype = 0;
    void *coords_ptrs[3] = {NULL, NULL, NULL};

    // assume x,y,z are all the same type
    DataType dtype = n_coords_compact["x"].dtype();

    if( dtype.is_float() )
    {
        coords_dtype = DB_FLOAT;
        coords_ptrs[0] = (void*)n_coords_compact["x"].as_float_ptr();
        coords_ptrs[1] = (void*)n_coords_compact["y"].as_float_ptr();
        if(num_coords == 3)
        {
            coords_ptrs[2] = (void*)n_coords_compact["z"].as_float_ptr();
        }
    }
    else if( dtype.is_double() )

    {
        coords_dtype = DB_DOUBLE;
        coords_ptrs[0] = (void*)n_coords_compact["x"].as_double_ptr();
        coords_ptrs[1] = (void*)n_coords_compact["y"].as_double_ptr();
        if(num_coords == 3)
        {
            coords_ptrs[2] = (void*)n_coords_compact["z"].as_double_ptr();
        }
    }
    else
    {
        // n_coords["x"].to_double_array(n_convert["x"]);
        // n_coords["y"].to_double_array(n_convert["y"]);
        CONDUIT_ERROR("coords data type not implemented, found " << 
                      dtype.name());
    }
    
    int ele_dims[3];
    ele_dims[0] = n_topo["elements/dims/i"].to_value();
    ele_dims[1] = n_topo["elements/dims/j"].to_value();
    ele_dims[2] = 0;

    index_t num_elems =  ele_dims[0]* ele_dims[1];

    if( n_topo["elements/dims"].has_path("k") )
    {
        ele_dims[2] = n_topo["elements/dims/k"].to_value();
        num_elems *= ele_dims[2];
    }

    // silo needs the node dims to define a structured grid
    int pts_dims[3];
    
    pts_dims[0] = ele_dims[0]+1;
    pts_dims[1] = ele_dims[1]+1;
    pts_dims[2] = 1;
     
    n_mesh_info[topo_name]["num_pts"].set(num_pts);
    n_mesh_info[topo_name]["num_elems"].set(num_elems);
    n_mesh_info[topo_name]["elements/i"] = ele_dims[0];
    n_mesh_info[topo_name]["elements/j"] = ele_dims[1];
    
    
    if(num_coords == 3)
    {
        n_mesh_info[topo_name]["elements/k"] = ele_dims[2];
        pts_dims[2] = ele_dims[2]+1;
    }

    
    int silo_error = DBPutQuadmesh(dbfile, // silo file ptr
                                   topo_name.c_str(), // mesh name
                                   (char**)&coordnames[0], // coord names
                                   coords_ptrs, // coords values
                                   pts_dims, //dims vals
                                   num_coords, // number of dims
                                   coords_dtype, // type of data array
                                   DB_NONCOLLINEAR, // DB_COLLINEAR or DB_NONCOLLINEAR
                                   state_optlist); // opt list

    CONDUIT_CHECK_SILO_ERROR(silo_error,
                             " DBPutQuadmesh");
}



//---------------------------------------------------------------------------//
void 
silo_mesh_write(Node &n,
                DBfile *dbfile,
                const std::string &silo_obj_path)
{
    int silo_error = 0;
    char silo_prev_dir[256];
    
    silo_error += DBGetDir(dbfile,silo_prev_dir);
    silo_error += DBMkDir(dbfile,silo_obj_path.c_str());
    silo_error += DBSetDir(dbfile,silo_obj_path.c_str());
    
    CONDUIT_CHECK_SILO_ERROR(silo_error,
                             " failed to make silo directory:"
                             << silo_obj_path);

    DBoptlist *state_optlist = silo_generate_state_optlist(n);
    
    Node n_mesh_info; // helps with bookkeeping for all topos
    
    NodeIterator topo_itr = n["topologies"].children();
    while(topo_itr.has_next())
    {
        Node &n_topo = topo_itr.next();
        
        std::string topo_name = topo_itr.path();
    
        std::string topo_type = n_topo["type"].as_string();

        n_mesh_info[topo_name]["type"].set(topo_type);
        

        
        // we need a zone list for a ucd mesh
        if(topo_type == "unstructured")
        {
            silo_write_ucd_zonelist(dbfile,
                                    topo_name,
                                    n_topo,
                                    n_mesh_info);
        }

        // make sure we have coordsets
        
        if(!n.has_path("coordsets"))
        {
             CONDUIT_ERROR( "mesh missing: coordsets");
        }
        
        // get this topo's coordset name
        std::string coordset_name = n_topo["coordset"].as_string();

        n_mesh_info[topo_name]["coordset"].set(coordset_name);
        
        // obtain the coordset with the name
        if(!n["coordsets"].has_path(coordset_name))
        {
             CONDUIT_ERROR( "mesh is missing coordset named "
                            << coordset_name 
                            << " for topology named "
                            << topo_name ); 
        }
    
        Node &n_coords = n["coordsets"][coordset_name];
    
        if(topo_type == "unstructured")
        {
            silo_write_ucd_mesh(dbfile,
                                topo_name,
                                n_coords,
                                state_optlist,
                                n_mesh_info);
        }
        else if (topo_type == "rectilinear")
        {
            silo_write_quad_rect_mesh(dbfile,
                                      topo_name,
                                      n_coords,
                                      state_optlist,
                                      n_mesh_info);
        }
        else if (topo_type == "uniform")
        {
            silo_write_quad_uniform_mesh(dbfile,
                                         topo_name,
                                         n_coords,
                                         state_optlist,
                                         n_mesh_info);

        }
        else if (topo_type == "structured")
        {
            silo_write_structured_mesh(dbfile,
                                       topo_name,
                                       n_topo,
                                       n_coords,
                                       state_optlist,
                                       n_mesh_info);
        }
        else if (topo_type == "points")
        {
            silo_write_pointmesh(dbfile,
                                 topo_name,
                                 n_coords,
                                 state_optlist,
                                 n_mesh_info);
        }
    }
    
    
    if(state_optlist)
    {
        silo_error = DBFreeOptlist(state_optlist);
    }
    
    CONDUIT_CHECK_SILO_ERROR(silo_error,
                             " freeing state optlist.");

    if (n.has_path("fields")) 
    {
        NodeIterator itr = n["fields"].children();

        while(itr.has_next())
        {
            Node &n_var = itr.next();
            std::string var_name = itr.path();
            silo_write_field(dbfile,
                             var_name,
                             n_var,
                             n_mesh_info);
            
        }
    }

    silo_error = DBSetDir(dbfile,silo_prev_dir);

    CONDUIT_CHECK_SILO_ERROR(silo_error,
                             " changing silo directory to previous path");
}


//---------------------------------------------------------------------------//
void 
silo_mesh_write(Node &node,
                const std::string &path)
{
    // check for ":" split
    std::string file_path;
    std::string silo_obj_base;
    conduit::utils::split_string(path,
                                 std::string(":"),
                                 file_path,
                                 silo_obj_base);

    /// If silo_obj_base is empty, we have a problem ... 
    if(silo_obj_base.size() == 0)
    {
        CONDUIT_ERROR("Invalid path for save: " << path);
    }

    silo_mesh_write(node,file_path,silo_obj_base);
}


//---------------------------------------------------------------------------//
void silo_mesh_write(Node &node,
                     const std::string &file_path,
                     const std::string &silo_obj_path)
{
    DBfile *dbfile = DBCreate(file_path.c_str(),
                              DB_CLOBBER,
                              DB_LOCAL,
                              NULL,
                              DB_HDF5);

    if(dbfile)
    {
        silo_mesh_write(node,dbfile,silo_obj_path);
    }
    else 
    {
        CONDUIT_ERROR("Error opening Silo file for writing: " << file_path );
        return;
    }
    
    if(DBClose(dbfile) != 0)
    {
        CONDUIT_ERROR("Error closing Silo file: " << file_path);
    }
}


}
//-----------------------------------------------------------------------------
// -- end conduit::relay::io --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::relay --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
