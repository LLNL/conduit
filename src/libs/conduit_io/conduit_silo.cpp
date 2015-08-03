//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2015, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see: http://cyrush.github.io/conduit.
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
/// file: conduit_silo.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_silo.hpp"

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
// -- begin conduit::io --
//-----------------------------------------------------------------------------
namespace io
{


//---------------------------------------------------------------------------//
void 
silo_save(const  Node &node,
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

    silo_save(node,file_path,silo_obj_base);
}

//---------------------------------------------------------------------------//
void
silo_load(const std::string &path,
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

    silo_load(file_path,silo_obj_base,node);
}


//---------------------------------------------------------------------------//
void silo_save(const  Node &node,
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
        silo_save(node,dbfile,silo_obj_path);
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
void silo_load(const std::string &file_path,
               const std::string &silo_obj_path,
               Node &n)
{
    DBfile *dbfile = DBOpen(file_path.c_str(), DB_HDF5, DB_READ);

    if(dbfile)
    {
        silo_load(dbfile,silo_obj_path,n);
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
void silo_save(const  Node &node,
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
void CONDUIT_IO_API silo_load(DBfile *dbfile,
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
silo_save_field(DBfile *dbfile, 
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
        int num_elems = n_mesh_info[topo_name]["num_elems"].value();
        int num_pts   = n_mesh_info[topo_name]["num_pts"].value();;


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
            CONDUIT_ERROR( "field " 
                            << var_name 
                            << "'s type not implemented, found " 
                            << dtype.name());
        }

        int silo_error = 0;
    
        if(mesh_type == "quads" || mesh_type == "tris")
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
        else if(mesh_type == "rectilinear" || mesh_type == "uniform")
        {
            int dims[3] = {0,0,0};
            int num_dims = 2;
            dims[0] = n_mesh_info[topo_name]["dims/i"].value();
            dims[1] = n_mesh_info[topo_name]["dims/j"].value();

            if(n_mesh_info[topo_name]["dims"].has_path("k"))
            {
                num_dims = 3;
                dims[2] = n_mesh_info[topo_name]["dims/k"].value();
            }

            if(centering == DB_ZONECENT)
            {
                dims[0]--;
                dims[1]--;
                if(num_dims ==3)
                {
                    dims[2]--;
                }
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
silo_save_ucd_zonelist(DBfile *dbfile, 
                       const std::string &topo_name,
                       Node &n_topo,
                       Node &n_mesh_info)
{
    // zoo will have to be a different path
    int shapetype[1] = {0};
    int shapesize[1] = {0};
    int shapecnt[1]  = {0};
    int conn_len  = 0;
    int *conn_ptr = NULL;
    int num_elems = 0;
    
    std::string topo_type = n_topo["type"].as_string();
    std::string coordset_name = n_topo["coordset"].as_string();
    
    Node &n_mesh_conn = n_topo["connectivity"];

    // convert to compact ints ... 
    Node n_conn;
    n_mesh_conn.compact_to(n_conn);

    if( topo_type == "quads")
    {
        num_elems    = n_conn.dtype().number_of_elements() / 4;
        shapetype[0] = DB_ZONETYPE_QUAD;
        shapesize[0] = 4;
        shapecnt[0]  = num_elems;
        conn_len     = n_conn.dtype().number_of_elements();
        conn_ptr     = n_conn.value();

    }
    else  if( topo_type == "tris")
    {
        num_elems    = n_conn.dtype().number_of_elements() / 3;
        shapetype[0] = DB_ZONETYPE_TRIANGLE;
        shapesize[0] = 3;
        shapecnt[0]  = num_elems;
        conn_len     = n_conn.dtype().number_of_elements();
        conn_ptr     = n_conn.value();

    }

    n_mesh_info[topo_name]["num_elems"].set(num_elems);

    std::string zlist_name = topo_name + "_connectivity";

    int silo_error = DBPutZonelist2(dbfile,  // silo file
                                    zlist_name.c_str() ,  // silo obj name
                                    num_elems,  // number of elements
                                    2,  // spatial dims
                                    conn_ptr,  // connectivity array 
                                    conn_len, // len of connectivity array
                                    0,  // base offset
                                    0,  // # ghosts low
                                    0,  // # ghosts high
                                    shapetype, // list of shapes ids
                                    shapesize, // number of points per shape id
                                    shapecnt,  // number of elements each shape id is used for
                                    1,  // number of shapes ids
                                    NULL); // optlist

    CONDUIT_CHECK_SILO_ERROR(silo_error,
                             " after saving ucd quad topology");

}

// we don't need this info .. 
//---------------------------------------------------------------------------//
void
mesh_topology_basics(const std::string &topo_name,
                     Node &n_topo,
                     Node &n_mesh_info)
{
}


//---------------------------------------------------------------------------//
void 
silo_save_ucd_mesh(DBfile *dbfile, 
                   const std::string &topo_name,
                   Node &n_coords,
                   DBoptlist *state_optlist,
                   Node &n_mesh_info)
{
    // also support interleaved:
    // xy, xyz 
    // convert these to separate coord arrays for silo 

    // check if we are 2d or 3d
    int num_coords = 2;    
    if(!n_coords.has_path("x"))
    {
        CONDUIT_ERROR( "mesh coordset missing: x");
    }
    if(!n_coords.has_path("y"))
    {
        CONDUIT_ERROR( "mesh coordset missing: y");
    }
    if(n_coords.has_path("z"))
    {
        num_coords = 3;
    }

    const char* coordnames[3] = {"x", "y", "z"};

    Node n_coords_compact;
    // compaction is necessary to support ragged arrays
    n_coords.compact_to(n_coords_compact);

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
silo_save_quad_rect_mesh(DBfile *dbfile, 
                        const std::string &topo_name,
                        Node &n_coords,
                        DBoptlist *state_optlist,
                        Node &n_mesh_info)
{
    // TODO: also support interleaved:
    // xy, xyz 
    // convert these to separate coord arrays for silo 
    Node &n_coords_rect = n_coords;

    // check if we are 2d or 3d
    int num_coords = 2;    
    if(!n_coords_rect.has_path("x"))
    {
        CONDUIT_ERROR( "mesh coordset missing: x");
    }
    if(!n_coords_rect.has_path("y"))
    {
        CONDUIT_ERROR( "mesh coordset missing: y");
    }
    if(n_coords_rect.has_path("z"))
    {
        num_coords = 3;
    }

    const char* coordnames[3] = {"x", "y", "z"};


    Node n_coords_compact;
    // compaction is necessary to support ragged arrays
    n_coords_rect.compact_to(n_coords_compact);


    int dims[3];
    dims[0] = n_coords_compact["x"].dtype().number_of_elements();
    dims[1] = n_coords_compact["y"].dtype().number_of_elements();
    dims[2] = 0;
    
    int num_pts = dims[0]* dims[1];
    int num_elems = (dims[0]-1)*(dims[1]-1);
    if(num_coords == 3)
    {
        num_pts   = num_pts * dims[2];
        num_elems = num_elems * (dims[2]-1);
    }
    
    n_mesh_info[topo_name]["num_pts"].set(num_pts);
    n_mesh_info[topo_name]["num_elems"].set(num_elems);
    n_mesh_info[topo_name]["dims/i"] = dims[0];
    n_mesh_info[topo_name]["dims/j"] = dims[1];
    if(num_coords == 3)
    {
        n_mesh_info[topo_name]["dims/k"] = dims[2];
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
                                   dims, //dims vals
                                   num_coords, // number of dims
                                   coords_dtype, // type of data array
                                   DB_COLLINEAR, // DB_COLLINEAR or DB_NONCOLLINEAR
                                   state_optlist); // opt list

    CONDUIT_CHECK_SILO_ERROR(silo_error,
                             " DBPutUcdmesh");

}


//---------------------------------------------------------------------------//
void 
silo_save_mesh(Node &n,
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
        if(topo_type == "quads" || topo_type == "tris")
        {
            silo_save_ucd_zonelist(dbfile,
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
    
        if(topo_type == "quads" || topo_type == "tris")
        {
            silo_save_ucd_mesh(dbfile,
                               topo_name,
                               n_coords,
                               state_optlist,
                               n_mesh_info);
        }
        else if (topo_type == "rectilinear")
        {
            silo_save_quad_rect_mesh(dbfile,
                                     topo_name,
                                     n_coords,
                                     state_optlist,
                                     n_mesh_info);
        }
        else if (topo_type == "uniform")
        {
            //silo_save_quad_uniform_mesh(dbfile,...)
        }
        else if (topo_type == "structured")
        {
            //silo_save_quad_structured_mesh(dbfile,...)
        }
        else if (topo_type == "point")
        {
            //silo_save_point_mesh(dbfile,...)
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
            silo_save_field(dbfile,
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
silo_save_mesh(Node &node,
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

    silo_save_mesh(node,file_path,silo_obj_base);
}


//---------------------------------------------------------------------------//
void silo_save_mesh(Node &node,
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
        silo_save_mesh(node,dbfile,silo_obj_path);
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


};
//-----------------------------------------------------------------------------
// -- end conduit::io --
//-----------------------------------------------------------------------------


};
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
