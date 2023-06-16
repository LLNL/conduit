// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_relay_io_silo.cpp
///
//-----------------------------------------------------------------------------

#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    #include "conduit_relay_mpi_io_silo.hpp"
#else
    #include "conduit_relay_io_silo.hpp"
#endif

//-----------------------------------------------------------------------------
// standard lib includes
//-----------------------------------------------------------------------------
#include <iostream>
#include <string.h>
#include <memory>
#include <map>

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint.hpp"
#include "conduit_blueprint_mesh_utils.hpp"
#include "conduit_fmt/conduit_fmt.h"

#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    #include "conduit_blueprint_mpi.hpp"
    #include "conduit_relay_mpi.hpp"
    #include "conduit_relay_mpi_io_blueprint.hpp"
#else
    #include "conduit_relay_io_blueprint.hpp"
#endif

//-----------------------------------------------------------------------------
// external lib includes
//-----------------------------------------------------------------------------
#include <silo.h>

#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
// Define an argument macro that adds the communicator argument.
#define CONDUIT_RELAY_COMMUNICATOR_ARG(ARG) ,ARG
#else
// Define an argument macro that does not add the communicator argument.
#define CONDUIT_RELAY_COMMUNICATOR_ARG(ARG) 
#endif

//-----------------------------------------------------------------------------
//
/// The CONDUIT_CHECK_SILO_ERROR macro is used to check error codes from silo.
//-----------------------------------------------------------------------------
#define CONDUIT_CHECK_SILO_ERROR( silo_err, msg )                   \
{                                                                   \
    if( (silo_err) != 0)                                            \
    {                                                               \
        std::ostringstream silo_err_oss;                            \
        silo_err_oss << "Silo Error code "                          \
            << (silo_err) << " " << DBErrString()                   \
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

#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
//-----------------------------------------------------------------------------
// -- begin conduit::relay::mpi --
//-----------------------------------------------------------------------------
namespace mpi
{
#endif

//-----------------------------------------------------------------------------
// -- begin conduit::relay::<mpi>::io --
//-----------------------------------------------------------------------------
namespace io
{

//---------------------------------------------------------------------------//
void
silo_write(const Node &node,
           const std::string &path)
{
    // check for ":" split
    std::string file_path;
    std::string silo_obj_base;
    conduit::utils::split_file_path(path,
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
    conduit::utils::split_file_path(path,
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
void silo_write(const Node &node,
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

    Generator node_gen(schema, "conduit_json", data);
    /// gen copy
    node_gen.walk(node);

    delete [] schema;
    delete [] data;
}


//-----------------------------------------------------------------------------
// -- begin conduit::relay::<mpi>::io::silo --
//-----------------------------------------------------------------------------
namespace silo
{

//-----------------------------------------------------------------------------
// -- begin conduit::relay::<mpi>::io::silo::detail --
//-----------------------------------------------------------------------------
namespace detail
{

template <class T, class Deleter>
class SiloObjectWrapper
{
private:
    T *obj;
    Deleter del;

public:
    SiloObjectWrapper(T *o, Deleter d) : 
        obj(o), del{d} {}
    T* getSiloObject() { return obj; }
    void setSiloObject(T *o) { obj = o; }
    ~SiloObjectWrapper()
    {
        if (obj)
        {
            del(obj);
        }
    }
};

template <class T, class Deleter>
class SiloObjectWrapperCheckError
{
private:
    T *obj;
    Deleter del;
    std::string errmsg = "";

public:
    SiloObjectWrapperCheckError(T *o, Deleter d, std::string err) : 
        obj(o), del{d}, errmsg{err} {}
    SiloObjectWrapperCheckError(T *o, Deleter d) : 
        obj(o), del{d} {}
    T* getSiloObject() { return obj; }
    void setSiloObject(T *o) { obj = o; }
    void setErrMsg(std::string newmsg) { errmsg = newmsg; }
    ~SiloObjectWrapperCheckError()
    {
        if (obj)
        {
            if (del(obj) != 0 && !errmsg.empty())
            {
                CONDUIT_ERROR(errmsg);
            }
        }
    }
};

class SiloTreePathGenerator
{
private:
    bool nameschemes;
    // TODO more work is required to support nameschemes

public:
    SiloTreePathGenerator(bool nameschemes_on) : nameschemes(nameschemes_on) {}
    void GeneratePaths(const std::string &path,
                       const std::string &relative_dir,
                       std::string &file_path,
                       std::string &silo_name)
    {
        conduit::utils::rsplit_file_path(path, ":", silo_name, file_path);
        if (silo_name.length() > 1 && silo_name[0] == '/')
        {
            silo_name = silo_name.substr(1);
        }
        if (!file_path.empty())
        {
            file_path = conduit::utils::join_file_path(relative_dir, file_path);
        }
    }
};

//-----------------------------------------------------------------------------
std::string sanitize_silo_varname(const std::string &varname)
{
    std::stringstream newvarname;
    for (uint i = 0; i < varname.size(); i ++)
    {
        if (std::isalnum(varname[i]))
            newvarname << varname[i];
        else
            newvarname << "_";
    }
    return newvarname.str();
}

}
//-----------------------------------------------------------------------------
// -- end conduit::relay::<mpi>::io::silo::detail --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
std::string
shapetype_to_string(int shapetype)
{
    if (shapetype == DB_ZONETYPE_BEAM)
        return "line";
    else if (shapetype == DB_ZONETYPE_TRIANGLE)
        return "tri";
    else if (shapetype == DB_ZONETYPE_QUAD)
        return "quad";
    else if (shapetype == DB_ZONETYPE_TET)
        return "tet";
    else if (shapetype == DB_ZONETYPE_HEX)
        return "hex";
    else if (shapetype == DB_ZONETYPE_PRISM)
        return "wedge";
    else if (shapetype == DB_ZONETYPE_PYRAMID)
        return "pyramid";
    else if (shapetype == DB_ZONETYPE_POLYHEDRON)
        return "polyhedral";
    else if (shapetype == DB_ZONETYPE_POLYGON)
        return "polygonal";

    CONDUIT_ERROR("Unsupported zone type " << shapetype);
    return "";
}

//---------------------------------------------------------------------------//
template<typename T>
void
silo_wedge_connectivity_to_conduit(Node &n_mesh_conn)
{
    const int conn_size = n_mesh_conn.dtype().number_of_elements();
    T *conn_ptr = n_mesh_conn.value();
    for (int i = 0; i < conn_size; i += 6)
    {
        auto conn0 = conn_ptr[i + 0];
        auto conn2 = conn_ptr[i + 2];
        auto conn4 = conn_ptr[i + 4];
        auto conn5 = conn_ptr[i + 5];
        conn_ptr[i + 0] = conn2;
        conn_ptr[i + 2] = conn5;
        conn_ptr[i + 4] = conn0;
        conn_ptr[i + 5] = conn4;
    }
}

//---------------------------------------------------------------------------//
template<typename T>
void
conduit_wedge_connectivity_to_silo(Node &n_mesh_conn)
{
    const int conn_size = n_mesh_conn.dtype().number_of_elements();
    T *conn_ptr = n_mesh_conn.value();
    for (int i = 0; i < conn_size; i += 6)
    {
        auto conn0 = conn_ptr[i + 0];
        auto conn2 = conn_ptr[i + 2];
        auto conn4 = conn_ptr[i + 4];
        auto conn5 = conn_ptr[i + 5];
        conn_ptr[i + 2] = conn0;
        conn_ptr[i + 5] = conn2;
        conn_ptr[i + 0] = conn4;
        conn_ptr[i + 4] = conn5;
    }
}

//---------------------------------------------------------------------------//
int get_coordset_silo_type(const std::string &sys)
{
    if (sys == "cartesian")
    {
        return DB_CARTESIAN;
    }
    else if (sys == "cylindrical")
    {
        return DB_CYLINDRICAL;
    }
    else if (sys == "spherical")
    {
        return DB_SPHERICAL;
    }
    CONDUIT_ERROR("Unrecognized coordinate system " << sys);
    return -1;
}

//---------------------------------------------------------------------------//
std::vector<const char *>
get_coordset_axis_labels(const int sys)
{
    std::vector<const char *> coordnames;
    if (sys == DB_CARTESIAN)
    {
        coordnames.push_back(conduit::blueprint::mesh::utils::CARTESIAN_AXES[0].c_str());
        coordnames.push_back(conduit::blueprint::mesh::utils::CARTESIAN_AXES[1].c_str());
        coordnames.push_back(conduit::blueprint::mesh::utils::CARTESIAN_AXES[2].c_str());
    }
    else if (sys == DB_CYLINDRICAL)
    {
        coordnames.push_back(conduit::blueprint::mesh::utils::CYLINDRICAL_AXES[0].c_str());
        coordnames.push_back(conduit::blueprint::mesh::utils::CYLINDRICAL_AXES[1].c_str());
        coordnames.push_back(nullptr);
    }
    else if (sys == DB_SPHERICAL)
    {
        coordnames.push_back(conduit::blueprint::mesh::utils::SPHERICAL_AXES[0].c_str());
        coordnames.push_back(conduit::blueprint::mesh::utils::SPHERICAL_AXES[1].c_str());
        coordnames.push_back(conduit::blueprint::mesh::utils::SPHERICAL_AXES[2].c_str());
    }
    else if (sys == DB_NUMERICAL)
    {
        CONDUIT_ERROR("Conduit Blueprint does not support DB_NUMERICAL coordinate systems.");
    }
    else if (sys == DB_OTHER)
    {
        CONDUIT_INFO("Encountered DB_OTHER, we will default to a cartesian coordinate system.");
        coordnames.push_back(conduit::blueprint::mesh::utils::CARTESIAN_AXES[0].c_str());
        coordnames.push_back(conduit::blueprint::mesh::utils::CARTESIAN_AXES[1].c_str());
        coordnames.push_back(conduit::blueprint::mesh::utils::CARTESIAN_AXES[2].c_str());
    }
    else
    {
        CONDUIT_ERROR("Invalid coordinate system " << sys);
    }
    return coordnames;
}

//-----------------------------------------------------------------------------
void
copy_point_coords(const int datatype,
                  void *coords[3],
                  int ndims,
                  int *dims,
                  const int coord_sys,
                  conduit::Node &node)
{
    ndims = ndims < 3 ? ndims : 3;
    std::vector<const char *> labels = get_coordset_axis_labels(coord_sys);
    if (coord_sys == DB_CYLINDRICAL && ndims >= 3)
    {
        CONDUIT_ERROR("Blueprint only supports 2D cylindrical coordinates");    
    }
    for (int i = 0; i < ndims; i ++)
    {
        if (coords[i] != NULL)
        {
            if (datatype == DB_DOUBLE)
            {
                node[labels[i]].set(static_cast<double *>(coords[i]), dims[i]);
            }
            else if (datatype == DB_FLOAT)
            {
                node[labels[i]].set(static_cast<float *>(coords[i]), dims[i]);
            }
            else
            {
                CONDUIT_ERROR("Unsupported mesh data type " << datatype);
            }
        }
        else
        {
            return;
        }
    }
}

//-----------------------------------------------------------------------------
void
add_offsets(DBzonelist *zones,
            conduit::Node &elements)
{
    int offset = 0;
    int *offset_arr = new int[zones->nzones];
    for (int i = 0; i < zones->nzones; ++i)
    {
        offset_arr[i] = offset;
        offset += zones->shapesize[i];
    }
    elements["offsets"].set(offset_arr, zones->nzones);
}

//-----------------------------------------------------------------------------
void
add_shape_info(DBzonelist *zones,
              conduit::Node &elements)
{
    for (int i = 0; i < zones->nshapes; ++i)
    {
        CONDUIT_ASSERT(zones->shapetype[0] == zones->shapetype[i],
                       "Expected a single shape type, got "
                           << zones->shapetype[0] << " and "
                           << zones->shapetype[i]);
    }

    elements["shape"] = shapetype_to_string(zones->shapetype[0]);
    elements["connectivity"].set(zones->nodelist, zones->lnodelist);
    if (zones->shapetype[0] == DB_ZONETYPE_PRISM)
    {
        // we must reorder the wedge connectivity b/c conduit uses the 
        // vtk ordering, NOT the silo ordering
        DataType dtype = elements["connectivity"].dtype();

        // swizzle the connectivity
        if (dtype.is_uint64())
        {
            silo_wedge_connectivity_to_conduit<uint64>(elements["connectivity"]);
        }
        else if (dtype.is_uint32())
        {
            silo_wedge_connectivity_to_conduit<uint32>(elements["connectivity"]);
        }
        else if (dtype.is_int64())
        {
            silo_wedge_connectivity_to_conduit<int64>(elements["connectivity"]);
        }
        else if (dtype.is_int32())
        {
            silo_wedge_connectivity_to_conduit<int32>(elements["connectivity"]);
        }
        else
        {
            CONDUIT_ERROR("Unsupported connectivity type in " << dtype.to_yaml());
        }
    }

    // TODO polytopal support
    if (zones->shapetype[0] == DB_ZONETYPE_POLYHEDRON)
    {
        CONDUIT_ERROR("Polyhedra not yet supported");
        elements["sizes"].set(zones->shapesize, zones->nzones);
        // TODO double check this approach
        add_offsets(zones, elements["subelements"]); 
    }
    if (zones->shapetype[0] == DB_ZONETYPE_POLYGON)
    {
        CONDUIT_ERROR("Polygons not yet supported");
        // TODO zones->shapesize is NOT zones->nzones elements long; see docs
        // TODO need to loop over the shapes array and expand it out to resemble the blueprint approach
        elements["sizes"].set(zones->shapesize, zones->nzones);
        add_offsets(zones, elements);
    }
}

//-----------------------------------------------------------------------------
// add complete topology and coordset entries to a mesh domain
void
read_ucdmesh_domain(DBucdmesh *ucdmesh_ptr,
                    const std::string &mesh_name,
                    const std::string &multimesh_name,
                    conduit::Node &mesh_domain)
{
    if (ucdmesh_ptr->zones)
    {
        CONDUIT_ASSERT(!ucdmesh_ptr->phzones,
                       "Both phzones and zones are defined in mesh "
                           << mesh_name);
        add_shape_info(ucdmesh_ptr->zones,
                       mesh_domain["topologies"][multimesh_name]["elements"]);
    }
    else if (ucdmesh_ptr->phzones)
    {
        // TODO implement support for phzones
        CONDUIT_ERROR("Silo ucdmesh phzones not yet supported");
        mesh_domain["topologies"][multimesh_name]["elements"]["shape"] =
            shapetype_to_string(DB_ZONETYPE_POLYHEDRON);
    }
    else
    {
        CONDUIT_ERROR("Neither phzones nor zones is defined in mesh "
                      << mesh_name);
    }

    mesh_domain["topologies"][multimesh_name]["coordset"] = multimesh_name;
    mesh_domain["coordsets"][multimesh_name]["type"] = "explicit";
    mesh_domain["topologies"][multimesh_name]["type"] = "unstructured";

    // explicit coords
    int dims[] = {ucdmesh_ptr->nnodes,
                  ucdmesh_ptr->nnodes,
                  ucdmesh_ptr->nnodes};

    copy_point_coords(ucdmesh_ptr->datatype,
                      ucdmesh_ptr->coords,
                      ucdmesh_ptr->ndims,
                      dims,
                      ucdmesh_ptr->coord_sys,
                      mesh_domain["coordsets"][multimesh_name]["values"]);
}

//-----------------------------------------------------------------------------
// add complete topology and coordset entries to a mesh domain
void
read_quadmesh_domain(DBquadmesh *quadmesh_ptr,
                     const std::string &multimesh_name,
                     conduit::Node &mesh_domain)
{
    int coordtype{quadmesh_ptr->coordtype};
    int ndims{quadmesh_ptr->ndims};
    int dims[] = {quadmesh_ptr->nnodes,
                  quadmesh_ptr->nnodes,
                  quadmesh_ptr->nnodes};
    int *real_dims = dims;

    if (coordtype == DB_COLLINEAR)
    {
        mesh_domain["coordsets"][multimesh_name]["type"] = "rectilinear";
        mesh_domain["topologies"][multimesh_name]["type"] = "rectilinear";
        real_dims = quadmesh_ptr->dims;
    }
    else if (coordtype == DB_NONCOLLINEAR)
    {
        mesh_domain["coordsets"][multimesh_name]["type"] = "explicit";
        mesh_domain["topologies"][multimesh_name]["type"] = "structured";

        // We subtract 1 from each of these because in silo these dims are node dims, not element dims
        mesh_domain["topologies"][multimesh_name]["elements/dims/i"] = quadmesh_ptr->dims[0] - 1;
        if (ndims > 1) mesh_domain["topologies"][multimesh_name]["elements/dims/j"] = quadmesh_ptr->dims[1] - 1;
        if (ndims > 2) mesh_domain["topologies"][multimesh_name]["elements/dims/k"] = quadmesh_ptr->dims[2] - 1;
    }
    else
    {
        CONDUIT_ERROR("Undefined coordtype in " << coordtype);
    }

    mesh_domain["topologies"][multimesh_name]["coordset"] = multimesh_name;

    // If the origin is not the default value, then we need to specify it
    if (quadmesh_ptr->base_index[0] != 0 && 
        quadmesh_ptr->base_index[1] != 0 && 
        quadmesh_ptr->base_index[2] != 0)
    {
        Node &origin = mesh_domain["topologies"][multimesh_name]["elements"]["origin"];
        origin["i"] = quadmesh_ptr->base_index[0];
        if (ndims > 1) origin["i"] = quadmesh_ptr->base_index[1];
        if (ndims > 2) origin["i"] = quadmesh_ptr->base_index[2];
    }

    copy_point_coords(quadmesh_ptr->datatype,
                      quadmesh_ptr->coords,
                      ndims,
                      real_dims,
                      quadmesh_ptr->coord_sys,
                      mesh_domain["coordsets"][multimesh_name]["values"]);
}


//-----------------------------------------------------------------------------
// add complete topology and coordset entries to a mesh domain
void
read_pointmesh_domain(DBpointmesh *pointmesh_ptr,
                      const std::string &multimesh_name,
                      conduit::Node &mesh_domain)
{
    mesh_domain["topologies"][multimesh_name]["type"] = "points";
    mesh_domain["topologies"][multimesh_name]["coordset"] = multimesh_name;
    mesh_domain["coordsets"][multimesh_name]["type"] = "explicit";
    int dims[] = {pointmesh_ptr->nels,
                  pointmesh_ptr->nels,
                  pointmesh_ptr->nels};

    copy_point_coords(pointmesh_ptr->datatype,
                      pointmesh_ptr->coords,
                      pointmesh_ptr->ndims,
                      dims,
                      DB_CARTESIAN,
                      mesh_domain["coordsets"][multimesh_name]["values"]);
}

//-----------------------------------------------------------------------------
template <class T>
void
assign_values(int nvals,
              int nels,
              void **vals,
              Node &field_values)
{
    if (nvals == 1)
    {
        field_values.set(static_cast<T *>(vals[0]), nels);
    }
    else
    {
        for (int i = 0; i < nvals; i ++)
        {
            // need to put the values under a vector component
            field_values[std::to_string(i)].set(static_cast<T *>(vals[i]), nels);
        }
    }
}

//-----------------------------------------------------------------------------
template <class T>
void
read_variable_domain(const T *var_ptr,
                     const std::string &var_name,
                     const std::string &multimesh_name,
                     conduit::Node &field)
{
    if (!var_ptr)
    {
        CONDUIT_ERROR("Error fetching variable " << var_name);
    }

    field["topology"] = multimesh_name;

    int datatype = var_ptr->datatype;
    if (datatype == DB_INT)
    {
        assign_values<int>(var_ptr->nvals, var_ptr->nels, 
                           var_ptr->vals, field["values"]);
    }
    else if (datatype == DB_SHORT)
    {
        assign_values<short>(var_ptr->nvals, var_ptr->nels, 
                             var_ptr->vals, field["values"]);
    }
    else if (datatype == DB_LONG)
    {
        assign_values<long>(var_ptr->nvals, var_ptr->nels, 
                            var_ptr->vals, field["values"]);
    }
    else if (datatype == DB_FLOAT)
    {
        assign_values<float>(var_ptr->nvals, var_ptr->nels, 
                             var_ptr->vals, field["values"]);
    }
    else if (datatype == DB_DOUBLE)
    {
        assign_values<double>(var_ptr->nvals, var_ptr->nels,
                              var_ptr->vals, field["values"]);
    }
    else if (datatype == DB_CHAR)
    {
        // implementation taken from assign_values
        if (var_ptr->nvals == 1)
        {
            field["values"].set_char_ptr(static_cast<char *>(var_ptr->vals[0]), var_ptr->nels);
        }
        else
        {
            for (int i = 0; i < var_ptr->nvals; i ++)
            {
                // need to put the values under a vector component
                field["values"][std::to_string(i)].set_char_ptr(static_cast<char *>(var_ptr->vals[0]), var_ptr->nels);
            }
        }
    }
    else if (datatype == DB_LONG_LONG)
    {
        assign_values<long long>(var_ptr->nvals, var_ptr->nels,
                                 var_ptr->vals, field["values"]);
    }
    else
    {
        CONDUIT_ERROR("Unsupported type in " << datatype);
    }
}

// TODO support material read
// //-----------------------------------------------------------------------------
// // Read a material domain from a Silo file.
// // 'file' must be a pointer into the file containing the material domain
// // 'mat_name' must be the name of the material within the file.
// //-----------------------------------------------------------------------------
// void
// read_material_domain(DBfile *file,
//                      std::string &mat_name,
//                      conduit::Node &matsets)
// {
//     DBmaterial *material_ptr;
//     if (!(material_ptr = DBGetMaterial(file, mat_name.c_str())))
//     {
//         CONDUIT_ERROR("Error fetching variable " << mat_name);
//     }

//     std::unique_ptr<DBmaterial, decltype(&DBFreeMaterial)> material{
//         material_ptr, &DBFreeMaterial};
//     conduit::Node &curr_matset = matsets[material_ptr->name];
//     curr_matset["topology"] = material_ptr->meshname;
//     for (int i = 0; i < material_ptr->nmat; ++i)
//     {
//         // material names may be NULL
//         std::string material_name;
//         if (material_ptr->matnames)
//         {
//             material_name = material_ptr->matnames[i];
//         }
//         else
//         {
//             // but matnos should always be
//             material_name = std::to_string(material_ptr->matnos[i]);
//         }
//         curr_matset["material_map"][material_name] = material_ptr->matnos[i];
//     }
//     // TODO: support multi-dimensional materials
//     CONDUIT_ASSERT(material_ptr->ndims == 1,
//                    "Only single-dimension materials supported, got "
//                        << material_ptr->ndims);
//     if (material_ptr->mixlen > 0)
//     {
//         // The struct has volume fractions.
//         // In this case, the struct is very confusing.
//         // If an entry in the `matlist` is negative, it implies that the
//         // associated zone has mixed materials, and `-(value) - 1` gives the
//         // first index into mix_vf and mix_mat for that zone. mix_next is then
//         // used to find the rest of the indices into mix_vf and mix_mat for
//         // the zone.
//         std::vector<double> volume_fractions;
//         std::vector<int> material_ids;
//         std::vector<int> sizes;
//         std::vector<int> offsets;
//         int curr_offset = 0;
//         for (int i = 0; i < material_ptr->dims[0]; ++i)
//         {
//             int matlist_entry = material_ptr->matlist[i];
//             if (matlist_entry >= 0)
//             {
//                 volume_fractions.push_back(1.0);
//                 material_ids.push_back(matlist_entry);
//                 sizes.push_back(1);
//                 offsets.push_back(curr_offset);
//                 curr_offset++;
//             }
//             else
//             {
//                 int mix_id = -(matlist_entry)-1;
//                 int curr_size = 0;
//                 while (mix_id >= 0)
//                 {
//                     material_ids.push_back(material_ptr->mix_mat[mix_id]);
//                     if (material_ptr->datatype == DB_DOUBLE)
//                     {
//                         volume_fractions.push_back(static_cast<double *>(
//                             material_ptr->mix_vf)[mix_id]);
//                     }
//                     else if (material_ptr->datatype == DB_FLOAT)
//                     {
//                         volume_fractions.push_back(
//                             static_cast<float *>(material_ptr->mix_vf)[mix_id]);
//                     }
//                     curr_size++;
//                     mix_id = material_ptr->mix_next[mix_id] - 1;
//                 }
//                 sizes.push_back(curr_size);
//                 offsets.push_back(curr_offset);
//                 curr_offset += curr_size;
//             }
//         }
//         curr_matset["material_ids"].set(material_ids.data(), material_ids.size());
//         curr_matset["volume_fractions"].set(volume_fractions.data(),
//                                        volume_fractions.size());
//         curr_matset["sizes"].set(sizes.data(), sizes.size());
//         curr_matset["offsets"].set(offsets.data(), offsets.size());
//     }
//     else
//     {
//         // TODO: remove, since this is just a special case of the above logic, I think?
//         // no volume fractions. All zones are single-material.
//         int arr_len = material_ptr->dims[0];
//         copy_and_assign(material_ptr->matlist,
//                         arr_len,
//                         curr_matset["material_ids"]);

//         double *volume_fractions = new double[arr_len];
//         int *sizes = new int[arr_len];
//         int *offsets = new int[arr_len];
//         for (int i = 0; i < arr_len; ++i)
//         {
//             volume_fractions[i] = 1.0;
//             offsets[i] = i;
//             sizes[i] = 1;
//         }
//         curr_matset["volume_fractions"].set(volume_fractions, arr_len);
//         curr_matset["sizes"].set(sizes, arr_len);
//         curr_matset["offsets"].set(offsets, arr_len);
//     }
// }

// //-----------------------------------------------------------------------------
// // Read a multimaterial from a Silo file.
// // 'root_file' should be the file containing the multivar entry
// // 'filemap' should be a mapping providing DBfile* for files which have
// //  already been opened.
// // 'dirname' should be the directory containing the root file, as if the
// // `dirname` command were called on the root file path. This directory is used
// // to concretize the paths given by the multimat.
// //-----------------------------------------------------------------------------
// void
// read_multimaterial(DBfile *root_file,
//                    std::map<std::string, std::unique_ptr<DBfile, decltype(&DBClose)>> &filemap,
//                    const std::string &dirname,
//                    DBmultimat *multimat,
//                    conduit::Node &mesh)
// {

//     std::string file_path, silo_name;
//     for (index_t i = 0; i < multimat->nmats; ++i)
//     {
//         Node &matsets = mesh[i]["matsets"];
//         split_silo_path(multimat->matnames[i], dirname, file_path, silo_name);
//         if (!file_path.empty())
//         {
//             read_material_domain(get_or_open(filemap, file_path),
//                                  silo_name,
//                                  matsets);
//         }
//         else
//         {
//             read_material_domain(root_file, silo_name, matsets);
//         }
//     }
// }

// //---------------------------------------------------------------------------//
// void
// read_all_multimats(DBfile *root_file,
//                   DBtoc *toc,
//                   std::map<std::string, std::unique_ptr<DBfile, decltype(&DBClose)>> &filemap,
//                   const std::string &dirname,
//                   const std::string &mmesh_name,
//                   int expected_domains,
//                   conduit::Node &mesh)
// {

//     for (int i = 0; i < toc->nmultimat; ++i)
//     {
//         std::unique_ptr<DBmultimat, decltype(&DBFreeMultimat)> multimat{
//             DBGetMultimat(root_file, toc->multimat_names[i]), &DBFreeMultimat};
//         if (!multimat.get()) {
//             multimat.release();
//             CONDUIT_ERROR("Error fetching multimaterial "
//                           << multimat.get()->matnames[i]);
//         }

//         if (multimat.get()->mmesh_name != NULL &&
//             multimat.get()->mmesh_name == mmesh_name)
//         {
//             CONDUIT_ASSERT(multimat.get()->nmats == expected_domains,
//                            "Domain count mismatch between multimaterial "
//                                << multimat.get()->matnames[i]
//                                << "and multimesh");
//             // read in the multimaterial and add it to the mesh Node
//             read_multimaterial(root_file,
//                                filemap,
//                                dirname,
//                                multimat.get(),
//                                mesh);
//         }
//     }
// }

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API
read_mesh(const std::string &root_file_path,
          Node &mesh
          CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm mpi_comm))
{
    Node opts;
#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    read_mesh(root_file_path,
              opts,
              mesh,
              mpi_comm);
#else
    read_mesh(root_file_path,
              opts,
              mesh);
#endif
}

//-----------------------------------------------------------------------------
bool
read_root_silo_index(const std::string &root_file_path,
                     const Node &opts,
                     Node &root_node, // output
                     std::string &multimesh_name, // output
                     std::ostringstream &error_oss) // output
{
    // clear output vars
    root_node.reset();
    multimesh_name = "";
    error_oss.str("");

    // first, make sure we can open the root file
    std::ifstream ifs;
    ifs.open(root_file_path.c_str());
    if(!ifs.is_open())
    {
        error_oss << "failed to open root file: " << root_file_path;
        return false;
    }
    ifs.close();

    // open silo file
    detail::SiloObjectWrapperCheckError<DBfile, decltype(&DBClose)> dbfile{
        DBOpen(root_file_path.c_str(), DB_UNKNOWN, DB_READ), 
        &DBClose, 
        "Error closing Silo file: " + root_file_path};
    if (! dbfile.getSiloObject())
    {
        error_oss << "Error opening Silo file for reading: " << root_file_path;
        return false;
    }

    // get table of contents
    DBtoc *toc = DBGetToc(dbfile.getSiloObject()); // shouldn't be free'd
    if (!toc)
    {
        error_oss << "Table of contents could not be extracted from file: " << root_file_path;
        return false;
    }
    // check for multimeshes
    if (toc->nmultimesh <= 0)
    {
        error_oss << "No multimesh found in file: " << root_file_path;
        return false;
    }

    // decide what multimesh to extract
    if(opts.has_child("mesh_name") && opts["mesh_name"].dtype().is_string())
    {
        multimesh_name = opts["mesh_name"].as_string();
    }

    // check multimesh name
    if (multimesh_name.empty())
    {
        multimesh_name = toc->multimesh_names[0];
    }
    else
    {
        bool found = false;
        for (int i = 0; i < toc->nmultimesh; i ++)
        {
            if (toc->multimesh_names[i] == multimesh_name)
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            error_oss << "No multimesh found matching " << multimesh_name;
            return false;
        }
    }

    // extract the multimesh
    detail::SiloObjectWrapper<DBmultimesh, decltype(&DBFreeMultimesh)> multimesh{
        DBGetMultimesh(dbfile.getSiloObject(), multimesh_name.c_str()), 
        &DBFreeMultimesh};
    if (! multimesh.getSiloObject())
    {
        error_oss << "Error opening multimesh " << multimesh_name;
        return false;
    }

    const int nblocks = multimesh.getSiloObject()->nblocks;
    root_node[multimesh_name]["nblocks"] = nblocks;

    bool nameschemes = false;
    if (!multimesh.getSiloObject()->meshnames || !multimesh.getSiloObject()->meshtypes)
    {
        nameschemes = true;
    }
    // TODO nameschemes
    if (nameschemes)
    {
        root_node[multimesh_name]["nameschemes"] = "yes";
        error_oss << "multimesh " << multimesh_name << " uses nameschemes which are not yet supported.";
        return false;
    }
    else
    {
        root_node[multimesh_name]["nameschemes"] = "no";
        root_node[multimesh_name]["mesh_types"].set(DataType::int64(nblocks));
        int64 *mesh_types = root_node[multimesh_name]["mesh_types"].value();
        for (int i = 0; i < nblocks; i ++)
        {
            // save the mesh name and mesh type
            Node &mesh_path = root_node[multimesh_name]["mesh_paths"].append();
            mesh_path.set(multimesh.getSiloObject()->meshnames[i]);
            mesh_types[i] = multimesh.getSiloObject()->meshtypes[i];
        }
    }

    // iterate thru the multivars and find the ones that are associated with
    // the chosen multimesh
    for (int i = 0; i < toc->nmultivar; i ++)
    {
        std::string multivar_name = toc->multivar_names[i];
        detail::SiloObjectWrapper<DBmultivar, decltype(&DBFreeMultivar)> multivar{
            DBGetMultivar(dbfile.getSiloObject(), multivar_name.c_str()), 
            &DBFreeMultivar};
        if (! multivar.getSiloObject())
        {
            error_oss << "Error opening multivar " << multivar_name;
            return false;
        }

        // does this variable use nameschemes?
        bool var_nameschemes = false;
        if (!multivar.getSiloObject()->varnames || !multivar.getSiloObject()->vartypes)
        {
            var_nameschemes = true;
            error_oss << "multivar " << multivar_name << " uses nameschemes which are not yet supported.";
            return false;
        }

        // is this multivar associated with a multimesh?
        bool multimesh_assoc = false;

        // there are two cases:
        // 1. the multivar is directly associated with a multimesh
        // 2. the components of the multivar are associatd with components of a multimesh

        // we begin with the second case:
        if (!multivar.getSiloObject()->mmesh_name)
        {
            // This multivar has no associated multimesh. 
            // We will assume it is associated with the multimesh
            // And then check later when we are actually reading vars
            multimesh_assoc = true;
        }
        // and then the first case
        else if (multivar.getSiloObject()->mmesh_name == multimesh_name)
        {
            multimesh_assoc = true;
        }

        if (multimesh_assoc)
        {
            if (multivar.getSiloObject()->nvars != nblocks)
            {
                CONDUIT_INFO("Domain count mismatch between multivar " +
                             multivar_name + " and multimesh " + 
                             multimesh_name + ". Skipping.");
                continue;
            }
            Node &var = root_node[multimesh_name]["vars"][multivar_name];
            // TODO var_nameschemes
            if (var_nameschemes)
            {
                var["nameschemes"] = "yes";
            }
            else
            {
                var["nameschemes"] = "no";
                var["var_types"].set(DataType::int64(nblocks));
                int64 *var_types = var["var_types"].value();
                for (int j = 0; j < nblocks; j ++)
                {
                    // save the mesh name and mesh type
                    Node &var_path = var["var_paths"].append();
                    var_path.set(multivar.getSiloObject()->varnames[j]);
                    var_types[j] = multivar.getSiloObject()->vartypes[j];
                }
            }
        }
    }

    // now set up state if necessary

    // look for dtime then time like VisIt
    if (DBInqVarExists(dbfile.getSiloObject(), "dtime"))
    {
        double dtime;
        DBReadVar(dbfile.getSiloObject(), "dtime", &dtime);
        root_node[multimesh_name]["state"]["time"] = dtime;
    }
    else if (DBInqVarExists(dbfile.getSiloObject(), "time"))
    {
        float ftime;
        DBReadVar(dbfile.getSiloObject(), "time", &ftime);
        root_node[multimesh_name]["state"]["time"] = (double) ftime;
    }

    if (DBInqVarExists(dbfile.getSiloObject(), "cycle"))
    {
        int cycle;
        DBReadVar(dbfile.getSiloObject(), "cycle", &cycle);
        root_node[multimesh_name]["state"]["cycle"] = cycle;
    }

    // our silo index should look like this:

    // mesh:
    //    state:
    //       cycle: 100
    //       time: 10
    //       dtime: 10
    //    nblocks: 5
    //    nameschemes: "no"
    //    mesh_paths:
    //       - "domain_000000.silo:mesh"
    //       - "domain_000001.silo:mesh" 
    //         ...
    //    mesh_types: [UCD_MESH, UCD_MESH, ...]
    //    vars:
    //       field:
    //          nameschemes: "no"
    //          var_paths:
    //             - "domain_000000.silo:field"
    //             - "domain_000001.silo:field"
    //               ...
    //          var_types: [DB_UCDVAR, DB_UCDVAR, ...]
    //       ...

    return true;
}

//-----------------------------------------------------------------------------
///
/// opts:
///      mesh_name: "{name}"
///          provide explicit mesh name, for cases where silo data includes
///           more than one mesh.
///
/// note: we have made the choice to read ONLY the multimesh with the name
/// mesh_name. We also read all multivariables which are associated with the
/// chosen multimesh.
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API
read_mesh(const std::string &root_file_path,
          const Node &opts,
          Node &mesh
          CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm mpi_comm))
{
    int par_rank = 0;
#if CONDUIT_RELAY_IO_MPI_ENABLED
    par_rank = relay::mpi::rank(mpi_comm);
    int par_size = relay::mpi::size(mpi_comm);
#endif

    int error = 0;
    std::ostringstream error_oss;
    std::string multimesh_name;
    Node root_node;

    // only read bp index on rank 0
    if(par_rank == 0)
    {
        if(!read_root_silo_index(root_file_path,
                                 opts,
                                 root_node,
                                 multimesh_name,
                                 error_oss))
        {
            error = 1;
        }
    }

#if CONDUIT_RELAY_IO_MPI_ENABLED
    Node n_local, n_global;
    n_local.set((int)error);
    relay::mpi::sum_all_reduce(n_local,
                               n_global,
                               mpi_comm);

    error = n_global.as_int();

    if(error == 1)
    {
        // we have a problem, broadcast string message
        // from rank 0 all ranks can throw an error
        n_global.set(error_oss.str());
        conduit::relay::mpi::broadcast_using_schema(n_global,
                                                    0,
                                                    mpi_comm);

        CONDUIT_ERROR(n_global.as_string());
    }
    else
    {
        // broadcast the mesh name and the bp index
        // from rank 0 to all ranks
        n_global.set(multimesh_name);
        conduit::relay::mpi::broadcast_using_schema(n_global,
                                                    0,
                                                    mpi_comm);
        multimesh_name = n_global.as_string();
        conduit::relay::mpi::broadcast_using_schema(root_node,
                                                    0,
                                                    mpi_comm);
    }
#else
    // non MPI case, throw error
    if(error == 1)
    {
        CONDUIT_ERROR(error_oss.str());
    }
#endif
    const Node &mesh_index = root_node[multimesh_name];

    // read all domains for given mesh
    int num_domains = mesh_index["nblocks"].to_int();

    std::ostringstream oss;
    int domain_start = 0;
    int domain_end = num_domains;

#if CONDUIT_RELAY_IO_MPI_ENABLED

    int read_size = num_domains / par_size;
    int rem = num_domains % par_size;
    if(par_rank < rem)
    {
        read_size++;
    }

    conduit::Node n_read_size;
    conduit::Node n_doms_per_rank;

    n_read_size.set_int32(read_size);

    relay::mpi::all_gather_using_schema(n_read_size,
                                        n_doms_per_rank,
                                        mpi_comm);
    int *counts = (int*)n_doms_per_rank.data_ptr();

    int rank_offset = 0;
    for(int i = 0; i < par_rank; ++i)
    {
        rank_offset += counts[i];
    }

    domain_start = rank_offset;
    domain_end = rank_offset + read_size;
#endif

    bool mesh_nameschemes = false;
    if (mesh_index.has_child("nameschemes") &&
        mesh_index["nameschemes"].as_string() == "yes")
    {
        mesh_nameschemes = true;
        CONDUIT_ERROR("TODO no support for nameschemes yet");
    }
    detail::SiloTreePathGenerator mesh_path_gen{mesh_nameschemes};

    std::string root_file_name, relative_dir;
    utils::rsplit_file_path(root_file_path, root_file_name, relative_dir);

    // If the root file is named OvlTop.silo, then there is a very good chance that
    // this file is valid overlink. Therefore, we must modify the paths we get from
    // the root node to reflect this.
    bool ovltop_case = false;
    if (root_file_name == "OvlTop.silo")
    {
        ovltop_case = true;
    }

    for (int domain_id = domain_start; domain_id < domain_end; domain_id ++)
    {
        //
        // Read Mesh
        //

        std::string silo_mesh_path = mesh_index["mesh_paths"][domain_id].as_string();
        int_accessor meshtypes = mesh_index["mesh_types"].value();
        int meshtype = meshtypes[domain_id];

        std::string mesh_name, mesh_domain_filename;
        mesh_path_gen.GeneratePaths(silo_mesh_path, relative_dir, mesh_domain_filename, mesh_name);

        if (mesh_name == "EMPTY")
        {
            continue;
        }

        std::string bottom_level_mesh_name, tmp;
        conduit::utils::rsplit_file_path(mesh_name, "/", bottom_level_mesh_name, tmp);

        // root only case
        if (mesh_domain_filename.empty())
        {
            mesh_domain_filename = root_file_path;
            // we are in the root file only case so overlink is not possible
            ovltop_case = false;
        }

        detail::SiloObjectWrapperCheckError<DBfile, decltype(&DBClose)> mesh_domain_file{
            nullptr, 
            &DBClose};

        if (ovltop_case)
        {
            // first, we will assume valid overlink, so
            // we need to move the mesh path to ../
            std::string old_mesh_domain_filename = mesh_domain_filename;
            std::string actual_filename, directory;
            conduit::utils::rsplit_file_path(mesh_domain_filename, actual_filename, directory);
            if (!directory.empty())
            {
                std::string dir_lvl_up, bottom_lvl_dir;
                conduit::utils::rsplit_file_path(directory, bottom_lvl_dir, dir_lvl_up);

                mesh_domain_filename = conduit::utils::join_file_path(dir_lvl_up, actual_filename);
            }
            mesh_domain_file.setSiloObject(DBOpen(mesh_domain_filename.c_str(), DB_UNKNOWN, DB_READ));
            mesh_domain_file.setErrMsg("Error closing Silo file: " + mesh_domain_filename);
            if (! mesh_domain_file.getSiloObject())
            {
                CONDUIT_INFO("Provided file is not valid Overlink; defaulting to absolute path rather than assumed path.")
                // this is not valid overlink so we default to what is in the path
                mesh_domain_filename = old_mesh_domain_filename;
                mesh_domain_file.setSiloObject(DBOpen(mesh_domain_filename.c_str(), DB_UNKNOWN, DB_READ));
                mesh_domain_file.setErrMsg("Error closing Silo file: " + mesh_domain_filename);
                if (! mesh_domain_file.getSiloObject())
                {
                    CONDUIT_ERROR("Error opening Silo file for reading: " << mesh_domain_filename);
                }
            }
        }
        else
        {
            mesh_domain_file.setSiloObject(DBOpen(mesh_domain_filename.c_str(), DB_UNKNOWN, DB_READ));
            mesh_domain_file.setErrMsg("Error closing Silo file: " + mesh_domain_filename);
            if (! mesh_domain_file.getSiloObject())
            {
                CONDUIT_ERROR("Error opening Silo file for reading: " << mesh_domain_filename);
            }
        }

        // this is for the blueprint mesh output
        std::string domain_path = conduit_fmt::format("domain_{:06d}", domain_id);

        if (meshtype == DB_UCDMESH)
        {
            detail::SiloObjectWrapper<DBucdmesh, decltype(&DBFreeUcdmesh)> ucdmesh{
                DBGetUcdmesh(mesh_domain_file.getSiloObject(), mesh_name.c_str()), 
                &DBFreeUcdmesh};
            if (!ucdmesh.getSiloObject())
            {
                // If we cannot fetch this mesh we will skip
                continue;
            }
            read_ucdmesh_domain(ucdmesh.getSiloObject(), 
                                mesh_name, 
                                multimesh_name, 
                                mesh[domain_path]);
        }
        else if (meshtype == DB_QUADMESH ||
                 meshtype == DB_QUADCURV ||
                 meshtype == DB_QUADRECT)
        {
            detail::SiloObjectWrapper<DBquadmesh, decltype(&DBFreeQuadmesh)> quadmesh{
                DBGetQuadmesh(mesh_domain_file.getSiloObject(), mesh_name.c_str()), 
                &DBFreeQuadmesh};
            if (!quadmesh.getSiloObject())
            {
                // If we cannot fetch this mesh we will skip
                continue;
            }
            read_quadmesh_domain(quadmesh.getSiloObject(), 
                                 multimesh_name, 
                                 mesh[domain_path]);
        }
        else if (meshtype == DB_POINTMESH)
        {
            detail::SiloObjectWrapper<DBpointmesh, decltype(&DBFreePointmesh)> pointmesh{
                DBGetPointmesh(mesh_domain_file.getSiloObject(), mesh_name.c_str()), 
                &DBFreePointmesh};
            if (!pointmesh.getSiloObject())
            {
                // If we cannot fetch this mesh we will skip
                continue;
            }
            read_pointmesh_domain(pointmesh.getSiloObject(), 
                                  multimesh_name, 
                                  mesh[domain_path]);
        }
        else
        {
            CONDUIT_ERROR("Unsupported mesh type " << meshtype);
        }

        // we know we were for sure successful (we didn't skip ahead to the next domain)
        // so we create the mesh_out now for good
        Node &mesh_out = mesh[domain_path];

        mesh_out["state"]["domain_id"] = domain_id;
        if (mesh_index.has_path("state/time"))
        {
            mesh_out["state"]["time"] = mesh_index["state"]["time"].as_double();
        }
        if (mesh_index.has_path("state/cycle"))
        {
            mesh_out["state"]["cycle"] = (index_t) mesh_index["state"]["cycle"].as_int();
        }

        //
        // Read Fields
        //

        // for each mesh domain, we would like to iterate through all the variables
        // and extract the same domain from them.
        if (mesh_index.has_child("vars"))
        {
            const Node &vars = mesh_index["vars"];
            auto var_itr = vars.children();
            while (var_itr.has_next())
            {
                const Node &n_var = var_itr.next();
                std::string multivar_name = var_itr.name();

                bool var_nameschemes = false;
                if (n_var.has_child("nameschemes") &&
                    n_var["nameschemes"].as_string() == "yes")
                {
                    var_nameschemes = true;
                    CONDUIT_ERROR("TODO no support for nameschemes yet");
                }
                detail::SiloTreePathGenerator var_path_gen{var_nameschemes};

                std::string silo_var_path = n_var["var_paths"][domain_id].as_string();
                int_accessor vartypes = n_var["var_types"].value();
                int vartype = vartypes[domain_id];

                std::string var_name, var_domain_filename;
                var_path_gen.GeneratePaths(silo_var_path, relative_dir, var_domain_filename, var_name);

                if (var_name == "EMPTY")
                {
                    // we choose not to write anything to blueprint
                    continue;
                }

                // root only case
                if (var_domain_filename.empty())
                {
                    var_domain_filename = root_file_path;
                    // we are in the root file only case so overlink is not possible
                    ovltop_case = false;
                }

                detail::SiloObjectWrapperCheckError<DBfile, decltype(&DBClose)> var_domain_file{
                    nullptr, 
                    &DBClose};
                DBfile *domain_file_to_use = nullptr;

                // handle ovltop.silo case
                if (ovltop_case)
                {
                    // first, we will assume valid overlink, so
                    // we need to move the var path to ../
                    std::string old_var_domain_filename = var_domain_filename;
                    std::string actual_filename, directory;
                    conduit::utils::rsplit_file_path(var_domain_filename, actual_filename, directory);
                    if (!directory.empty())
                    {
                        std::string dir_lvl_up, bottom_lvl_dir;
                        conduit::utils::rsplit_file_path(directory, bottom_lvl_dir, dir_lvl_up);

                        var_domain_filename = conduit::utils::join_file_path(dir_lvl_up, actual_filename);
                    }

                    // if the var domain is stored in the same file as the mesh domain then we
                    // can reuse the open file ptr
                    if (var_domain_filename == mesh_domain_filename)
                    {
                        domain_file_to_use = mesh_domain_file.getSiloObject();
                    }
                    // otherwise we need to open our own file
                    else
                    {
                        var_domain_file.setSiloObject(DBOpen(var_domain_filename.c_str(), DB_UNKNOWN, DB_READ));
                        var_domain_file.setErrMsg("Error closing Silo file: " + var_domain_filename);
                        if (! (domain_file_to_use = var_domain_file.getSiloObject()))
                        {
                            CONDUIT_INFO("Provided file is not valid Overlink; defaulting to absolute path rather than assumed path.")
                            // this is not valid overlink so we default to what is in the path
                            var_domain_filename = old_var_domain_filename;

                            // if the var domain is stored in the same file as the mesh domain then we
                            // can reuse the open file ptr
                            if (var_domain_filename == mesh_domain_filename)
                            {
                                domain_file_to_use = mesh_domain_file.getSiloObject();
                            }
                            // otherwise we need to open our own file
                            else
                            {
                                var_domain_file.setSiloObject(DBOpen(var_domain_filename.c_str(), DB_UNKNOWN, DB_READ));
                                var_domain_file.setErrMsg("Error closing Silo file: " + var_domain_filename);
                                if (! (domain_file_to_use = var_domain_file.getSiloObject()))
                                {
                                    CONDUIT_ERROR("Error opening Silo file for reading: " << var_domain_filename);
                                }
                            }
                        }
                    }
                }
                else
                {
                    // if the var domain is stored in the same file as the mesh domain then we
                    // can reuse the open file ptr
                    if (var_domain_filename == mesh_domain_filename)
                    {
                        domain_file_to_use = mesh_domain_file.getSiloObject();
                    }
                    // otherwise we need to open our own file
                    else
                    {
                        var_domain_file.setSiloObject(DBOpen(var_domain_filename.c_str(), DB_UNKNOWN, DB_READ));
                        var_domain_file.setErrMsg("Error closing Silo file: " + var_domain_filename);
                        if (! (domain_file_to_use = var_domain_file.getSiloObject()))
                        {
                            CONDUIT_ERROR("Error opening Silo file for reading: " << var_domain_filename);
                        }
                    }
                }
                // end handling of ovltop.silo case

                if (vartype == DB_UCDVAR)
                {
                    // create ucd var
                    detail::SiloObjectWrapper<DBucdvar, decltype(&DBFreeUcdvar)> ucdvar{
                        DBGetUcdvar(domain_file_to_use, var_name.c_str()),
                        &DBFreeUcdvar};

                    // If we cannot fetch this var we will skip
                    if (!ucdvar.getSiloObject())
                    {
                        continue;
                    }

                    // check that this var is associated with the mesh
                    std::string var_meshname = ucdvar.getSiloObject()->meshname;
                    if (var_meshname.length() > 1 && var_meshname[0] == '/')
                    {
                        var_meshname = var_meshname.substr(1);
                    }
                    if (var_meshname != bottom_level_mesh_name)
                    {
                        CONDUIT_INFO("DB_UCDVAR " + var_name + " is not "
                                     "associated with mesh " + var_meshname +
                                     ". Skipping.");
                        continue;
                    }

                    // create an entry for this field in the output
                    Node &field_out = mesh_out["fields"][multivar_name];

                    // handle association
                    field_out["association"] = ucdvar.getSiloObject()->centering == DB_ZONECENT ? "element" : "vertex";
                    
                    // call subroutine for handling the rest
                    read_variable_domain<DBucdvar>(ucdvar.getSiloObject(), 
                                                   var_name, 
                                                   multimesh_name, 
                                                   field_out);
                }
                else if (vartype == DB_QUADVAR)
                {
                    // create quad var
                    detail::SiloObjectWrapper<DBquadvar, decltype(&DBFreeQuadvar)> quadvar{
                        DBGetQuadvar(domain_file_to_use, var_name.c_str()), 
                        &DBFreeQuadvar};

                    // If we cannot fetch this var we will skip
                    if (!quadvar.getSiloObject())
                    {
                        continue;
                    }

                    // check that this var is associated with the mesh
                    std::string var_meshname = quadvar.getSiloObject()->meshname;
                    if (var_meshname.length() > 1 && var_meshname[0] == '/')
                    {
                        var_meshname = var_meshname.substr(1);
                    }
                    if (var_meshname != bottom_level_mesh_name)
                    {
                        CONDUIT_INFO("DB_QUADVAR " + var_name + " is not "
                                     "associated with mesh " + var_meshname +
                                     ". Skipping.");
                        continue;
                    }

                    // create an entry for this field in the output
                    Node &field_out = mesh_out["fields"][multivar_name];

                    // handle association
                    field_out["association"] = quadvar.getSiloObject()->centering == DB_NODECENT ? "vertex" : "element";

                    // call subroutine for handling the rest
                    read_variable_domain<DBquadvar>(quadvar.getSiloObject(), 
                                                    var_name, 
                                                    multimesh_name, 
                                                    field_out);
                }
                else if (vartype == DB_POINTVAR)
                {
                    // create point var
                    detail::SiloObjectWrapper<DBmeshvar, decltype(&DBFreeMeshvar)> meshvar{
                        DBGetPointvar(domain_file_to_use, var_name.c_str()), 
                        &DBFreeMeshvar};

                    // If we cannot fetch this var we will skip
                    if (!meshvar.getSiloObject())
                    {
                        continue;
                    }

                    // check that this var is associated with the mesh
                    std::string var_meshname = meshvar.getSiloObject()->meshname;
                    if (var_meshname.length() > 1 && var_meshname[0] == '/')
                    {
                        var_meshname = var_meshname.substr(1);
                    }
                    if (var_meshname != bottom_level_mesh_name)
                    {
                        CONDUIT_INFO("DB_POINTVAR " + var_name + " is not "
                                     "associated with mesh " + var_meshname +
                                     ". Skipping.");
                        continue;
                    }

                    // create an entry for this field in the output
                    Node &field_out = mesh_out["fields"][multivar_name];

                    // handle association
                    field_out["association"] = "vertex";

                    // call subroutine for handling the rest
                    read_variable_domain<DBmeshvar>(meshvar.getSiloObject(), 
                                                    var_name, 
                                                    multimesh_name, 
                                                    field_out);
                }
                else
                {
                    CONDUIT_ERROR("Unsupported variable type " << vartype);
                }
            }
        }
        // TODO read multimaterials
    }
}

//-----------------------------------------------------------------------------
// The load semantics, the mesh node is reset before reading.
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void load_mesh(const std::string &root_file_path,
               conduit::Node &mesh
               CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm mpi_comm))
{
    mesh.reset();

#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    read_mesh(root_file_path,
              mesh,
              mpi_comm);
#else
    read_mesh(root_file_path,
              mesh);
#endif
}

//-----------------------------------------------------------------------------
///
/// opts:
///      mesh_name: "{name}"
///          provide explicit mesh name, for cases where silo data includes
///           more than one mesh.
///
/// note: we have made the choice to read ONLY the multimesh with the name
/// mesh_name. We also read all multivariables which are associated with the
/// chosen multimesh.
//-----------------------------------------------------------------------------
void load_mesh(const std::string &root_file_path,
               const conduit::Node &opts,
               conduit::Node &mesh
               CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm mpi_comm))
{
    mesh.reset();

#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    read_mesh(root_file_path,
              opts,
              mesh,
              mpi_comm);
#else
    read_mesh(root_file_path,
              opts,
              mesh);
#endif
}

//---------------------------------------------------------------------------//
int dtype_to_silo_type(DataType dtype)
{
    if (dtype.is_float())
    {
        return DB_FLOAT;
    }
    else if (dtype.is_double())
    {
        return DB_DOUBLE;
    }
    else if (dtype.is_int())
    {
        return DB_INT;
    }
    else if (dtype.is_long())
    {
        return DB_LONG;
    }
    else if (dtype.is_long_long())
    {
        return DB_LONG_LONG;
    }
    else if (dtype.is_char())
    {
        return DB_CHAR;
    }
    else if (dtype.is_short())
    {
        return DB_SHORT;
    }
    return DB_NOTYPE;
}

//---------------------------------------------------------------------------//
void silo_write_field(DBfile *dbfile,
                      const std::string &var_name,
                      const Node &n_var,
                      const bool overlink,
                      const int local_num_domains,
                      const int local_domain_index,
                      const int global_domain_id,
                      Node &local_type_info,
                      Node &n_mesh_info)
{
    if (!n_var.has_path("topology"))
    {
        CONDUIT_INFO("Skipping this variable because we are "
                     "missing a linked topology: "
                      << "fields/" << var_name << "/topology");
        return;
    }

    const std::string topo_name = n_var["topology"].as_string();

    if (!n_mesh_info.has_path(topo_name))
    {
        CONDUIT_INFO("Skipping this variable because the linked "
                     "topology is invalid: "
                      << "fields/" << var_name
                      << "/topology: " << topo_name);
        return;
    }

    std::string mesh_type = n_mesh_info[topo_name]["type"].as_string();
    int num_elems = n_mesh_info[topo_name]["num_elems"].to_value();
    int num_pts = n_mesh_info[topo_name]["num_pts"].to_value();

    int centering = 0;
    int num_values = 0;

    if (!n_var.has_path("association"))
    {
        CONDUIT_ERROR("Missing association! "
                      << "fields/" << var_name << "/association");
    }

    const std::string association = n_var["association"].as_string();
    if (association == "element")
    {
        centering = DB_ZONECENT;
        num_values = num_elems;
    }
    else if (association == "vertex")
    {
        centering = DB_NODECENT;
        num_values = num_pts;
    }
    else
    {
        CONDUIT_ERROR("Unknown association in " << association);
    }

    if (!n_var.has_path("values"))
    {
        CONDUIT_ERROR("Missing field data ! "
                      << "fields/" << var_name << "/values");
    }

    // we compact to support a strided array cases
    Node n_values;
    n_var["values"].compact_to(n_values);

    DataType dtype = n_values.dtype();
    int vals_type = DB_NOTYPE;
    int nvars = 0;
    std::vector<std::string> comp_name_strings;
    std::vector<const char *> comp_name_ptrs;
    std::vector<const void *> comp_vals_ptrs;

    // if we have vector/tensor values instead
    if (dtype.is_object())
    {
        nvars = n_values.number_of_children();
        if (nvars > 0)
        {
            vals_type = dtype_to_silo_type(n_values[0].dtype());
            if (vals_type == DB_NOTYPE)
            {
                // skip the field if we don't support its type
                CONDUIT_INFO("skipping field "
                             << var_name
                             << ", since its type is not implemented, found "
                             << dtype.name());
                return;
            }
        }

        auto val_itr = n_var["values"].children();
        while (val_itr.has_next())
        {
            const Node &n_comp = val_itr.next();
            std::string comp_name = val_itr.name();

            if (vals_type != dtype_to_silo_type(n_comp.dtype()))
            {
                CONDUIT_ERROR("Inconsistent values types across vector components in field " << var_name);
            }

            comp_name_strings.push_back(comp_name);
            comp_name_ptrs.push_back(comp_name_strings.back().c_str());
            comp_vals_ptrs.push_back(n_comp.element_ptr(0));
        }
    }
    else
    {
        nvars = 1;
        vals_type = dtype_to_silo_type(dtype);
        if (vals_type == DB_NOTYPE)
        {
            // skip the field if we don't support its type
            CONDUIT_INFO("skipping field "
                         << var_name
                         << ", since its type is not implemented, found "
                         << dtype.name());
            return;
        }
        comp_name_strings.push_back("unused");
        comp_name_ptrs.push_back(comp_name_strings.back().c_str());
        comp_vals_ptrs.push_back(n_values.element_ptr(0));
    }

    std::string safe_meshname;
    if (overlink)
    {
        safe_meshname = "MESH";
    }
    else
    {
        safe_meshname = detail::sanitize_silo_varname(topo_name);
    }

    int var_type;

    int silo_error = 0;
    if (mesh_type == "unstructured")
    {
        // save the var type
        var_type = DB_UCDVAR;

        silo_error = DBPutUcdvar(dbfile, // Database file pointer
                                 detail::sanitize_silo_varname(var_name).c_str(), // variable name
                                 safe_meshname.c_str(), // mesh name
                                 nvars, // number of variable components
                                 comp_name_ptrs.data(), // variable component names
                                 comp_vals_ptrs.data(), // the data values
                                 num_values, // number of elements
                                 NULL, // mixed data arrays
                                 0, // lenght of mixed data arrays
                                 vals_type, // Datatype of the variable
                                 centering, // centering (nodal or zonal)
                                 NULL); // optlist
    }
    else if (mesh_type == "rectilinear" || 
             mesh_type == "uniform" ||
             mesh_type == "structured")
    {
        int dims[3] = {0, 0, 0};
        int num_dims = 2;

        dims[0] = n_mesh_info[topo_name]["elements/i"].value();
        dims[1] = n_mesh_info[topo_name]["elements/j"].value();

        if (n_mesh_info[topo_name]["elements"].has_path("k"))
        {
            num_dims = 3;
            dims[2] = n_mesh_info[topo_name]["elements/k"].value();
        }

        if (centering == DB_NODECENT)
        {
            dims[0] += 1;
            dims[1] += 1;
            dims[2] += 1;
        }

        // save the var type
        var_type = DB_QUADVAR;

        silo_error = DBPutQuadvar(dbfile, // Database file pointer
                                  detail::sanitize_silo_varname(var_name).c_str(), // variable name
                                  safe_meshname.c_str(), // mesh name
                                  nvars, // number of variable components
                                  comp_name_ptrs.data(), // variable component names
                                  comp_vals_ptrs.data(), // the data values
                                  dims, // the dimensions of the data
                                  num_dims, // number of dimensions
                                  NULL, // mixed data arrays
                                  0, // length of mixed data arrays
                                  vals_type, // Datatype of the variable
                                  centering, // centering (nodal or zonal)
                                  NULL); // optlist
    }
    else if (mesh_type == "points") 
    {
        // save the var type
        var_type = DB_POINTVAR;

        silo_error = DBPutPointvar(dbfile, // Database file pointer.
                                   detail::sanitize_silo_varname(var_name).c_str(),  // variable name
                                   safe_meshname.c_str(), // mesh name
                                   nvars, // number of variable components
                                   comp_vals_ptrs.data(), // data values
                                   num_pts, // Number of elements (points)
                                   vals_type, // Datatype of the variable
                                   NULL); // optlist
    }
    else
    {
        CONDUIT_ERROR("only DBPutQuadvar + DBPutUcdvar + DBPutPointvar var are supported");
    }

    CONDUIT_CHECK_SILO_ERROR(silo_error,
                             " after creating field " << var_name);

    // bookkeeping
    if (! local_type_info["vars"].has_child(var_name))
    {
        local_type_info["vars"][var_name]["domain_ids"].set(DataType::index_t(local_num_domains));
        index_t_array domain_ids = local_type_info["vars"][var_name]["domain_ids"].value();
        domain_ids.fill(-1); // we want missing domains to have -1 and not 0 to avoid confusion
        local_type_info["vars"][var_name]["types"].set(DataType::index_t(local_num_domains));
    }
    index_t_array domain_ids = local_type_info["vars"][var_name]["domain_ids"].value();
    domain_ids[local_domain_index] = global_domain_id;
    index_t_array var_types = local_type_info["vars"][var_name]["types"].value();
    var_types[local_domain_index] = var_type;
}

//---------------------------------------------------------------------------//
int
assign_coords_ptrs(void *coords_ptrs[3],
                   int ndims,
                   conduit::Node &n_coords_compact,
                   char const * const coordnames[])
{
    DataType dtype = n_coords_compact[coordnames[0]].dtype();
    CONDUIT_ASSERT(dtype.id() == n_coords_compact[coordnames[1]].dtype().id(),
                   "all coordinate arrays must have same type, got " << dtype.to_string()
                    << " and " << n_coords_compact[coordnames[1]].dtype().to_string());
    if (ndims == 3)
    {
        CONDUIT_ASSERT(dtype.id() == n_coords_compact[coordnames[2]].dtype().id(),
                       "all coordinate arrays must have same type, got " << dtype.to_string()
                        << " and " << n_coords_compact[coordnames[2]].dtype().to_string());
        coords_ptrs[2] = n_coords_compact[coordnames[2]].element_ptr(0);
    }
    coords_ptrs[0] = n_coords_compact[coordnames[0]].element_ptr(0);
    coords_ptrs[1] = n_coords_compact[coordnames[1]].element_ptr(0);

    if (dtype.is_float())
    {
        return DB_FLOAT;
    }
    else if (dtype.is_double())
    {
        return DB_DOUBLE;
    }
    else
    {
        CONDUIT_ERROR("coords data type not implemented, found "
                      << dtype.name());
        return -1;
    }
}

//---------------------------------------------------------------------------//
// compaction is necessary to support ragged arrays
void compact_coords(const Node &n_coords,
                    Node &n_coords_compact)
{
    // are we already compact?
    if (n_coords["values"].dtype().is_compact())
    {
        n_coords_compact.set_external(n_coords["values"]);
    }
    else
    {
        auto val_itr = n_coords["values"].children();
        while (val_itr.has_next())
        {
            const Node &n_val = val_itr.next();
            std::string label = val_itr.name();
            // is this piece already compact?
            if (n_coords["values"][label].dtype().is_compact())
            {
                n_coords_compact[label].set_external(n_val);
            }
            else
            {
                n_val.compact_to(n_coords_compact[label]);
            }
        }
    }
}

//---------------------------------------------------------------------------//
// calculates and checks the number of points for an explicit coordset
int get_explicit_num_pts(const Node &n_vals)
{
    auto val_itr = n_vals.children();
    if (!val_itr.has_next())
    {
        CONDUIT_ERROR("Cannot count the number of points because no points given.");
    }
    const Node &n_first_val = val_itr.next();
    int num_pts = n_first_val.dtype().number_of_elements();
    while (val_itr.has_next())
    {
        const Node &n_val = val_itr.next();
        if (num_pts != n_val.dtype().number_of_elements())
        {
            CONDUIT_ERROR("Number of points in explicit coordset does not match between dimensions.");
        }
    }
    return num_pts;
}

//---------------------------------------------------------------------------//
void silo_write_ucd_zonelist(DBfile *dbfile,
                             const std::string &topo_name,
                             const Node &n_topo,
                             Node &n_mesh_info) 
{
    Node ucd_zlist;

    index_t num_shapes = 1;
    ucd_zlist["shapetype"].set(DataType::c_int(1));
    ucd_zlist["shapesize"].set(DataType::c_int(1));
    ucd_zlist["shapecnt"].set(DataType::c_int(1));

    const Node &n_elements = n_topo["elements"];
    std::string coordset_name = n_topo["coordset"].as_string();

    if (!n_elements.dtype().is_object())
    {
        CONDUIT_ERROR("Invalid elements for 'unstructured' case");
    }

    int *shapetype = ucd_zlist["shapetype"].value();
    int *shapesize = ucd_zlist["shapesize"].value();
    int *shapecnt = ucd_zlist["shapecnt"].value();

    int total_num_elems = 0;

    Node n_conn;

    std::string topo_shape = n_elements["shape"].as_string();

    Node n_mesh_conn;
    
    // We are using the vtk ordering for our wedges; silo wedges (prisms)
    // expect a different ordering. Thus before we output to silo, we must
    // change the ordering of each of our wedges.
    if (topo_shape == "wedge")
    {
        n_mesh_conn.set(n_elements["connectivity"]);
        DataType dtype = n_mesh_conn.dtype();
        // swizzle the connectivity
        if (dtype.is_uint64())
        {
            conduit_wedge_connectivity_to_silo<uint64>(n_mesh_conn);
        }
        else if (dtype.is_uint32())
        {
            conduit_wedge_connectivity_to_silo<uint32>(n_mesh_conn);
        }
        else if (dtype.is_int64())
        {
            conduit_wedge_connectivity_to_silo<int64>(n_mesh_conn);
        }
        else if (dtype.is_int32())
        {
            conduit_wedge_connectivity_to_silo<int32>(n_mesh_conn);
        }
        else
        {
            CONDUIT_ERROR("Unsupported connectivity type in " << dtype.to_yaml());
        }
    }
    else
    {
        n_mesh_conn.set_external(n_elements["connectivity"]);
    }

    // convert to compact ints ...
    n_mesh_conn.compact_to(n_conn);

    if (topo_shape == "quad")
    {
        int num_elems = n_mesh_conn.dtype().number_of_elements() / 4;
        shapetype[0] = DB_ZONETYPE_QUAD;
        shapesize[0] = 4;
        shapecnt[0] = num_elems;
        total_num_elems += num_elems;

    }
    else if (topo_shape == "tri")
    {
        int num_elems = n_mesh_conn.dtype().number_of_elements() / 3;
        shapetype[0] = DB_ZONETYPE_TRIANGLE;
        shapesize[0] = 3;
        shapecnt[0] = num_elems;
        total_num_elems += num_elems;
    }
    else if (topo_shape == "hex")
    {
        int num_elems = n_mesh_conn.dtype().number_of_elements() / 8;
        shapetype[0] = DB_ZONETYPE_HEX;
        shapesize[0] = 8;
        shapecnt[0] = num_elems;
        total_num_elems += num_elems;

    }
    else if (topo_shape == "tet")
    {
        int num_elems = n_mesh_conn.dtype().number_of_elements() / 4;
        shapetype[0] = DB_ZONETYPE_TET;
        shapesize[0] = 4;
        shapecnt[0] = num_elems;
        total_num_elems += num_elems;
    }
    else if( topo_shape == "wedge")
    {
        int num_elems    = n_mesh_conn.dtype().number_of_elements() / 6;
        shapetype[0] = DB_ZONETYPE_PRISM;
        shapesize[0] = 6;
        shapecnt[0]  = num_elems;
        total_num_elems  += num_elems;
    }
    else if( topo_shape == "pyramid")
    {
        int num_elems    = n_mesh_conn.dtype().number_of_elements() / 5;
        shapetype[0] = DB_ZONETYPE_PYRAMID;
        shapesize[0] = 5;
        shapecnt[0]  = num_elems;
        total_num_elems  += num_elems;
    }
    else if (topo_shape == "line")
    {
        int num_elems = n_mesh_conn.dtype().number_of_elements() / 2;
        shapetype[0] = DB_ZONETYPE_BEAM;
        shapesize[0] = 2;
        shapecnt[0] = num_elems;
        total_num_elems += num_elems;
    }
    else
    {
        // TODO add polygons and polyhedra and mixed
        CONDUIT_ERROR("Unsupported topo shape " << topo_shape);
    }

    // Final Compaction
    Node n_conn_final;
    n_conn.compact_to(n_conn_final);

    int conn_len = n_conn_final.total_bytes_compact() / sizeof(int);
    int *conn_ptr = (int *)n_conn_final.data_ptr();

    n_mesh_info[topo_name]["num_elems"].set(total_num_elems);

    std::string zlist_name = topo_name + "_connectivity";

    int silo_error =
        DBPutZonelist2(dbfile,             // silo file
                       detail::sanitize_silo_varname(zlist_name).c_str(), // silo obj name
                       total_num_elems,    // number of elements
                       2,                  // spatial dims
                       conn_ptr,           // connectivity array
                       conn_len,           // len of connectivity array
                       0,                  // base offset
                       0,                  // # ghosts low
                       0,                  // # ghosts high
                       shapetype,          // list of shapes ids
                       shapesize,          // number of points per shape id
                       shapecnt, // number of elements each shape id is used for
                       num_shapes, // number of shapes ids
                       NULL);      // optlist

    CONDUIT_CHECK_SILO_ERROR(silo_error, " after saving ucd quad topology");
}

//---------------------------------------------------------------------------//
void silo_write_quad_rect_mesh(DBfile *dbfile,
                               const std::string &topo_name,
                               const Node &n_topo,
                               const Node &n_coords,
                               DBoptlist *state_optlist,
                               const int ndims,
                               char const * const coordnames[],
                               const bool overlink,
                               Node &n_mesh_info) 
{
    Node n_coords_compact;
    compact_coords(n_coords, n_coords_compact);

    int pts_dims[3];
    pts_dims[0] = n_coords_compact[coordnames[0]].dtype().number_of_elements();
    pts_dims[1] = n_coords_compact[coordnames[1]].dtype().number_of_elements();
    pts_dims[2] = 1;

    int num_pts = pts_dims[0] * pts_dims[1];
    int num_elems = (pts_dims[0] - 1) * (pts_dims[1] - 1);
    if (ndims == 3)
    {
        pts_dims[2] = n_coords_compact[coordnames[2]].dtype().number_of_elements();
        num_pts *= pts_dims[2];
        num_elems *= (pts_dims[2] - 1);
        n_mesh_info[topo_name]["elements/k"] = pts_dims[2] - 1;
    }

    n_mesh_info[topo_name]["num_pts"].set(num_pts);
    n_mesh_info[topo_name]["num_elems"].set(num_elems);
    n_mesh_info[topo_name]["elements/i"] = pts_dims[0] - 1;
    n_mesh_info[topo_name]["elements/j"] = pts_dims[1] - 1;

    void *coords_ptrs[3] = {NULL, NULL, NULL};
    int coords_dtype = assign_coords_ptrs(coords_ptrs,
                                          ndims,
                                          n_coords_compact,
                                          coordnames);

    int base_index[] = {0,0,0};
    if (n_topo.has_path("elements/origin"))
    {
        base_index[0] = n_topo["elements/origin/i"].as_int();
        base_index[1] = n_topo["elements/origin/j"].as_int();
        base_index[2] = n_topo["elements/origin/k"].as_int();

        CONDUIT_CHECK_SILO_ERROR( DBAddOption(state_optlist,
                                              DBOPT_BASEINDEX,
                                              base_index),
                                  "Error adding option");
    }

    std::string safe_meshname;
    if (overlink)
    {
        safe_meshname = "MESH";
    }
    else
    {
        safe_meshname = detail::sanitize_silo_varname(topo_name);
    }

    int silo_error =
        DBPutQuadmesh(dbfile,                      // silo file ptr
                      safe_meshname.c_str(), // mesh name
                      coordnames, // coord names
                      coords_ptrs,                 // coords values
                      pts_dims,                    // dims vals
                      ndims,                       // number of dims
                      coords_dtype,                // type of data array
                      DB_COLLINEAR,   // DB_COLLINEAR or DB_NONCOLLINEAR
                      state_optlist); // opt list

    CONDUIT_CHECK_SILO_ERROR(silo_error, " DBPutQuadmesh");
}

//---------------------------------------------------------------------------//
void silo_write_ucd_mesh(DBfile *dbfile,
                         const std::string &topo_name,
                         DBoptlist *optlist,
                         const int ndims,
                         const int num_pts,
                         char const * const coordnames[],
                         const void *coords_ptrs,
                         const int coords_dtype,
                         const bool overlink,
                         Node &n_mesh_info)
{
    int num_elems = n_mesh_info[topo_name]["num_elems"].value();

    // TODO there is a different approach for polyhedral zone lists
    std::string zlist_name = topo_name + "_connectivity";

    std::string safe_meshname;
    if (overlink)
    {
        safe_meshname = "MESH";
    }
    else
    {
        safe_meshname = detail::sanitize_silo_varname(topo_name);
    }

    int silo_error = DBPutUcdmesh(dbfile,                      // silo file ptr
                                  safe_meshname.c_str(), // mesh name
                                  ndims,                       // number of dims
                                  coordnames, // coord names
                                  coords_ptrs,                 // coords values
                                  num_pts,            // number of points
                                  num_elems,          // number of elements
                                  detail::sanitize_silo_varname(zlist_name).c_str(), // zone list name
                                  NULL,               // face list names
                                  coords_dtype,       // type of data array
                                  optlist);     // opt list

    CONDUIT_CHECK_SILO_ERROR(silo_error, " DBPutUcdmesh");
}

//---------------------------------------------------------------------------//
void silo_write_structured_mesh(DBfile *dbfile,
                                const std::string &topo_name,
                                const Node &n_topo,
                                DBoptlist *optlist,
                                const int ndims,
                                char const * const coordnames[],
                                const void *coords_ptrs,
                                const int coords_dtype,
                                const bool overlink,
                                Node &n_mesh_info) 
{
    int ele_dims[3];
    ele_dims[0] = n_topo["elements/dims/i"].to_value();
    ele_dims[1] = n_topo["elements/dims/j"].to_value();
    ele_dims[2] = 0;

    index_t num_elems = ele_dims[0] * ele_dims[1];

    if (n_topo["elements/dims"].has_path("k"))
    {
        ele_dims[2] = n_topo["elements/dims/k"].to_value();
        num_elems *= ele_dims[2];
    }

    // silo needs the node dims to define a structured grid
    int pts_dims[3];
    pts_dims[0] = ele_dims[0] + 1;
    pts_dims[1] = ele_dims[1] + 1;
    pts_dims[2] = 1;

    n_mesh_info[topo_name]["num_elems"].set(num_elems);
    n_mesh_info[topo_name]["elements/i"] = ele_dims[0];
    n_mesh_info[topo_name]["elements/j"] = ele_dims[1];

    if (ndims == 3)
    {
        n_mesh_info[topo_name]["elements/k"] = ele_dims[2];
        pts_dims[2] = ele_dims[2] + 1;
    }

    int base_index[] = {0,0,0};
    if (n_topo.has_path("elements/origin"))
    {
        base_index[0] = n_topo["elements/origin/i"].as_int();
        base_index[1] = n_topo["elements/origin/j"].as_int();
        if (ndims == 3)
        {
            base_index[2] = n_topo["elements/origin/k"].as_int();
        }
        

        CONDUIT_CHECK_SILO_ERROR( DBAddOption(optlist,
                                              DBOPT_BASEINDEX,
                                              base_index),
                                  "Error adding option");
    }

    std::string safe_meshname;
    if (overlink)
    {
        safe_meshname = "MESH";
    }
    else
    {
        safe_meshname = detail::sanitize_silo_varname(topo_name);
    }

    int silo_error =
        DBPutQuadmesh(dbfile,                // silo file ptr
                      safe_meshname.c_str(), // mesh name
                      coordnames,            // coord names
                      coords_ptrs,           // coords values
                      pts_dims,              // dims vals
                      ndims,                 // number of dims
                      coords_dtype,          // type of data array
                      DB_NONCOLLINEAR,       // DB_COLLINEAR (rectilinear grid) or DB_NONCOLLINEAR (structured grid)
                      optlist);              // opt list

    CONDUIT_CHECK_SILO_ERROR(silo_error, " DBPutQuadmesh");
}

//---------------------------------------------------------------------------//
void silo_write_pointmesh(DBfile *dbfile,
                          const std::string &topo_name,
                          DBoptlist *optlist,
                          const int ndims,
                          const int num_pts,
                          const void *coords_ptrs,
                          const int coords_dtype,
                          const bool overlink,
                          Node &n_mesh_info)
{
    n_mesh_info[topo_name]["num_elems"].set(num_pts);

    std::string safe_meshname;
    if (overlink)
    {
        safe_meshname = "MESH";
    }
    else
    {
        safe_meshname = detail::sanitize_silo_varname(topo_name);
    }

    int silo_error = DBPutPointmesh(dbfile,                // silo file ptr
                                    safe_meshname.c_str(), // mesh name
                                    ndims,                 // num_dims
                                    coords_ptrs,           // coords values
                                    num_pts,               // num eles = num pts
                                    coords_dtype,          // type of data array
                                    optlist);              // opt list

    CONDUIT_CHECK_SILO_ERROR(silo_error, " after saving DBPutPointmesh");
}

//---------------------------------------------------------------------------//
void silo_write_topo(const Node &n,
                     const std::string &topo_name,
                     Node &n_mesh_info,
                     const bool overlink,
                     const int local_num_domains,
                     const int local_domain_index,
                     const int global_domain_id,
                     Node &local_type_info,
                     DBfile *dbfile)
{
    const Node &n_topo = n["topologies"][topo_name];
    std::string topo_type = n_topo["type"].as_string();

    n_mesh_info[topo_name]["type"].set(topo_type);

    bool unstructured_points = false;
    if (topo_type == "unstructured")
    {
        std::string ele_shape = n_topo["elements/shape"].as_string();
        if (ele_shape != "point")
        {
            // we need a zone list for a ucd mesh
            silo_write_ucd_zonelist(dbfile,
                                    topo_name,
                                    n_topo,
                                    n_mesh_info);
        }
        else
        {
            unstructured_points = true;
            topo_type = "points";
            n_mesh_info[topo_name]["type"].set(topo_type);
        }
    }

    // make sure we have coordsets

    if (!n.has_path("coordsets"))
    {
        CONDUIT_ERROR("mesh missing: coordsets");
    }

    // get this topo's coordset name
    std::string coordset_name = n_topo["coordset"].as_string();

    n_mesh_info[topo_name]["coordset"].set(coordset_name);

    // obtain the coordset with the name
    if (!n["coordsets"].has_path(coordset_name))
    {
        CONDUIT_ERROR("mesh is missing coordset named "
                      << coordset_name << " for topology named "
                      << topo_name);
    }

    const Node &n_coords = n["coordsets"][coordset_name];

    // check dims
    int ndims = conduit::blueprint::mesh::utils::coordset::dims(n_coords);
    CONDUIT_ASSERT(2 <= ndims && ndims <= 3, "Dimension count not accepted: " << ndims);

    // get coordsys info
    std::string coordsys = conduit::blueprint::mesh::utils::coordset::coordsys(n_coords);
    int silo_coordsys_type = get_coordset_silo_type(coordsys);
    std::vector<const char *> silo_coordset_axis_labels = get_coordset_axis_labels(silo_coordsys_type);
    // create optlist
    detail::SiloObjectWrapperCheckError<DBoptlist, decltype(&DBFreeOptlist)> optlist{
        DBMakeOptlist(1),
        &DBFreeOptlist,
        "Error freeing state optlist."};
    if (!optlist.getSiloObject())
    {
        CONDUIT_ERROR("Error creating optlist");
    }
    CONDUIT_CHECK_SILO_ERROR( DBAddOption(optlist.getSiloObject(),
                                          DBOPT_COORDSYS,
                                          &silo_coordsys_type),
                             "error adding coordsys option");

    int mesh_type;

    if (topo_type == "unstructured" ||
        topo_type == "structured" ||
        topo_type == "points")
    {
        // check for explicit coords
        if (n_coords["type"].as_string() != "explicit")
        {
            CONDUIT_ERROR("Expected an explicit coordset when writing " << topo_type 
                          << " mesh " << topo_name);
        }

        // compact arrays
        Node n_coords_compact, new_coords;
        
        // here we handle the unstructured points case:
        if (unstructured_points)
        {
            // we need to change the coords to only have the ones that are used
            int_accessor conn = n_topo["elements"]["connectivity"].value();
            const int num_elem = conn.number_of_elements();
            
            new_coords["values"][silo_coordset_axis_labels[0]].set(DataType::float64(num_elem));
            new_coords["values"][silo_coordset_axis_labels[1]].set(DataType::float64(num_elem));
            if (ndims == 3)
            {
                new_coords["values"][silo_coordset_axis_labels[2]].set(DataType::float64(num_elem));
            }

            for (int conn_index = 0; conn_index < num_elem; conn_index ++)
            {
                int old_coord_index = conn[conn_index];

                double_array new_x_coords = new_coords["values"][silo_coordset_axis_labels[0]].value();
                double_accessor old_x_coords = n_coords["values"][silo_coordset_axis_labels[0]].value();
                new_x_coords[conn_index] = old_x_coords[old_coord_index];
                
                double_array new_y_coords = new_coords["values"][silo_coordset_axis_labels[1]].value();
                double_accessor old_y_coords = n_coords["values"][silo_coordset_axis_labels[1]].value();
                new_y_coords[conn_index] = old_y_coords[old_coord_index];

                if (ndims == 3)
                {
                    double_array new_z_coords = new_coords["values"][silo_coordset_axis_labels[2]].value();
                    double_accessor old_z_coords = n_coords["values"][silo_coordset_axis_labels[2]].value();
                    new_z_coords[conn_index] = old_z_coords[old_coord_index];
                }
            }

            compact_coords(new_coords, n_coords_compact);
        }
        else
        {
            compact_coords(n_coords, n_coords_compact);
        }

        // get num pts
        const int num_pts = get_explicit_num_pts(n_coords_compact);
        n_mesh_info[topo_name]["num_pts"].set(num_pts);

        // get coords ptrs
        void *coords_ptrs[3] = {NULL, NULL, NULL};
        int coords_dtype = assign_coords_ptrs(coords_ptrs,
                                              ndims,
                                              n_coords_compact,
                                              silo_coordset_axis_labels.data());

        if (topo_type == "unstructured")
        {
            mesh_type = DB_UCDMESH;
            silo_write_ucd_mesh(dbfile, topo_name,
                                optlist.getSiloObject(), 
                                ndims, num_pts, silo_coordset_axis_labels.data(),
                                coords_ptrs, coords_dtype,
                                overlink, n_mesh_info);
        }
        else if (topo_type == "structured")
        {
            mesh_type = DB_QUADMESH;
            silo_write_structured_mesh(dbfile, topo_name, n_topo,
                                       optlist.getSiloObject(), 
                                       ndims, silo_coordset_axis_labels.data(),
                                       coords_ptrs, coords_dtype,
                                       overlink, n_mesh_info);
        }
        else if (topo_type == "points")
        {
            mesh_type = DB_POINTMESH;
            silo_write_pointmesh(dbfile, topo_name,
                                 optlist.getSiloObject(), 
                                 ndims, num_pts,
                                 coords_ptrs, coords_dtype,
                                 overlink, n_mesh_info);
        }
    }
    else if (topo_type == "rectilinear")
    {
        mesh_type = DB_QUADMESH;
        silo_write_quad_rect_mesh(dbfile, topo_name,
                                  n_topo, n_coords,
                                  optlist.getSiloObject(), 
                                  ndims, silo_coordset_axis_labels.data(),
                                  overlink, n_mesh_info);
    }
    else if (topo_type == "uniform")
    {
        mesh_type = DB_QUADMESH;
        // silo doesn't have a direct path for a uniform mesh
        // we need to convert its implicit uniform coords to
        // implicit rectilinear coords

        Node n_rect;
        Node &n_rect_coords = n_rect["coordsets"][coordset_name];
        Node &n_rect_topo = n_rect["topologies"][topo_name];
        conduit::blueprint::mesh::topology::uniform::to_rectilinear(
            n_topo, n_rect_topo, n_rect_coords);

        silo_write_quad_rect_mesh(dbfile, topo_name,
                                  n_rect_topo, n_rect_coords,
                                  optlist.getSiloObject(), 
                                  ndims, silo_coordset_axis_labels.data(),
                                  overlink, n_mesh_info);
    }
    else
    {
        CONDUIT_ERROR("Unknown topo type in " << topo_type);
    }

    // bookkeeping
    if (! local_type_info["meshes"].has_child(topo_name))
    {
        local_type_info["meshes"][topo_name]["domain_ids"].set(DataType::index_t(local_num_domains));
        index_t_array domain_ids = local_type_info["meshes"][topo_name]["domain_ids"].value();
        domain_ids.fill(-1); // we want missing domains to have -1 and not 0 to avoid confusion
        local_type_info["meshes"][topo_name]["types"].set(DataType::index_t(local_num_domains));
    }
    index_t_array domain_ids = local_type_info["meshes"][topo_name]["domain_ids"].value();
    domain_ids[local_domain_index] = global_domain_id;
    index_t_array topo_types = local_type_info["meshes"][topo_name]["types"].value();
    topo_types[local_domain_index] = mesh_type;
}

//---------------------------------------------------------------------------//
void silo_mesh_write(const Node &n, 
                     DBfile *dbfile,
                     const std::string &silo_obj_path,
                     const std::string &ovl_topo_name,
                     const int local_num_domains,
                     const int local_domain_index,
                     const int global_domain_id,
                     Node &local_type_info,
                     const bool overlink)
{
    int silo_error = 0;
    char silo_prev_dir[256];
    if (!silo_obj_path.empty())
    {
        silo_error += DBGetDir(dbfile, silo_prev_dir);

        std::string dir; 
        std::stringstream ss(silo_obj_path);
        while (getline(ss, dir, '/'))
        {
            DBMkDir(dbfile, dir.c_str()); // if this fails we want to keep going
            silo_error += DBSetDir(dbfile, dir.c_str());
        }
        CONDUIT_CHECK_SILO_ERROR(silo_error,
                                 " failed to make silo directory: "
                                 << silo_obj_path);
    }

    Node n_mesh_info;

    if (overlink)
    {
        if (n["topologies"].has_child(ovl_topo_name))
        {
            // we choose one topo to write out: ovl_topo_name
            silo_write_topo(n,
                            ovl_topo_name,
                            n_mesh_info,
                            overlink,
                            local_num_domains,
                            local_domain_index,
                            global_domain_id,
                            local_type_info,
                            dbfile);
        }
    }
    else
    {
        // we write out all topos
        auto topo_itr = n["topologies"].children();
        while (topo_itr.has_next())
        {
            topo_itr.next();
            std::string topo_name = topo_itr.name();
            silo_write_topo(n,
                            topo_name,
                            n_mesh_info,
                            overlink,
                            local_num_domains,
                            local_domain_index,
                            global_domain_id,
                            local_type_info,
                            dbfile);
        }
    }

    if (n.has_path("fields")) 
    {
        auto itr = n["fields"].children();
        while (itr.has_next())
        {
            const Node &n_var = itr.next();
            std::string var_name = itr.name();
            if (! overlink || n_var["topology"].as_string() == ovl_topo_name)
            {
                silo_write_field(dbfile,
                                 var_name,
                                 n_var,
                                 overlink,
                                 local_num_domains,
                                 local_domain_index,
                                 global_domain_id,
                                 local_type_info,
                                 n_mesh_info);
            }
        }
    }

    if (!silo_obj_path.empty()) 
    {
        silo_error = DBSetDir(dbfile, silo_prev_dir);
        CONDUIT_CHECK_SILO_ERROR(silo_error,
                                 " changing silo directory to previous path");
    }
}

//-----------------------------------------------------------------------------
void
generate_silo_names(const Node &n_mesh_state,
                    const std::string &silo_path,
                    const std::string &safe_name,
                    const int num_files,
                    const int global_num_domains,
                    const bool root_only,
                    const Node &types_for_mesh_or_var,
                    const int default_type,
                    std::vector<std::string> &name_strings,
                    std::vector<int> &types)
{
    int_accessor stored_types = types_for_mesh_or_var.value();
    for (index_t i = 0; i < global_num_domains; i ++)
    {
        std::string silo_name;

        // determine which domain
        index_t d;
        if (n_mesh_state.has_path("partition_map/domain"))
        {
            index_t_array part_map_domain_vals = n_mesh_state["partition_map"]["domain"].value();
            d = part_map_domain_vals[i];
        }
        else
        {
            d = i;
        }

        // we are missing a domain
        if (stored_types[d] == -1)
        {
            silo_name = "EMPTY";

            types.push_back(default_type);
        }
        else
        {
            // we have three cases, just as we had in write_mesh
            // we don't want to be making any choices here, just using 
            // what was already decided in write_mesh

            // single file case
            if (root_only)
            {
                if (global_num_domains == 1)
                {
                    silo_name = conduit_fmt::format(silo_path, safe_name);
                }
                else
                {
                    silo_name = conduit_fmt::format(silo_path, d, safe_name);
                }
            }
            // num domains == num files case
            else if (global_num_domains == num_files)
            {
                silo_name = conduit_fmt::format(silo_path, d, safe_name);
            }
            // m to n case
            else
            {
                // determine which file
                index_t f;
                if (n_mesh_state.has_path("partition_map/file"))
                {
                    index_t_array part_map_file_vals = n_mesh_state["partition_map"]["file"].value();
                    f = part_map_file_vals[i];
                }
                else
                {
                    f = i;
                }

                silo_name = conduit_fmt::format(silo_path, f, d, safe_name);
            }

            types.push_back(stored_types[d]);
        }

        // we create the silo names
        name_strings.push_back(silo_name);
    }
}

//-----------------------------------------------------------------------------
void write_multimesh(DBfile *dbfile,
                     const Node &n_mesh,
                     const std::string &topo_name,
                     const conduit::Node &root,
                     const int global_num_domains,
                     const std::string &multimesh_name,
                     const Node &root_type_info_meshes,
                     const bool overlink)
{
    const int num_files = root["number_of_files"].as_int32();
    const bool root_only = root["file_style"].as_string() == "root_only";

    const Node &n_topo = n_mesh["topologies"][topo_name];
    std::string topo_type = n_topo["type"].as_string();

    std::string safe_meshname;
    if (overlink)
    {
        safe_meshname = "MESH";
    }
    else
    {
        safe_meshname = detail::sanitize_silo_varname(topo_name);
    }

    std::string silo_path = root["silo_path"].as_string();

    std::vector<std::string> domain_name_strings;
    
    std::vector<int> mesh_types;
    generate_silo_names(n_mesh["state"],
                        silo_path,
                        safe_meshname,
                        num_files,
                        global_num_domains,
                        root_only,
                        root_type_info_meshes[topo_name],
                        DB_QUADMESH, // the default if we have an empty domain
                        domain_name_strings,
                        mesh_types);

    // package up char ptrs for silo
    std::vector<const char *> domain_name_ptrs;
    for (index_t i = 0; i < domain_name_strings.size(); i ++)
    {
        domain_name_ptrs.push_back(domain_name_strings[i].c_str());
    }

    // create state optlist
    detail::SiloObjectWrapperCheckError<DBoptlist, decltype(&DBFreeOptlist)> state_optlist{
        DBMakeOptlist(3), 
        &DBFreeOptlist,
        "Error freeing state optlist."};
    if (!state_optlist.getSiloObject())
    {
        CONDUIT_ERROR("Error creating state optlist");
    }

    int cycle;
    float ftime;
    double dtime;
    if (n_mesh.has_child("state"))
    {
        int silo_error = 0;
        const Node &n_state = n_mesh["state"];
        if (n_state.has_child("cycle"))
        {
            cycle = n_state["cycle"].to_int();
            silo_error += DBAddOption(state_optlist.getSiloObject(),
                                      DBOPT_CYCLE,
                                      &cycle);
        }
        if (n_state.has_child("time"))
        {
            ftime = n_state["time"].to_float();
            silo_error += DBAddOption(state_optlist.getSiloObject(),
                                      DBOPT_TIME,
                                      &ftime);
            dtime = n_state["time"].to_double();
            silo_error += DBAddOption(state_optlist.getSiloObject(),
                                      DBOPT_DTIME,
                                      &dtime);
        }
        CONDUIT_CHECK_SILO_ERROR(silo_error,
                                 " creating state optlist (time, cycle) ");
    }

    // TODO add dboptions for nameschemes

    CONDUIT_CHECK_SILO_ERROR(
        DBPutMultimesh(
            dbfile,
            detail::sanitize_silo_varname(multimesh_name).c_str(),
            global_num_domains,
            domain_name_ptrs.data(),
            mesh_types.data(),
            state_optlist.getSiloObject()),
        "Error putting multimesh corresponding to topo: " << topo_name);
}

//-----------------------------------------------------------------------------
void write_multimeshes(DBfile *dbfile,
                       const std::string &opts_out_mesh_name,
                       const std::string &ovl_topo_name,
                       const Node &root,
                       const Node &root_type_info_meshes,
                       const bool overlink)
{
    const int global_num_domains = root["number_of_domains"].as_int32();
    const Node &n_mesh = root["blueprint_index"][opts_out_mesh_name];

    // these should be the same b/c the num domains the bp index was given
    // was global_num_domains
    if (global_num_domains != n_mesh["state/number_of_domains"].as_int64())
    {
        CONDUIT_ERROR("Domain count mismatch");
    }

    // write only the chosen mesh for overlink case
    if (overlink)
    {
        write_multimesh(dbfile,
                        n_mesh,
                        ovl_topo_name,
                        root,
                        global_num_domains,
                        opts_out_mesh_name, // "MMESH"
                        root_type_info_meshes,
                        overlink);
    }
    // write all meshes for nonoverlink case
    else
    {
        auto topo_itr = n_mesh["topologies"].children();
        while (topo_itr.has_next())
        {
            topo_itr.next();
            std::string topo_name = topo_itr.name();
            std::string multimesh_name = opts_out_mesh_name + "_" + topo_name;
            write_multimesh(dbfile,
                            n_mesh,
                            topo_name,
                            root,
                            global_num_domains,
                            multimesh_name,
                            root_type_info_meshes,
                            overlink);
        }
    }
}

//-----------------------------------------------------------------------------
// TODO support multimaterial write
// void
// write_multimaterial(DBfile *root,
//                     const std::string &mmat_name,
//                     const std::string &mmesh_name,
//                     std::vector<std::string> mat_domains) 
// {
//     std::vector<const char *> domain_name_ptrs;
//     detail::SiloObjectWrapperCheckError<DBoptlist, decltype(&DBFreeOptlist)> optlist{
//             DBMakeOptlist(1),
//             &DBFreeOptlist,
//             "Error freeing optlist."};
//     if (!optlist.getSiloObject())
//     {
//         CONDUIT_ERROR("Error creating options");
//     }

//     // have to const_cast because converting to void *
//     CONDUIT_CHECK_SILO_ERROR( DBAddOption(optlist.getSiloObject(),
//                                           DBOPT_MMESH_NAME,
//                                           const_cast<char *>(mmesh_name.c_str())),
//                               "Error creating options for putting multimat");
    
//     for (auto domain : mat_domains) 
//     {
//         domain_name_ptrs.push_back(domain.c_str());
//     }

//     CONDUIT_CHECK_SILO_ERROR( DBPutMultimat(root,
//                                             detail::sanitize_silo_varname(mmat_name).c_str(),
//                                             mat_domains.size(),
//                                             domain_name_ptrs.data(),
//                                             optlist.getSiloObject()),
//                               "Error putting multimaterial");
// }

//-----------------------------------------------------------------------------
void
write_multivars(DBfile *dbfile,
                const std::string &opts_mesh_name,
                const std::string &ovl_topo_name,
                const Node &root,
                const Node &root_type_info_vars,
                const bool overlink)
{
    const int num_files = root["number_of_files"].as_int32();
    const int global_num_domains = root["number_of_domains"].as_int32();
    const Node &n_mesh = root["blueprint_index"][opts_mesh_name];
    const bool root_only = root["file_style"].as_string() == "root_only";

    // these should be the same b/c the num domains the bp index was given
    // was global_num_domains
    if (global_num_domains != n_mesh["state/number_of_domains"].as_int64())
    {
        CONDUIT_ERROR("Domain count mismatch");
    }

    if (n_mesh.has_child("fields"))
    {
        auto field_itr = n_mesh["fields"].children();
        while (field_itr.has_next())
        {
            const Node &n_var = field_itr.next();
            std::string var_name = field_itr.name();

            std::string linked_topo_name = n_var["topology"].as_string();

            if (! overlink || linked_topo_name == ovl_topo_name)
            {
                std::string linked_topo_type = n_mesh["topologies"][linked_topo_name]["type"].as_string();

                std::string safe_varname = detail::sanitize_silo_varname(var_name); 
                std::string safe_linked_topo_name = detail::sanitize_silo_varname(linked_topo_name);
                std::string silo_path = root["silo_path"].as_string();

                std::vector<std::string> var_name_strings;
                std::vector<int> var_types;
                generate_silo_names(n_mesh["state"],
                                    silo_path,
                                    safe_varname,
                                    num_files,
                                    global_num_domains,
                                    root_only,
                                    root_type_info_vars[var_name],
                                    DB_QUADVAR, // the default if we have an empty domain
                                    var_name_strings,
                                    var_types);

                // package up char ptrs for silo
                std::vector<const char *> var_name_ptrs;
                for (index_t i = 0; i < var_name_strings.size(); i ++)
                {
                    var_name_ptrs.push_back(var_name_strings[i].c_str());
                }

                detail::SiloObjectWrapperCheckError<DBoptlist, decltype(&DBFreeOptlist)> optlist{
                    DBMakeOptlist(1),
                    &DBFreeOptlist,
                    "Error freeing optlist."};
                if (!optlist.getSiloObject())
                    CONDUIT_ERROR("Error creating options");

                std::string multimesh_name, multivar_name;
                if (overlink)
                {
                    multimesh_name = opts_mesh_name;
                    multivar_name = safe_varname;
                }
                else
                {
                    multimesh_name = opts_mesh_name + "_" + safe_linked_topo_name;
                    multivar_name = opts_mesh_name + "_" + safe_varname;
                }

                // have to const_cast because converting to void *
                CONDUIT_CHECK_SILO_ERROR( DBAddOption(optlist.getSiloObject(),
                                                      DBOPT_MMESH_NAME,
                                                      const_cast<char *>(multimesh_name.c_str())),
                                          "Error creating options for putting multivar");

                CONDUIT_CHECK_SILO_ERROR(
                    DBPutMultivar(
                        dbfile,
                        multivar_name.c_str(),
                        global_num_domains,
                        var_name_ptrs.data(),
                        var_types.data(),
                        optlist.getSiloObject()),
                    "Error putting multivar corresponding to field: " << var_name);
            }        
        }
    }
}

//-----------------------------------------------------------------------------
/// The following options can be passed via the opts Node:
//-----------------------------------------------------------------------------
/// opts:
///
///      file_style: "default", "root_only", "multi_file", "overlink"
///            when # of domains == 1,  "default"   ==> "root_only"
///            else,                    "default"   ==> "multi_file"
///
///      silo_type: "default", "pdb", "hdf5", "unknown"
///            when the file we are writing to exists, "default" ==> "unknown"
///            else,                                   "default" ==> "hdf5"
///         note: these are additional silo_type options that we could add 
///         support for in the future:
///           "hdf5_sec2", "hdf5_stdio", "hdf5_mpio", "hdf5_mpiposix", "taurus"
///
///      suffix: "default", "cycle", "none"
///            when cycle is present,  "default"   ==> "cycle"
///            else,                   "default"   ==> "none"
///
///      mesh_name:  (used if present, default ==> "mesh")
///
///      ovl_topo_name: (used if present, default ==> "")
///
///      number_of_files:  {# of files}
///            when "multi_file" or "overlink":
///                 <= 0, use # of files == # of domains
///                  > 0, # of files == number_of_files
///
/// Note: 
///  In the non-overlink case...
///   1) We have made the choice to output ALL topologies as multimeshes. 
///   2) We prepend the provided mesh_name to each of these topo names. We do 
///      this to avoid a name collision in the root only + single domain case.
///      We do this across all cases for the sake of consistency. We also use 
///      the mesh_name as the name of the silo directory within each silo file
///      where data is stored.
///   3) ovl_topo_name is ignored if provided.
///  In the overlink case...
///   1) We have made the choice to output only ONE topology as a multimesh.
///   2) mesh_name is ignored if provided and changed to "MMESH"
///   3) ovl_topo_name is the name of the topo we are outputting. If it is not
///      provided, we choose the first topology in the blueprint.
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API write_mesh(const conduit::Node &mesh,
                                  const std::string &path,
                                  const conduit::Node &opts
                                  CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm mpi_comm))
{
    // The assumption here is that everything is multi domain

    std::string opts_file_style    = "default";
    std::string opts_suffix        = "default";
    std::string opts_out_mesh_name = "mesh"; // used only for the non-overlink case
    std::string opts_ovl_topo_name = ""; // used only for the overlink case
    std::string opts_silo_type     = "default";
    int         opts_num_files     = -1;
    bool        opts_truncate      = false;
    int         silo_type          = DB_HDF5;
    std::set<std::string> filelist;

    // check for + validate file_style option
    if(opts.has_child("file_style") && opts["file_style"].dtype().is_string())
    {
        opts_file_style = opts["file_style"].as_string();

        if(opts_file_style != "default" && 
           opts_file_style != "root_only" &&
           opts_file_style != "multi_file" &&
           opts_file_style != "overlink")
        {
            CONDUIT_ERROR("write_mesh invalid file_style option: \"" 
                          << opts_file_style << "\"\n"
                          " expected: \"default\", \"root_only\", "
                          "\"multi_file\", or \"overlink\"");
        }
    }

    // check for + validate suffix option
    if(opts.has_child("suffix") && opts["suffix"].dtype().is_string())
    {
        opts_suffix = opts["suffix"].as_string();

        if(opts_suffix != "default" && 
           opts_suffix != "cycle" &&
           opts_suffix != "none" )
        {
            CONDUIT_ERROR("write_mesh invalid suffix option: \"" 
                          << opts_suffix << "\"\n"
                          " expected: \"default\", \"cycle\", or \"none\"");
        }
    }
    
    // check for + validate mesh_name option
    if(opts.has_child("mesh_name") && opts["mesh_name"].dtype().is_string())
    {
        opts_out_mesh_name = opts["mesh_name"].as_string();
    }

    // check for + validate ovl_topo_name option
    // only used for overlink case
    if(opts.has_child("ovl_topo_name") && opts["ovl_topo_name"].dtype().is_string())
    {
        opts_ovl_topo_name = opts["ovl_topo_name"].as_string();
    }

    // check for number_of_files, 0 or -1 implies #files => # domains
    if(opts.has_child("number_of_files") && opts["number_of_files"].dtype().is_integer())
    {
        opts_num_files = (int) opts["number_of_files"].to_int();
    }

    // check for truncate (overwrite)
    if(opts.has_child("truncate") && opts["truncate"].dtype().is_string())
    {
        const std::string ow_string = opts["truncate"].as_string();
        if(ow_string == "true")
            opts_truncate = true;
    }

    // check for + validate silo_type option
    if (opts.has_child("silo_type") && opts["silo_type"].dtype().is_string())
    {
        opts_silo_type = opts["silo_type"].as_string();

        // TODO if we were to add additional silo_type options in the future,
        // they would need to be added here.
        if(opts_silo_type != "default" && 
           opts_silo_type != "pdb" &&
           opts_silo_type != "hdf5" &&
           opts_silo_type != "unknown" )
        {
            CONDUIT_ERROR("write_mesh invalid suffix option: \"" 
                          << opts_silo_type << "\"\n"
                          " expected: \"default\", \"pdb\", \"hdf5\", or \"unknown\"");
        }
    }

    if (opts_silo_type == "default")
    {
        silo_type = DB_HDF5;
        // "default" logic will be handled later, once we know
        // what the `root_filename` is.
    }
    else if (opts_silo_type == "pdb")
    {
        silo_type = DB_PDB;
    }
    else if (opts_silo_type == "hdf5")
    {
        silo_type = DB_HDF5;
    }
    else if (opts_silo_type == "unknown") 
    {
        silo_type = DB_UNKNOWN;
    }
    // TODO these are the additional silo_type options we could add support 
    // for in the future.
    // else if (opts_silo_type == "hdf5_sec2")
    // {
    //     silo_type = DB_HDF5_SEC2;
    // }
    // else if (opts_silo_type == "hdf5_stdio")
    // {
    //     silo_type = DB_HDF5_STDIO;
    // }
    // else if (opts_silo_type == "hdf5_mpio")
    // {
    //     silo_type = DB_HDF5_MPIO;
    // }
    // else if (opts_silo_type == "hdf5_mpiposix")
    // {
    //     silo_type = DB_HDF5_MPIPOSIX; 
    // }
    // else if (opts_silo_type == "taurus") 
    // {
    //     silo_type = DB_TAURUS;
    // }

    // more will happen for this case later
    if (opts_file_style == "overlink")
    {
        CONDUIT_INFO("Overlink is not yet fully supported. Outputted files "
                     "with this option will be missing several components "
                     "that Overlink requires.")
        opts_suffix = "none"; // force no suffix for overlink case
    }

    int num_files = opts_num_files;

#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    // nodes used for MPI comm (share them for many operations)
    Node n_local, n_reduced;
#endif

    // -----------------------------------------------------------
    // make sure some MPI task has data
    // -----------------------------------------------------------
    Node multi_dom;
#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    bool is_valid = conduit::relay::mpi::io::blueprint::clean_mesh(mesh, multi_dom, mpi_comm);
#else
    bool is_valid = conduit::relay::io::blueprint::clean_mesh(mesh, multi_dom);
#endif

    int par_rank = 0;
    int par_size = 1;
    // we may not have any domains so init to max
    int cycle = std::numeric_limits<int>::max();

    int local_boolean = is_valid ? 1 : 0;
    int global_boolean = local_boolean;


#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    par_rank = relay::mpi::rank(mpi_comm);
    par_size = relay::mpi::size(mpi_comm);

    // reduce to check to see if any valid data exists

    n_local = (int)cycle;
    relay::mpi::sum_all_reduce(n_local,
                               n_reduced,
                               mpi_comm);

    global_boolean = n_reduced.as_int();

#endif

    if(global_boolean == 0)
    {
      CONDUIT_INFO("Silo save: no valid data exists. Skipping save");
      return;
    }

    // -----------------------------------------------------------
    // get the number of local domains and the cycle info
    // -----------------------------------------------------------

    int local_num_domains = (int)multi_dom.number_of_children();
    // figure out what cycle we are
    if(local_num_domains > 0 && is_valid)
    {
        Node dom = multi_dom.child(0);
        if(!dom.has_path("state/cycle"))
        {
            if(opts_suffix == "cycle")
            {
                static std::map<std::string,int> counters;
                CONDUIT_INFO("Silo save: no 'state/cycle' present."
                             " Defaulting to counter");
                cycle = counters[path];
                counters[path]++;
            }
            else
            {
                opts_suffix = "none";
            }
        }
        else if(opts_suffix == "cycle")
        {
            cycle = dom["state/cycle"].to_int();
        }
        else if(opts_suffix == "default")
        {
            cycle = dom["state/cycle"].to_int();
            opts_suffix = "cycle";
        }
    }

#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    // reduce to get the cycle (some tasks might not have domains)
    n_local = (int)cycle;

    relay::mpi::min_all_reduce(n_local,
                               n_reduced,
                               mpi_comm);

    cycle = n_reduced.as_int();

    // we also need to have all mpi tasks agree on the `opts_suffix`
    // checking the first mpi task with domains should be sufficient.
    // find first
    n_local   = local_num_domains;
    n_reduced.reset();
    
    relay::mpi::all_gather(n_local,
                           n_reduced,
                           mpi_comm);

    index_t_accessor counts = n_reduced.value();
    index_t idx = -1;
    index_t i =0;
    NodeConstIterator counts_itr = n_reduced.children();
    while(counts_itr.has_next() && idx < 0)
    {
        const Node &curr = counts_itr.next();
        index_t count = curr.to_index_t();
        if(count > 0)
        {
            idx = i;
        }
        i++;
    }

    // now broadcast from idx
    Node n_opts_suffix;
    if(par_rank == idx)
    {
        n_opts_suffix = opts_suffix;
    }

    conduit::relay::mpi::broadcast_using_schema(n_opts_suffix,
                                                idx,
                                                mpi_comm);

    opts_suffix = n_opts_suffix.as_string();

#endif
    
    // -----------------------------------------------------------
    // find the # of global domains
    // -----------------------------------------------------------
    int global_num_domains = (int)local_num_domains;

#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    n_local = local_num_domains;

    relay::mpi::sum_all_reduce(n_local,
                               n_reduced,
                               mpi_comm);

    global_num_domains = n_reduced.as_int();
#endif

    if(global_num_domains == 0)
    {
      if(par_rank == 0)
      {
          CONDUIT_WARN("There no data to save. Doing nothing.");
      }
      return;
    }

    std::string output_dir = "";

    // resolve file_style == default
    // 
    // default implies multi_file if more than one domain
    if(opts_file_style == "default")
    {
        if( global_num_domains > 1)
        {
            opts_file_style = "multi_file";
        }
        else // other wise, use root only
        {
            opts_file_style = "root_only";
        }
    }

    // -----------------------------------------------------------
    // handle overlink-specific rules
    // -----------------------------------------------------------
    if (opts_file_style == "overlink")
    {
        // for overlink, things are different.
        // we are only going to write out one multimesh, with the name "MMESH"
        // the topo we will choose is the one named opts_ovl_topo_name, or, if that does
        // not exist, it is the first topo we can find.
        const Node &dom = multi_dom.child(0);

        if (dom.has_child("topologies"))
        {
            const Node &dom_topos = dom["topologies"];
            if (!dom_topos.has_child(opts_ovl_topo_name))
            {
                if (par_rank == 0)
                {
                    CONDUIT_INFO("Silo save: overlink: topo name not provided or not found.");
                }

                if (dom_topos.number_of_children() > 0)
                {
                    opts_ovl_topo_name = dom_topos.children().next().name();
                    if (par_rank == 0)
                    {
                        CONDUIT_INFO("Silo save: overlink: topo name defaulting to " + opts_ovl_topo_name);
                    }
                }
                else
                {
                    if (par_rank == 0)
                    {
                        CONDUIT_WARN("Silo save: overlink: No topologies to save. Doing nothing.");
                    }
                    return;
                }
            }
            // else we are good, the provided mesh name is valid
        }
        else
        {
            if (par_rank == 0)
            {
                CONDUIT_WARN("Silo save: overlink: No topologies to save. Doing nothing.");
            }
            return;
        }

        opts_out_mesh_name = "MMESH";
    }

    // ----------------------------------------------------
    // if using multi_file or overlink, create output dir
    // ----------------------------------------------------
    if (opts_file_style == "multi_file" ||
        opts_file_style == "overlink")
    {
        // setup the directory
        output_dir = path;

        // at this point for suffix, we should only see
        // cycle or none -- default has been resolved
        if (opts_suffix == "cycle")
        {
            output_dir += conduit_fmt::format(".cycle_{:06d}",cycle);
        }

        bool dir_ok = false;

        // let rank zero handle dir creation
        if (par_rank == 0)
        {
            // check if the dir exists
            dir_ok = utils::is_directory(output_dir);
            if (!dir_ok)
            {
                // if not try to let rank zero create it
                dir_ok = utils::create_directory(output_dir);
            }
        }

        // make sure everyone knows if dir creation was successful 

#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
        // use an mpi sum to check if the dir exists
        n_local = dir_ok ? 1 : 0;

        relay::mpi::sum_all_reduce(n_local,
                                   n_reduced,
                                   mpi_comm);

        dir_ok = (n_reduced.as_int() == 1);
#endif

        if(!dir_ok)
        {
            CONDUIT_ERROR("Error: failed to create directory " << output_dir);
        }
    }

    // ----------------------------------------------------
    // setup root file name
    // ----------------------------------------------------
    std::string root_filename;
    if (opts_file_style == "overlink")
    {
        root_filename = utils::join_file_path(output_dir, "OvlTop.silo");
    }
    else
    {
        root_filename = path;

        // at this point for suffix, we should only see 
        // cycle or none -- default has been resolved
        if(opts_suffix == "cycle")
        {
            root_filename += conduit_fmt::format(".cycle_{:06d}",cycle);
        }

        root_filename += ".root";
    }

    // ----------------------------------------------------
    // check silo type now that root file name is known
    // ----------------------------------------------------
    // if the file exists and we are not truncating
    if (utils::is_file(root_filename) && !opts_truncate)
    {
        // then silo type must be unknown
        if (silo_type != DB_UNKNOWN)
        {
            silo_type = DB_UNKNOWN;
            CONDUIT_INFO("Overriding silo type to DB_UNKNOWN because the "
                         "file already exists and truncation is disabled.");
        }
    }
    else // the file does not exist or we are truncating
    {
        // then silo type can be anything except unknown
        if (silo_type == DB_UNKNOWN)
        {
            // silo type can be anything except unknown
            silo_type = DB_HDF5;
            CONDUIT_INFO("Overriding chosen silo type (DB_UNKNOWN) to DB_HDF5 "
                         "because either the file does not exist or "
                         "truncation is enabled.");
        }
    }

    // zero or negative (default cases), use one file per domain
    if(num_files <= 0)
    {
        num_files = global_num_domains;
    }

    // if global domains > num_files, warn and use one file per domain
    if(global_num_domains < num_files)
    {
        CONDUIT_INFO("Requested more files than actual domains, "
                     "writing one file per domain");
        num_files = global_num_domains;
    }

    // new style bp index partition_map
    // NOTE: the part_map is inited during write process for N domains
    // to M files case.
    // Other cases are simpler and are created when root file is written
    Node output_partition_map;

    Node local_type_info;
    // our local type info is going to look like this:
    // meshes:
    //   mesh1:
    //     domain_ids: [5, 53, 74, ...]
    //     types: [quadmesh, ucdmesh, quadmesh, ...]
    //   mesh2:
    //     domain_ids: [5, 53, 74, ...]
    //     types: [pointmesh, pointmesh, pointmesh, ...]
    // vars:
    //   var1:
    //     domain_ids: [5, 53, 74, ...]
    //     types: [quadvar, ucdvar, quadvar, ...]
    //   var2:
    //     domain_ids: [5, 53, 74, ...]
    //     types: [pointvar, pointvar, pointvar, ...]
    // each array is local_num_domains long

    // at this point for file_style,
    // default has been resolved, we need to just handle:
    //   root_only, multi_file
    if(opts_file_style == "root_only")
    {
        // write out local domains, since all tasks will
        // write to single file in this case, we need baton.
        // the outer loop + par_rank == current_writer implements
        // the baton.

        Node local_root_file_created;
        Node global_root_file_created;
        local_root_file_created.set((int)0);
        global_root_file_created.set((int)0);

        for(int current_writer=0; current_writer < par_size; current_writer++)
        {
            if(par_rank == current_writer)
            {
                detail::SiloObjectWrapperCheckError<DBfile, decltype(&DBClose)> dbfile{
                    nullptr, 
                    &DBClose, 
                    "Error closing Silo file: " + root_filename};

                for(int i = 0; i < local_num_domains; ++i)
                {
                    // if truncate, first rank to touch the file needs
                    // to open at
                    if (!dbfile.getSiloObject()
                        && (global_root_file_created.as_int() == 0)
                        && opts_truncate)
                    {
                        dbfile.setSiloObject(DBCreate(root_filename.c_str(), DB_CLOBBER, DB_LOCAL, NULL, silo_type));
                        if (!dbfile.getSiloObject())
                        {
                            CONDUIT_ERROR("Error opening Silo file for writing: " << root_filename );
                        }
                        local_root_file_created.set((int)1);
                    }

                    if (!dbfile.getSiloObject())
                    {
                        if (utils::is_file(root_filename))
                        {
                            dbfile.setSiloObject(DBOpen(root_filename.c_str(), silo_type, DB_APPEND));
                        }
                        else
                        {
                            dbfile.setSiloObject(DBCreate(root_filename.c_str(), DB_CLOBBER, DB_LOCAL, NULL, silo_type));
                        }
                        
                        if (!dbfile.getSiloObject())
                        {
                            CONDUIT_ERROR("Error opening Silo file for writing: " << root_filename);
                        }
                    }

                    const Node &dom = multi_dom.child(i);
                    // figure out the proper mesh path the file
                    std::string mesh_path;

                    uint64 domain = dom["state/domain_id"].to_uint64();
                    if (global_num_domains == 1)
                    {
                        // no domain prefix, write to mesh name
                        mesh_path = opts_out_mesh_name;
                    }
                    else
                    {
                        // multiple domains, we need to use a domain prefix
                        mesh_path = conduit_fmt::format("domain_{:06d}/{}",
                                                        domain,
                                                        opts_out_mesh_name);
                        // we cannot have overlink in the root_only case so no need to handle it
                    }
                    silo_mesh_write(dom,
                                    dbfile.getSiloObject(),
                                    mesh_path,
                                    opts_ovl_topo_name,
                                    local_num_domains,
                                    i, // local domain index
                                    domain, // global domain id
                                    local_type_info,
                                    opts_file_style == "overlink");
                }
            }

        // Reduce to sync up (like a barrier) and solve first writer need
        #ifdef CONDUIT_RELAY_IO_MPI_ENABLED
            mpi::max_all_reduce(local_root_file_created,
                                global_root_file_created,
                                mpi_comm);
        #else
            global_root_file_created.set(local_root_file_created);
        #endif
        }
    }
    else if (global_num_domains == num_files)
    {
        // write out each domain
        // writes are independent, so no baton here
        for (int i = 0; i < local_num_domains; ++i)
        {
            const Node &dom = multi_dom.child(i);
            uint64 domain = dom["state/domain_id"].to_uint64();

            std::string output_file;
            if (opts_file_style == "overlink")
            {
                output_file = conduit::utils::join_file_path(output_dir,
                                                conduit_fmt::format("domain{:d}.silo",
                                                                    domain));
            }
            else
            {
                output_file = conduit::utils::join_file_path(output_dir,
                                                conduit_fmt::format("domain_{:06d}.silo",
                                                                    domain));
            }

            // properly support truncate vs non truncate

            detail::SiloObjectWrapperCheckError<DBfile, decltype(&DBClose)> dbfile{
                nullptr, 
                &DBClose,
                "Error closing Silo file: " + output_file};

            if (opts_truncate || !utils::is_file(output_file))
            {
                dbfile.setSiloObject(DBCreate(output_file.c_str(), DB_CLOBBER, DB_LOCAL, NULL, silo_type));
            }
            else
            {
                dbfile.setSiloObject(DBOpen(output_file.c_str(), silo_type, DB_APPEND));
            }
            if (!dbfile.getSiloObject())
            {
                CONDUIT_ERROR("Error opening Silo file for writing: " << output_file );
            }

            std::string mesh_path = opts_file_style == "overlink" ? "" : opts_out_mesh_name;

            // write to mesh name subpath
            silo_mesh_write(dom, 
                            dbfile.getSiloObject(), 
                            mesh_path, 
                            opts_ovl_topo_name,
                            local_num_domains,
                            i, // local domain index
                            domain, // global domain id
                            local_type_info,
                            opts_file_style == "overlink");
        }
    }
    else // more complex case, N domains to M files
    {
        //
        // recall: we have re-labeled domain ids from 0 - > N-1, however
        // some mpi tasks may have no data.
        //

        // books we keep:
        Node books;
        books["local_domain_to_file"].set(DataType::index_t(local_num_domains));
        books["local_domain_status"].set(DataType::index_t(local_num_domains));

        // batons
        books["local_file_batons"].set(DataType::index_t(num_files));
        books["global_file_batons"].set(DataType::index_t(num_files));

        // used to track first touch
        books["local_file_created"].set(DataType::index_t(num_files));
        books["global_file_created"].set(DataType::index_t(num_files));

        // size local # of domains
        index_t_array local_domain_to_file = books["local_domain_to_file"].value();
        index_t_array local_domain_status  = books["local_domain_status"].value();

        // size num total files
        /// batons
        index_t_array local_file_batons    = books["local_file_batons"].value();
        index_t_array global_file_batons   = books["global_file_batons"].value();
        /// file created flags
        index_t_array local_file_created    = books["local_file_created"].value();
        index_t_array global_file_created   = books["global_file_created"].value();


        Node d2f_map;
        blueprint::gen_domain_to_file_map(global_num_domains,
                                          num_files,
                                          books);

        // generate part map
        // use global_d2f is what we need for "file" part of part_map
        output_partition_map["file"] = books["global_domain_to_file"];
        output_partition_map["domain"].set(DataType::index_t(global_num_domains));
        index_t_array part_map_domain_vals = output_partition_map["domain"].value();
        for(index_t i=0; i < global_num_domains; i++)
        {
            part_map_domain_vals[i] = i;
        }

        index_t_accessor global_d2f = books["global_domain_to_file"].value();

        // init our local map and status array
        for(int d = 0; d < local_num_domains; ++d)
        {
            const Node &dom = multi_dom.child(d);
            uint64 domain = dom["state/domain_id"].to_uint64();
            // local domain index to file map
            local_domain_to_file[d] = global_d2f[domain];
            local_domain_status[d] = 1; // pending (1), vs done (0)
        }

        //
        // Round and round we go, will we deadlock I believe no :-)
        //
        // Here is how this works:
        //  At each round, if a rank has domains pending to write to a file,
        //  we put the rank id in the local file_batons vec.
        //  This vec is then mpi max'ed, and the highest rank
        //  that needs access to each file will write this round.
        //
        //  When a rank does not need to write to a file, we
        //  put -1 for this rank.
        //
        //  During each round, max of # files writers are participating
        //
        //  We are done when the mpi max of the batons is -1 for all files.
        //

        bool another_twirl = true;
        int twirls = 0;

        int local_all_is_good  = 1;
        int global_all_is_good = 1;

        books["local_all_is_good"].set_external(&local_all_is_good,1);
        books["global_all_is_good"].set_external(&global_all_is_good,1);

        std::string local_io_exception_msg = "";

        while(another_twirl)
        {
            // update baton requests
            for(int f = 0; f < num_files; ++f)
            {
                // reset file baton logic, look
                // to see if any local domains are
                // destined for this file
                local_file_batons[f] = -1;
                for(int d = 0; d < local_num_domains; ++d)
                {
                    // do we need to write this domain,
                    // and if so is it going to the file
                    // f
                    if(local_domain_status[d] == 1 &&
                       local_domain_to_file[d] == f)
                    {
                        local_file_batons[f] = par_rank;
                    }
                }
            }

            // mpi max file batons array
            #ifdef CONDUIT_RELAY_IO_MPI_ENABLED
                mpi::max_all_reduce(books["local_file_batons"],
                                    books["global_file_batons"],
                                    mpi_comm);
            #else
                global_file_batons.set(local_file_batons);
            #endif

            // mpi max file created array
            #ifdef CONDUIT_RELAY_IO_MPI_ENABLED
                mpi::max_all_reduce(books["local_file_created"],
                                    books["global_file_created"],
                                    mpi_comm);
            #else
                global_file_created.set(local_file_created);
            #endif


            // we now have valid batons (global_file_batons)
            for(int f = 0; f < num_files && local_all_is_good == 1 ; ++f)
            {
                // check if this rank has the global baton for this file
                if( global_file_batons[f] == par_rank )
                {
                    // check the domains this rank has pending
                    for(int d = 0; d < local_num_domains && local_all_is_good == 1; ++d)
                    {
                        if(local_domain_status[d] == 1 &&  // pending
                           local_domain_to_file[d] == f) // destined for this file
                        {
                            // now is the time to write!
                            // pattern is:
                            //  file_%06llu.{protocol}:/domain_%06llu/...
                            const Node &dom = multi_dom.child(d);
                            uint64 domain_id = dom["state/domain_id"].to_uint64();

                            // construct file name and path name
                            std::string file_name, curr_path;
                            if (opts_file_style == "overlink")
                            {
                                file_name = conduit_fmt::format("domfile{:d}.silo", f);
                                curr_path = conduit_fmt::format("domain{:d}/{}",
                                                                domain_id,
                                                                opts_out_mesh_name);
                            }
                            else
                            {
                                file_name = conduit_fmt::format("file_{:06d}.silo", f);
                                curr_path = conduit_fmt::format("domain_{:06d}/{}",
                                                                domain_id,
                                                                opts_out_mesh_name);
                            }

                            std::string output_file = conduit::utils::join_file_path(output_dir,
                                                                                     file_name);

                            try
                            {
                                detail::SiloObjectWrapperCheckError<DBfile, decltype(&DBClose)> dbfile{
                                    nullptr, 
                                    &DBClose,
                                    "Error closing Silo file: " + output_file};
                                // if truncate == true check if this is the first time we are
                                // touching file, and use DBCREATE w/ DB_CLOBBER
                                Node open_opts;
                                if(opts_truncate && global_file_created[f] == 0)
                                {
                                    if(!dbfile.getSiloObject())
                                    {
                                        dbfile.setSiloObject(DBCreate(output_file.c_str(), DB_CLOBBER, DB_LOCAL, NULL, silo_type));
                                        if (!dbfile.getSiloObject())
                                            CONDUIT_ERROR("Error opening Silo file for writing: " << output_file );
                                    }
                                    local_file_created[f]  = 1;
                                    global_file_created[f] = 1;
                                }
                                
                                if(!dbfile.getSiloObject())
                                {
                                    if (utils::is_file(output_file))
                                    {
                                        dbfile.setSiloObject(DBOpen(output_file.c_str(), silo_type, DB_APPEND));
                                    }
                                    else
                                    {
                                        dbfile.setSiloObject(DBCreate(output_file.c_str(), DB_CLOBBER, DB_LOCAL, NULL, silo_type));
                                    }
                                    if (!dbfile.getSiloObject())
                                    {
                                        CONDUIT_ERROR("Error opening Silo file for writing: " << output_file);
                                    }
                                }

                                // CONDUIT_INFO("rank " << par_rank << " output_file"
                                //              << output_file << " path " << path);

                                silo_mesh_write(dom, 
                                                dbfile.getSiloObject(), 
                                                curr_path, 
                                                opts_ovl_topo_name,
                                                local_num_domains,
                                                d, // local domain index
                                                domain_id, // global domain id
                                                local_type_info,
                                                opts_file_style == "overlink");
                                
                                // update status, we are done with this doman
                                local_domain_status[d] = 0;
                            }
                            catch(conduit::Error &e)
                            {
                                local_all_is_good = 0;
                                local_io_exception_msg = e.message();
                            }
                        }
                    }
                }
            }

            // if any I/O errors happened stop and have all
            // tasks bail out with an exception (to avoid hangs)
            #ifdef CONDUIT_RELAY_IO_MPI_ENABLED
                mpi::min_all_reduce(books["local_all_is_good"],
                                    books["global_all_is_good"],
                                    mpi_comm);
            #else
                global_all_is_good = local_all_is_good;
            #endif

            if(global_all_is_good == 0)
            {
                std::string emsg = "Failed to write mesh data on one more more ranks.";

                if(!local_io_exception_msg.empty())
                {
                     emsg += conduit_fmt::format("Exception details from rank {}: {}.",
                                                 par_rank, local_io_exception_msg);
                }
                CONDUIT_ERROR(emsg);
            }

            // // If you need to debug the baton algorithm,
            // // uncomment to examine the books:
            // if(par_rank == 0)
            // {
            //    std::cout << "[" << par_rank << "] "
            //              << " twirls: " << twirls
            //              << " details\n"
            //              << books.to_yaml();
            // }

            // check if we have another round
            // stop when all batons are -1
            another_twirl = false;

            for(int f = 0; f < num_files && !another_twirl; ++f)
            {
                // if any entry is not -1, we still have more work to do
                if(global_file_batons[f] != -1)
                {
                    another_twirl = true;
                    twirls++;
                }
            }
        }
    }

    int root_file_writer = 0;
    if(local_num_domains == 0)
    {
        root_file_writer = -1;
    }
#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    // Rank 0 could have an empty domain, so we have to check
    // to find someone with a data set to write out the root file.
    Node out;
    out = local_num_domains;
    Node rcv;

    mpi::all_gather_using_schema(out, rcv, mpi_comm);
    root_file_writer = -1;
    int* res_ptr = (int*)rcv.data_ptr();
    for(int i = 0; i < par_size; ++i)
    {
        if(res_ptr[i] != 0)
        {
            root_file_writer = i;
            break;
        }
    }

    MPI_Barrier(mpi_comm);
#endif

    if(root_file_writer == -1)
    {
        // this should not happen. global doms is already 0
        CONDUIT_WARN("Relay: there are no domains to write out");
    }

    // generate the bp index
    Node local_bp_idx, bp_idx;
    if(local_num_domains > 0)
    {
        ::conduit::blueprint::mesh::generate_index(multi_dom,
                                                   opts_out_mesh_name,
                                                   global_num_domains,
                                                   local_bp_idx);
    }
    // handle mpi case. 
    // this logic is from the mpi ver of mesh index gen
    // it is duplicated here b/c we dont want a circular dep
    // between conduit_blueprint_mpi and conduit_relay_io_mpi
#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    // NOTE: do to save vs write cases, these updates should be
    // single mesh only
    Node gather_bp_idx;
    relay::mpi::all_gather_using_schema(local_bp_idx,
                                        gather_bp_idx,
                                        mpi_comm);

    // union all entries into final index that reps
    // all domains
    NodeConstIterator itr = gather_bp_idx.children();
    while(itr.has_next())
    {
        const Node &curr = itr.next();
        bp_idx[opts_out_mesh_name].update(curr);
    }
#else
    // NOTE: do to save vs write cases, these updates should be
    // single mesh only
    bp_idx[opts_out_mesh_name] = local_bp_idx;
#endif

    Node global_type_info;
#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    relay::mpi::gather_using_schema(local_type_info,
                                    global_type_info,
                                    root_file_writer,
                                    mpi_comm);
#else
    global_type_info.append().set_external(local_type_info);
#endif

    // root_file_writer will now write out the root file
    if(par_rank == root_file_writer)
    {
        // we will gather type info into one place and organize it
        // by the end we should have a root_type_info that looks like this:
        // (one entry in each list for each domain)
        // 
        // meshes:
        //   mesh1: ucdmesh, ucdmesh, ...
        //   mesh2: ucdmesh, pointmesh, ...
        //   mesh3: quadmesh, quadmesh, ...
        //   ...
        // vars:
        //   var1: ucdvar, ucdvar, ...
        //   var2: ucdvar, pointvar, ...
        //   var3: quadvar, quadvar, ...
        //   ...

        Node root_type_info;
        Node &root_type_info_meshes = root_type_info["meshes"];
        Node &root_type_info_vars = root_type_info["vars"];

        auto type_info_itr = global_type_info.children();
        while (type_info_itr.has_next())
        {
            // type info from a particular MPI rank
            const Node &type_info_from_rank = type_info_itr.next();
            
            if (type_info_from_rank.has_child("meshes"))
            {
                auto read_meshes_itr = type_info_from_rank["meshes"].children();
                while (read_meshes_itr.has_next())
                {
                    const Node &read_mesh_type_info = read_meshes_itr.next();
                    const std::string read_mesh_name = read_meshes_itr.name();

                    if (!root_type_info_meshes.has_child(read_mesh_name)) 
                    {
                        root_type_info_meshes[read_mesh_name].set(DataType::index_t(global_num_domains));
                        index_t_array root_mesh_types = root_type_info_meshes[read_mesh_name].value();
                        root_mesh_types.fill(-1); // empty domains get -1
                    }
                    // the global domain ids array is of length local domain ids
                    // local domain ids index into it to read global domain ids out
                    index_t_accessor global_domain_ids = read_mesh_type_info["domain_ids"].value();
                    index_t_accessor read_mesh_types = read_mesh_type_info["types"].value();

                    // this is where we are writing the data to
                    index_t_array root_mesh_types = root_type_info_meshes[read_mesh_name].value();

                    for (index_t local_domain_id = 0; local_domain_id < global_domain_ids.number_of_elements(); local_domain_id ++)
                    {
                        index_t global_domain_index = global_domain_ids[local_domain_id];
                        // we initialized the array to be all -1, so if we are missing a domain
                        // we will have -1. Thus we should write -1 so I know later that we are
                        // missing a domain.
                        if (global_domain_index != -1)
                        {
                            root_mesh_types[global_domain_index] = read_mesh_types[local_domain_id];
                        }
                    }
                }
            }

            if (type_info_from_rank.has_child("vars"))
            {
                auto read_vars_itr = type_info_from_rank["vars"].children();
                while (read_vars_itr.has_next())
                {
                    const Node &read_var_type_info = read_vars_itr.next();
                    const std::string read_var_name = read_vars_itr.name();

                    if (!root_type_info_vars.has_child(read_var_name)) 
                    {
                        root_type_info_vars[read_var_name].set(DataType::index_t(global_num_domains));
                        index_t_array root_var_types = root_type_info_vars[read_var_name].value();
                        root_var_types.fill(-1); // empty domains get -1
                    }
                    // the global domain ids array is of length local domain ids
                    // local domain ids index into it to read global domain ids out
                    index_t_accessor global_domain_ids = read_var_type_info["domain_ids"].value();
                    index_t_accessor read_var_types = read_var_type_info["types"].value();

                    // this is where we are writing the data to
                    index_t_array root_var_types = root_type_info_vars[read_var_name].value();

                    for (index_t local_domain_id = 0; local_domain_id < global_domain_ids.number_of_elements(); local_domain_id ++)
                    {
                        index_t global_domain_index = global_domain_ids[local_domain_id];
                        // we initialized the array to be all -1, so if we are missing a domain
                        // we will have -1. Thus we should write -1 so I know later that we are
                        // missing a domain.
                        if (global_domain_index != -1)
                        {
                            root_var_types[global_domain_index] = read_var_types[local_domain_id];
                        }
                    }
                }
            }
        }

        std::string output_silo_path;

        // single file case
        if (opts_file_style == "root_only")
        {
            if (global_num_domains == 1)
            {
                output_silo_path = opts_out_mesh_name + "/{}";
            }
            else
            {
                output_silo_path = "domain_{:06d}/" + opts_out_mesh_name + "/{}";
            }

            // generate part map (we only need domain for this case)
            output_partition_map["domain"].set(DataType::index_t(global_num_domains));
            index_t_array part_map_domain_vals = output_partition_map["domain"].value();
            for (index_t i = 0; i < global_num_domains; i ++)
            {
                part_map_domain_vals[i] = i;
            }
        }
        else
        {
            std::string output_dir_base, output_dir_path;
            utils::rsplit_file_path(output_dir,
                                    output_dir_base,
                                    output_dir_path);

            // num domains == num files case
            if (global_num_domains == num_files)
            {
                // generate partition map
                output_partition_map["file"].set(DataType::index_t(global_num_domains));
                output_partition_map["domain"].set(DataType::index_t(global_num_domains));
                index_t_array part_map_file_vals   = output_partition_map["file"].value();
                index_t_array part_map_domain_vals = output_partition_map["domain"].value();

                for (index_t i = 0; i < global_num_domains; i ++)
                {
                    // file id == domain id
                    part_map_file_vals[i]   = i;
                    part_map_domain_vals[i] = i;
                }

                if (opts_file_style == "overlink")
                {
                    output_silo_path = utils::join_file_path(output_dir_base, "domain{:d}.silo:{}");
                }
                else
                {
                    output_silo_path = utils::join_file_path(output_dir_base, "domain_{:06d}.silo") + ":"
                                     + opts_out_mesh_name + "/{}";
                }
            }
            // m to n case
            else
            {
                // we generated the partition map earlier

                if (opts_file_style == "overlink")
                {
                    output_silo_path = utils::join_file_path(output_dir_base, "domfile{:d}.silo:domain{:d}/{}");
                }
                else
                {
                    output_silo_path = utils::join_file_path(output_dir_base, "file_{:06d}.silo") + ":"
                                     + "domain_{:06d}" + "/" 
                                     + opts_out_mesh_name + "/{}";
                }
            }
        }

        /////////////////////////////
        // mesh partition map
        /////////////////////////////
        // example of cases (for opts_out_mesh_name == "mesh"):
        // root only, single domain
        // silo_path: "mesh/{}"
        //
        // root only, multi domain
        // silo_path: "domain_{:06d}/mesh/{}"
        // partition_map:
        //   domain: [0, 1, 2, 3, 4 ]
        //
        // # domains == # files:
        // silo_path: "out/domain_{:06d}.silo:mesh/{}"
        // partition_map:
        //   file:  [ 0, 1, 2, 3, 4 ]
        //   domain: [ 0, 1, 2, 3, 4 ]
        //
        // N domains to M files:
        // silo_path: "out/file_{:06d}.silo:domain_{:06d}/mesh/{}"
        // partition_map:
        //   file:  [ 0, 0, 1, 2, 2 ]
        //   domain: [ 0, 1, 2, 3, 4 ]
        //
        // N domains to M files (non trivial domain order):
        // silo_path: "out/file_{:06d}.silo:domain_{:06d}/mesh/{}"
        // partition_map:
        //    file:  [ 0, 0, 1, 2, 2 ]
        //    domain: [ 4, 0, 3, 2, 1 ]

        if (output_partition_map.number_of_children() > 0 )
        {
            bp_idx[opts_out_mesh_name]["state/partition_map"] = output_partition_map;
        }

        Node root;
        root["blueprint_index"].set(bp_idx);

        root["protocol/name"]    = "silo";
        root["protocol/version"] = CONDUIT_VERSION;

        root["number_of_files"]  = num_files;
        root["number_of_domains"]  = global_num_domains;

        root["silo_path"] = output_silo_path;
        root["file_style"] = opts_file_style;

        detail::SiloObjectWrapperCheckError<DBfile, decltype(&DBClose)> dbfile{
            nullptr, 
            &DBClose,
            "Error closing Silo file: " + root_filename};

        // if not root only, this is the first time we are writing 
        // to the root file -- make sure to properly support truncate
        if(opts_file_style != "root_only" && opts_truncate)
        {
            if(!dbfile.getSiloObject())
            {
                dbfile.setSiloObject(DBCreate(root_filename.c_str(), DB_CLOBBER, DB_LOCAL, NULL, silo_type));
                if (!dbfile.getSiloObject())
                {
                    CONDUIT_ERROR("Error opening Silo file for writing: " << root_filename);
                }
            }
        }

        if(!dbfile.getSiloObject())
        {
            if (utils::is_file(root_filename))
            {
                dbfile.setSiloObject(DBOpen(root_filename.c_str(), silo_type, DB_APPEND));
            }
            else
            {
                dbfile.setSiloObject(DBCreate(root_filename.c_str(), DB_CLOBBER, DB_LOCAL, NULL, silo_type));
            }
            
            if (!dbfile.getSiloObject())
            {
                CONDUIT_ERROR("Error opening Silo file for writing: " << root_filename);
            }
        }

        write_multimeshes(dbfile.getSiloObject(), 
                          opts_out_mesh_name, 
                          opts_ovl_topo_name, 
                          root, 
                          root_type_info["meshes"],
                          opts_file_style == "overlink");
        write_multivars(dbfile.getSiloObject(), 
                        opts_out_mesh_name, 
                        opts_ovl_topo_name, 
                        root, 
                        root_type_info["vars"],
                        opts_file_style == "overlink");
        // write_multimaterials(); // TODO

    }

    // barrier at end of work to avoid file system race
    // (non root task could write the root file in write_mesh, 
    // but root task is always the one to read the root file
    // in read_mesh.

    #ifdef CONDUIT_RELAY_IO_MPI_ENABLED
        MPI_Barrier(mpi_comm);
    #endif
}

//-----------------------------------------------------------------------------
// Write a blueprint mesh to silo
//-----------------------------------------------------------------------------
/// These methods assume `mesh` is a valid blueprint mesh.
///
/// Note: These methods use "write" semantics, they will append to existing
///       files.
///
///
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API write_mesh(const conduit::Node &mesh,
                                  const std::string &path
                                  CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm mpi_comm)) 
{
    // empty opts
    Node opts;
#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    write_mesh(mesh, path, opts, mpi_comm);
#else
    write_mesh(mesh, path, opts);
#endif
}

//-----------------------------------------------------------------------------
// Save a blueprint mesh to silo
//-----------------------------------------------------------------------------
/// These methods assume `mesh` is a valid blueprint mesh.
///
/// Note: These methods use "save" semantics, they will overwrite existing
///       files.
///
///
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save_mesh(const conduit::Node &mesh,
                                 const std::string &path
                                 CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm mpi_comm)) 
{
    // empty opts
    Node opts;
#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    save_mesh(mesh, path, opts, mpi_comm);
#else
    save_mesh(mesh, path, opts);
#endif
}

//-----------------------------------------------------------------------------
/// The following options can be passed via the opts Node:
//-----------------------------------------------------------------------------
/// opts:
///
///      file_style: "default", "root_only", "multi_file", "overlink"
///            when # of domains == 1,  "default"   ==> "root_only"
///            else,                    "default"   ==> "multi_file"
///
///      silo_type: "default", "pdb", "hdf5", "unknown"
///            when the file we are writing to exists, "default" ==> "unknown"
///            else,                                   "default" ==> "hdf5"
///         note: these are additional silo_type options that we could add 
///         support for in the future:
///           "hdf5_sec2", "hdf5_stdio", "hdf5_mpio", "hdf5_mpiposix", "taurus"
///
///      suffix: "default", "cycle", "none"
///            when cycle is present,  "default"   ==> "cycle"
///            else,                   "default"   ==> "none"
///
///      mesh_name:  (used if present, default ==> "mesh")
///
///      ovl_topo_name: (used if present, default ==> "")
///
///      number_of_files:  {# of files}
///            when "multi_file" or "overlink":
///                 <= 0, use # of files == # of domains
///                  > 0, # of files == number_of_files
///
/// Note: 
///  In the non-overlink case...
///   1) We have made the choice to output ALL topologies as multimeshes. 
///   2) We prepend the provided mesh_name to each of these topo names. We do 
///      this to avoid a name collision in the root only + single domain case.
///      We do this across all cases for the sake of consistency. We also use 
///      the mesh_name as the name of the silo directory within each silo file
///      where data is stored.
///   3) ovl_topo_name is ignored if provided.
///  In the overlink case...
///   1) We have made ther choice to output only one topology as a multimesh.
///   2) mesh_name is ignored if provided and changed to "MMESH"
///   3) ovl_topo_name is the name of the topo we are outputting. If it is not
///      provided, we choose the first topology in the blueprint.
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save_mesh(const conduit::Node &mesh,
                                 const std::string &path,
                                 const conduit::Node &opts
                                 CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm mpi_comm)) 
{
    // we force overwrite to true, so we need a copy of the const opts passed.
    Node save_opts;
    save_opts.set(opts);
    save_opts["truncate"] = "true";

#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    write_mesh(mesh, path, save_opts, mpi_comm);
#else
    write_mesh(mesh, path, save_opts);
#endif
}

}
//-----------------------------------------------------------------------------
// -- end conduit::relay::<mpi>::io::silo --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::relay::<mpi>::io --
//-----------------------------------------------------------------------------

#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
}
//-----------------------------------------------------------------------------
// -- end conduit::relay::mpi --
//-----------------------------------------------------------------------------
#endif

}
//-----------------------------------------------------------------------------
// -- end conduit::relay --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
