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


//---------------------------------------------------------------------------//
void
silo_mesh_write(const Node &node,
                const std::string &path)
{
    // check for ":" split
    std::string file_path;
    std::string silo_obj_base;
    conduit::utils::split_file_path(path,
                                    std::string(":"),
                                    file_path,
                                    silo_obj_base);

    silo_mesh_write(node,file_path,silo_obj_base);
}


//---------------------------------------------------------------------------//
void silo_mesh_write(const Node &node,
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
        silo::silo_mesh_write(node,dbfile,silo_obj_path);
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


//-----------------------------------------------------------------------------
// -- begin conduit::relay::<mpi>::io::silo --
//-----------------------------------------------------------------------------
namespace silo
{

//-----------------------------------------------------------------------------
std::string sanitize_silo_varname(const std::string &varname)
{
    std::string newvarname = "";
    std::for_each(varname.begin(), varname.end(), 
        [](char const &c)
        {
            newvarname += (std::isalnum(c) || strchr("_", c) ? c : "_");
        });
    return newvarname;
}

//-----------------------------------------------------------------------------
// Fetch the DBfile * associated with 'filename' from 'filemap'.
// If the map does not contain an entry for 'filename', open
// the file and add it to the map before returning the pointer.
// 'type' should be either DB_READ or DB_APPEND.
//-----------------------------------------------------------------------------
DBfile *
get_or_open(std::map<std::string, std::unique_ptr<DBfile, decltype(&DBClose)>> &filemap,
            const std::string &filename,
            int mode = DB_READ)
{

    DBfile *fileptr;
    auto search = filemap.find(filename);
    
    if (search != filemap.end())
    {
        return search->second.get();
    }
    else
    {
        if (!(fileptr = DBOpen(filename.c_str(), DB_UNKNOWN, mode)))
        {
            CONDUIT_ERROR("Error opening silo file " << filename);
        }
        
        filemap.emplace(std::piecewise_construct,
                        std::make_tuple(filename),
                        std::make_tuple(fileptr, &DBClose));
        return fileptr;
    }
}

//-----------------------------------------------------------------------------
// Fetch the DBfile * associated with 'filename' from 'filemap'.
// If the map does not contain an entry for 'filename', but the file
// exists, open the file and add it to the map before returning the pointer.
// If the file is not in the map and does not exist, create the file and add
// it to the map before returning the pointer.
// 'type' should be one of the DB_HDF5* or DB_PDB* constants.
//-----------------------------------------------------------------------------
DBfile *
get_or_create(std::map<std::string,
              std::unique_ptr<DBfile, decltype(&DBClose)>> &filemap,
              const std::string &filename,
              int type)
{
    DBfile *fileptr;
    if (conduit::utils::is_file(filename))
    {
        return get_or_open(filemap, filename, DB_APPEND);
    }

    if (!(fileptr = DBCreate(filename.c_str(), DB_CLOBBER, DB_LOCAL, NULL, type)))
    {
        CONDUIT_ERROR("Error creating silo file " << filename);
    }

    filemap.emplace(std::piecewise_construct,
                    std::make_tuple(filename),
                    std::make_tuple(fileptr, &DBClose));
    return fileptr;
}

//-----------------------------------------------------------------------------
// Split a silo path into file path and silo name components.
// If there is no file path component (because the path points to an entry in
// the same file) the file path component will be empty.
//-----------------------------------------------------------------------------
void
split_silo_path(const std::string &path,
                const std::string &relative_dir,
                std::string &file_path,
                std::string &silo_name)
{

    conduit::utils::rsplit_file_path(path, ":", silo_name, file_path);
    if (!file_path.empty())
    {
        file_path = conduit::utils::join_file_path(relative_dir, file_path);
    }
}

//-----------------------------------------------------------------------------
std::string
shapetype_to_string(int shapetype)
{
    if (shapetype == DB_ZONETYPE_BEAM)
        return "line";
    if (shapetype == DB_ZONETYPE_TRIANGLE)
        return "tri";
    if (shapetype == DB_ZONETYPE_QUAD)
        return "quad";
    if (shapetype == DB_ZONETYPE_TET)
        return "tet";
    if (shapetype == DB_ZONETYPE_HEX)
        return "hex";
    if (shapetype == DB_ZONETYPE_POLYHEDRON)
        return "polyhedral";
    if (shapetype == DB_ZONETYPE_POLYGON)
        return "polygonal";

    CONDUIT_ERROR("Unsupported zone type " << shapetype);
    return "";
}

//-----------------------------------------------------------------------------
// copy data and assign it to a Node
template <typename T>
void
copy_and_assign(T *data,
                int data_length,
                conduit::Node &target)
{
    T *data_copy = new T[data_length];
    memcpy(data_copy, data, data_length * sizeof(T));
    target.set(data_copy, data_length);
}

//-----------------------------------------------------------------------------
template <typename T>
void
copy_point_coords(void *coords[3],
                  int ndims,
                  int *dims,
                  int coord_sys,
                  conduit::Node &node)
{

    ndims = ndims < 3 ? ndims : 3;
    const std::vector<std::string> labels;
    if (coord_sys == DB_CARTESIAN)
    {
        labels = CARTESIAN_AXES;
    }
    else if (coord_sys == DB_CYLINDRICAL)
    {
        labels = CYLINDRICAL_AXES;
        if (ndims >= 3)
        {
            CONDUIT_ERROR("Blueprint only supports 2D cylindrical coordinates");
        }
    }
    else if (coord_sys == DB_SPHERICAL)
    {
        labels = SPHERICAL_AXES;
    }
    else
    {
        CONDUIT_ERROR("Unsupported coordinate system " << coord_sys);
    }
    
    for (int i = 0; i < ndims; i++)
    {
        if (coords[i] != NULL)
        {
            copy_and_assign(static_cast<T *>(coords[i]),
                            dims[i],
                            node[labels[i]]);
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
    copy_and_assign(zones->nodelist,
                    zones->lnodelist,
                    elements["connectivity"]);
    if (zones->shapetype[0] == DB_ZONETYPE_POLYHEDRON)
    {
        // TODO: support polyhedra
        CONDUIT_ERROR("Polyhedra not yet supported");
        copy_and_assign(zones->shapesize, zones->nzones, elements["sizes"]);
        // TODO: no idea if this is right
        add_offsets(zones, elements["subelements"]); 
    }
    if (zones->shapetype[0] == DB_ZONETYPE_POLYGON)
    {
        copy_and_assign(zones->shapesize, zones->nzones, elements["sizes"]);
        add_offsets(zones, elements);
    }
}

//-----------------------------------------------------------------------------
// add complete topology and coordset entries to a mesh domain
void
read_ucdmesh_domain(DBfile *file,
                    std::string &mesh_name,
                    conduit::Node &mesh_domain)
{
    DBucdmesh *ucdmesh_ptr;
    if (!(ucdmesh_ptr = DBGetUcdmesh(file, mesh_name.c_str())))
    {
        CONDUIT_ERROR("Error fetching mesh " << mesh_name);
    }
    
    std::unique_ptr<DBucdmesh, decltype(&DBFreeUcdmesh)> ucdmesh{
        ucdmesh_ptr, &DBFreeUcdmesh};

    std::string name{ucdmesh_ptr->name};
    if (ucdmesh_ptr->zones)
    {
        CONDUIT_ASSERT(!ucdmesh_ptr->phzones,
                       "Both phzones and zones are defined in mesh "
                           << mesh_name);
        add_shape_info(ucdmesh_ptr->zones,
                       mesh_domain["topologies"][name]["elements"]);
    }
    else if (ucdmesh_ptr->phzones)
    {
        // TODO: implement support for phzones
        CONDUIT_ERROR("Silo ucdmesh phzones not yet supported");
        mesh_domain["topologies"][name]["elements"]["shape"] =
            shapetype_to_string(DB_ZONETYPE_POLYHEDRON);

    }
    else
    {
        CONDUIT_ERROR("Neither phzones nor zones is defined in mesh "
                      << mesh_name);
    }

    mesh_domain["topologies"][name]["coordset"] = name;
    mesh_domain["coordsets"][name]["type"] = "explicit";
    mesh_domain["topologies"][name]["type"] = "unstructured";

    int dims[] = {ucdmesh_ptr->nnodes,
                  ucdmesh_ptr->nnodes,
                  ucdmesh_ptr->nnodes};

    if (ucdmesh_ptr->datatype == DB_DOUBLE)
    {
        copy_point_coords<double>(ucdmesh_ptr->coords,
                                  ucdmesh_ptr->ndims,
                                  dims,
                                  ucdmesh_ptr->coord_sys,
                                  mesh_domain["coordsets"][name]["values"]);
    }
    else if (ucdmesh_ptr->datatype == DB_FLOAT)
    {
        copy_point_coords<float>(ucdmesh_ptr->coords,
                                 ucdmesh_ptr->ndims,
                                 dims,
                                 ucdmesh_ptr->coord_sys,
                                 mesh_domain["coordsets"][name]["values"]);
    }
    else 
    {
        CONDUIT_ERROR("Unsupported mesh data type " << ucdmesh_ptr->datatype);
    }
}

//-----------------------------------------------------------------------------
// add complete topology and coordset entries to a mesh domain
void
read_quadmesh_domain(DBfile *file,
                     std::string &mesh_name,
                     conduit::Node &mesh_domain)
{
    DBquadmesh *quadmesh_ptr;
    if (!(quadmesh_ptr = DBGetQuadmesh(file, mesh_name.c_str())))
    {
        CONDUIT_ERROR("Error fetching mesh " << mesh_name);
    }
    
    std::unique_ptr<DBquadmesh, decltype(&DBFreeQuadmesh)> quadmesh{
        quadmesh_ptr, &DBFreeQuadmesh};
    
    std::string name{quadmesh_ptr->name};

    int coordtype{quadmesh_ptr->coordtype};
    int ndims{quadmesh_ptr->ndims};

    if (coordtype == DB_COLLINEAR)
    {
        mesh_domain["coordsets"][name]["type"] = "rectilinear";
        mesh_domain["topologies"][name]["type"] = "rectilinear";
    }
    else if (coordtype == DB_NONCOLLINEAR)
    {
        mesh_domain["coordsets"][name]["type"] = "explicit";
        mesh_domain["topologies"][name]["type"] = "structured";
        mesh_domain["topologies"][name]["elements/dims/i"] = quadmesh_ptr->dims[0];
        if (ndims > 1) mesh_domain["topologies"][name]["elements/dims/j"] = quadmesh_ptr->dims[1];
        if (ndims > 2) mesh_domain["topologies"][name]["elements/dims/k"] = quadmesh_ptr->dims[2];
    }
    else
    {
        CONDUIT_ERROR("Undefined coordtype in " << coordtype);
    }

    mesh_domain["topologies"][name]["coordset"] = name;

    // If the origin is not the default value
    if (quadmesh_ptr->base_index[0] != 0 && quadmesh_ptr->base_index[1] != 0 && quadmesh_ptr->base_index[2] != 0)
    {
        // then we need to specify it
        mesh_domain["topologies"][name]["elements/origin/i"] = quadmesh_ptr->base_index[0];
        if (ndims > 1) mesh_domain["topologies"][name]["elements/origin/j"] = quadmesh_ptr->base_index[1];
        if (ndims > 2) mesh_domain["topologies"][name]["elements/origin/k"] = quadmesh_ptr->base_index[2];
    }
    
    if (quadmesh_ptr->datatype == DB_DOUBLE)
    {
        copy_point_coords<double>(quadmesh_ptr->coords,
                                  ndims,
                                  quadmesh_ptr->dims,
                                  quadmesh_ptr->coord_sys,
                                  mesh_domain["coordsets"][name]["values"]);
    }
    else if (quadmesh_ptr->datatype == DB_FLOAT)
    {
        copy_point_coords<float>(quadmesh_ptr->coords,
                                 ndims,
                                 quadmesh_ptr->dims,
                                 quadmesh_ptr->coord_sys,
                                 mesh_domain["coordsets"][name]["values"]);
    }
    else
    {
        CONDUIT_ERROR("Unsupported mesh data type " << quadmesh_ptr->datatype);
    }
}

//-----------------------------------------------------------------------------
// add complete topology and coordset entries to a mesh domain
void
read_pointmesh_domain(DBfile *file,
                      std::string &mesh_name,
                      conduit::Node &mesh_domain)
{
    DBpointmesh *pointmesh_ptr;
    if (!(pointmesh_ptr = DBGetPointmesh(file, mesh_name.c_str())))
    {
        CONDUIT_ERROR("Error fetching mesh " << mesh_name);
    }
    
    std::string name{pointmesh_ptr->name};
    std::unique_ptr<DBpointmesh, decltype(&DBFreePointmesh)> pointmesh{
        pointmesh_ptr, &DBFreePointmesh};

    mesh_domain["topologies"][name]["type"] = "points";
    mesh_domain["topologies"][name]["coordset"] = name;
    mesh_domain["coordsets"][name]["type"] = "explicit";
    int dims[] = { pointmesh_ptr->nels,
                   pointmesh_ptr->nels,
                   pointmesh_ptr->nels};
    
    if (pointmesh_ptr->datatype == DB_DOUBLE)
    {
        copy_point_coords<double>(pointmesh_ptr->coords,
                                  pointmesh_ptr->ndims,
                                  dims,
                                  DB_CARTESIAN,
                                  mesh_domain["coordsets"][name]["values"]);
    }
    else if (pointmesh_ptr->datatype == DB_FLOAT)
    {
        copy_point_coords<float>(pointmesh_ptr->coords,
                                 pointmesh_ptr->ndims,
                                 dims,
                                 DB_CARTESIAN,
                                 mesh_domain["coordsets"][name]["values"]);
    }
    else
    {
        CONDUIT_ERROR("Unsupported mesh data type " << pointmesh_ptr->datatype);
    }
}

//-----------------------------------------------------------------------------
// Read a multimesh domain, switching on the type.
// 'file' must be a pointer into the file containing the mesh, and 'mesh_name'
// must be the mesh's name
//-----------------------------------------------------------------------------
void
read_mesh_domain(DBfile *file,
                 std::string &mesh_name,
                 conduit::Node &mesh_domain,
                 int meshtype)
{
    if (meshtype == DB_UCDMESH)
        return read_ucdmesh_domain(file, mesh_name, mesh_domain);
    if (meshtype == DB_QUADMESH)
        return read_quadmesh_domain(file, mesh_name, mesh_domain);
    if (meshtype == DB_POINTMESH)
        return read_pointmesh_domain(file, mesh_name, mesh_domain);
    if (meshtype == DB_CSGMESH)
        CONDUIT_ERROR("CSG meshes are not supported by Blueprint");

    CONDUIT_ERROR("Unsupported mesh type " << meshtype);
}

//-----------------------------------------------------------------------------
// Read a multimesh from a Silo file.
// 'root_file' should be the file containing the multivar entry
// 'filemap' should be a mapping providing DBfile* for files which have
//  already been opened.
// 'dirname' should be the directory containing the root file, as if the
// `dirname` command were called on the root file path. This directory is used
// to concretize the paths given by the multivar.
//-----------------------------------------------------------------------------
void
read_multimesh( DBfile *root_file,
                std::map<std::string, std::unique_ptr<DBfile, decltype(&DBClose)>> &filemap,
                std::string &dirname,
                DBmultimesh *multimesh,
                conduit::Node &mesh)
{

    std::string file_path, silo_name;
    for (index_t i = 0; i < multimesh->nblocks; ++i)
    {
        Node &entry = mesh.append();
        split_silo_path(multimesh->meshnames[i], dirname, file_path, silo_name);
        if (!file_path.empty())
        {
            read_mesh_domain(get_or_open(filemap, file_path),
                             silo_name,
                             entry,
                             multimesh->meshtypes[i]);
        }else
        {
            read_mesh_domain(root_file,
                             silo_name,
                             entry,
                             multimesh->meshtypes[i]);
        }
    }
}

//-----------------------------------------------------------------------------
void
apply_centering(int centering, conduit::Node &field)
{
    if (centering == DB_NODECENT) {
        field["association"] = "vertex";
    } else if (centering == DB_ZONECENT) {
        field["association"] = "element";
    } else {
        CONDUIT_ERROR("Unsupported field association " << centering);
    }
}

//-----------------------------------------------------------------------------
// add a set of data arrays to a Node.
template <typename T>
void
apply_values(void **vals,
             int num_arrays,
             int num_elems,
             conduit::Node &values)
{
    for (int i = 0; i < num_arrays; ++i)
    {
        copy_and_assign(static_cast<T *>(vals[i]), num_elems, values);
    }
}

//-----------------------------------------------------------------------------
// Read a quad variable from a Silo file.
// 'file' must be a pointer into the file containing the variable domain
// 'var_name' must be the name of the variable within the file.
//-----------------------------------------------------------------------------
void
read_quadvariable_domain(DBfile *file,
                         std::string &var_name,
                         conduit::Node &field)
{
    DBquadvar *quadvar_ptr;
    if (!(quadvar_ptr = DBGetQuadvar(file, var_name.c_str())))
    {
        CONDUIT_ERROR("Error fetching quad variable " << var_name);
    }
    std::unique_ptr<DBquadvar, decltype(&DBFreeQuadvar)> quadvar{
        quadvar_ptr, &DBFreeQuadvar};
    std::string name{quadvar_ptr->name};
    field["topology"] = std::string(quadvar_ptr->meshname);

    apply_centering(quadvar_ptr->centering, field);
    
    if (quadvar_ptr->datatype == DB_FLOAT)
    {
        apply_values<float>(quadvar_ptr->vals, quadvar_ptr->nvals,
                            quadvar_ptr->nels, field["values"]);
    }
    else if (quadvar_ptr->datatype == DB_DOUBLE)
    {
        apply_values<double>(quadvar_ptr->vals, quadvar_ptr->nvals,
                             quadvar_ptr->nels, field["values"]);
    }
}

//-----------------------------------------------------------------------------
// Read a UCD variable domain from a Silo file.
// 'file' must be a pointer into the file containing the variable domain
// 'var_name' must be the name of the variable within the file.
//-----------------------------------------------------------------------------
void
read_ucdvariable_domain(DBfile *file,
                        std::string &var_name,
                        conduit::Node &field)
{
    DBucdvar *ucdvar_ptr;
    if (!(ucdvar_ptr = DBGetUcdvar(file, var_name.c_str())))
    {
        CONDUIT_ERROR("Error fetching ucd variable " << var_name);
    }

    std::unique_ptr<DBucdvar, decltype(&DBFreeUcdvar)> ucdvar{ucdvar_ptr,
                                                              &DBFreeUcdvar};
    std::string name{ucdvar_ptr->name};
    field["topology"] = std::string(ucdvar_ptr->meshname);
    apply_centering(ucdvar_ptr->centering, field);
    
    if (ucdvar_ptr->datatype == DB_FLOAT)
    {
        apply_values<float>(ucdvar_ptr->vals, ucdvar_ptr->nvals,
                            ucdvar_ptr->nels, field["values"]);
    }
    else if (ucdvar_ptr->datatype == DB_DOUBLE)
    {
        apply_values<double>(ucdvar_ptr->vals, ucdvar_ptr->nvals,
                             ucdvar_ptr->nels, field["values"]);
    }
}

//-----------------------------------------------------------------------------
// Read a pointvariable domain from a Silo file.
// 'file' must be a pointer into the file containing the variable domain
// 'var_name' must be the name of the variable within the file.
//-----------------------------------------------------------------------------
void
read_pointvariable_domain(DBfile *file,
                          std::string &var_name,
                          conduit::Node &field)
{
    DBmeshvar *meshvar_ptr;
    if (!(meshvar_ptr = DBGetPointvar(file, var_name.c_str())))
        CONDUIT_ERROR("Error fetching variable " << var_name);

    std::unique_ptr<DBmeshvar, decltype(&DBFreeMeshvar)> meshvar{
        meshvar_ptr, &DBFreeMeshvar};
    std::string name{meshvar_ptr->name};
    field["topology"] = std::string(meshvar_ptr->meshname);

    apply_centering(meshvar_ptr->centering, field);

    if (meshvar_ptr->datatype == DB_FLOAT)
    {
        apply_values<float>(meshvar_ptr->vals, meshvar_ptr->nvals,
                            meshvar_ptr->nels, field["values"]);
    }
    else if (meshvar_ptr->datatype == DB_DOUBLE)
    {
        apply_values<double>(meshvar_ptr->vals, meshvar_ptr->nvals,
                             meshvar_ptr->nels, field["values"]);
    }
}

//-----------------------------------------------------------------------------
// Read a multivar domain, switching on the type.
// 'file' must be a pointer into the file containing the variable domain
// 'var_name' must be the name of the variable within the file.
//-----------------------------------------------------------------------------
void
read_variable_domain(DBfile *file, std::string &var_name,
                     conduit::Node &field, int vartype)
{
    if (vartype == DB_UCDVAR)
        return read_ucdvariable_domain(file, var_name, field);
    if (vartype == DB_QUADVAR)
        return read_quadvariable_domain(file, var_name, field);
    if (vartype == DB_POINTVAR)
        return read_pointvariable_domain(file, var_name, field);
    if (vartype == DB_CSGVAR)
        CONDUIT_ERROR("CSG Variables not supported by Blueprint");

    CONDUIT_ERROR("Unsupported variable type " << vartype);
}

//-----------------------------------------------------------------------------
// Read a multivar from a Silo file.
// 'root_file' should be the file containing the multivar entry
// 'filemap' should be a mapping providing DBfile* for files which have
//  already been opened.
// 'dirname' should be the directory containing the root file, as if the
// `dirname` command were called on the root file path. This directory is used
// to concretize the paths given by the multivar.
//-----------------------------------------------------------------------------
void
read_multivar(DBfile *root_file,
              std::map<std::string, std::unique_ptr<DBfile, decltype(&DBClose)>> &filemap,
              const std::string &dirname,
              DBmultivar *multivar,
              conduit::Node &mesh)
{
    std::string file_path, silo_name;
    for (index_t i = 0; i < multivar->nvars; ++i)
    {
        split_silo_path(multivar->varnames[i], dirname, file_path, silo_name);
        Node &field = mesh[i]["fields"][silo_name];
        if (!file_path.empty())
        {
            read_variable_domain(get_or_open(filemap, file_path), silo_name,
                                field, multivar->vartypes[i]);
        }
        else
        {
            read_variable_domain(root_file, silo_name, field,
                                 multivar->vartypes[i]);
        }
    }
}

//-----------------------------------------------------------------------------
void
read_all_multivars(DBfile *root_file,
                   DBtoc *toc,
                   std::map<std::string, std::unique_ptr<DBfile, decltype(&DBClose)>> &filemap,
                   const std::string &dirname,
                   const std::string &mmesh_name,
                   int expected_domains,
                   conduit::Node &mesh)
{
    for (int i = 0; i < toc->nmultivar; ++i)
    {
        std::unique_ptr<DBmultivar, decltype(&DBFreeMultivar)> multivar{
            DBGetMultivar(root_file, toc->multivar_names[i]), &DBFreeMultivar};
        if (!multivar.get())
        {
            multivar.release();
            CONDUIT_ERROR("Error fetching multivar "
                          << multivar.get()->varnames[i]);
        }

        if (multivar.get()->mmesh_name != NULL &&
            multivar.get()->mmesh_name == mmesh_name)
        {
            CONDUIT_ASSERT(multivar.get()->nvars == expected_domains,
                           "Domain count mismatch between multivar "
                               << multivar.get()->varnames[i]
                               << "and multimesh");
            // read in the multivar and add it to the mesh Node
            read_multivar(root_file, filemap, dirname, multivar.get(), mesh);
        }
    }
}

//-----------------------------------------------------------------------------
// Read a material domain from a Silo file.
// 'file' must be a pointer into the file containing the material domain
// 'mat_name' must be the name of the material within the file.
//-----------------------------------------------------------------------------
void
read_material_domain(DBfile *file,
                     std::string &mat_name,
                     conduit::Node &matsets)
{
    DBmaterial *material_ptr;
    if (!(material_ptr = DBGetMaterial(file, mat_name.c_str())))
    {
        CONDUIT_ERROR("Error fetching variable " << mat_name);
    }

    std::unique_ptr<DBmaterial, decltype(&DBFreeMaterial)> material{
        material_ptr, &DBFreeMaterial};
    conduit::Node &curr_matset = matsets[material_ptr->name];
    curr_matset["topology"] = material_ptr->meshname;
    for (int i = 0; i < material_ptr->nmat; ++i)
    {
        // material names may be NULL
        std::string material_name;
        if (material_ptr->matnames)
        {
            material_name = material_ptr->matnames[i];
        }
        else
        {
            // but matnos should always be
            material_name = std::to_string(material_ptr->matnos[i]);
        }
        curr_matset["material_map"][material_name] = material_ptr->matnos[i];
    }
    // TODO: support multi-dimensional materials
    CONDUIT_ASSERT(material_ptr->ndims == 1,
                   "Only single-dimension materials supported, got "
                       << material_ptr->ndims);
    if (material_ptr->mixlen > 0)
    {
        // The struct has volume fractions.
        // In this case, the struct is very confusing.
        // If an entry in the `matlist` is negative, it implies that the
        // associated zone has mixed materials, and `-(value) - 1` gives the
        // first index into mix_vf and mix_mat for that zone. mix_next is then
        // used to find the rest of the indices into mix_vf and mix_mat for
        // the zone.
        std::vector<double> volume_fractions;
        std::vector<int> material_ids;
        std::vector<int> sizes;
        std::vector<int> offsets;
        int curr_offset = 0;
        for (int i = 0; i < material_ptr->dims[0]; ++i)
        {
            int matlist_entry = material_ptr->matlist[i];
            if (matlist_entry >= 0)
            {
                volume_fractions.push_back(1.0);
                material_ids.push_back(matlist_entry);
                sizes.push_back(1);
                offsets.push_back(curr_offset);
                curr_offset++;
            }
            else
            {
                int mix_id = -(matlist_entry)-1;
                int curr_size = 0;
                while (mix_id >= 0)
                {
                    material_ids.push_back(material_ptr->mix_mat[mix_id]);
                    if (material_ptr->datatype == DB_DOUBLE)
                    {
                        volume_fractions.push_back(static_cast<double *>(
                            material_ptr->mix_vf)[mix_id]);
                    }
                    else if (material_ptr->datatype == DB_FLOAT)
                    {
                        volume_fractions.push_back(
                            static_cast<float *>(material_ptr->mix_vf)[mix_id]);
                    }
                    curr_size++;
                    mix_id = material_ptr->mix_next[mix_id] - 1;
                }
                sizes.push_back(curr_size);
                offsets.push_back(curr_offset);
                curr_offset += curr_size;
            }
        }
        curr_matset["material_ids"].set(material_ids.data(), material_ids.size());
        curr_matset["volume_fractions"].set(volume_fractions.data(),
                                       volume_fractions.size());
        curr_matset["sizes"].set(sizes.data(), sizes.size());
        curr_matset["offsets"].set(offsets.data(), offsets.size());
    }
    else
    {
        // TODO: remove, since this is just a special case of the above logic, I think?
        // no volume fractions. All zones are single-material.
        int arr_len = material_ptr->dims[0];
        copy_and_assign(material_ptr->matlist,
                        arr_len,
                        curr_matset["material_ids"]);

        double *volume_fractions = new double[arr_len];
        int *sizes = new int[arr_len];
        int *offsets = new int[arr_len];
        for (int i = 0; i < arr_len; ++i)
        {
            volume_fractions[i] = 1.0;
            offsets[i] = i;
            sizes[i] = 1;
        }
        curr_matset["volume_fractions"].set(volume_fractions, arr_len);
        curr_matset["sizes"].set(sizes, arr_len);
        curr_matset["offsets"].set(offsets, arr_len);
    }
}

//-----------------------------------------------------------------------------
// Read a multimaterial from a Silo file.
// 'root_file' should be the file containing the multivar entry
// 'filemap' should be a mapping providing DBfile* for files which have
//  already been opened.
// 'dirname' should be the directory containing the root file, as if the
// `dirname` command were called on the root file path. This directory is used
// to concretize the paths given by the multimat.
//-----------------------------------------------------------------------------
void
read_multimaterial(DBfile *root_file,
                   std::map<std::string, std::unique_ptr<DBfile, decltype(&DBClose)>> &filemap,
                   const std::string &dirname,
                   DBmultimat *multimat,
                   conduit::Node &mesh)
{

    std::string file_path, silo_name;
    for (index_t i = 0; i < multimat->nmats; ++i)
    {
        Node &matsets = mesh[i]["matsets"];
        split_silo_path(multimat->matnames[i], dirname, file_path, silo_name);
        if (!file_path.empty())
        {
            read_material_domain(get_or_open(filemap, file_path),
                                 silo_name,
                                 matsets);
        }
        else
        {
            read_material_domain(root_file, silo_name, matsets);
        }
    }
}

//---------------------------------------------------------------------------//
void
read_all_multimats(DBfile *root_file,
                  DBtoc *toc,
                  std::map<std::string, std::unique_ptr<DBfile, decltype(&DBClose)>> &filemap,
                  const std::string &dirname,
                  const std::string &mmesh_name,
                  int expected_domains,
                  conduit::Node &mesh)
{

    for (int i = 0; i < toc->nmultimat; ++i)
    {
        std::unique_ptr<DBmultimat, decltype(&DBFreeMultimat)> multimat{
            DBGetMultimat(root_file, toc->multimat_names[i]), &DBFreeMultimat};
        if (!multimat.get()) {
            multimat.release();
            CONDUIT_ERROR("Error fetching multimaterial "
                          << multimat.get()->matnames[i]);
        }

        if (multimat.get()->mmesh_name != NULL &&
            multimat.get()->mmesh_name == mmesh_name)
        {
            CONDUIT_ASSERT(multimat.get()->nmats == expected_domains,
                           "Domain count mismatch between multimaterial "
                               << multimat.get()->matnames[i]
                               << "and multimesh");
            // read in the multimaterial and add it to the mesh Node
            read_multimaterial(root_file,
                               filemap,
                               dirname,
                               multimat.get(),
                               mesh);
        }
    }
}

//---------------------------------------------------------------------------//
void CONDUIT_RELAY_API
read_mesh(const std::string &root_file_path,
          conduit::Node &mesh)
{
    Node opts;
    read_mesh(root_file_path, opts, mesh);
}

//-----------------------------------------------------------------------------
///
/// opts:
///      mesh_name: "{name}"
///          provide explicit mesh name, for cases where bp data includes
///           more than one mesh.
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API read_mesh(const std::string &root_file_path,
                                 const conduit::Node &opts,
                                 conduit::Node &mesh)
{

    int i;
    std::string mmesh_name;
    std::string dirname;
    std::map<std::string, std::unique_ptr<DBfile, decltype(&DBClose)>> filemap;
    // get the directory of the root silo file, for concretizing paths found
    // within the root silo file
    conduit::utils::rsplit_file_path(root_file_path, mmesh_name, dirname);
    DBfile *silofile = get_or_open(filemap, root_file_path);
    DBtoc *toc = DBGetToc(silofile); // shouldn't be free'd
    // get the multimesh
    CONDUIT_ASSERT(toc->nmultimesh > 0, "No multimesh found in file");
    if (!opts.has_path("mesh_name"))
    {
        mmesh_name = toc->multimesh_names[0];
    }
    else
    {
        CONDUIT_ASSERT(opts["mesh_name"].dtype().is_string(),
                       "opts['mesh_name'] must be a string");
        for (i = 0; i < toc->nmultimesh; ++i)
        {
            if (toc->multimesh_names[i] == opts["mesh_name"].as_string())
            {
                mmesh_name = toc->multimesh_names[i];
                break;
            }
        }
        CONDUIT_ERROR("No multimesh found matching "
                      << opts["mesh_name"].as_string());
    }
    std::unique_ptr<DBmultimesh, decltype(&DBFreeMultimesh)> multimesh{
        DBGetMultimesh(silofile, mmesh_name.c_str()), &DBFreeMultimesh};
    
    if (!multimesh.get())
    {
        multimesh.release();
        CONDUIT_ERROR("Error fetching multimesh " << mmesh_name);
    }
    // read in the multimesh and add it to the mesh Node
    read_multimesh(silofile, filemap, dirname, multimesh.get(), mesh);
    // get the multivars matching the multimesh
    read_all_multivars(silofile, toc, filemap, dirname,
        mmesh_name, multimesh.get()->nblocks, mesh);
    // get the multimaterials matching the multimesh
    read_all_multimats(silofile, toc, filemap, dirname, mmesh_name,
                        multimesh.get()->nblocks, mesh);
}

//---------------------------------------------------------------------------//
void CONDUIT_RELAY_API
load_mesh(const std::string &root_file_path,
          conduit::Node &mesh)
{
    Node opts;
    load_mesh(root_file_path, opts, mesh);
}

//-----------------------------------------------------------------------------
///
/// opts:
///      mesh_name: "{name}"
///          provide explicit mesh name, for cases where silo data includes
///           more than one mesh.
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API
load_mesh(const std::string &root_file_path,
          const conduit::Node &opts,
          conduit::Node &mesh)
{
    mesh.reset();
    read_mesh(root_file_path, opts, mesh);
}

//---------------------------------------------------------------------------//
DBoptlist *
silo_generate_state_optlist(const Node &n)
{
    DBoptlist *res = NULL;

    if (n.has_path("state"))
    {
        int silo_error = 0;
        const Node &n_state = n["state"];
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
// return a pair where the first entry is the coordset type
// and the second is the labels for the coordinates
std::pair<int, const std::vector<std::string>>
get_coordset_type_labels(const Node &values)
{
    std::string sys =
        conduit::blueprint::mesh::utils::coordset::coordsys(values);
    if (sys == "cartesian")
        return std::make_pair(DB_CARTESIAN, CARTESIAN_AXES);
    else if (sys == "cylindrical")
        return std::make_pair(DB_CYLINDRICAL, CYLINDRICAL_AXES);
    else if (sys == "spherical")
        return std::make_pair(DB_SPHERICAL, SPHERICAL_AXES);
    else
        CONDUIT_ERROR("Unrecognized coordinate system " << sys);
}

//---------------------------------------------------------------------------//
void silo_write_field(DBfile *dbfile,
                      const std::string &var_name,
                      const Node &n_var,
                      Node &n_mesh_info)
{

    if (!n_var.has_path("topology"))
    {
        CONDUIT_ERROR("Missing linked topology! "
                      << "fields/" << var_name << "/topology");
    }

    const std::string topo_name = n_var["topology"].as_string();

    if (!n_mesh_info.has_path(topo_name))
    {
        CONDUIT_ERROR("Invalid linked topology! "
                      << "fields/" << var_name
                      << "/topology: " << topo_name);
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

    // create a name
    int vals_type = 0;
    void *vals_ptr = NULL;

    DataType dtype = n_var["values"].dtype();

    if (dtype.is_float())
    {
        vals_type = DB_FLOAT;
        vals_ptr = (void *)n_values.as_float_ptr();
    }
    else if (dtype.is_double())
    {
        vals_type = DB_DOUBLE;
        vals_ptr = (void *)n_values.as_double_ptr();
    }
    else if (dtype.is_int())
    {
        vals_type = DB_INT;
        vals_ptr = (void *)n_values.as_int_ptr();
    }
    else if (dtype.is_long())
    {
        vals_type = DB_LONG;
        vals_ptr = (void *)n_values.as_long_ptr();
    }
    else if (dtype.is_long_long())
    {
        vals_type = DB_LONG_LONG;
        vals_ptr = (void *)n_values.as_long_long_ptr();
    }
    else if (dtype.is_char())
    {
        vals_type = DB_CHAR;
        vals_ptr = (void *)n_values.as_char_ptr();
    }
    else if (dtype.is_short())
    {
        vals_type = DB_SHORT;
        vals_ptr = (void *)n_values.as_short_ptr();
    }
    else
    {
        // skip the field if we don't support its type
        CONDUIT_INFO("skipping field "
                     << var_name
                     << ", since its type is not implemented, found "
                     << dtype.name());
        continue;
    }

    int silo_error = 0;

    if (mesh_type == "unstructured")
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

        silo_error = DBPutQuadvar1( dbfile,
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
    else if (mesh_type == "points") 
    {

        silo_error = DBPutPointvar1(dbfile, // dbfile Database file pointer.
                                    var_name.c_str(),  // variable name
                                    topo_name.c_str(), // mesh name
                                    vals_ptr,          // data values
                                    num_pts, // Number of elements (points).
                                    vals_type, // Datatype of the variable.
                                    NULL);
    }
    else
    {
        CONDUIT_ERROR("only DBPutQuadvar1 + DBPutUcdvar1 + DBPutPointvar1 var are supported");
    }

    CONDUIT_CHECK_SILO_ERROR(silo_error,
                             " after creating field " << var_name);
}

//---------------------------------------------------------------------------//
int
assign_coords_ptrs(void *coords_ptrs[3],
                   int ndims,
                   conduit::Node &n_coords_compact,
                   const std::vector<std::string> &coordsys_labels)
{

    DataType dtype = n_coords_compact[coordsys_labels[0]].dtype();
    CONDUIT_ASSERT(dtype.compatible(n_coords_compact[coordsys_labels[1]].dtype()),
                   "all coordinate arrays must have same type, got " << dtype.to_string()
                    << " and " << n_coords_compact[coordsys_labels[1]].dtype().to_string());
    if (ndims == 3)
    {
        CONDUIT_ASSERT(dtype.compatible(n_coords_compact[coordsys_labels[2]].dtype()),
                       "all coordinate arrays must have same type");
    }

    if (dtype.is_float())
    {
        coords_ptrs[0] = (void *)n_coords_compact[coordsys_labels[0]].as_float_ptr();
        coords_ptrs[1] = (void *)n_coords_compact[coordsys_labels[1]].as_float_ptr();
        if (ndims == 3)
            coords_ptrs[2] = (void *)n_coords_compact[coordsys_labels[2]].as_float_ptr();
        return DB_FLOAT;
    }
    else if (dtype.is_double())
    {
        
        coords_ptrs[0] = (void *)n_coords_compact[coordsys_labels[0]].as_double_ptr();
        coords_ptrs[1] = (void *)n_coords_compact[coordsys_labels[1]].as_double_ptr();
        if (ndims == 3)
            coords_ptrs[2] = (void *)n_coords_compact[coordsys_labels[2]].as_double_ptr();
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
void silo_write_pointmesh(DBfile *dbfile,
                          const std::string &topo_name,
                          const Node &n_coords,
                          DBoptlist *state_optlist,
                          Node &n_mesh_info) 
{

    int ndims = conduit::blueprint::mesh::utils::coordset::dims(n_coords);
    auto coordsys_type_labels = get_coordset_type_labels(n_coords);
    CONDUIT_CHECK_SILO_ERROR( DBAddOption(state_optlist,
                                          DBOPT_COORDSYS,
                                          &coordsys_type_labels.first),
                             "error adding coordsys option");

    Node n_coords_compact;
    // compaction is necessary to support ragged arrays
    n_coords["values"].compact_to(n_coords_compact);

    int num_pts = n_coords_compact[coordsys_type_labels.second[0]].dtype().number_of_elements();

    n_mesh_info[topo_name]["num_pts"].set(num_pts);
    n_mesh_info[topo_name]["num_elems"].set(num_pts);

    void *coords_ptrs[3] = {NULL, NULL, NULL};
    int coords_dtype = assign_coords_ptrs(coords_ptrs,
                                          ndims,
                                          n_coords_compact,
                                          coordsys_type_labels.second);

    int silo_error = DBPutPointmesh(dbfile,            // silo file ptr
                                    topo_name.c_str(), // mesh name
                                    ndims,             // num_dims
                                    coords_ptrs,       // coords values
                                    num_pts,           // num eles = num pts
                                    coords_dtype,      // type of data array
                                    state_optlist);    // opt list

    CONDUIT_CHECK_SILO_ERROR(silo_error, " after saving DBPutPointmesh");
}

//---------------------------------------------------------------------------//
void silo_write_ucd_zonelist(DBfile *dbfile,
                             const std::string &topo_name,
                             const Node &n_topo,
                             Node &n_mesh_info) 
{
    Node ucd_zlist;

    index_t num_shapes = 0;
    ucd_zlist["shapetype"].set(DataType::c_int(1));
    ucd_zlist["shapesize"].set(DataType::c_int(1));
    ucd_zlist["shapecnt"].set(DataType::c_int(1));

    const Node &n_elements = n_topo["elements"];
    std::string coordset_name = n_topo["coordset"].as_string();

    bool shape_list = true;

    if (n_elements.dtype().is_object())
    {
        // simple path case
        num_shapes = 1;
    } 
    else if (n_elements.dtype().is_list())
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
    int *shapecnt = ucd_zlist["shapecnt"].value();

    int total_num_elems = 0;

    const Node *shape_block = &n_elements;
    Node n_conn;

    for (index_t i = 0; i < num_shapes; i++)
    {
        if (shape_list) 
        {
            // TODO: This is wrong, re work silo logic post bp verify merge
            // const Node *shape_block = n_elements.child_ptr(i);
        }

        std::string topo_shape = shape_block->fetch("shape").as_string();

        const Node &n_mesh_conn = shape_block->fetch("connectivity");

        // convert to compact ints ...
        if (shape_list)
        {
            n_mesh_conn.compact_to(n_conn.append());
        }
        else 
        {
            n_mesh_conn.compact_to(n_conn);
        }

        if (topo_shape == "quad")
        {
            // TODO: check for explicit # of elems
            int num_elems = n_mesh_conn.dtype().number_of_elements() / 4;
            shapetype[i] = DB_ZONETYPE_QUAD;
            shapesize[i] = 4;
            shapecnt[i] = num_elems;
            total_num_elems += num_elems;

        }
        else if (topo_shape == "tri")
        {
            // TODO: check for explicit # of elems
            int num_elems = n_mesh_conn.dtype().number_of_elements() / 3;
            shapetype[i] = DB_ZONETYPE_TRIANGLE;
            shapesize[i] = 3;
            shapecnt[i] = num_elems;
            total_num_elems += num_elems;
        }
        else if (topo_shape == "hex")
        {
            // TODO: check for explicit # of elems
            int num_elems = n_mesh_conn.dtype().number_of_elements() / 8;
            shapetype[i] = DB_ZONETYPE_HEX;
            shapesize[i] = 8;
            shapecnt[i] = num_elems;
            total_num_elems += num_elems;

        }
        else if (topo_shape == "tet")
        {
            // TODO: check for explicit # of elems
            int num_elems = n_mesh_conn.dtype().number_of_elements() / 4;
            shapetype[i] = DB_ZONETYPE_TET;
            shapesize[i] = 4;
            shapecnt[i] = num_elems;
            total_num_elems += num_elems;
        }
        else if (topo_shape == "line")
        {
            // TODO: check for explicit # of elems
            int num_elems = n_mesh_conn.dtype().number_of_elements() / 2;
            shapetype[i] = DB_ZONETYPE_BEAM;
            shapesize[i] = 2;
            shapecnt[i] = num_elems;
            total_num_elems += num_elems;
        }
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
                       zlist_name.c_str(), // silo obj name
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
void silo_write_ucd_mesh(DBfile *dbfile,
                         const std::string &topo_name,
                         const Node &n_coords,
                         DBoptlist *state_optlist,
                         Node &n_mesh_info)
{

    // check if we are 2d or 3d
    int ndims = conduit::blueprint::mesh::utils::coordset::dims(n_coords);
    auto coordsys_type_labels = get_coordset_type_labels(n_coords);

    CONDUIT_CHECK_SILO_ERROR( DBAddOption(state_optlist,
                                          DBOPT_COORDSYS,
                                          &coordsys_type_labels.first),
                              "Failed to create coordsystem labels");

    Node n_coords_compact;
    // compaction is necessary to support ragged arrays
    n_coords["values"].compact_to(n_coords_compact);

    int num_pts = n_coords_compact[coordsys_type_labels.second[0]].dtype().number_of_elements();
    // TODO: check that y & z have the same number of points

    n_mesh_info[topo_name]["num_pts"].set(num_pts);

    void *coords_ptrs[3] = {NULL, NULL, NULL};
    int coords_dtype = assign_coords_ptrs(coords_ptrs,
                                          ndims,
                                          n_coords_compact,
                                          coordsys_type_labels.second);

    int num_elems = n_mesh_info[topo_name]["num_elems"].value();

    std::string zlist_name = topo_name + "_connectivity";

    char const * const coordnames[3] = {coordsys_type_labels.second[0].c_str(),
                                        coordsys_type_labels.second[1].c_str(),
                                        coordsys_type_labels.second[2].c_str()};

    int silo_error = DBPutUcdmesh(dbfile,                      // silo file ptr
                                  topo_name.c_str(),           // mesh name
                                  ndims,                       // number of dims
                                  coordnames, // coord names
                                  coords_ptrs,                 // coords values
                                  num_pts,            // number of points
                                  num_elems,          // number of elements
                                  zlist_name.c_str(), // zone list name
                                  NULL,               // face list names
                                  coords_dtype,       // type of data array
                                  state_optlist);     // opt list

    CONDUIT_CHECK_SILO_ERROR(silo_error, " DBPutUcdmesh");
}

//---------------------------------------------------------------------------//
void silo_write_quad_rect_mesh(DBfile *dbfile,
                               const std::string &topo_name,
                               const Node &n_coords,
                               DBoptlist *state_optlist,
                               Node &n_mesh_info) 
{
    // TODO: also support interleaved:
    // xy, xyz

    // check if we are 2d or 3d
    int ndims = conduit::blueprint::mesh::utils::coordset::dims(n_coords);
    auto coordsys_type_labels = get_coordset_type_labels(n_coords);
    CONDUIT_CHECK_SILO_ERROR( DBAddOption(state_optlist,
                                          DBOPT_COORDSYS,
                                          &coordsys_type_labels.first),
                              "Failed to create coordsystem labels");

    Node n_coords_compact;
    // compaction is necessary to support ragged arrays
    n_coords["values"].compact_to(n_coords_compact);

    int pts_dims[3];
    pts_dims[0] = n_coords_compact[coordsys_type_labels.second[0]].dtype().number_of_elements();
    pts_dims[1] = n_coords_compact[coordsys_type_labels.second[1]].dtype().number_of_elements();
    pts_dims[2] = 1;

    int num_pts = pts_dims[0] * pts_dims[1];
    int num_elems = (pts_dims[0] - 1) * (pts_dims[1] - 1);
    if (ndims == 3)
    {
        pts_dims[2] = n_coords_compact[coordsys_type_labels.second[2]].dtype().number_of_elements();
        num_pts *= pts_dims[2];
        num_elems *= (pts_dims[2] - 1);
    }

    n_mesh_info[topo_name]["num_pts"].set(num_pts);
    n_mesh_info[topo_name]["num_elems"].set(num_elems);
    n_mesh_info[topo_name]["elements/i"] = pts_dims[0] - 1;
    n_mesh_info[topo_name]["elements/j"] = pts_dims[1] - 1;
    
    if (ndims == 3) 
    {
        n_mesh_info[topo_name]["elements/k"] = pts_dims[2] - 1;
    }

    void *coords_ptrs[3] = {NULL, NULL, NULL};
    int coords_dtype = assign_coords_ptrs(coords_ptrs,
                                          ndims,
                                          n_coords_compact,
                                          coordsys_type_labels.second);

    char const * const coordnames[3] = {coordsys_type_labels.second[0].c_str(),
                                        coordsys_type_labels.second[1].c_str(),
                                        coordsys_type_labels.second[2].c_str()};

    int silo_error =
        DBPutQuadmesh(dbfile,                      // silo file ptr
                      topo_name.c_str(),           // mesh name
                      coordnames, // coord names
                      coords_ptrs,                 // coords values
                      pts_dims,                    // dims vals
                      ndims,                       // number of dims
                      coords_dtype,                // type of data array
                      DB_COLLINEAR,   // DB_COLLINEAR or DB_NONCOLLINEAR
                      state_optlist); // opt list

    CONDUIT_CHECK_SILO_ERROR(silo_error, " DBPutUcdmesh");
}

//---------------------------------------------------------------------------//
void silo_write_structured_mesh(DBfile *dbfile,
                                const std::string &topo_name,
                                const Node &n_topo,
                                const Node &n_coords,
                                DBoptlist *state_optlist,
                                Node &n_mesh_info) 
{
    // also support interleaved:
    // xy, xyz

    // check if we are 2d or 3d
    int ndims = conduit::blueprint::mesh::utils::coordset::dims(n_coords);

    CONDUIT_ASSERT(2 <= ndims && ndims <= 3, "Dimension count not accepted: " << ndims);
    auto coordsys_type_labels = get_coordset_type_labels(n_coords);
    CONDUIT_CHECK_SILO_ERROR( DBAddOption(state_optlist,
                                          DBOPT_COORDSYS,
                                          &coordsys_type_labels.first),
                              "Error adding option");

    Node n_coords_compact;
    // compaction is necessary to support ragged arrays
    n_coords["values"].compact_to(n_coords_compact);

    int num_pts = n_coords_compact[coordsys_type_labels.second[0]].dtype().number_of_elements();
    CONDUIT_ASSERT(num_pts ==
                   n_coords_compact[coordsys_type_labels.second[1]].dtype().number_of_elements(),
                   "element count mismatch");

    n_mesh_info[topo_name]["num_pts"].set(num_pts);

    void *coords_ptrs[3] = {NULL, NULL, NULL};

    int coords_dtype = assign_coords_ptrs(coords_ptrs,
                                          ndims,
                                          n_coords_compact,
                                          coordsys_type_labels.second);

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

    n_mesh_info[topo_name]["num_pts"].set(num_pts);
    n_mesh_info[topo_name]["num_elems"].set(num_elems);
    n_mesh_info[topo_name]["elements/i"] = ele_dims[0];
    n_mesh_info[topo_name]["elements/j"] = ele_dims[1];

    if (ndims == 3)
    {
        n_mesh_info[topo_name]["elements/k"] = ele_dims[2];
        pts_dims[2] = ele_dims[2] + 1;
    }

    char const * const coordnames[3] = {coordsys_type_labels.second[0].c_str(),
                                        coordsys_type_labels.second[1].c_str(),
                                        coordsys_type_labels.second[2].c_str()};

    int silo_error =
        DBPutQuadmesh(dbfile,                      // silo file ptr
                      topo_name.c_str(),           // mesh name
                      coordnames, // coord names
                      coords_ptrs,                 // coords values
                      pts_dims,                    // dims vals
                      ndims,                       // number of dims
                      coords_dtype,                // type of data array
                      DB_NONCOLLINEAR, // DB_COLLINEAR or DB_NONCOLLINEAR
                      state_optlist);  // opt list

    CONDUIT_CHECK_SILO_ERROR(silo_error, " DBPutQuadmesh");
}

//---------------------------------------------------------------------------//
void silo_mesh_write(const Node &n, 
                     DBfile *dbfile,
                     const std::string &silo_obj_path)
{
    int silo_error = 0;
    char silo_prev_dir[256];

    if (!silo_obj_path.empty())
    {
        silo_error += DBGetDir(dbfile, silo_prev_dir);
        silo_error += DBMkDir(dbfile, silo_obj_path.c_str());
        silo_error += DBSetDir(dbfile, silo_obj_path.c_str());

        CONDUIT_CHECK_SILO_ERROR(silo_error,
                                 " failed to make silo directory:"
                                 << silo_obj_path);
    }

    DBoptlist *state_optlist = silo_generate_state_optlist(n);

    Node n_mesh_info;

    auto topo_itr = n["topologies"].children();
    while (topo_itr.has_next())
    {
        const Node &n_topo = topo_itr.next();

        std::string topo_name = topo_itr.name();

        std::string topo_type = n_topo["type"].as_string();

        n_mesh_info[topo_name]["type"].set(topo_type);

        if (topo_type == "unstructured")
        {
            std::string ele_shape = n_topo["elements/shape"].as_string();
            if( ele_shape != "point")
            {
                // we need a zone list for a ucd mesh
                silo_write_ucd_zonelist(dbfile,
                                        topo_name,
                                        n_topo,
                                        n_mesh_info);
            }
            else
            {
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

        if (topo_type == "unstructured")
        {
            silo_write_ucd_mesh(dbfile, topo_name, n_coords,
                                state_optlist, n_mesh_info);
        }
        else if (topo_type == "rectilinear") 
        {
            silo_write_quad_rect_mesh(dbfile, topo_name, n_coords,
                                      state_optlist, n_mesh_info);
        }
        else if (topo_type == "uniform") 
        {
            // silo doesn't have a direct path for a uniform mesh
            // we need to convert its implicit uniform coords to
            // implicit rectilinear coords

            Node n_rect;
            Node &n_rect_coords = n_rect["coordsets"][coordset_name];
            Node &n_rect_topo = n_rect["topologies"][topo_name];
            blueprint::mesh::topology::uniform::to_rectilinear(
                n_topo, n_rect_topo, n_rect_coords);

            silo_write_quad_rect_mesh(dbfile, topo_name, n_rect_coords,
                                      state_optlist, n_mesh_info);

        }
        else if (topo_type == "structured")
        {
            silo_write_structured_mesh(dbfile, topo_name, n_topo, n_coords,
                                       state_optlist, n_mesh_info);
        }
        else if (topo_type == "points")
        {
            silo_write_pointmesh(dbfile, topo_name, n_coords,
                                 state_optlist, n_mesh_info);
        }
    }

    if (n.has_path("fields")) 
    {
        auto itr = n["fields"].children();
        while (itr.has_next())
        {
            const Node &n_var = itr.next();
            std::string var_name = itr.name();

            silo_write_field(dbfile, var_name, n_var, n_mesh_info);
        }
    }

    if(state_optlist)
    {
        silo_error = DBFreeOptlist(state_optlist);
    }

    CONDUIT_CHECK_SILO_ERROR(silo_error,
                             " freeing state optlist.");

    if (!silo_obj_path.empty()) 
    {
        silo_error = DBSetDir(dbfile, silo_prev_dir);

        CONDUIT_CHECK_SILO_ERROR(silo_error,
                                 " changing silo directory to previous path");
    }
}

//-----------------------------------------------------------------------------
// return the directory within a silo file where a mesh domain's info
// (mesh object, materials, fields, ...) will be stored
std::string
get_domain_silo_directory(int domain,
                          int nfiles,
                          int ndomains)
{
    if (ndomains < nfiles)
    {
        return "domain" + std::to_string(domain);
    }
    else
    {
        return "";
    }
}

//-----------------------------------------------------------------------------
// return the path to the file where a mesh domain will be stored
std::string
get_domain_file(int curr_domain,
                int ndomains,
                int nfiles,
                const std::string &root_file) 
{
    if (nfiles <= 0) 
    {
        return root_file;
    }
    else if (ndomains == nfiles)
    {
        return "domain" + std::to_string(curr_domain) + ".silo";
    }
    else
    {
        // ndomains > nfiles
        // see overlink spec
        int remainder = ndomains % nfiles;
        int a = 1 + int(ndomains / nfiles);
        if (curr_domain < remainder * a)
        {
            return "domfile" + std::to_string(int(curr_domain / a)) + ".silo";
        }
        else
        {
            return "domfile" + std::to_string(int((curr_domain - remainder) / (a - 1))) + ".silo";
        }
    }
}

//-----------------------------------------------------------------------------
std::string
get_mesh_domain_name(const conduit::Node &topo, bool overlink) 
{
    CONDUIT_ASSERT(topo.number_of_children() == 1,
                   "Multiple topologies not supported");
    if (overlink)
    {
        return "MESH";
    }

    return topo.children().next().name();
}

//-----------------------------------------------------------------------------
void write_multimesh(DBfile *root,
                     const std::string &mmesh_name,
                     std::vector<std::string> &mesh_domain_names,
                     std::vector<int> &mesh_domain_types) 
{
    std::vector<const char *> domain_name_ptrs;

    for (auto domain : mesh_domain_names)
    {
        domain_name_ptrs.push_back(domain.c_str());
    }

    CONDUIT_CHECK_SILO_ERROR( DBPutMultimesh(root,
                                             mmesh_name.c_str(),
                                             domain_name_ptrs.size(),
                                             domain_name_ptrs.data(),
                                             mesh_domain_types.data(),
                                             NULL),
                              "Error putting multimesh");
}

//-----------------------------------------------------------------------------
void
write_multimaterial(DBfile *root,
                    const std::string &mmat_name,
                    const std::string &mmesh_name,
                    std::vector<std::string> mat_domains) 
{
    std::vector<const char *> domain_name_ptrs;
    std::unique_ptr<DBoptlist, decltype(&DBFreeOptlist)> optlist{ DBMakeOptlist(1),
                                                                  &DBFreeOptlist};

    if (!optlist.get())
    {
        optlist.release();
        CONDUIT_ERROR("Error creating options");
    }

    // have to const_cast because converting to void *
    CONDUIT_CHECK_SILO_ERROR( DBAddOption(optlist.get(),
                                          DBOPT_MMESH_NAME,
                                          const_cast<char *>(mmesh_name.c_str())),
                              "Error creating options for putting multimat");
    
    for (auto domain : mat_domains) 
    {
        domain_name_ptrs.push_back(domain.c_str());
    }

    CONDUIT_CHECK_SILO_ERROR( DBPutMultimat(root,
                                            mmat_name.c_str(),
                                            mat_domains.size(),
                                            domain_name_ptrs.data(),
                                            optlist.get()),
                              "Error putting multimaterial");
}

//-----------------------------------------------------------------------------
void
write_multivar(DBfile *root,
               const std::string &mvar_name,
               const std::string &mmesh_name,
               std::vector<std::pair<std::string, int>> var_domains)
{
    std::vector<const char *> domain_name_ptrs;
    std::unique_ptr<DBoptlist, decltype(&DBFreeOptlist)> optlist{DBMakeOptlist(1),
                                                                 &DBFreeOptlist};
    std::vector<int> domain_var_types;
    
    if (!optlist.get())
    {
        optlist.release();
        CONDUIT_ERROR("Error creating options");
    }

    // have to const_cast because converting to void *
    CONDUIT_CHECK_SILO_ERROR( DBAddOption(optlist.get(),
                                          DBOPT_MMESH_NAME,
                                          const_cast<char *>(mmesh_name.c_str())),
                              "Error creating options for putting multivar");
    for (auto domain : var_domains)
    {
        domain_name_ptrs.push_back(domain.first.c_str());
        domain_var_types.push_back(domain.second);
    }
    CONDUIT_CHECK_SILO_ERROR( DBPutMultivar(root,
                                            mvar_name.c_str(),
                                            var_domains.size(),
                                            domain_name_ptrs.data(),
                                            domain_var_types.data(),
                                            optlist.get()),
                            "Error putting multivar");
}

//-----------------------------------------------------------------------------
///      silo_type: "default", "pdb", "hdf5", "hdf5_sec2", "hdf5_stdio",
///                 "hdf5_mpio", "hdf5_mpiposix", "taurus", "unknown"
///            when 'path' exists, "default" ==> "unknown"
///            else,               "default" ==> "hdf5"
int
parse_type_option(const std::string &path,
                  const conduit::Node &opts)
{
    int type = 0;

    if (conduit::utils::is_file(path)) 
    {
        type = DB_UNKNOWN;
    }
    else
    {
        type = DB_HDF5;
    }
    
    if (opts.has_path("silo_type"))
    {
        std::string silo_type = opts["silo_type"].as_string();
        if (silo_type == "pdb")
        {
            type = DB_PDB;
        }
        else if (silo_type == "hdf5")
        {
            type = DB_HDF5;
        }
        else if (silo_type == "hdf5_sec2")
        {
            type = DB_HDF5_SEC2;
        }
        else if (silo_type == "hdf5_stdio")
        {
            type = DB_HDF5_STDIO;
        }
        else if (silo_type == "hdf5_mpio")
        {
            type = DB_HDF5_MPIO;
        }
        // TODO when can I uncomment this
        // else if (silo_type == "hdf5_mpiposix")
        // {
        //     type = DB_HDF5_MPIPOSIX; 
        // }
        else if (silo_type == "taurus") 
        {
            type = DB_TAURUS;
        }
        else if (silo_type == "unknown") 
        {
            type = DB_UNKNOWN;
        }
        else
        {
            CONDUIT_ASSERT(silo_type == "default",
                           "Unrecognized 'silo_type' option " << silo_type);
        }
    }
    return type;
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
///      silo_type: "default", "pdb", "hdf5", "hdf5_sec2", "hdf5_stdio",
///                 "hdf5_mpio", "hdf5_mpiposix", "taurus", "unknown"
///            when 'path' exists, "default" ==> "unknown"
///            else,               "default" ==> "hdf5"
///
///      suffix: "default", "cycle", "none"
///            when # of domains == 1,  "default"   ==> "none"
///            else,                    "default"   ==> "cycle"
///
///      mesh_name:  (used if present, default ==> "mesh")
///
///      number_of_files:  {# of files}
///            when "multi_file" or "overlink":
///                 <= 0, use # of files == # of domains
///                  > 0, # of files == number_of_files
///
// TODO check that all opts are honored
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API write_mesh(const conduit::Node &mesh,
                                  const std::string &path,
                                  const conduit::Node &opts,
                                  CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm mpi_comm))
{
    // The assumption here is that everything is multi domain

    std::string opts_file_style = "default";
    std::string opts_suffix     = "default";
    std::string opts_mesh_name  = "mesh";
    int         opts_num_files  = -1;
    bool        opts_truncate   = false;
    bool        overlink        = false;

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
                          "\"multi_file\", or \"overlink\",");
        }

    }

    // check for + validate silo_type option
    // TODO

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
        opts_mesh_name = opts["mesh_name"].as_string();
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

    // special logic for overlink
    if (overlink = opts_file_style == "overlink")
    {
        opts_mesh_name = "MMESH";
        opts_file_style == "multi_file";
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
    bool is_valid = detail::clean_mesh(mesh, multi_dom, mpi_comm);
#else
    bool is_valid = detail::clean_mesh(mesh, multi_dom);
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

    // ----------------------------------------------------
    // if using multi_file, create output dir
    // ----------------------------------------------------
    if(opts_file_style == "multi_file")
    {
        // setup the directory
        output_dir = path;
        // at this point for suffix, we should only see
        // cycle or none -- default has been resolved
        if(opts_suffix == "cycle")
        {
            // TODO check with cyrus - do we want to have cycle_649320 in the dir name?
            output_dir += conduit_fmt::format(".cycle_{:06d}",cycle);
        }

        bool dir_ok = false;

        // let rank zero handle dir creation
        if(par_rank == 0)
        {
            // check if the dir exists
            dir_ok = utils::is_directory(output_dir);
            if(!dir_ok)
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
    std::string root_filename = path;

    // at this point for suffix, we should only see 
    // cycle or none -- default has been resolved
    if(opts_suffix == "cycle")
    {
        root_filename += conduit_fmt::format(".cycle_{:06d}",cycle);
    }

    root_filename += ".silo";

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

    // at this point for file_style,
    // default has been resolved, we need to just handle:
    //   root_only, multi_file
    if(opts_file_style == "root_only")
    {
        // if truncate, first touch needs to open the file with
        //          open_opts["mode"] = "wt";

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
                for(int i = 0; i < local_num_domains; ++i)
                {
                    // if truncate, first rank to touch the file needs
                    // to open at
                    Node open_opts;
                    if( (global_root_file_created.as_int() == 0) 
                        && opts_truncate)
                    {
                        CONDUIT_ERROR("TODO Truncate case not yet implemented");
                        // Node open_opts;
                        // open_opts["mode"] = "wt";
                        // // TODO I need a way to support truncate?
                        // // How to open a file with opts for silo?
                        // hnd.open(root_filename,file_protocol,open_opts);
                        // local_root_file_created.set((int)1);
                    }
                    
                    if(!hnd.is_open())
                    {
                        hnd.open(root_filename,file_protocol);
                    }

                    const Node &dom = multi_dom.child(i);
                    // figure out the proper mesh path the file
                    std::string mesh_path = "";

                    if(global_num_domains == 1)
                    {
                        // no domain prefix, write to mesh name
                        mesh_path = opts_mesh_name;
                    }
                    else
                    {
                        // multiple domains, we need to use a domain prefix
                        uint64 domain = dom["state/domain_id"].to_uint64();
                        mesh_path = conduit_fmt::format("domain_{:06d}/{}",
                                                        domain,
                                                        opts_mesh_name);
                    }
                    // TODO help where did my dbfile come from!
                    silo_mesh_write(dom, dbfile, mesh_path);
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
    else if(global_num_domains == num_files)
    {
        // write out each domain
        // writes are independent, so no baton here
        for(int i = 0; i < local_num_domains; ++i)
        {
            const Node &dom = multi_dom.child(i);
            uint64 domain = dom["state/domain_id"].to_uint64();

            std::string output_file  = conduit::utils::join_file_path(output_dir,
                                                conduit_fmt::format("domain_{:06d}.{}:{}",
                                                                    domain,
                                                                    file_protocol,
                                                                    opts_mesh_name));
            // properly support truncate vs non truncate
            if(opts_truncate)
            {
                // TODO help I'm scared
                relay::io::save(dom, output_file);
            }
            else
            {
                relay::io::save_merged(dom, output_file);
            }
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
        books["local_domain_to_file"].set(DataType::int32(local_num_domains));
        books["local_domain_status"].set(DataType::int32(local_num_domains));

        // batons
        books["local_file_batons"].set(DataType::int32(num_files));
        books["global_file_batons"].set(DataType::int32(num_files));

        // used to track first touch
        books["local_file_created"].set(DataType::int32(num_files));
        books["global_file_created"].set(DataType::int32(num_files));

        // size local # of domains
        int32_array local_domain_to_file = books["local_domain_to_file"].value();
        int32_array local_domain_status  = books["local_domain_status"].value();

        // size num total files
        /// batons
        int32_array local_file_batons    = books["local_file_batons"].value();
        int32_array global_file_batons   = books["global_file_batons"].value();
        /// file created flags
        int32_array local_file_created    = books["local_file_created"].value();
        int32_array global_file_created   = books["global_file_created"].value();


        Node d2f_map;
        detail::gen_domain_to_file_map(global_num_domains,
                                       num_files,
                                       books);
        int32_array global_d2f = books["global_domain_to_file"].value();

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
                for(int d = 0; d < local_num_domains; ++d)
                {
                    if(local_domain_status[d] == 1)
                        local_file_batons[f] = par_rank;
                    else
                        local_file_batons[f] = -1;
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
                        // reuse this handle for all domains in the file
                        relay::io::IOHandle hnd;
                        if(local_domain_status[d] == 1 &&  // pending
                           local_domain_to_file[d] == f) // destined for this file
                        {
                            // now is the time to write!
                            // pattern is:
                            //  file_%06llu.{protocol}:/domain_%06llu/...
                            const Node &dom = multi_dom.child(d);
                            uint64 domain_id = dom["state/domain_id"].to_uint64();

                            // construct file name
                            std::string file_name = conduit_fmt::format(
                                                        "file_{:06d}.{}",
                                                        f,
                                                        file_protocol);

                            std::string output_file = conduit::utils::join_file_path(output_dir,
                                                                                     file_name);

                            // now the path in the file, and domain id
                            std::string curr_path = conduit_fmt::format(
                                                            "domain_{:06d}/{}",
                                                             domain_id,
                                                             opts_mesh_name);

                            try
                            {
                                // if truncate == true check if this is the first time we are
                                // touching file, and use wt
                                Node open_opts;
                                if(opts_truncate && global_file_created[f] == 0)
                                {
                                   open_opts["mode"] = "wt";

                                   local_file_created[f]  = 1;
                                   global_file_created[f] = 1;
                                }

                                if(!hnd.is_open())
                                {
                                    hnd.open(output_file, open_opts);
                                }

                                // CONDUIT_INFO("rank " << par_rank << " output_file"
                                //              << output_file << " path " << path);

                                // TODO help me get a dbfile!
                                silo_mesh_write(dom, dbfile, curr_path);
                                
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
            // If you  need to debug the baton alog:
            // std::cout << "[" << par_rank << "] "
            //              << " twirls: " << twirls
            //              << " details\n"
            //              << books.to_yaml();

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

    // root_file_writer will now write out the root file
    if(par_rank == root_file_writer)
    {
        std::string output_dir_base, output_dir_path;
        conduit::utils::rsplit_file_path(output_dir,
                                         output_dir_base,
                                         output_dir_path);

        std::string output_tree_pattern;
        std::string output_file_pattern;

        if(opts_file_style == "root_only")
        {
            output_file_pattern = root_filename;
            if(global_num_domains == 1)
            {
                output_tree_pattern = "/";
            }
            else
            {
                output_tree_pattern = "/domain_%06d/";
            }
        }
        else if(global_num_domains == num_files)
        {
            output_tree_pattern = "/";
            output_file_pattern = conduit::utils::join_file_path(
                                                output_dir_base,
                                                "domain_%06d." + file_protocol);
        }
        else
        {
            output_tree_pattern = "/domain_%06d";
            output_file_pattern = conduit::utils::join_file_path(
                                                output_dir_base,
                                                "file_%06d." + file_protocol);
        }

        Node root;
        Node &bp_idx = root["blueprint_index"];

        // TODO: Use MPI ver vs providing the domains?
        ::conduit::blueprint::mesh::generate_index(multi_dom.child(0),
                                                   opts_mesh_name,
                                                   global_num_domains,
                                                   bp_idx[opts_mesh_name]);

        // work around conduit and manually add state fields
        if(multi_dom.child(0).has_path("state/cycle"))
        {
          bp_idx[ opts_mesh_name + "/state/cycle"] = multi_dom.child(0)["state/cycle"].to_int32();
        }

        if(multi_dom.child(0).has_path("state/time"))
        {
          bp_idx[opts_mesh_name + "/state/time"] = multi_dom.child(0)["state/time"].to_double();
        }

        root["protocol/name"]    = file_protocol;
        root["protocol/version"] = CONDUIT_VERSION;

        root["number_of_files"]  = num_files;
        root["number_of_trees"]  = global_num_domains;

        // TODO: make sure this is relative
        root["file_pattern"]     = output_file_pattern;
        root["tree_pattern"]     = output_tree_pattern;

        relay::io::IOHandle hnd;

        // if not root only, this is the first time we are writing 
        // to the root file -- make sure to properly support truncate
        Node open_opts;
        if(opts_file_style != "root_only" && opts_truncate)
        {
            open_opts["mode"] = "wt";
        }

        hnd.open(root_filename, file_protocol, open_opts);
        hnd.write(root);
        // TODO here is where we write multimesh and multivars
        hnd.close();
    }

    // barrier at end of work to avoid file system race
    // (non root task could write the root file in write_mesh, 
    // but root task is always the one to read the root file
    // in read_mesh.

    #ifdef CONDUIT_RELAY_IO_MPI_ENABLED
        MPI_Barrier(mpi_comm);
    #endif
}


void CONDUIT_RELAY_API write_mesh_OUTDATED(const conduit::Node &mesh,
                                  const std::string &path,
                                  const conduit::Node &opts)
{
    // TODO scrap this and use the blueprint version instead, fill in where it uses iohandle for silo calls

    int i, type, ndomains, nfiles;
    std::string mmesh_name = "mesh";
    bool overlink = false;
    DBfile *silofile;
    std::map<std::string, std::unique_ptr<DBfile, decltype(&DBClose)>> filemap;
    std::vector<const conduit::Node *> domains = blueprint::mesh::domains(mesh);
    // nfiles is the number of non-root files, so 0 implies root-only
    // nfiles will == # domains unless:
    // a) root_only option passed
    // b) no style option passed and domains == 1
    // c) number_of_files > 0 specified
    nfiles = ndomains = domains.size();

    // parse out options
    type = parse_type_option(path, opts);
    if (opts.has_path("mesh_name"))
    {
        mmesh_name = opts["mesh_name"].as_string();
    }

    if (opts.has_path("file_style"))
    {
        std::string file_style = opts["file_style"].as_string();
        if (file_style == "overlink")
        {
            mmesh_name = "MMESH";
            overlink = true;
        }
        else if (file_style == "multi")
        {
            // nothing to do
        }
        else if (file_style == "root_only")
        {
            nfiles = 0;
        }
        else
        {
            CONDUIT_ASSERT(file_style == "default",
                           "Unrecognized file_style option " << file_style);
            if (nfiles == 1)
            {
                nfiles = 0;
            }
        }
    }
    else
    {
        if (nfiles == 1)
        {
            nfiles = 0;
        }
    }
    
    if (opts.has_path("number_of_files") && nfiles > 0) 
    {
        if (opts["number_of_files"].as_int() > 0 && nfiles > 0) 
        {
            nfiles = opts["number_of_files"].as_int();
        }

        CONDUIT_ASSERT(nfiles <= ndomains, "More files ("
                                               << nfiles << ") than domains ("
                                               << ndomains << ") not allowed");
    }

    // end option parsing
    silofile = get_or_create(filemap, path, type);

    std::vector<std::string> silo_mesh_paths;
    std::vector<int> silo_mesh_types;
    std::map<std::string, std::vector<std::string>> silo_material_paths;
    std::map<std::string, std::vector<std::pair<std::string, int>>> silo_variable_paths;

    for (i = 0; i < ndomains; ++i)
    {
        std::string domain_file = get_domain_file(i, ndomains, nfiles, path);
        std::string silo_dir = get_domain_silo_directory(i, nfiles, ndomains);
        const Node *dom = domains[i];
        Node fix_name_collisions, replace_field_names;

        std::string mesh_domain_name = conduit::utils::join_path(
            silo_dir, 
            get_mesh_domain_name(
                    (*dom)["topologies"],
                    overlink));

        // TODO
        // instead of renaming everything to domain_000000 + lalalala
        // we want to make subdirectories called domain_000000 and then put the things in there
        // we don't need to change the topo names

        // maybe just rework this entirely so that it does "write_topo" and then "write_field"

        // see blueprint write mesh too! - steal it and change this up to work for silo

        int var_type;

        // prepare multivar paths
        if (dom->has_path("fields"))
        {
            auto field_itr = (*dom)["fields"].children();

            while (field_itr.has_next())
            {
                const Node &curr_field = field_itr.next();
                std::string var_name{field_itr.name()};
                // TODO fix this so that it writes to fixed width of 6
                std::string new_var_name{VN("domain_00000" + std::to_string(i) + "_" + var_name)};

                replace_field_names["fields"][new_var_name].set_external(curr_field);

                bool singleFile{nfiles == 0};

                std::stringstream tmp;
                if (singleFile)
                {
                    // CONDUIT_ERROR("Uh oh spaghettio");
                    // tmp << "block" << i << "/" << VN(var_name);
                    tmp << new_var_name;
                }
                else
                {
                    tmp << domain_file << ":" << new_var_name;
                }

                silo_variable_paths[var_name].push_back(std::make_pair(tmp.str(), var_type));
            }
        }
        // TODO material paths as well

        silo_mesh_write(*dom,
                        get_or_create(filemap, domain_file, type), 
                        silo_dir);

        if (domain_file == path) 
        {
            // domain is in root file
            silo_mesh_paths.push_back(mesh_domain_name);
        }
        else
        {
            // domain is not in root file
            silo_mesh_paths.push_back(domain_file + ":" + mesh_domain_name);
        }
    }
    // We always will want a multimesh so this is unconditional
    write_multimesh(silofile, mmesh_name, silo_mesh_paths, silo_mesh_types);
    if (silo_material_paths.size() > 0) 
    {
        for (const auto &pair : silo_material_paths) 
        {
            CONDUIT_ASSERT( pair.second.size() == static_cast<std::size_t>(ndomains),
                           "material " << pair.first << " not specified for all domains");
            write_multimaterial(silofile, pair.first, mmesh_name, pair.second);
        }
    }

    if (silo_variable_paths.size() > 0) 
    {
        for (const auto &pair : silo_variable_paths)
        {
            CONDUIT_ASSERT(
                pair.second.size() == static_cast<std::size_t>(ndomains),
                "variable " << pair.first << " not specified for all domains");
            write_multivar(silofile, pair.first, mmesh_name, pair.second);
        }
    }
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
                                  const std::string &path) 
{
    // empty opts
    Node opts;
    write_mesh(mesh, path, opts);
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
                                 const std::string &path) 
{
    // empty opts
    Node opts;
    save_mesh(mesh, path, opts);
}

//-----------------------------------------------------------------------------
/// The following options can be passed via the opts Node:
//-----------------------------------------------------------------------------
/// opts:
///      TODO
///
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save_mesh(const conduit::Node &mesh,
                                 const std::string &path,
                                 const conduit::Node &opts) 
{
    conduit::utils::remove_path_if_exists(path);
    write_mesh(mesh, path, opts);
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
