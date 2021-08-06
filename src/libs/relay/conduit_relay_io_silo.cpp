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

//-----------------------------------------------------------------------------
// -- begin conduit::relay::<mpi>::io::silo --
//-----------------------------------------------------------------------------
namespace silo
{

//---------------------------------------------------------------------------//
void
save_mesh(const Node &mesh,
          const std::string &root_file_path)
{
    // TODO: 'save_mesh' will output the information in the given Blueprint-compliant
    // 'mesh' node to a Silo tree at 'path' with the following structure:
    // - 'root_file_path': for example, /path/to/root.silo
    // - '/path/to/root/Index.silo': mesh index file
    // - '/path/to/root/(name).silo': domain file for (name) domain
    // - '/path/to/root.silo': symbolic link to '/path/to/root/Index.silo'
    //
    // If files exist at any of the above paths when 'save_mesh' is called,
    // they should be removed prior to processing.
}

//---------------------------------------------------------------------------//
void
load_mesh(const std::string &root_file_path,
          conduit::Node &mesh)
{
    // TODO: 'load_mesh' will read the mesh information encoded in the Silo
    // root file at the given path and load this information into 'mesh'
    // as a Blueprint-compliant structure
    // - 'root_file_path': for example, /path/to/root.silo
    // - '/path/to/root/Index.silo': the name of the index file pointed to by
    //   'root.silo'
    // - '/path/to/root/(name).silo': translated into a Blueprint domain
    //   at path (name)
    //
    // If any data exists in the 'mesh' node when this function is called, it
    // should be removed prior to processing (e.g. via 'mesh.reset()').
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
