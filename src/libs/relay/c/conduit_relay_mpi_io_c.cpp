// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_relay_mpi_io_c.cpp
///
//-----------------------------------------------------------------------------
#include "conduit_relay_mpi_io.h"

#include "conduit.hpp"
#include "conduit_relay_mpi_io.hpp"
#include "conduit_cpp_to_c.hpp"

//-----------------------------------------------------------------------------
// -- begin extern C
//-----------------------------------------------------------------------------

extern "C" {

using namespace conduit;

//-----------------------------------------------------------------------------
void
conduit_relay_mpi_io_about(conduit_node *cnode, MPI_Fint comm)
{
    Node *n = cpp_node(cnode);
    relay::mpi::io::about(*n, MPI_Comm_f2c(comm));
}

//-----------------------------------------------------------------------------
void
conduit_relay_mpi_io_initialize(MPI_Fint comm)
{
    relay::mpi::io::initialize(MPI_Comm_f2c(comm));
}

//-----------------------------------------------------------------------------
void
conduit_relay_mpi_io_finalize(MPI_Fint comm)
{
    relay::mpi::io::finalize(MPI_Comm_f2c(comm));
}

//-----------------------------------------------------------------------------
void
conduit_relay_mpi_io_save(conduit_node *cnode,
                          const char *path,
                          const char *protocol,
                          conduit_node *coptions,
                          MPI_Fint comm)
{
    Node *n = cpp_node(cnode);
    Node *opt = cpp_node(coptions);

    std::string path_str;
    std::string protocol_str;
    
    if(path != NULL)
        path_str = std::string(path);

    if(protocol != NULL)
        protocol_str = std::string(protocol);    
    
    
    if(opt != NULL)
    {
        relay::mpi::io::save(*n,
                             path_str,
                             protocol_str,
                             *opt,
                             MPI_Comm_f2c(comm));
    }
    else
    {
        relay::mpi::io::save(*n,
                             path_str,
                             protocol_str,
                             MPI_Comm_f2c(comm));
    }
}

//-----------------------------------------------------------------------------
void
conduit_relay_mpi_io_save_merged(conduit_node *cnode,
                                 const char *path,
                                 const char *protocol,
                                 conduit_node *coptions,
                                 MPI_Fint comm)
{
    Node *n = cpp_node(cnode);
    Node *opt = cpp_node(coptions);
    
    std::string path_str;
    std::string protocol_str;
    
    if(path != NULL)
        path_str = std::string(path);

    if(protocol != NULL)
        protocol_str = std::string(protocol);    
    
    
    if(opt != NULL)
    {
        relay::mpi::io::save_merged(*n,
                                    path_str,
                                    protocol_str,
                                    *opt,
                                    MPI_Comm_f2c(comm));
    }
    else
    {
        relay::mpi::io::save_merged(*n,
                                    path_str,
                                    protocol_str,
                                    MPI_Comm_f2c(comm));
    }
}


//-----------------------------------------------------------------------------
void conduit_relay_mpi_io_add_step(conduit_node *cnode,
                                   const char *path,
                                   const char *protocol,
                                   conduit_node *coptions,
                                   MPI_Fint comm)
{
    Node *n = cpp_node(cnode);
    Node *opt = cpp_node(coptions);
    
    std::string path_str;
    std::string protocol_str;
    
    if(path != NULL)
        path_str = std::string(path);

    if(protocol != NULL)
        protocol_str = std::string(protocol);
    
    if(opt != NULL)
    {
        relay::mpi::io::add_step(*n,
                                 path_str,
                                 protocol_str,
                                 *opt,
                                 MPI_Comm_f2c(comm));
    }
    else
    {
        relay::mpi::io::add_step(*n,
                                 path_str,
                                 protocol_str,
                                 MPI_Comm_f2c(comm));
    }
}

//-----------------------------------------------------------------------------
void
conduit_relay_mpi_io_load(const char *path,
                          const char *protocol, 
                          conduit_node *coptions,
                          conduit_node *cnode,
                          MPI_Fint comm)
{
    Node *n = cpp_node(cnode);
    Node *opt = cpp_node(coptions);
    
    std::string path_str;
    std::string protocol_str;
    
    if(path != NULL)
        path_str = std::string(path);

    if(protocol != NULL)
        protocol_str = std::string(protocol);
    
    if(opt != NULL)
    {
        relay::mpi::io::load(path_str,
                             protocol_str,
                             *opt,
                             *n,
                             MPI_Comm_f2c(comm));
    }
    else
    {
        relay::mpi::io::load(path_str,
                             protocol_str,
                             *n,
                             MPI_Comm_f2c(comm));        
    }
}

//-----------------------------------------------------------------------------
void
conduit_relay_mpi_io_load_step_and_domain(const char *path,
                                          const char *protocol,
                                          int step,
                                          int domain,
                                          conduit_node *coptions,
                                          conduit_node *cnode,
                                          MPI_Fint comm)
{
    Node *n = cpp_node(cnode);
    Node *opt = cpp_node(coptions);
    
    std::string path_str;
    std::string protocol_str;
    
    if(path != NULL)
        path_str = std::string(path);

    if(protocol != NULL)
        protocol_str = std::string(protocol);
    
    
    if(opt != NULL)
    {
        relay::mpi::io::load(path_str,
                             protocol_str,
                             step,
                             domain,
                             *opt,
                             *n,
                             MPI_Comm_f2c(comm));
    }
    else
    {
        relay::mpi::io::load(path_str,
                             protocol_str,
                             step,
                             domain,
                             *n,
                             MPI_Comm_f2c(comm));
    }
}

//-----------------------------------------------------------------------------
int
conduit_relay_mpi_io_query_number_of_domains(const char *path, MPI_Fint comm)
{
    return relay::mpi::io::query_number_of_domains(std::string(path),
                                                   MPI_Comm_f2c(comm));
}

//-----------------------------------------------------------------------------
int
conduit_relay_mpi_io_query_number_of_steps(const char *path, MPI_Fint comm)
{
    return relay::mpi::io::query_number_of_steps(std::string(path),
                                                 MPI_Comm_f2c(comm));
}

}
//-----------------------------------------------------------------------------
// -- end extern C
//-----------------------------------------------------------------------------
