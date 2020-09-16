// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_relay_mpi_io.h
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_RELAY_MPI_IO_H
#define CONDUIT_RELAY_MPI_IO_H

#include <mpi.h>

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit.h"
#include "conduit_relay_exports.h"
#include "conduit_relay_config.h"

//-----------------------------------------------------------------------------
// -- begin extern C
//-----------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

//-----------------------------------------------------------------------------
// -- conduit_relay io c interface  --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API conduit_relay_mpi_io_about(conduit_node *cnode,
                                                  MPI_Fint comm);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API conduit_relay_mpi_io_initialize(MPI_Fint comm);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API conduit_relay_mpi_io_finalize(MPI_Fint comm);

///
/// ``save`` works like a 'set' to the file.
///

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API conduit_relay_mpi_io_save(conduit_node *cnode,
                                                 const char *path,
                                                 const char *protocol,
                                                 conduit_node *copt,
                                                 MPI_Fint comm);

///
/// ``save_merged`` works like an update to the file.
///
                                                 
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API conduit_relay_mpi_io_save_merged(conduit_node *cnode,
                                                        const char *path,
                                                        const char *protocol,
                                                        conduit_node *copt,
                                                        MPI_Fint comm);

///
/// ``add_step`` adds a new step of data to the file.
///

void CONDUIT_RELAY_API conduit_relay_mpi_io_add_step(conduit_node *cnode,
                                                     const char *path,
                                                     const char *protocol,
                                                     conduit_node *coptions,
                                                     MPI_Fint comm);

///
/// ``load`` works like a 'set', the node is reset and then populated
///

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API conduit_relay_mpi_io_load(const char *path,
                                                 const char *protocol,
                                                 conduit_node *options,
                                                 conduit_node *node,
                                                 MPI_Fint comm);


//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API conduit_relay_mpi_io_load_step_and_domain(const char *path,
                                                                 const char *protocol,
                                                                 int step,
                                                                 int domain,
                                                                 conduit_node *options,
                                                                 conduit_node *node,
                                                                 MPI_Fint comm);

///
/// ``query_number_of_domains`` return the number of time steps.
///
//-----------------------------------------------------------------------------
int CONDUIT_RELAY_API conduit_relay_mpi_io_query_number_of_steps(
                                                  const char *path,
                                                  MPI_Fint comm);

///
/// ``query_number_of_domains`` return the number of domains.
///
//-----------------------------------------------------------------------------
int CONDUIT_RELAY_API conduit_relay_mpi_io_query_number_of_domains(
                                                  const char *path,
                                                  MPI_Fint comm);

#ifdef __cplusplus
}
#endif
//-----------------------------------------------------------------------------
// -- end extern C
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// -- end header guard ifdef
//-----------------------------------------------------------------------------
#endif
