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
/// file: conduit_relay_io.h
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_RELAY_IO_H
#define CONDUIT_RELAY_IO_H

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
void CONDUIT_RELAY_API conduit_relay_io_about(conduit_node *cnode);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API conduit_relay_io_initialize(void);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API conduit_relay_io_finalize(void);

///
/// ``save`` works like a 'set' to the file.
///

//-----------------------------------------------------------------------------
// conduit_relay_io_save
// For simpler use cases, accepts NULL for protocol and options
void CONDUIT_RELAY_API conduit_relay_io_save(conduit_node *cnode,
                                             const char *path,
                                             const char *protocol,
                                             conduit_node *coptions);

///
/// ``save_merged`` works like an update to the file.
///

//-----------------------------------------------------------------------------
// conduit_relay_io_save_merged 
// For simpler use cases, accepts NULL for protocol and options
void CONDUIT_RELAY_API conduit_relay_io_save_merged(conduit_node *cnode,
                                                    const char *path,
                                                    const char *protocol,
                                                    conduit_node *coptions);
                                                     

//-----------------------------------------------------------------------------
///
/// ``add_step`` adds a new time step of data to the file.
///
// void CONDUIT_RELAY_API conduit_relay_io_add_step(conduit_node *cnode,
//                                                  const char *path);

//-----------------------------------------------------------------------------
// conduit_relay_io_add_step                                        
// For simpler use cases, accepts NULL for protocol and options
void CONDUIT_RELAY_API conduit_relay_io_add_step(conduit_node *cnode,
                                                 const char *path,
                                                 const char *protocol,
                                                 conduit_node *coptions);

///
/// ``load`` works like a 'set', the node is reset and then populated
///

//-----------------------------------------------------------------------------
// conduit_relay_io_load
// For simpler use cases, accepts NULL for protocol and options
void CONDUIT_RELAY_API conduit_relay_io_load(const char *path,
                                             const char *protocol,
                                             conduit_node *coptions,
                                             conduit_node *cnode);

//-----------------------------------------------------------------------------
// conduit_relay_io_load_step_and_domain
// For simpler use cases, accepts NULL for protocol and options
void CONDUIT_RELAY_API conduit_relay_io_load_step_and_domain(const char *path,
                                                             const char *protocol,
                                                             int step,
                                                             int domain,
                                                             conduit_node *coptions,
                                                             conduit_node *cnode);

///
/// ``query_number_of_domains`` return the number of time steps.
///
//-----------------------------------------------------------------------------
int CONDUIT_RELAY_API conduit_relay_io_query_number_of_steps(const char *path);

///
/// ``query_number_of_domains`` return the number of domains.
///
//-----------------------------------------------------------------------------
int CONDUIT_RELAY_API conduit_relay_io_query_number_of_domains(
                                                            const char *path);

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
