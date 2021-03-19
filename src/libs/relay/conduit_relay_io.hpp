// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_relay_io.hpp
///
//-----------------------------------------------------------------------------


#ifndef CONDUIT_RELAY_IO_HPP
#define CONDUIT_RELAY_IO_HPP

//-----------------------------------------------------------------------------
// conduit lib include
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "conduit_relay_exports.h"
#include "conduit_relay_config.h"
#include "conduit_relay_io_identify_protocol.hpp"

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
// -- begin conduit::relay::io --
//-----------------------------------------------------------------------------
namespace io
{

//-----------------------------------------------------------------------------
/// The about methods construct human readable info about how relay io was
/// configured.
//-----------------------------------------------------------------------------
std::string CONDUIT_RELAY_API about();
void        CONDUIT_RELAY_API about(conduit::Node &res);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API initialize();

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API finalize();

///
/// ``save`` works like a 'set' to the file.
///

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save(const Node &node,
                            const std::string &path);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save(const Node &node,
                            const std::string &path,
                            const Node &options);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save(const Node &node,
                            const std::string &path,
                            const std::string &protocol);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save(const Node &node,
                            const std::string &path,
                            const std::string &protocol,
                            const Node &options);

///
/// ``save_merged`` works like an update to the file.
///

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save_merged(const Node &node,
                                   const std::string &path);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save_merged(const Node &node,
                                   const std::string &path,
                                   const Node &options);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save_merged(const Node &node,
                                   const std::string &path,
                                   const std::string &protocol);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save_merged(const Node &node,
                                   const std::string &path,
                                   const std::string &protocol,
                                   const Node &options);


///
/// ``add_step`` adds a new time step of data to the file.
///

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API add_step(const Node &node,
                                const std::string &path);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API add_step(const Node &node,
                                const std::string &path,
                                const std::string &protocol);


//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API add_step(const Node &node,
                                const std::string &path,
                                const std::string &protocol,
                                const Node &options);

///
/// ``load`` works like a 'set', the node is reset and then populated
///

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API load(const std::string &path,
                            Node &node);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API load(const std::string &path,
                            const Node &options,
                            Node &node);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API load(const std::string &path,
                            const std::string &protocol,
                            Node &node);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API load(const std::string &path,
                            const std::string &protocol,
                            const Node &options,
                            Node &node);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API load(const std::string &path,
                            const std::string &protocol,
                            int step,
                            int domain,
                            Node &node);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API load(const std::string &path,
                            const std::string &protocol,
                            int step,
                            int domain,
                            const Node &options,
                            Node &node);

///
/// ``load_merged`` works like an update, for the object case, entries are read
///  into the node. If the node is already in the OBJECT_T role, children are
///  added
///

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API load_merged(const std::string &path,
                                   Node &node);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API load_merged(const std::string &path,
                                   const Node &options,
                                   Node &node);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API load_merged(const std::string &path,
                                   const std::string &protocol,
                                   Node &node);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API load_merged(const std::string &path,
                                   const std::string &protocol,
                                   const Node &options,
                                   Node &node);

///
/// ``query_number_of_steps`` return the number of steps.
///
//-----------------------------------------------------------------------------
int CONDUIT_RELAY_API query_number_of_steps(const std::string &path);

///
/// ``query_number_of_domains`` return the number of domains.
///
//-----------------------------------------------------------------------------
int CONDUIT_RELAY_API query_number_of_domains(const std::string &path);

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


#endif
