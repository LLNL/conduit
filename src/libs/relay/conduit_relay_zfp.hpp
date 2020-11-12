// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_relay_zfp.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_RELAY_ZFP_HPP
#define CONDUIT_RELAY_ZFP_HPP

//-----------------------------------------------------------------------------
// external lib includes
//-----------------------------------------------------------------------------
#include "zfparray1.h"
#include "zfparray2.h"
#include "zfparray3.h"

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_relay_io.hpp"

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

static const std::string ZFP_HEADER_FIELD_NAME = "zfp_header";
static const std::string ZFP_COMPRESSED_DATA_FIELD_NAME = "zfp_compressed_data";

zfp::array* CONDUIT_RELAY_API unwrap_zfparray(const Node &node);

int CONDUIT_RELAY_API wrap_zfparray(const zfp::array* arr,
                                    Node &node);

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
