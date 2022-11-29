// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_annotations.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_ANNOTATIONS_HPP
#define CONDUIT_ANNOTATIONS_HPP

//-----------------------------------------------------------------------------
// -- conduit includes --
//-----------------------------------------------------------------------------
#include "conduit_core.hpp"

//-----------------------------------------------------------------------------
#if defined(CONDUIT_USE_CALIPER)
#include <caliper/cali.h>
#endif

//-----------------------------------------------------------------------------
//
/// CONDUIT_ANNOTATE_ZZZ macros are used for caliper performance annotations.
//
//-----------------------------------------------------------------------------
#if defined(CONDUIT_USE_CALIPER)
#define CONDUIT_ANNOTATE_MARK_BEGIN( name ) CALI_MARK_BEGIN( name )
#define CONDUIT_ANNOTATE_MARK_END( name ) CALI_MARK_END( name )
#define CONDUIT_ANNOTATE_MARK_FUNCTION CALI_CXX_MARK_FUNCTION
#define CONDUIT_ANNOTATE_MARK_SCOPE( name )  CALI_CXX_MARK_SCOPE( name )
#else // these are empty when caliper is not enabled
#define CONDUIT_ANNOTATE_MARK_BEGIN( name )
#define CONDUIT_ANNOTATE_MARK_END( name )
#define CONDUIT_ANNOTATE_MARK_FUNCTION
#define CONDUIT_ANNOTATE_MARK_SCOPE( name )
#endif

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

// fwd declare Node
class Node;

//-----------------------------------------------------------------------------
// -- begin conduit::annotations --
//-----------------------------------------------------------------------------
namespace annotations
{
  //---------------------------------------------------------------------------
  /// Caliper performance annotations environment management.
  ///
  /// Setup and tear down of Caliper.
  /// 
  /// These are all noops when Caliper is not enabled.
  ///
  /// These routines are optional, targeted for cases where caliper env vars
  /// are not used or a client code does not setup caliper itself.
  ///
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  /// Report if conduit as built with caliper support
  //---------------------------------------------------------------------------
  bool CONDUIT_API supported();

  //---------------------------------------------------------------------------
  /// Initialize performance annotations
  //---------------------------------------------------------------------------
  void CONDUIT_API initialize();

  //---------------------------------------------------------------------------
  /// Initialize performance annotations with options
  /// opts:
  ///   config: (optional caliper config string)
  ///   services: (optional caliper services string)
  ///   output_file: (optional string with output filename)
  //---------------------------------------------------------------------------
  void CONDUIT_API initialize(const Node &opts);

  //---------------------------------------------------------------------------
  /// Flush performance annotations
  //---------------------------------------------------------------------------
  void CONDUIT_API flush();

  //---------------------------------------------------------------------------
  /// Finalize performance annotations
  //---------------------------------------------------------------------------
  void CONDUIT_API finalize();

}
//-----------------------------------------------------------------------------
// -- end conduit::annotations --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------


#endif
