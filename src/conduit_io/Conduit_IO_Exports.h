//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC
// Produced at the Lawrence Livermore National Laboratory. 
// 
// All rights reserved.
// 
// This source code cannot be distributed without further review from 
// Lawrence Livermore National Laboratory.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: Conduit_IO_Exports.h
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_IO_EXPORTS_H
#define CONDUIT_IO_EXPORTS_H

//-----------------------------------------------------------------------------
// -- define proper lib exports for various platforms -- 
//-----------------------------------------------------------------------------
#if defined(_WIN32)
#if defined(CONDUIT_IO_EXPORTS) || defined(conduit_io_EXPORTS)
#define CONDUIT_IO_API __declspec(dllexport)
#else
#define CONDUIT_IO_API __declspec(dllimport)
#endif
#if defined(_MSC_VER)
// Turn off warning about lack of DLL interface
#pragma warning(disable:4251)
// Turn off warning non-dll class is base for dll-interface class.
#pragma warning(disable:4275)
// Turn off warning about identifier truncation
#pragma warning(disable:4786)
#endif
#else
# if __GNUC__ >= 4 && (defined(CONDUIT_IO_EXPORTS) || defined(conduit_io_EXPORTS))
#   define CONDUIT_IO_API __attribute__ ((visibility("default")))
# else
#   define CONDUIT_IO_API /* hidden by default */
# endif
#endif

#endif



