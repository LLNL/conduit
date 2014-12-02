/*****************************************************************************
* Copyright (c) 2014, Lawrence Livermore National Security, LLC
* Produced at the Lawrence Livermore National Laboratory. 
* 
* All rights reserved.
* 
* This source code cannot be distributed without further review from 
* Lawrence Livermore National Laboratory.
*****************************************************************************/

///
/// file: Conduit_Exports.h
///

#ifndef CONDUIT_EXPORTS_H
#define CONDUIT_EXPORTS_H

#if defined(_WIN32)
#if defined(CONDUIT_EXPORTS) || defined(conduit_EXPORTS)
#define CONDUIT_API __declspec(dllexport)
#else
#define CONDUIT_API __declspec(dllimport)
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
# if __GNUC__ >= 4 && (defined(CONDUIT_EXPORTS) || defined(conduit_EXPORTS))
#   define CONDUIT_API __attribute__ ((visibility("default")))
# else
#   define CONDUIT_API /* hidden by default */
# endif
#endif

#endif



