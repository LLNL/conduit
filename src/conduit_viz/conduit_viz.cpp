//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2015, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see https://lc.llnl.gov/conduit/.
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
/// file: conduit_viz.cpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// std lib includes
//-----------------------------------------------------------------------------
#if defined(CONDUIT_PLATFORM_WINDOWS)
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include <string.h>

//-----------------------------------------------------------------------------
// external lib includes
//-----------------------------------------------------------------------------
#include "civetweb.h"

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_viz.hpp"
#include "Conduit_Viz_Config.hpp"

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::viz --
//-----------------------------------------------------------------------------

namespace viz 
{

// TODO refactor to move this server into a class

Node *theNode;
bool visualizerIsRunning;

//---------------------------------------------------------------------------//
int
handle_get_schema(struct mg_connection *conn,
                  void *cbdata)
{
    mg_printf(conn, "%s",theNode->schema().to_json(true).c_str());
    return 1;
}

//---------------------------------------------------------------------------//
// Handles a request from the client for a specific value in the Node
int
handle_get_value(struct mg_connection *conn,
                 void *cbdata)
{
    char post_data[2048];
    char cpath[2048];

    const struct mg_request_info *ri = mg_get_request_info(conn);
    int post_data_len = mg_read(conn, post_data, sizeof(post_data));

    mg_get_var(post_data, post_data_len, "cpath", cpath, sizeof(cpath));
    mg_printf(conn, "{ \"datavalue\": %s }",
              theNode->fetch(cpath).to_json(false).c_str());
    return 1;
}

//---------------------------------------------------------------------------//
int 
handle_release(struct mg_connection *conn,
               void *cbdata)
{
    visualizerIsRunning = false;
    return 1;
}
    

//---------------------------------------------------------------------------//
// Event handler, deals with requests other than GETs to page resources.
// int
// ev_handler(struct mg_connection *conn,
//            enum mg_event ev)
// {
//     switch (ev)
//     {
//         case MG_AUTH:
//         {
//             return MG_TRUE;
//         }
//         case MG_REQUEST:
//         {
//             if (!strcmp(conn->uri, "/api/get-schema"))
//             {
//                 handle_get_schema(conn);
//                 return MG_TRUE;
//             }
//
//             if (!strcmp(conn->uri, "/api/get-value"))
//             {
//                 handle_get_value(conn);
//                 return MG_TRUE;
//             }
//
//             if (!strcmp(conn->uri, "/api/kill-server"))
//             {
//                 visualizerIsRunning = false;
//                 return MG_TRUE;
//             }
//         }
//         default:
//         {
//             return MG_FALSE;
//         }
//     }
// }

//---------------------------------------------------------------------------//
void
visualize(Node *n)
{
    theNode = n;

    const char * options[] = { "document_root", CONDUIT_VIZ_CLIENT_ROOT,
                               "listening_ports", "8080", 0
                             };

    struct mg_callbacks callbacks;
    struct mg_context *ctx;

    memset(&callbacks, 0, sizeof(callbacks));
    ctx = mg_start(&callbacks, 0, options);
    
    mg_set_request_handler(ctx, "/api/get-schema", handle_get_schema, 0);
    mg_set_request_handler(ctx, "/api/get-value", handle_get_value, 0);
    mg_set_request_handler(ctx, "/api/kill-server", handle_release, 0);

    visualizerIsRunning = true;

    printf("Visualizer is running!\n");
    
    while (visualizerIsRunning)
    {
#if defined(CONDUIT_PLATFORM_WINDOWS)
        Sleep(1000);
#else
        sleep(10);
#endif
    }

    printf("Visualizer has been closed; execution continuing...\n");
}


};
//-----------------------------------------------------------------------------
// -- end conduit::viz --
//-----------------------------------------------------------------------------



};
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
