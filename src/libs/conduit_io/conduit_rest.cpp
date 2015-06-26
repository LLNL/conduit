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
/// file: conduit_rest.cpp
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
#include "CivetServer.h"

#define BUFFERSIZE 65536
#include "b64/encode.h"

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_rest.hpp"
#include "Conduit_IO_Config.hpp"

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::io --
//-----------------------------------------------------------------------------

namespace io 
{

//-----------------------------------------------------------------------------
// -- begin conduit::io::rest --
//-----------------------------------------------------------------------------

namespace rest 
{
    
//-----------------------------------------------------------------------------
// simple interface to launch a blocking REST server    
//-----------------------------------------------------------------------------
void
serve(Node *n,
      index_t port)

{
    RESTServer srv;
    srv.serve(n,true,port);
}



//-----------------------------------------------------------------------------
// -- Internal REST Server Interface Classes -
//-----------------------------------------------------------------------------

class RESTHandler : public CivetHandler
{
public:
        
        //---------------------------------------------------------------------------//
        RESTHandler(RESTServer &server,
                    Node *node)
        : m_server(&server),
          m_node(node)
        {
            // empty
        }

        //---------------------------------------------------------------------------//
        ~RESTHandler()
        {
            // empty
        }
        
        
        //---------------------------------------------------------------------------//
        bool handleAll(CivetServer *server,
                       struct mg_connection *conn) 
        {
            const struct mg_request_info *req_info = mg_get_request_info(conn);
            
            std::string uri(req_info->uri);
            std::string uri_cmd;
            std::string uri_next;
            utils::rsplit_string(uri,"/",uri_cmd,uri_next);
            
            if(uri_cmd == "get-schema")
            {
                return handle_get_schema(server,conn);
            }
            else if(uri_cmd == "get-value")
            {
                return handle_get_value(server,conn);                
            }
            else if(uri_cmd == "get-base64-json")
            {
                return handle_get_base64_json(server,conn);
            }
            else if(uri_cmd == "kill-server")
            {
                return handle_release(server,conn);
            }
            else
            {
                // TODO: unknown REST command
            }

            return true;
        }
        
        //---------------------------------------------------------------------------//
        bool handlePost(CivetServer *server,
                       struct mg_connection *conn) 
        {
            return handleAll(server,conn);
        }
        
        //---------------------------------------------------------------------------//
        bool handleGet(CivetServer *server,
                       struct mg_connection *conn) 
        {
            return handleAll(server,conn);
        }
        
        //---------------------------------------------------------------------------//
        // Handles a request from the client for the node's schema.
        bool handle_get_schema(CivetServer *server,
                               struct mg_connection *conn)
        {
            mg_printf(conn, "%s",m_node->schema().to_json(true).c_str());
            return 1;
        }

        //---------------------------------------------------------------------------//
        // Handles a request from the client for a specific value in the node.
        bool handle_get_value(CivetServer *server,
                              struct mg_connection *conn)
        {
            // TODO size checks?
            char post_data[2048];
            char cpath[2048];

            int post_data_len = mg_read(conn, post_data, sizeof(post_data));

            mg_get_var(post_data, post_data_len, "cpath", cpath, sizeof(cpath));

            mg_printf(conn, "{ \"datavalue\": %s }",
                      m_node->fetch(cpath).to_json(false).c_str());
            return true;
        }

        //---------------------------------------------------------------------------//
        // Handles a request from the client for a compact, base64 encoded version
        // of the node.
        bool handle_get_base64_json(CivetServer *server,
                                    struct mg_connection *conn)
        {
            std::string b64_json = m_node->to_base64_json();
            mg_printf(conn, "%s",b64_json.c_str());
            return true;
        }

        //---------------------------------------------------------------------------//
        // Handles a request from the client to shutdown the REST server
        bool handle_release(CivetServer *server,
                            struct mg_connection *conn)
        {
            m_server->shutdown();
            return true;
        }
    
        
  private:
      RESTServer *m_server;
      Node       *m_node;

};


//-----------------------------------------------------------------------------
RESTServer::RESTServer()
: m_server(NULL),
  m_handler(NULL),
  m_port(""),
  m_running(false)
{
    
}

//-----------------------------------------------------------------------------
RESTServer::~RESTServer()
{
    shutdown();
}

//-----------------------------------------------------------------------------
bool
RESTServer::is_running() const
{
    return m_running;
}

//-----------------------------------------------------------------------------
void
RESTServer::serve(Node *node,
                  bool block,
                  index_t port)
{
    if(is_running())
    {
        CONDUIT_ERROR("RESTServer is already running");
    }
        
    std::ostringstream oss;
    oss << port;
    m_port = oss.str();
    const char *options[] = { "document_root", CONDUIT_REST_CLIENT_ROOT,
                              "listening_ports", m_port.c_str(),
                              "num_threads", "2",
                               NULL};

    m_handler = new RESTHandler(*this,node);
    try
    {
    
        m_server = new CivetServer(options);

    }
    catch(CivetException except)
    {
        // Catch Civet Exception and use Conduit's error handling mech.
        CONDUIT_ERROR("RESTServer failed to bind civet server on port " << port);
    }
    
    // check for valid context    
    const struct mg_context *ctx = m_server->getContext();
    if(ctx == NULL)
    {
         CONDUIT_ERROR("RESTServer failed to bind civet server on port " << port);
    }else
    {
        CONDUIT_INFO("conduit::io::RESTServer instance active on port: " 
                     << port);
    }

    m_server->addHandler("/api/*",m_handler);

    m_running = true;

    if(block)
    {
        while(is_running()) // wait for shutdown()
        {
#if defined(CONDUIT_PLATFORM_WINDOWS)
        Sleep(1000);
#else
        sleep(10);
#endif
        }
    }
}

//-----------------------------------------------------------------------------
void     
RESTServer::shutdown()
{
    if(is_running())
    {
        CONDUIT_INFO("closing conduit::io::RESTServer instance on port: " 
                     << m_port);
        
        m_running = false;
        delete m_server;
        delete m_handler;


        m_handler = NULL;
        m_server  = NULL;
    }
}



};
//-----------------------------------------------------------------------------
// -- end conduit::io::rest --
//-----------------------------------------------------------------------------


};
//-----------------------------------------------------------------------------
// -- end conduit::io --
//-----------------------------------------------------------------------------


};
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
