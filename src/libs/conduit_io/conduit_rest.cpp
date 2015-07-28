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

class RequestHandler : public CivetHandler
{
public:
        
        //---------------------------------------------------------------------------//
        RequestHandler(RESTServer &server,
                       Node *node)
        : m_server(&server),
          m_node(node)
        {
            // empty
        }

        //---------------------------------------------------------------------------//
        ~RequestHandler()
        {
            // cleanup any alloced web socket instances
            for(size_t i=0; i < m_sockets.size(); i++)
            {
                delete m_sockets[i];
            }
        }
        
        
        //---------------------------------------------------------------------------//
        // Main handler, dispatches to proper api handlers. 
        //---------------------------------------------------------------------------//
        bool handle_request(CivetServer *server,
                            struct mg_connection *conn) 
        {
            const struct mg_request_info *req_info = mg_get_request_info(conn);
            
            std::string uri(req_info->uri);
            std::string uri_cmd;
            std::string uri_next;
            utils::rsplit_string(uri,"/",uri_cmd,uri_next);
            
            if(uri_cmd == "get-schema")
            {
                return handle_rest_get_schema(server,conn);
            }
            else if(uri_cmd == "get-value")
            {
                return handle_rest_get_value(server,conn);                
            }
            else if(uri_cmd == "get-base64-json")
            {
                return handle_rest_get_base64_json(server,conn);
            }
            else if(uri_cmd == "kill-server")
            {
                return handle_rest_shutdown(server,conn);
            }
            else
            {
                CONDUIT_INFO("Unknown REST request uri:" << uri_cmd );
                // TODO: what behavior does returning false this trigger?
                return false;
            }
            return true;
        }
        
        //---------------------------------------------------------------------------//
        // wire all CivetHandler handlers to "handle_request"
        //---------------------------------------------------------------------------//
        bool
        handlePost(CivetServer *server,
                   struct mg_connection *conn) 
        {
            return handle_request(server,conn);
        }
        
        //---------------------------------------------------------------------------//
        // wire all CivetHandler handlers to "handle_request"
        //---------------------------------------------------------------------------//
        bool
        handleGet(CivetServer *server,
                  struct mg_connection *conn) 
        {
            return handle_request(server,conn);
        }
        
        //---------------------------------------------------------------------------//
        // Handles a request from the client for the node's schema.
        //---------------------------------------------------------------------------//
        bool
        handle_rest_get_schema(CivetServer *server,
                          struct mg_connection *conn)
        {
            mg_printf(conn, "%s",m_node->schema().to_json(true).c_str());
            return 1;
        }

        //---------------------------------------------------------------------------//
        // Handles a request from the client for a specific value in the node.
        //---------------------------------------------------------------------------//
        bool
        handle_rest_get_value(CivetServer *server,
                              struct mg_connection *conn)
        {
            // TODO size checks?
            char post_data[2048];
            char cpath[2048];

            int post_data_len = mg_read(conn, post_data, sizeof(post_data));

            mg_get_var(post_data, post_data_len, "cpath", cpath, sizeof(cpath));

            mg_printf(conn, "{ \"datavalue\": %s }",
                      m_node->fetch(cpath).to_json().c_str());
            return true;
        }

        //---------------------------------------------------------------------------//
        // Handles a request from the client for a compact, base64 encoded version
        // of the node.
        //---------------------------------------------------------------------------//
        bool
        handle_rest_get_base64_json(CivetServer *server,
                                    struct mg_connection *conn)
        {
            std::string b64_json = m_node->to_json("base64_json");
            mg_printf(conn, "%s",b64_json.c_str());
            return true;
        }

        //---------------------------------------------------------------------------//
        // Handles a request from the client to shutdown the REST server
        //---------------------------------------------------------------------------//
        bool
        handle_rest_shutdown(CivetServer *server,
                             struct mg_connection *conn)
        {
            m_server->shutdown();
            return true;
        }
        
        
        //---------------------------------------------------------------------------//
        // Handlers for WebSockets
        // These aren't exposed via the CivetWeb C++ interface, so the 
        // process is a bit more complex.
        //---------------------------------------------------------------------------//

        //---------------------------------------------------------------------------//
        // static callback used when a web socket initially connects
        //---------------------------------------------------------------------------//
        static int 
        handle_websocket_connect(const struct mg_connection *conn,
                                 void *cbdata)
        {
            CONDUIT_INFO("conduit::io::RESTServer WebSocket Connected");
            return 0;
        }

        //---------------------------------------------------------------------------//
        // static callback used when a web socket connection becomes active
        //---------------------------------------------------------------------------//
        static void
        handle_websocket_ready(struct mg_connection *conn,
                               void *cbdata)
        {
            struct mg_context *ctx  = mg_get_context(conn);
            RequestHandler *handler = (RequestHandler *)cbdata;

            // lock context while we add a new websocket
            mg_lock_context(ctx);
            {
            
                WebSocket *ws = new WebSocket();
                ws->set_connection(conn);
                handler->m_sockets.push_back(ws);

                // send connection successful message
                // TODO: locking semantics for WebSocket::send ?
                Node n;
                n["type"] = "info";
                n["message"] = "websocket ready!";
                ws->send(n);
            }
            // unlock context
            mg_unlock_context(ctx);
        }

        //---------------------------------------------------------------------------//
        // static callback used when a web socket connection drops
        //---------------------------------------------------------------------------//
        static int
        handle_websocket_recv(struct mg_connection *conn,
                              int bits,
                              char *data,
                              size_t len,
                              void *cbdata)
        {
            struct mg_context *ctx  = mg_get_context(conn);
            RequestHandler *handler = (RequestHandler *)cbdata;
            
            //lock context as we search
            WebSocket *ws = NULL;
            mg_lock_context(ctx);
            {
                ws = handler->find_socket_for_connection(conn);
                // TODO: call future recv handler.
                
                std::string schema(data,len);
                try
                {
                    // parse with pure json parser first
                    // 
                    // this will be what we want in most cases
                    //
                    Generator g(schema,"json");
                    Node n;
                    g.walk(n);

                    CONDUIT_INFO("WebSocket rcvd message:" << n.to_json());
                }
                catch(conduit::Error e)
                {
                    CONDUIT_INFO("Error parsing JSON response from browser\n" 
                                 << e.message());
                }

            }// unlock context
            mg_unlock_context(ctx);
            
            if(ws == NULL)
            {
                CONDUIT_ERROR("Bad websocket state");
            }
            return 1;
        }
        
        //---------------------------------------------------------------------------//
        // static callback used when a web socket connection drops
        //---------------------------------------------------------------------------//
        static void
        handle_websocket_close(const struct mg_connection *conn,
                               void *cbdata)
        {
            struct mg_context *ctx  = mg_get_context(conn);
            RequestHandler *handler = (RequestHandler *)cbdata;
            
            WebSocket *ws = NULL;
            // lock context while we cleanup the websocket
            mg_lock_context(ctx);
            {
                ws = handler->find_socket_for_connection(conn);
                if(ws != NULL)
                {
                    ws->set_connection(NULL);
                }
            }
            // unlock context
            mg_unlock_context(ctx);
            
            if(ws == NULL)
            {
                CONDUIT_ERROR("Bad websocket state");
            }

            CONDUIT_INFO("conduit::io::RESTServer WebSocket Disconnected");
        }
        
        //---------------------------------------------------------------------------//
        // loops over active web sockets and returns the instance
        // that is linked to the given connection
        //---------------------------------------------------------------------------//
        WebSocket *
        find_socket_for_connection(const struct mg_connection *conn)
            
        {
            WebSocket *res = NULL;
            if(conn != NULL)
            {
                // loop over websockets and return the one associated with 
                // the passed civet connection 
                for(size_t i=0; i< m_sockets.size();i++)
                {
                    if(m_sockets[i]->m_connection == conn)
                    {
                        res = m_sockets[i];
                    }
                }
            }
            return res;
        }


  private:
      RESTServer                 *m_server;
      Node                       *m_node;
      std::vector<WebSocket*>     m_sockets;
};

//-----------------------------------------------------------------------------
// WebSocket Class Implementation
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
WebSocket::WebSocket()
: m_connection(NULL)
{
    // empty
}


//-----------------------------------------------------------------------------
WebSocket::~WebSocket()
{
    // empty
}


//-----------------------------------------------------------------------------
void
WebSocket::set_connection(mg_connection *connection)
{
    m_connection = connection;
}

//-----------------------------------------------------------------------------
bool
WebSocket::is_connected() const
{
    return m_connection != NULL;
}

//-----------------------------------------------------------------------------
void
WebSocket::send(const Node &data,
                const std::string &protocol)
{
    if(m_connection == NULL)
    {
        CONDUIT_INFO("attempt to write to bad websocket connection");
        return;
    }

    // convert our node to json using the requested conduit protocol
    std::ostringstream oss;
    data.to_json_stream(oss,protocol);
    
    // get a pointer to our message data and its length
    const char   *msg = oss.str().c_str();
    size_t        msg_len = oss.str().size();
    
    // send our message
    mg_websocket_write(m_connection,
                       WEBSOCKET_OPCODE_TEXT,
                       msg,
                       msg_len);
}


//-----------------------------------------------------------------------------
// RESTServer Class Implementation
//-----------------------------------------------------------------------------

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
        CONDUIT_INFO("RESTServer instance is already running");
        return;
    }

    // create our request handler
    m_handler = new RequestHandler(*this,node);

    // civetweb takes strings as arguments
    // convert the port number to a string.
    std::ostringstream oss;
    oss << port;
    m_port = oss.str();

    // setup
    const char *options[] = { "document_root", CONDUIT_REST_CLIENT_ROOT,
                              "listening_ports", m_port.c_str(),
                              "num_threads", "2",
                               NULL};



    try
    {
    
        m_server = new CivetServer(options);

    }
    catch(CivetException except)
    {
        // Catch Civet Exception and use Conduit's error handling mech.
        CONDUIT_ERROR("RESTServer failed to bind civet server on port " 
                      << port);
    }
    
    // check for valid context    
    const struct mg_context *ctx = m_server->getContext();
    if(ctx == NULL)
    {
         CONDUIT_ERROR("RESTServer failed to bind civet server on port " 
                       << port);
    }else
    {
        CONDUIT_INFO("conduit::io::RESTServer instance active on port: " 
                     << port);
    }

    // setup REST handlers
    m_server->addHandler("/api/*",m_handler);
    
    //
    // setup web socket handlers
    //
    mg_set_websocket_handler((struct mg_context*)ctx,
                             "/websocket",
                             // static handlers for the web socket case
                             RequestHandler::handle_websocket_connect,
                             RequestHandler::handle_websocket_ready,
                             RequestHandler::handle_websocket_recv,
                             RequestHandler::handle_websocket_close,
                             // pass our handler instance as context for 
                             // the static callbacks
                             m_handler);

    // signal we are valid
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
