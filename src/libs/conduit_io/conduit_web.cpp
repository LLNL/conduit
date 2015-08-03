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
// For details, see: http://cyrush.github.io/conduit.
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
#include <string.h>

//-----------------------------------------------------------------------------
// external lib includes
//-----------------------------------------------------------------------------
#include "civetweb.h"
#include "CivetServer.h"

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_web.hpp"
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
    WebServer srv;
    srv.serve(n,true,port);
}

};
//-----------------------------------------------------------------------------
// -- end conduit::io::rest --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// -- Internal REST Server Interface Classes -
//-----------------------------------------------------------------------------

class RequestHandler : public CivetHandler
{
public:
        
        //---------------------------------------------------------------------------//
        RequestHandler(WebServer &server)
        : m_server(&server),
          m_node(NULL)
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
        // Bridge to preserve our old interface
        //---------------------------------------------------------------------------//
        void set_node(Node *node)
        {
            m_node = node;
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
            if(m_node != NULL)
            {
                mg_printf(conn, "%s",m_node->schema().to_json(true).c_str());
            }
            return true;
        }

        //---------------------------------------------------------------------------//
        // Handles a request from the client for a specific value in the node.
        //---------------------------------------------------------------------------//
        bool
        handle_rest_get_value(CivetServer *server,
                              struct mg_connection *conn)
        {
            if(m_node != NULL)
            {
                // TODO size checks?
                char post_data[2048];
                char cpath[2048];

                int post_data_len = mg_read(conn, post_data, sizeof(post_data));

                mg_get_var(post_data, post_data_len, "cpath", cpath, sizeof(cpath));

                mg_printf(conn, "{ \"datavalue\": %s }",
                          m_node->fetch(cpath).to_json().c_str());
            }
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
            if(m_node != NULL)
            {
                std::string b64_json = m_node->to_json("base64_json");
                mg_printf(conn, "%s",b64_json.c_str());
            }
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
            CONDUIT_INFO("conduit::io::WebServer WebSocket Connected");
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

            CONDUIT_INFO("conduit::io::WebServer WebSocket Disconnected");
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

        //---------------------------------------------------------------------------//
        // returns the first active websocket. 
        // waits for a new websocket connection if none are active.
        //---------------------------------------------------------------------------//
        WebSocket *
        websocket(index_t ms_poll,
                  index_t ms_timeout)
        {
            WebSocket *res = NULL;
            
            // check if an active websocket exists
            m_server->lock_context();
            {
                for(size_t i=0; i < m_sockets.size() && res == NULL; i++)
                {
                    if(m_sockets[i]->is_connected())
                    {
                        res = m_sockets[i];
                    }
                }
            }
            m_server->unlock_context();
            
            // if we don't have a websocket connected, wait for one
            if(res == NULL)
            {
                res = accept_websocket(ms_poll,ms_timeout);
            }
            
            return res;
        }

        //---------------------------------------------------------------------------//
        // waits for a new websocket connection
        //---------------------------------------------------------------------------//
        WebSocket *
        accept_websocket(index_t ms_poll,
                         index_t ms_timeout)
        {
            WebSocket *res = NULL;

            index_t ms_total = 0;
            size_t  start_num_sockets = 0;
            size_t  curr_num_sockets  = 0;

            // check for bad or un-inited context
            if(m_server->context() == NULL)
            {
                return NULL;
            }

            // we will need to lock the context while we check for new websocket
            m_server->lock_context();
            {
                start_num_sockets = m_sockets.size();
            }
            m_server->unlock_context();

            curr_num_sockets = start_num_sockets;

            while(curr_num_sockets == start_num_sockets && 
                  ms_total <= ms_timeout)
            {
                utils::sleep(ms_poll);
                ms_total += ms_poll;

                m_server->lock_context();
                {
                    curr_num_sockets = m_sockets.size();
                    if(curr_num_sockets != start_num_sockets)
                    {
                        // we will return the last socket added
                        res = m_sockets[curr_num_sockets-1];
                    }
                }
                m_server->unlock_context();
            }

            return res;
        }


  private:
      WebServer                 *m_server;
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
// WebServer Class Implementation
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
WebServer::WebServer()
: m_server(NULL),
  m_handler(NULL),
  m_port(""),
  m_running(false)
{
    m_handler = new RequestHandler(*this);
}

//-----------------------------------------------------------------------------
WebServer::~WebServer()
{
    shutdown();
}

//-----------------------------------------------------------------------------
bool
WebServer::is_running() const
{
    return m_running;
}

//-----------------------------------------------------------------------------
WebSocket *
WebServer::websocket(index_t ms_poll,
                      index_t ms_timeout)
{
    return m_handler->websocket(ms_poll,ms_timeout);
}




//-----------------------------------------------------------------------------
void
WebServer::lock_context()
{
    struct mg_context *ctx = context();
    if(ctx != NULL)
    {
        mg_lock_context(ctx);
    }
}

//-----------------------------------------------------------------------------
void
WebServer::unlock_context()
{
    struct mg_context *ctx = context();
    if(ctx != NULL)
    {
        mg_unlock_context(ctx);
    }
}




//-----------------------------------------------------------------------------
void
WebServer::serve(Node *node,
                  bool block,
                  index_t port)
{
    m_handler->set_node(node);

    // call general serve routine
    serve(utils::join_file_path(CONDUIT_WEB_CLIENT_ROOT,"rest_client"),
          port);
    
    if(block)
    {
        // wait for shutdown()
        while(is_running()) 
        {
            utils::sleep(100);
        }
    }
}


//-----------------------------------------------------------------------------
mg_context *
WebServer::context()
{
    // NOTE: casting away const here.
    return (struct mg_context *) m_server->getContext();
}

//-----------------------------------------------------------------------------
void
WebServer::serve(const std::string &doc_root,
                  index_t port)
{
    if(is_running())
    {
        CONDUIT_INFO("WebServer instance is already running");
        return;
    }

    // civetweb takes strings as arguments
    // convert the port number to a string.
    std::ostringstream oss;
    oss << port;
    m_port = oss.str();

    CONDUIT_INFO("Starting WebServer instance with doc root = " << doc_root);

    // setup civetweb options
    const char *options[] = { "document_root",   doc_root.c_str(),
                              "listening_ports", m_port.c_str(),
                              "num_threads",     "2",
                               NULL};

    try
    {
        m_server = new CivetServer(options);
    }
    catch(CivetException except)
    {
        // Catch Civet Exception and use Conduit's error handling mech.
        CONDUIT_ERROR("WebServer failed to bind civet server on port " 
                      << port);
    }
    
    // check for valid context    
    mg_context *ctx = context();
    if(ctx == NULL)
    {
         CONDUIT_ERROR("WebServer failed to bind civet server on port " 
                       << port);
    }else
    {
        CONDUIT_INFO("conduit::io::WebServer instance active on port: " 
                     << port);
    }

    // setup REST handlers
    m_server->addHandler("/api/*",m_handler);
    
    // setup web socket handlers
    mg_set_websocket_handler(ctx,
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

}

//-----------------------------------------------------------------------------
void     
WebServer::shutdown()
{
    if(is_running())
    {
        CONDUIT_INFO("closing conduit::io::WebServer instance on port: " 
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
// -- end conduit::io --
//-----------------------------------------------------------------------------


};
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
