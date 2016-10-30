//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
/// file: conduit_relay_web_node_viewer_server.cpp
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
#include "conduit_relay_config.h"
#include "conduit_relay_web.hpp"
#include "conduit_relay_web_node_viewer_server.hpp"

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
// -- begin conduit::relay::web --
//-----------------------------------------------------------------------------
namespace web
{

//-----------------------------------------------------------------------------
// -- Viewer Request Handler  -
//-----------------------------------------------------------------------------

NodeViewerRequestHandler::NodeViewerRequestHandler()
: WebRequestHandler(),
  m_node(NULL)
{
    // empty
}

NodeViewerRequestHandler::~NodeViewerRequestHandler()
{
    // empty
}

//---------------------------------------------------------------------------//
bool
NodeViewerRequestHandler::handle_post(WebServer *server,
                                       struct mg_connection *conn) 
{
    return handle_request(server,conn);
}


//---------------------------------------------------------------------------//
bool
NodeViewerRequestHandler::handle_get(WebServer *server,
                                     struct mg_connection *conn) 
{
    return handle_request(server,conn);
}

//---------------------------------------------------------------------------//
// Main handler, dispatches to proper api requests.
//---------------------------------------------------------------------------//
bool
NodeViewerRequestHandler::handle_request(WebServer *server,
                                         struct mg_connection *conn) 
{
    const struct mg_request_info *req_info = mg_get_request_info(conn);
    
    std::string uri(req_info->uri);
    std::string uri_cmd;
    std::string uri_next;
    utils::rsplit_string(uri,"/",uri_cmd,uri_next);
    
    if(uri_cmd == "get-schema")
    {
        return handle_get_schema(conn);
    }
    else if(uri_cmd == "get-value")
    {
        return handle_get_value(conn);
    }
    else if(uri_cmd == "get-base64-json")
    {
        return handle_get_base64_json(conn);
    }
    else if(uri_cmd == "kill-server")
    {
        return handle_shutdown(server);
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
// Handles a request from the client for the node's schema.
//---------------------------------------------------------------------------//
bool
NodeViewerRequestHandler::handle_get_schema(struct mg_connection *conn)
{
    if(m_node != NULL)
    {
        mg_printf(conn, "%s",m_node->schema().to_json(true).c_str());
    }
    else
    {
        CONDUIT_WARN("rest request for schema of NULL Node");
        return false;
    }
    return true;
}

//---------------------------------------------------------------------------//
// Handles a request from the client for a specific value in the node.
//---------------------------------------------------------------------------//
bool
NodeViewerRequestHandler::handle_get_value(struct mg_connection *conn)
{
    if(m_node != NULL)
    {
        // TODO size checks?
        char post_data[2048];
        char cpath[2048];

        int post_data_len = mg_read(conn, post_data, sizeof(post_data));
        
        // TODO: path instead of cpath?
        
        mg_get_var(post_data, post_data_len, "cpath", cpath, sizeof(cpath));
        // TODO: value instead of datavalue
        mg_printf(conn, "{ \"datavalue\": %s }",
                  m_node->fetch(cpath).to_json().c_str());
    }
    else
    {
        CONDUIT_WARN("rest request for value of NULL Node");
        return false;
    }
    return true;
}

//---------------------------------------------------------------------------//
// Handles a request from the client for a compact, base64 encoded version
// of the node.
//---------------------------------------------------------------------------//
bool
NodeViewerRequestHandler::handle_get_base64_json(struct mg_connection *conn)
{
    if(m_node != NULL)
    {
        std::ostringstream oss;
        m_node->to_json_stream(oss,"conduit_base64_json");
        mg_printf(conn, "%s",oss.str().c_str());
    }
    else
    {
        CONDUIT_WARN("rest request for base64 json of NULL Node");
        return false;
    }
    
    return true;
}

//---------------------------------------------------------------------------//
// Handles a request from the client to shutdown the REST server
//---------------------------------------------------------------------------//
bool
NodeViewerRequestHandler::handle_shutdown(WebServer *server)
{
    server->shutdown();
    return true;
}

//---------------------------------------------------------------------------//
// Sets the Node Object to view
//---------------------------------------------------------------------------//
void
NodeViewerRequestHandler::set_node(Node *node)
{
    m_node = node;
}


//---------------------------------------------------------------------------//
// Node Viewer Server Methods
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
NodeViewerServer::NodeViewerServer()
: WebServer()
{
    set_request_handler(new NodeViewerRequestHandler());
    set_document_root(utils::join_file_path(web_client_root_directory(),
                                            "node_viewer"));
    
}

//---------------------------------------------------------------------------//
NodeViewerServer::~NodeViewerServer()
{
    shutdown();
}

//---------------------------------------------------------------------------//
void
NodeViewerServer::set_node(Node *data)
{
    NodeViewerRequestHandler *req_handler=(NodeViewerRequestHandler*)handler();
    req_handler->set_node(data);
}


}
//-----------------------------------------------------------------------------
// -- end conduit::relay::web --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::relay --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
