// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

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
        mg_printf(conn,
                  "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n");
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
        mg_printf(conn,
                  "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n");
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
        
        mg_printf(conn,
                  "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n");
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
