// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_relay_web_node_viewer_server.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_RELAY_WEB_NODE_VIEWER_SERVER_HPP
#define CONDUIT_RELAY_WEB_NODE_VIEWER_SERVER_HPP

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"

#include "conduit_relay_exports.h"
#include "conduit_relay_web.hpp"

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
// -- Viewer Web Request Handler  -
//-----------------------------------------------------------------------------
class CONDUIT_RELAY_API NodeViewerRequestHandler : public WebRequestHandler
{
public:
                   NodeViewerRequestHandler();
                  ~NodeViewerRequestHandler();
    
    virtual bool   handle_post(WebServer *server,
                               struct mg_connection *conn);

    virtual bool   handle_get(WebServer *server,
                              struct mg_connection *conn);
                              
    void           set_node(Node *node);

private:
    // catch all, used for any post or get
    bool           handle_request(WebServer *server,
                                  struct mg_connection *conn);
    // handlers for specific commands 
    bool           handle_get_schema(struct mg_connection *conn);
    bool           handle_get_value(struct mg_connection *conn);
    bool           handle_get_base64_json(struct mg_connection *conn);
    bool           handle_shutdown(WebServer *server);

    // holds the node to visualize 
    Node          *m_node;
};

//-----------------------------------------------------------------------------
// -- Node Viewer Web Server -
//-----------------------------------------------------------------------------

class CONDUIT_RELAY_API NodeViewerServer : public WebServer
{
public:
    
             NodeViewerServer();
    virtual ~NodeViewerServer();

    void    set_node(Node *node);

};


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

#endif 



