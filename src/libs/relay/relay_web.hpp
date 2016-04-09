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
/// file: conduit_web.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_RELAY_WEB_HPP
#define CONDUIT_RELAY_WEB_HPP

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"

#include "relay_exports.hpp"

//
// forward declare civetweb types so we don't need the 
// civetweb headers in our public interface. 
// 

class CivetServer;
struct mg_context;
struct mg_connection;

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
// -- Web Server Request Handler Interface -
//-----------------------------------------------------------------------------

// forward declare WebServer class
class WebServer;

// interface used to create concrete server instances.  
class CONDUIT_RELAY_API WebRequestHandler
{
public:
                  WebRequestHandler();
    virtual      ~WebRequestHandler();

    virtual bool  handle_post(WebServer *server,
                              struct mg_connection *conn);

    virtual bool  handle_get(WebServer *server,
                             struct mg_connection *conn);
};

//-----------------------------------------------------------------------------
// -- Web Server Interface -
//-----------------------------------------------------------------------------

// forward declare websocket interface class
class WebSocket;
// forward declare internal handler class
class CivetDispatchHandler;

class CONDUIT_RELAY_API WebServer
{
public:

                WebServer();
    virtual    ~WebServer();

    /// TODO: too many default args here, may want to change the interface
    /// to make things cleaner in the future.
    
    /// convenience case that uses the default request handler
    void        serve(const std::string &doc_root,
                      index_t port = 8080,
                      const std::string &ssl_cert_file = std::string(""),
                      const std::string &auth_domain   = std::string(""),
                      const std::string &auth_file     = std::string(""));

    /// general case, supporting a user provided request handler
    void        serve(const std::string &doc_root,
                      WebRequestHandler *dispatch, // takes ownership?
                      index_t port = 8080,
                      const std::string &ssl_cert_file = std::string(""),
                      const std::string &auth_domain   = std::string(""),
                      const std::string &auth_file     = std::string(""));
    
    void        shutdown();
    
    bool        is_running() const;

    /// returns the first active websocket, if none are active, blocks
    /// until a websocket connection is established.
    ///
    ///  ms_poll specifies the number of microseconds for each poll attempt
    ///  ms_timeout specifies the total time out in microseconds
    WebSocket  *websocket(index_t ms_poll = 100,
                          index_t ms_timeout = 60000);

    WebRequestHandler *handler();
    
    mg_context *context();
    void        lock_context();
    void        unlock_context();

private:
    WebRequestHandler      *m_handler;
    
    std::string             m_doc_root;
    std::string             m_port;
    std::string             m_ssl_cert_file;
    std::string             m_auth_domain;
    std::string             m_auth_file;
    bool                    m_running;

    CivetServer            *m_server;
    CivetDispatchHandler   *m_dispatch;

};

//-----------------------------------------------------------------------------
/// -- WebSocket Connection Interface -
//-----------------------------------------------------------------------------
//
/// The lifetimes of our WebSocket instances are managed by the 
/// WebServer and its RequestHandler instance
// 
class CONDUIT_RELAY_API WebSocket
{
public:
    friend class CivetDispatchHandler;
    
    void           send(const Node &data,
                        const std::string &protocol="json");

    // todo: receive? 

    bool           is_connected() const;

    mg_context    *context();
    void           lock_context();
    void           unlock_context();

private:
                   WebSocket();
    virtual       ~WebSocket();

    void           set_connection(mg_connection *connection);

    mg_connection  *m_connection;
};


}
//-----------------------------------------------------------------------------
// -- end conduit::relay::relay --
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



