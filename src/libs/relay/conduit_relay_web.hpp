//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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
/// file: conduit_conduit_web.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_RELAY_WEB_HPP
#define CONDUIT_RELAY_WEB_HPP

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"

#include "conduit_relay_exports.h"

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
/// -- Returns path to root directory for web client resources --
/// Source path:     ${CMAKE_CURRENT_SOURCE_DIR}/web_clients
/// Installed path:  {install_prefix}/share/conduit/web_client/
//-----------------------------------------------------------------------------
std::string CONDUIT_RELAY_API web_client_root_directory();

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


    /// the root path on the file system to server
    /// 
    /// Note: must be set before starting the server, or an error will
    /// be thrown
    void set_document_root(const std::string &doc_root);
    
    /// the request handler ( WebServer takes ownership of this)
    ///
    /// If not set, the default request handler is used
    ///
    void set_request_handler(WebRequestHandler *handler);

    /// the ip address to run the server on
    ///   default address: 127.0.0.1 (localhost)
    void set_bind_address(const std::string &addy);

    /// the port to run the server on
    ///   default port:    9000
    void set_port(int port);

    /// options for htpasswd authentication 
    ///   defaults: {not used}
    void set_htpasswd_auth_domain(const std::string &domain);
    void set_htpasswd_auth_file(const std::string &htpasswd_file);

    /// option for using ssl certificate file to enable https
    ///   default: {not used}
    void set_ssl_certificate_file(const std::string &cert_file);
    
    /// options for entangle auth and port forwarding setup
    ///   defaults: {not used}
    void set_entangle_output_base(const std::string &obase);
    void set_entangle_gateway(const std::string &gateway);

    /// calls the entangle python script to generate a new 
    /// password, htpasswd file and setup json files for 
    /// clients to use with entangle to establish ssh tunnels.
    ///   
    ///   runs conduit_relay_entangle.py --register
    ///   calls this->set_htpasswd_domain("127.0.0.1")
    ///   calls this->set_htpasswd_file({entangle_obase}.htpasswd)
    void entangle_register();
    
    
    /// start server, if block is true enters a wait loop that
    /// waits for the server to close
    void serve(bool block = false);

    virtual void shutdown();
    
    bool        is_running() const;

    std::string bind_address() const;
    int         port()    const;


    /// returns the first active websocket, if none are active, blocks
    /// until a websocket connection is established.
    ///
    ///  ms_poll specifies the number of microseconds for each poll attempt
    ///  ms_timeout specifies the total time out in microseconds
    WebSocket  *websocket(index_t ms_poll = 100,
                          index_t ms_timeout = 60000);

    /// returns the request handler used by this server instance
    WebRequestHandler *handler();
    
    /// access to civetweb 
    mg_context *context();
    void        lock_context();
    void        unlock_context();

private:
    WebRequestHandler      *m_handler;
    
    std::string             m_doc_root;
    std::string             m_address;
    int                     m_port;

    std::string             m_ssl_cert_file;

    std::string             m_htpasswd_auth_domain;
    std::string             m_htpasswd_auth_file;

    std::string             m_entangle_obase;
    std::string             m_entangle_gateway;
    bool                    m_using_entangle;

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



