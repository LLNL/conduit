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
// For details, see: http://llnl.github.io/conduit/.
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
/// file: conduit_rest.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_REST_HPP
#define CONDUIT_REST_HPP

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "Conduit_IO_Exports.hpp"

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
/// Simple interface to launch a blocking REST server    
//-----------------------------------------------------------------------------
void CONDUIT_IO_API serve(Node *n,
                          index_t port = 8080);

};
//-----------------------------------------------------------------------------
// -- end conduit::io::rest --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// -- Web Server Interface -
//-----------------------------------------------------------------------------

// forward declare websocket interface class
class WebSocket;

// forward declare internal handler class
class RequestHandler;

class CONDUIT_IO_API WebServer
{
public:

                WebServer();
    virtual    ~WebServer();
    
    void        serve(const std::string &doc_root,
                      index_t port = 8080);

    // note: this variant of serve is to specific to the 
    // the visualizer client use case.
    void        serve(Node *data,
                      bool block=false,
                      index_t port = 8080);

    void        shutdown();
    
    bool        is_running() const;


    // returns the first active websocket, if non are active, blocks
    // until a websocket connection is established.
    WebSocket  *websocket(index_t ms_poll = 100,
                          index_t ms_timeout = 60000);

    mg_context *context();
    void        lock_context();
    void        unlock_context();

private:
    CivetServer            *m_server;
    RequestHandler         *m_handler;
    
    std::string             m_port;
    bool                    m_running;
};

//-----------------------------------------------------------------------------
/// -- WebSocket Connection Interface -
//-----------------------------------------------------------------------------
//
/// The lifetimes of our WebSocket instances are managed by the 
/// WebServer and its RequestHandler instance
// 
class CONDUIT_IO_API WebSocket
{
public:
    friend class RequestHandler;
    
    void           send(const Node &data,
                        const std::string &protocol="json");

    bool           is_connected() const;

    mg_context    *context();
    void           lock_context();
    void           unlock_context();

private:
             WebSocket();
    virtual ~WebSocket();
    
    void     set_connection(mg_connection *connection);

    mg_connection  *m_connection;
};





};
//-----------------------------------------------------------------------------
// -- end conduit::io --
//-----------------------------------------------------------------------------



};
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

#endif 



