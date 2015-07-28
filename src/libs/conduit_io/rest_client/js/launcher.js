/*
###############################################################################
# Copyright (c) 2014-2015, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-666778
# 
# All rights reserved.
# 
# This file is part of Conduit. 
# 
# For details, see https://lc.llnl.gov/conduit/.
# 
# Please also read conduit/LICENSE
# 
# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, 
#   this list of conditions and the disclaimer below.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
# 
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
# POSSIBILITY OF SUCH DAMAGE.
# 
###############################################################################
*/
var visualizer;

var request = new XMLHttpRequest();
request.open('POST', '/api/get-schema', true);
request.onload = function () {
  if (request.status >= 200 && request.status < 400) {
    var node = JSON.parse(request.response.replace(/nan/ig, "null"));
    visualizer = new Visualizer(node);
    visualizer.run();
  } else {
    console.log("Server responded with error code: ", request.status);
  }
};

request.onerror = function () {
  console.log("Connection error");
};

request.send();

var b64_req = new XMLHttpRequest();
b64_req.open('POST', '/api/get-base64-json', true);
b64_req.onload = function ()
{
    if (b64_req.status >= 200 && b64_req.status < 400) 
    {
        var node = JSON.parse(b64_req.response);
        console.log("base64-json result:", b64_req.response);
    }
    else 
    {
      console.log("Server responded with error code: ", b64_req.status);
    }
};

b64_req.onerror = function ()
{
  console.log("Connection error");
};

b64_req.send();

function websocket_test()
{
    var wsproto = (location.protocol === 'https:') ? 'wss:' : 'ws:';
    connection = new WebSocket(wsproto + '//' + window.location.host + '/websocket');
    
    connection.onmessage = function (msg) 
    {
        console.log('WebSocket message' + msg.data);
        connection.send('{"type":"info","message":"response from browser"}');
    }
      
    connection.onerror = function (error)
    {
        console.log('WebSocket error');
        connection.close();
    }
}

websocket_test();




