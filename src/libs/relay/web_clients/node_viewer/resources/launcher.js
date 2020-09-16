/*
# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
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




