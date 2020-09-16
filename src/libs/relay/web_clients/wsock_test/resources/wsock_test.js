/*
# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
*/

function websocket_test()
{
    var wsproto = (location.protocol === 'https:') ? 'wss:' : 'ws:';
    connection = new WebSocket(wsproto + '//' + window.location.host + '/websocket');
    
    connection.onmessage = function (msg) 
    {
        var data;
        try
        {
            data=JSON.parse(msg.data);
            if(data.type == "image")
            {
                $("#image_display").html("<img src='" + data.data + "'/>");
                $("#image_display").show();
                $("#count_display").text("Current Count = " +data.count);
                $("#status_display").html("<font color=blue>[status=connected]</font>");

                var response = {type: "info",
                                message: "response from browser, count = " + data.count};

                connection.send(JSON.stringify(response));
            }
        }
        catch(e)
        {
             //caught an error in the above code, simply display to the status element
            $("#status_display").html("<font color=red>[status=error] " + e + "</font>");
        }
    }
      
    connection.onerror = function (error)
    {
        console.log('WebSocket error');
        console.log(error)
        connection.close();
        $("#status_display").html("<font color=red>[status=error]</font>");
    }
    
    connection.onclose = function (error)
    {
        console.log('WebSocket closed');
        console.log(error)
        connection.close();
        $("#status_display").html("<font color=orange>[status=disconnected]</font>");
    }
}

websocket_test();




