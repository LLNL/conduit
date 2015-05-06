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
