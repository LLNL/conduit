/*
# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
*/

var bytesToString = function (bytes) {
	var suffixes = ["B", "kB", "MB", "GB"];
	var sufInd = 0;
	var scale = 1024;
	while (bytes > scale) {
		sufInd += 1;
		scale *= 1024;
	}
	return (1024*(bytes/scale)).toFixed(sufInd) + " " + suffixes[sufInd];
};

var nodeSize = function (d) {
	if (!d.size) {
		if (d.leaf) {
			d.size = d.number_of_elements * d.element_bytes;
		} else {
      var kids = d.children ? d.children : d._children;
      d.size = kids.map(nodeSize).reduce(function (a, b) { return a + b; });
		}
	}
	return d.size;
};

var nodeLogSize = function (d) {
	if (!d.logsize) {
		if (d.leaf) {
      var size = nodeSize(d);
      if (size === 0) {
        d.logsize = 0;
      } else {
        d.logsize = Math.log(size)/Math.log(1024);
      }
		} else {
      var kids = d.children ? d.children : d._children;
      d.logsize = kids.map(nodeLogSize).reduce(function (a, b) { return a + b; });
		}
	}
	return d.logsize;
};

var nodeOffset = function (d) {
	if (d.offset === null) {
		var kids = d.children ? d.children : d._children;
		var leastOffset = Infinity;
		for (var i = 0; i < kids.length; i++) {
			if (nodeOffset(kids[i]) < leastOffset) {
				leastOffset = nodeOffset(kids[i]);
			}
		}
		d.offset = leastOffset;
	}
	return d.offset;
};

var getNodeValue = function (d, callback) {
	if (!d.datavalue && d.leaf && d.length !== 0) {
    var request = new XMLHttpRequest();
    request.open('POST', '/api/get-value', true);
    request.onload = function () {
      if (request.status >= 200 && request.status < 400) {
        json = JSON.parse(request.response.replace(/nan/ig, "null"));
        d.datavalue = json.datavalue;
        callback(d.datavalue);
      } else {
        callback("Server error code " + request.status);
      }
    };

    request.onerror = function () {
      callback("Server connection error");
    };

    request.send("cpath=" + d.cpath);
	} else {
    callback(d.datavalue);
  }
};

var visit = function (parent, visitFn, childrenFn) {
  if (!parent) return;
  visitFn(parent);
  var children = childrenFn(parent);
  if (children) {
    var count = children.length;
    for (var i = 0; i < count; i++) {
      visit(children[i], visitFn, childrenFn);
    }
  }
};

var flatten_d3Tree = function (tree) {
  var children;
  if (tree.children) {
    children = tree.children;
    //delete tree.children;
  } else if (tree._children) {
    children = tree._children;
    //delete tree._children;
  }

  flattened = [tree];
  if (children) {
    for (var i in children) {
      flattened = flattened.concat(flatten_d3Tree(children[i]));
    }
  }
  return flattened;
};

var conduit_to_d3_tree = function (node) {
	var children = [];
	for (var child in node) {
		if (node.hasOwnProperty(child)) {
			children.push(conduit_to_d3_tree_recursive(child, child, node[child]));
		}
	}
	return {
		name: "root",
		children: children,
		cpath: ""
	};
};

var conduit_to_d3_tree_recursive = function (cpath, name, parent) {
	// base case: the parent is a leaf
	if (parent.hasOwnProperty("dtype")) {
		parent.name = name;
		parent.leaf = true;
		parent.cpath = cpath;
		return parent;
	} else {
		var children = [];
		for (var child in parent) {
			if (parent.hasOwnProperty(child)) {
				children.push(conduit_to_d3_tree_recursive(cpath+"/"+child, child, parent[child]));
			}
		}
		return {
			name: name,
			children: children,
			cpath: cpath,
      leaf: false
		};
	}
};
