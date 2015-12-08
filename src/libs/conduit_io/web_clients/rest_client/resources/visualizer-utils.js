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
# For details, see: http://llnl.github.io/conduit/.
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
			d.size = d.length * d.element_bytes;
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
