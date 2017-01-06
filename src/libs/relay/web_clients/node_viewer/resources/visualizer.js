/*
###############################################################################
# Copyright (c) 2014-2017, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-666778
# 
# All rights reserved.
# 
# This file is part of Conduit. 
# 
# For details, see: http://software.llnl.gov/conduit/.
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

function Visualizer (json) {
  this.init(json);
}

Visualizer.prototype = {
  init: function (json) {
    this.root = conduit_to_d3_tree(json);
  },

  run: function () {
    var self = this;
    this.tree = new Tree(this.root);
    this.treemap = new TreeMap(this.root);
    this.searchTable = new SearchTable(this.root);
    this.valueTable = new ValueTable();
    window.addEventListener("resize", self.treemap.resize.bind(self.treemap));
    window.addEventListener("resize", self.tree.resize.bind(self.tree));
    window.addEventListener("resize", self.searchTable.resize.bind(self.searchTable));
    window.addEventListener("resize", self.valueTable.resize.bind(self.valueTable));
    this.startToolbarListeners();
    this.searchMode = false;
    this.globalUpdate();
  },

  startToolbarListeners: function () {
    var self = this;
    // radio buttons that change sizing
    d3.selectAll(".sizing").on("change", function change() {
      var value;
      if (this.value === "log") {
        value = function (d) { return nodeLogSize(d); };
      } else if (this.value === "equal") {
        value = function(d) { return 1; };
      } else {
        value = function(d) { return nodeSize(d); };
      }

      self.treemap.treemap.value(value);
      self.treemap.update();
    });

    // radio buttons that change sorting order
    d3.selectAll(".sorting").on("change", function change() {
      var value;
      if (this.value === "alphabetical") {
        value = function(a, b) {
          if (a.name.toLowerCase() < b.name.toLowerCase()) {
            return -1;
          } else if (a.name.toLowerCase() > b.name.toLowerCase()) {
            return 1;
          } else {
            return 0;
          }
        };
      } else if (this.value === "size") {
        value = function(a, b) {
          if (nodeSize(a) < nodeSize(b)) {
            return -1;
          } else if (nodeSize(a) > nodeSize(b)) {
            return 1;
          } else {
            return 0;
          }
        };
      } else {
        value = function(a, b) {
          return nodeOffset(a) - nodeOffset(b);
        };
      }

      self.tree.tree.sort(value);
      self.globalUpdate();
    });

    // text field, radio buttons and submit button for filtering
    d3.select(".filterUpdate").on("click", function change() {
      var textField = d3.select(".filterInput");
      self.searchTerm = textField.property("value");

      var filterRadio = d3.selectAll(".filter");
      self.searchAttribute = filterRadio.filter(function () {
        return this.checked;
      }).property("value");
      self.searchMode = true;
      self.globalUpdate();
    });

    // button that quits visualizer
    d3.select("#killButton").on("click", function () {
      var request = new XMLHttpRequest();
      request.open('POST', '/api/kill-server', true);
      request.onload = function () {
        if (request.status >= 200 && request.status < 400) {
        } else {
          console.log("Server responded with error code: ", request.status);
        }
      };

      request.onerror = function () {
        console.log("Connection error");
      };

      request.send();

      d3.select("#visualizer").style("opacity", 0.2);
      d3.select("#goodbye").style("display", "block");
    });
  },

  globalUpdate: function (d) {
    // first update both the tree and treemap
    // note that the tree comes first, so the treemap can't do anything
    //    to the data that would change the tree.
    this.tree.update(d);
    this.treemap.update();
    this.searchTable.updateSingle(d);
    this.valueTable.updateSingle(d);
    if (this.searchMode) {
      this.tree.highlightSearch(this.searchAttribute, this.searchTerm);
      this.treemap.highlightSearch(this.searchAttribute, this.searchTerm);
      this.searchTable.updateSearch(this.searchAttribute, this.searchTerm);
    } else {
      this.tree.highlightPath();
      this.treemap.highlightPath();
    }
  },

  prevPath: [],

  highlightPath: function (d) {
    this.searchMode = false;
    // un-highlight the other things
    this.clearHighlight();
    // highlight the new path
    d.highlight = true;
    this.prevPath = [d];
    var p = d.parent;
    while (p) {
      p.highlight = true;
      this.prevPath.push(p);
      p = p.parent;
    }
  },

  clearHighlight: function () {
    for (var i = 0; i < this.prevPath.length; i++) {
      this.prevPath[i].highlight = false;
    }
  }
};

