/*
###############################################################################
# Copyright (c) 2014-2018, Lawrence Livermore National Security, LLC.
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

function Tree (root) {
  this.init(root);
  this.toggle(root);
  this.update(root);
}

Tree.prototype = {
  // creates a tree, bound to the #d3tree div
  init: function (root) {
    var self = this;
    // height, width, margin variables
    var margin = {top: 0, right: 20, bottom: 0, left: 20};
    var width = d3.select(".d3tree").node().clientWidth;
    var height = 350;
    this.viewerWidth = width - margin.left - margin.right;
    this.viewerHeight = height - margin.top - margin.bottom;
    this.i = 0;
    this.root = root;
    this.root.x0 = 0;
    this.root.y0 = 0;
    this.maxLabelLength = 0;

    // layout the tree
    this.tree = d3.layout.tree()
        .size([self.viewerHeight, self.viewerWidth]);

    this.diagonal = d3.svg.diagonal()
        .projection(function(d) { return [d.y, d.x]; });

    this.zoomListener = d3.behavior.zoom().scaleExtent([0.1, 3]).on("zoom", self.zoom.bind(self));
    this.zoomListener.size([self.viewerHeight, self.viewerWidth]);

    this.baseSvg = d3.select("#d3tree").append("svg")
        .attr("width", self.viewerWidth)
        .attr("height", self.viewerHeight)
        .attr("class", "overlay")
        .call(self.zoomListener);

    // get the maximum label length by visiting every node
    visit(this.root, function (d) {
      self.maxLabelLength = Math.max(d.name.length, self.maxLabelLength);
    }, function (d) {
      return d.children && d.children.length > 0 ? d.children : null;
    });

    // bind to the div
    this.svgGroup = this.baseSvg.append("g");
    this.zoomListener.translate([this.viewerWidth/2, 0]);
    this.zoomListener.event(this.baseSvg);
  },

  update: function (source) {
    var self = this;
    var duration = 0;

    this.updateHeight();

    // Compute the new tree layout.
    var nodes = this.tree.nodes(this.root).reverse();

    // Set widths between levels based on maxLabelLength
    nodes.forEach(function (d) { d.y = d.depth * (self.maxLabelLength * 10); });

    // Update the nodes…
    var treenode = this.svgGroup.selectAll("g.treenode")
        .data(nodes, function(d) { return d.id || (d.id = ++self.i); });

    // Enter any new nodes at the parent's previous position.
    var nodeEnter = treenode.enter().append("svg:g")
        .attr("class", "treenode")
        .attr("transform", function(d) { return "translate(" + source.y0 + "," + source.x0 + ")"; })
        .on("click", function(d) { self.toggle(d); visualizer.highlightPath(d); visualizer.globalUpdate(d); });

    nodeEnter.append("svg:circle")
        .attr("r", 1e-6);

    nodeEnter.append("svg:text")
        .attr("x", function(d) { return d.children || d._children ? -10 : 10; })
        .attr("dy", ".35em")
        .attr("text-anchor", function(d) { return d.children || d._children ? "end" : "start"; })
        .text(function(d) { return d.name; });

    // Transition nodes to their new position.
    var nodeUpdate = treenode.transition()
        .duration(duration)
        .attr("transform", function(d) { return "translate(" + d.y + "," + d.x + ")"; });

    nodeUpdate.select("circle")
        .attr("r", 4.5);
        //.style("fill", function(d) {
        //  if (d.highlight) {
        //    return "#f00";
        //  } else if (d._children) {
        //    return "lightsteelblue";
        //  }
        //  return "#fff";
        //});

    nodeUpdate.select("text")
        .style("fill-opacity", 1);

    // Transition exiting nodes to the parent's new position.
    var nodeExit = treenode.exit().transition()
        .duration(duration)
        .attr("transform", function(d) { return "translate(" + source.y + "," + source.x + ")"; })
        .remove();

    nodeExit.select("circle")
        .attr("r", 1e-6);

    nodeExit.select("text")
        .style("fill-opacity", 1e-6);

    // Update the links…
    var link = this.svgGroup.selectAll("path.link")
        .data(self.tree.links(nodes), function(d) { return d.target.id; });

    // Enter any new links at the parent's previous position.
    link.enter().insert("svg:path", "g")
        .attr("class", "link")
        .attr("d", function(d) {
          var o = {x: source.x0, y: source.y0};
          return self.diagonal({source: o, target: o});
        })
      .transition()
        .duration(duration)
        .attr("d", self.diagonal);

    // Transition links to their new position.
    link.transition()
        .duration(duration)
        .attr("d", self.diagonal);

    // Transition exiting nodes to the parent's new position.
    link.exit().transition()
        .duration(duration)
        .attr("d", function(d) {
          var o = {x: source.x, y: source.y};
          return self.diagonal({source: o, target: o});
        })
        .remove();

    // Stash the old positions for transition.
    nodes.forEach(function(d) {
      d.x0 = d.x;
      d.y0 = d.y;
    });

    if (source) {
      this.centerNode(source);
    }
  },

  updateHeight: function () {
    // Compute the new height of the tree by determining the largest
    // number of nodes at any level of the tree, and scaling accordingly
    var self = this;
    var levelWidth = [1];
    var childCount = function (level, n) {
      if (n.children && n.children.length > 0) {
        if (levelWidth.length <= level + 1) { levelWidth.push(0); }
        levelWidth[level + 1] += n.children.length;
        n.children.forEach(function (d) {
          childCount(level + 1, d);
        });
      }
    }
    childCount(0, this.root);
    var newHeight = d3.max([self.viewerHeight, d3.max(levelWidth) * 25]); // 25 px per line
    this.tree = this.tree.size([newHeight, self.viewerWidth]);
  },

  resize: function () {
    var self = this;
    var margin = {top: 0, right: 20, bottom: 0, left: 20};
    var width = d3.select(".d3tree").node().clientWidth;
    this.viewerWidth = width - margin.left - margin.right;
    this.baseSvg = this.baseSvg.attr("width", self.viewerWidth);
    this.zoomListener.size([self.viewerWidth, self.viewerHeight]);
  },

  centerNode: function (source) {
    scale = this.zoomListener.scale();
    x = -source.y0;
    y = -source.x0;
    x = x * scale + this.viewerWidth / 2;
    y = y * scale + this.viewerHeight / 2;
    this.zoomListener.scale(scale);
    this.zoomListener.translate([x, y]);
    this.zoomListener.event(this.baseSvg);
  },

  toggle: function (d) {
    if (d.children) {
      d._children = d.children;
      d.children = null;
    } else {
      d.children = d._children;
      d._children = null;
    }
  },

  toggleAll: function (d) {
    if (d.children) {
      d.children.forEach(toggleAll);
      toggle(d);
    }
  },

  highlightSearch: function (attribute, term) {
    var searchResults;
    this.svgGroup.selectAll("g.treenode").select("circle")
        .style("fill", function(d) {
          if (d[attribute] && d[attribute].match(term)) {
            return "#f00";
          } else if (d._children) {
            return "lightsteelblue";
          }
          return "#fff";
        });
    this.svgGroup.selectAll("path.link")
        .style("stroke", "#ccc");
  },

  highlightPath: function () {
    // highlight a clicked-on cell
    this.svgGroup.selectAll("g.treenode").select("circle")
        .style("fill", function(d) {
          if (d.highlight) {
            return "#f00";
          } else if (d._children) {
            return "lightsteelblue";
          }
          return "#fff";
        });
    this.svgGroup.selectAll("path.link")
        .style("stroke", function (d) { return d.target.highlight ? "#f00" : "#ccc"; });
  },

  zoom: function () {
    this.svgGroup.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
  },

  childCount: function (level, n) {
    if (n.children && n.children.length > 0) {
      if (levelWidth.length <= level + 1) { levelWidth.push(0); }
      levelWidth[level + 1] += n.children.length;
      n.children.forEach(function (d) {
        childCount(level + 1, d);
      });
    }
  }
};
