/*
# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
*/

function TreeMap (root) {
  this.init(root);
  this.update();
}

TreeMap.prototype = {
  init: function (root) {
    var self = this;
    this.root = root;

    var margin = {top: 10, right: 10, bottom: 10, left: 10};
    var width = d3.select(".d3tree").node().clientWidth;
    var height = 350;
    this.width = width - margin.left - margin.right;
    this.height = height - margin.top - margin.bottom;

    //this.color = d3.scale.category20c();
    this.color = d3.scale.log()
        .base(2)
        .domain([1, Math.pow(2,15)])
        .range(["red", "orange", "yellow", "green", "blue", "purple"]);

    this.treemap = d3.layout.treemap()
        .size([self.width, self.height])
        .sort(function(a, b) { return nodeOffset(a) - nodeOffset(b); })
        .value(function(d) { return nodeSize(d); });

    this.div = d3.select("#d3treemap")
        .style("position", "relative")
        .style("width", (this.width + margin.left + margin.right) + "px")
        .style("height", (this.height + margin.top + margin.bottom) + "px")
        .style("left", margin.left + "px")
        .style("top", margin.top + "px");

    this.div.datum(root).selectAll(".treemapnode").data(self.treemap.nodes)
      .enter().append("div")
        .attr("class", "treemapnode")
        .call(self.position, self.width, self.height)
        .style("background", function(d) { return d.children ? null : self.color(nodeSize(d)); })
        .text(function(d) { return d.children ? null : d.name; })
        .on("click", function (d) { visualizer.highlightPath(d); visualizer.globalUpdate(d); });
  },

  resize: function () {
    var margin = {top: 40, right: 10, bottom: 10, left: 10};
    var width = d3.select(".d3treemap").node().clientWidth;
    this.width = width - margin.left - margin.right;
    this.treemap.size([this.width, this.height]);
    this.div
        .style("width", (this.width + margin.left + margin.right) + "px")
    this.update();
  },

  position: function (selection, width, height) {
    this.style("left", function(d) { return width - Math.max(0, d.dx - 1) - d.x + "px"; })
        .style("top", function(d) { return height - Math.max(0, d.dy - 1) - d.y + "px"; })
        .style("width", function(d) { return Math.max(0, d.dx - 1) + "px"; })
        .style("height", function(d) { return Math.max(0, d.dy - 1) + "px"; });
  },

  update: function () {
    var self = this;
    var treemapnode = this.div.datum(this.root).selectAll(".treemapnode").data(self.treemap.nodes);
    treemapnode
      .enter()
        .append("div")
        .attr("class", "treemapnode")
        .on("click", function (d) { visualizer.highlightPath(d); visualizer.globalUpdate(d); });
    treemapnode
      .exit()
        .remove();
    treemapnode
        // correctly position cells
        .call(self.position, self.width, self.height)
        //correctly color cells
        .style("background", function(d) { return d.children ? null : self.color(nodeSize(d)); })
        // display the name of every leaf cell
        .text(function(d) { return d.children ? null : d.name; });
  },

  highlightSearch: function (attribute, term) {
    this.div.datum(this.root).selectAll(".treemapnode")
        .style("border-color", "grey")
        .style("border-width", function(d) {
          if (d.children) {
            return null;
          } else if (d[attribute] && d[attribute].match(term)) {
            return "5px";
          } else {
            return "0px";
          }
        });
  },

  highlightPath: function () {
    // highlight a clicked-on cell
    this.div.datum(this.root).selectAll(".treemapnode")
        .style("border-color", "black")
        .style("border-width", function(d) {
          if (d.children) {
            return null;
          } else if (d.highlight) {
            return "5px";
          } else {
            return "0px";
          }
        });
  }

};

