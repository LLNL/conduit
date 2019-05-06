/*
###############################################################################
# Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

function SearchTable (root) {
  this.init(root);
}

SearchTable.prototype = {
  // column names
  columns: ["abbrpath", "dtype", "number_of_elements", "size"],
  // creates a table, bound to the #d3table div
  init: function (root) {
    var self = this;
    // set the root
    this.root = root;
    // create a flattened version of the root for
    // showing filter results
    this.flatroot = flatten_d3Tree(this.root);
    // create the table
    this.table = d3.select("#d3searchtable").append("table").classed({
      "pure-table": true,
      "pure-table-bordered": true,
      "pure-table-striped": true
    });
    this.thead = this.table.append("thead");
    this.tbody = this.table.append("tbody");
    // create the header row
    this.thead.append("tr")
      .selectAll("th")
      .data(this.columns)
      .enter()
      .append("th")
        .text(function(column) { return self.memberNameToColumnName(column); });

    // get the correct size for our body columns
    this.colWidths = [];
    this.thead.selectAll("th").each(function () {
      self.colWidths.push(d3.select(this).style("width"));
    });
  },

  memberNameToColumnName: function (memberName) {
    var dict = {
      name: "Name",
      cpath: "Path",
      abbrpath: "Path",
      dtype: "Type",
      number_of_elements: "# of Elements",
      size: "Size",
      offset: "Offset",
      endianness: "Endianness"
    };
    return dict[memberName];
  },

  resize: function () {
    var self = this;
    this.colWidths = [];
    this.thead.selectAll("th").each(function () {
      self.colWidths.push(d3.select(this).style("width"));
    });

    for (var i = 0; i < this.colWidths.length; i++) {
      if (i == this.colWidths.length-1) {
        width = parseInt(this.colWidths[i]);
        width = width - 16;
        this.colWidths[i] = width + "px";
      }
      this.tbody.selectAll("td:nth-child("+(i+1)+")").style("width", this.colWidths[i]);
    }
    this.tbody.style("width", this.thead.style("width"));
    this.tbody.selectAll("tr").style("width", this.thead.style("width"));
  },

  // given results of a search or selection, displays them in the table
  update: function (d) {
    var self = this;
    // update the rows
    var rows = this.tbody.selectAll("tr").data(d);
    rows
      .enter()
      .append("tr");
    rows
      .exit().remove();
    // update the columns
    var cells = rows.selectAll("td")
      .data(function(row) {
        return self.columns.map(function(column) {
            console.log(column);
          if (column == 'abbrpath') {
            split = row.cpath.split("/");
            if (split.length > 4) {
              return '/' + split.slice(0, 2).join('/') + '/.../' + split.slice(-2,-1) + '/<b>' + split.slice(-1) + '</b>';
            } else {
              if (split.length > 1) {
                prefix = '/' + split.slice(0, -1).join('/');
                return prefix + '/<b>' + split.slice(-1) + '</b>';
              } else {
                return  '/<b>' + split[0] + '</b>';
              }
            }
          }
          if (column == 'size') {
            return bytesToString(nodeSize(row));
          }
          return row[column];
        });
      });
    cells
      .html(function (d) { return d; });
    cells
      .enter()
      .append("td")
      .html(function (d) { return d; });
    cells
      .exit();

    this.resize();
  },

  updateSingle: function (d) {
    if (d) {
      this.update([d]);
    }
  },

  updateSearch: function (attribute, term) {
    var matches = [];
    for (var i in this.flatroot) {
      var d = this.flatroot[i];
      if (d[attribute] && d[attribute].match(term)) {
        matches.push(d);
      }
    }
    this.update(matches);
  }
};
