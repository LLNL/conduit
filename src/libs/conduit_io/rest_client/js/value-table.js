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

function ValueTable (root) {
  this.init(root);
}

ValueTable.prototype = {
  // column names
  columns: ["index", "value"],
  // creates a table, bound to the #d3table div
  init: function () {
    var width = d3.select("#valueTableContainer").node().clientWidth;
    d3.select("#d3valuetable").style("width", width+"px");
  },

  memberNameToColumnName: function (memberName) {
    var dict = {
      index: "Index",
      value: "Value"
    };
    return dict[memberName];
  },

  resize: function () {
    var width = d3.select("#valueTableContainer").node().clientWidth;
    d3.select("#d3valuetable").style("width", width+"px");
    if (this.table) {
      this.table.setup();
    }
  },

  displayValue: function (value) {
    if ((typeof value) !== "object") {
      value = [value];
    }

    var painter = new fattable.Painter();
    painter.fillCell = function(cellDiv, data) {
      cellDiv.textContent = data.content;
      if (data.rowId % 2 === 0) {
        cellDiv.className = "even";
      }
      else {
        cellDiv.className = "odd";
      }
    };
    painter.fillCellPending = function(cellDiv, data) {
      cellDiv.textContent = "";
      cellDiv.className = "pending";
    };
    var model = new fattable.SyncTableModel();
    model.getCellSync = function (i, j) {
      var data = {
        rowId: i
      };
      if (j === 0) { data.content = i; }
      if (j == 1) { data.content = value[i]; }
      return data;
    };
    model.getHeaderSync = function (j) {
      if (j === 0) { return "Index"; }
      if (j == 1) { return "Value"; }
    };
    this.table = fattable({
      "container": "#d3valuetable",
      "painter": painter,    // your painter (see below)
      "model": model,          // model describing your data (see below)
      "nbRows": value.length,     // overall number of rows
      "rowHeight": 24,       // constant row height (px)
      "headerHeight": 24,   // height of the header (px)
      "columnWidths": [70, 1000] // array of column width (px)
    });
  },

  // given results of a search or selection, displays them in the table
  update: function (d) {
    var self = this;
    getNodeValue(d[0], self.displayValue.bind(self));
  },

  updateSingle: function (d) {
    if (d) {
      this.update([d]);
    }
  },

  updateSearch: function (attribute, term) {
  }
};
