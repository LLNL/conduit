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
