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

