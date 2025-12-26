(function () {
  "use strict";

  // Component purpose: client-side enhancements for dashboard tables.

  // Section: Table search filtering (lightweight, no backend calls).
  function initTableSearch() {
    var input = document.getElementById("txSearch");
    var table = document.getElementById("transactionsTable");

    if (!input || !table) {
      return;
    }

    var tbody = table.querySelector("tbody");
    if (!tbody) {
      return;
    }

    var rows = Array.prototype.slice.call(tbody.querySelectorAll("tr"));

    // TODO: Debounce input for large tables.
    input.addEventListener("input", function () {
      var query = (input.value || "").toLowerCase().trim();

      rows.forEach(function (row) {
        var text = (row.textContent || "").toLowerCase();
        row.style.display =
          query === "" || text.indexOf(query) !== -1 ? "" : "none";
      });
    });
  }

  // Section: Bootstrapping.
  document.addEventListener("DOMContentLoaded", function () {
    initTableSearch();
  });
})();
