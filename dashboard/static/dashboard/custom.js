(function () {
  "use strict";

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

    input.addEventListener("input", function () {
      var q = (input.value || "").toLowerCase().trim();
      var rows = tbody.querySelectorAll("tr");

      rows.forEach(function (tr) {
        var text = (tr.textContent || "").toLowerCase();
        tr.style.display = q === "" || text.indexOf(q) !== -1 ? "" : "none";
      });
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    initTableSearch();
  });
})();
