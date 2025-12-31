(function () {
  function onReady(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn);
    } else {
      fn();
    }
  }

  onReady(function () {
    const form = document.getElementById("score-form");
    const progress = document.getElementById("score-progress");
    const submit = document.getElementById("score-submit");
    const searchInput = document.getElementById("txSearch");
    const table = document.getElementById("transactionsTable");

    if (searchInput && table) {
      searchInput.addEventListener("input", function () {
        const q = searchInput.value.trim().toLowerCase();
        const rows = Array.from(table.querySelectorAll("tbody tr"));
        rows.forEach(function (row) {
          const text = row.textContent.toLowerCase();
          row.style.display = text.includes(q) ? "" : "none";
        });
      });
    }

    if (form) {
      form.addEventListener("submit", function () {
        if (progress) {
          progress.classList.remove("d-none");
          progress.textContent =
            "Scoring in progress... The first run may take up to a minute while models load.";
        }
        if (submit) {
          submit.disabled = true;
          submit.dataset.originalText = submit.innerText;
          submit.innerText = "Scoring...";
        }

        // Watchdog: if nothing happens after 120s, reset UI
        setTimeout(function () {
          if (submit && submit.disabled) {
            submit.disabled = false;
            submit.innerText =
              submit.dataset.originalText || "Validate and score";
          }
          if (progress) {
            progress.classList.remove("alert-info");
            progress.classList.add("alert-danger");
            progress.textContent =
              "This run took unusually long. If it keeps spinning, reload the page or lower CSV size.";
          }
        }, 120000); // 2 min fallback
      });
    }
  });
})();
