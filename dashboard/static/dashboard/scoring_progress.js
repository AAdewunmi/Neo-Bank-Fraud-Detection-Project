// dashboard/static/dashboard/scoring_progress.js
/**
 * Show a visible "processing" message as soon as the user submits the Validate and score form.
 *
 * This keeps the UI from looking frozen while the server:
 * - reads the uploaded CSV
 * - loads ML artefacts
 * - (first run) downloads and initialises the embeddings model
 */
(function () {
  function enableScoringProgressUI() {
    var form = document.getElementById("validate-score-form");
    if (!form) {
      return;
    }

    form.addEventListener("submit", function (e) {
      var status = document.getElementById("scoring-status");
      if (status) {
        status.classList.remove("d-none");
      }

      var buttons = form.querySelectorAll("button, input[type='submit']");
      buttons.forEach(function (btn) {
        btn.setAttribute("aria-busy", "true");
        btn.disabled = true;
      });

      // Let the browser paint the status message before the request blocks the tab.
      e.preventDefault();
      window.requestAnimationFrame(function () {
        form.submit();
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", enableScoringProgressUI);
  } else {
    enableScoringProgressUI();
  }
})();
