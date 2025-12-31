"""
Render Model Insights plots for the latest supervised fraud run.

Selection rules
- Prefer fraud section when present.
- Fall back to fraud_synthetic when real labels are not available.

Outputs
- artefacts/fraud_pr_curve.png
- artefacts/fraud_threshold_tradeoff.png
- Optional copy into neobank_site/static/artefacts for local dashboard display
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import shutil

from ml.metrics.plots import pr_curve_from_metrics, threshold_precision_recall
from ml.training.utils import load_registry


def _latest_metrics_path(reg: Dict[str, Any]) -> Optional[str]:
    """
    Resolve the latest metrics JSON path from the registry.
    """
    for section in ["fraud", "fraud_synthetic"]:
        block = reg.get(section, {})
        latest = block.get("latest")
        if latest and latest in block:
            entry = block[latest]
            mp = entry.get("metrics_path")
            if isinstance(mp, str) and mp:
                return mp
    return None


def _copy_to_static(src: Path, static_dir: Path) -> None:
    """
    Copy a PNG into the configured static directory.
    """
    static_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, static_dir / src.name)


# def main() -> None:
#     reg = load_registry("model_registry.json")
#     metrics_path = _latest_metrics_path(reg)

#     if not metrics_path:
#         print("No supervised fraud metrics found in registry.")
#         print("Run the supervised fraud trainer first.")
#         return

#     artefacts = Path("artefacts")
#     artefacts.mkdir(exist_ok=True)

#     pr_png = artefacts / "fraud_pr_curve.png"
#     thr_png = artefacts / "fraud_threshold_tradeoff.png"

#     pr_curve_from_metrics(metrics_path, str(pr_png))
#     threshold_precision_recall(metrics_path, str(thr_png))

#     static_dir = Path("neobank_site") / "static" / "artefacts"
#     _copy_to_static(pr_png, static_dir)
#     _copy_to_static(thr_png, static_dir)

#     print(f"Wrote: {pr_png} and {thr_png}")
#     print(f"Copied to static: {static_dir}")


# if __name__ == "__main__":
#     main()
