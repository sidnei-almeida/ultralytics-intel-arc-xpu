"""Template for training on a custom dataset using Intel Arc XPU.

Point ``DATA_YAML`` to your own dataset descriptor and tune the hyper-
parameters below. The defaults are a sensible starting point for an
Intel Arc A750 / A770 class GPU.
"""

from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL       = "yolo11s.pt"           # any Ultralytics checkpoint
DATA_YAML   = "path/to/your/data.yaml"
EPOCHS      = 100
IMGSZ       = 640
BATCH       = 8
WORKERS     = 0                       # raise once stable
AMP         = False                   # start with AMP off, flip on if stable
PROJECT     = "runs/train-xpu"
NAME        = "yolo11s-custom"

# ---------------------------------------------------------------------------


def main() -> None:
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise SystemExit("Intel XPU not available.")

    print(f"→ Torch {torch.__version__}")
    print(f"→ XPU   {torch.xpu.get_device_name(0)}")
    print(f"→ Data  {DATA_YAML}")

    if not Path(DATA_YAML).exists():
        raise SystemExit(
            f"Dataset descriptor not found: {DATA_YAML}\n"
            "Edit DATA_YAML at the top of this script."
        )

    model = YOLO(MODEL)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=torch.device("xpu"),
        amp=AMP,
        workers=WORKERS,
        project=PROJECT,
        name=NAME,
        plots=True,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
