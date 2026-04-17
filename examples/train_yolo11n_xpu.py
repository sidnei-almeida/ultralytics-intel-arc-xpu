"""Quick smoke test: train YOLO11n on the tiny ``coco8`` dataset on Intel XPU.

Use this immediately after applying the patch to confirm that ``device='xpu'``
is wired end to end.

    python examples/train_yolo11n_xpu.py
"""

from __future__ import annotations

import torch
from ultralytics import YOLO


def main() -> None:
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise SystemExit(
            "Intel XPU is not available in this PyTorch build. "
            "Install a PyTorch release with XPU support first."
        )

    print(f"Torch: {torch.__version__}")
    print(f"Device: {torch.xpu.get_device_name(0)}")

    model = YOLO("yolo11n.pt")

    model.train(
        data="coco8.yaml",
        epochs=1,
        imgsz=640,
        batch=4,
        device=torch.device("xpu"),
        amp=False,       # disable AMP for the first smoke test
        plots=False,
        workers=0,       # safer on Linux XPU during bring-up
    )


if __name__ == "__main__":
    main()
