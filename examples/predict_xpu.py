"""Run inference on a folder of images using Intel Arc XPU.

    python examples/predict_xpu.py --source path/to/images --weights yolo11n.pt
"""

from __future__ import annotations

import argparse

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolo11n.pt", help="Ultralytics checkpoint")
    p.add_argument("--source",  default="https://ultralytics.com/images/bus.jpg",
                   help="Image / video / folder / URL")
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--save",    action="store_true", help="Save annotated outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise SystemExit("Intel XPU not available.")

    print(f"→ XPU {torch.xpu.get_device_name(0)}")

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        device=torch.device("xpu"),
        conf=args.conf,
        imgsz=args.imgsz,
        save=args.save,
        verbose=True,
    )

    total = sum(len(r.boxes) for r in results)
    print(f"✓ Detected {total} object(s) across {len(results)} frame(s).")


if __name__ == "__main__":
    main()
