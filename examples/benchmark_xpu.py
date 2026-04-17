"""Micro-benchmark: measure inference throughput on Intel Arc XPU.

Reports steady-state FPS after a warm-up phase. Use it to sanity-check
install health or to compare driver / PyTorch versions.
"""

from __future__ import annotations

import argparse
import time

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolo11n.pt")
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--batch",   type=int, default=1)
    p.add_argument("--warmup",  type=int, default=10)
    p.add_argument("--iters",   type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise SystemExit("Intel XPU not available.")

    device = torch.device("xpu")
    print(f"→ XPU   {torch.xpu.get_device_name(0)}")
    print(f"→ Torch {torch.__version__}")
    print(f"→ Cfg   imgsz={args.imgsz} batch={args.batch} iters={args.iters}")

    model = YOLO(args.weights)
    model.to(device)

    x = torch.rand(args.batch, 3, args.imgsz, args.imgsz, device=device)

    for _ in range(args.warmup):
        _ = model.model(x)
    torch.xpu.synchronize()

    start = time.perf_counter()
    for _ in range(args.iters):
        _ = model.model(x)
    torch.xpu.synchronize()
    elapsed = time.perf_counter() - start

    fps = args.iters * args.batch / elapsed
    ms  = elapsed / args.iters * 1000
    print(f"✓ {fps:7.2f} FPS   ({ms:6.2f} ms/iter, batch={args.batch})")


if __name__ == "__main__":
    main()
