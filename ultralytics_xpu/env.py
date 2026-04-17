"""Environment inspection helpers.

These utilities centralize the checks performed by the CLI before attempting
any patch/restore operation, so that the UI layer stays free of branching
logic and only cares about rendering.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EnvReport:
    """Structured snapshot of the runtime environment."""

    python_version: str
    executable: str
    in_venv: bool
    venv_path: Optional[str]
    torch_version: Optional[str]
    torch_xpu_available: bool
    xpu_device_name: Optional[str]
    ultralytics_version: Optional[str]
    ultralytics_path: Optional[Path]
    errors: list[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return (
            self.in_venv
            and self.torch_version is not None
            and self.ultralytics_path is not None
        )


def _detect_venv() -> tuple[bool, Optional[str]]:
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        return True, venv
    # Fallback: sys.prefix differs from base_prefix when inside a venv
    base = getattr(sys, "base_prefix", sys.prefix)
    if base != sys.prefix:
        return True, sys.prefix
    return False, None


def collect() -> EnvReport:
    """Gather a full environment report without raising."""
    in_venv, venv_path = _detect_venv()
    errors: list[str] = []

    torch_version: Optional[str] = None
    xpu_available = False
    xpu_name: Optional[str] = None
    try:
        import torch  # type: ignore

        torch_version = torch.__version__
        if hasattr(torch, "xpu"):
            try:
                if torch.xpu.is_available():
                    xpu_available = True
                    try:
                        xpu_name = torch.xpu.get_device_name(0)
                    except Exception as exc:  # pragma: no cover - best effort
                        errors.append(f"xpu.get_device_name failed: {exc}")
            except Exception as exc:  # pragma: no cover - best effort
                errors.append(f"xpu.is_available failed: {exc}")
    except Exception as exc:
        errors.append(f"torch import failed: {exc}")

    ul_version: Optional[str] = None
    ul_path: Optional[Path] = None
    try:
        import ultralytics  # type: ignore

        ul_version = getattr(ultralytics, "__version__", "unknown")
        ul_path = Path(os.path.dirname(ultralytics.__file__))
    except Exception as exc:
        errors.append(f"ultralytics import failed: {exc}")

    return EnvReport(
        python_version=sys.version.split()[0],
        executable=sys.executable,
        in_venv=in_venv,
        venv_path=venv_path,
        torch_version=torch_version,
        torch_xpu_available=xpu_available,
        xpu_device_name=xpu_name,
        ultralytics_version=ul_version,
        ultralytics_path=ul_path,
        errors=errors,
    )


def target_files(base: Path) -> dict[str, Path]:
    """Return the files touched by the patch, keyed by short name."""
    return {
        "torch_utils": base / "utils" / "torch_utils.py",
        "trainer": base / "engine" / "trainer.py",
        "validator": base / "engine" / "validator.py",
    }
