"""Patch logic for enabling Intel Arc XPU inside Ultralytics.

The module is deliberately framework-agnostic: it returns structured results
(``StepResult``) so that any frontend — CLI, TUI, or Python API — can render
them however it likes.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Optional

from .env import target_files


class Status(str, Enum):
    OK = "ok"
    SKIP = "skip"
    WARN = "warn"
    ERROR = "error"


@dataclass
class StepResult:
    """Outcome of a single patch step, consumable by any UI layer."""

    name: str
    status: Status
    detail: str = ""
    file: Optional[Path] = None


@dataclass
class PatchReport:
    steps: list[StepResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not any(s.status is Status.ERROR for s in self.steps)

    def add(self, step: StepResult) -> StepResult:
        self.steps.append(step)
        return step


Progress = Callable[[StepResult], None]


# ---------------------------------------------------------------------------
# Backups
# ---------------------------------------------------------------------------

def _backup(path: Path) -> StepResult:
    if not path.exists():
        return StepResult(
            name=f"backup:{path.name}",
            status=Status.ERROR,
            detail=f"Target file does not exist: {path}",
            file=path,
        )
    bak = path.with_suffix(path.suffix + ".bak")
    if bak.exists():
        return StepResult(
            name=f"backup:{path.name}",
            status=Status.SKIP,
            detail=f"Backup already exists ({bak.name})",
            file=path,
        )
    shutil.copy2(path, bak)
    return StepResult(
        name=f"backup:{path.name}",
        status=Status.OK,
        detail=f"Backup created → {bak.name}",
        file=path,
    )


# ---------------------------------------------------------------------------
# Text transformations
# ---------------------------------------------------------------------------

def _apply_replacements(
    path: Path,
    replacements: Iterable[tuple[str, str]],
) -> list[StepResult]:
    text = path.read_text(encoding="utf-8")
    original = text
    results: list[StepResult] = []

    for idx, (old, new) in enumerate(replacements, start=1):
        label = f"patch:{path.name}#{idx}"
        if old in text:
            text = text.replace(old, new)
            results.append(
                StepResult(label, Status.OK, "Snippet replaced", path)
            )
        elif new in text:
            results.append(
                StepResult(label, Status.SKIP, "Already patched", path)
            )
        else:
            results.append(
                StepResult(
                    label,
                    Status.WARN,
                    "Original snippet not found (file may differ from supported version)",
                    path,
                )
            )

    if text != original:
        path.write_text(text, encoding="utf-8")
    return results


# ---------------------------------------------------------------------------
# File-specific patches
# ---------------------------------------------------------------------------

def _patch_torch_utils(path: Path) -> list[StepResult]:
    results = _apply_replacements(
        path,
        [
            (
                'cpu = device == "cpu"\n    mps = device in {"mps", "mps:0"}',
                'cpu = device == "cpu"\n    mps = device in {"mps", "mps:0"}\n'
                '    xpu = device in {"xpu", "xpu:0"}',
            ),
            ("if cpu or mps:", "if cpu or mps or xpu:"),
            (
                "if not cpu and not mps and torch.cuda.is_available():",
                "if not cpu and not mps and not xpu and torch.cuda.is_available():",
            ),
            (
                "elif mps and TORCH_2_0 and torch.backends.mps.is_available():",
                'elif xpu and hasattr(torch, "xpu") and torch.xpu.is_available():\n'
                '        s += "XPU\\n"\n'
                '        arg = "xpu"\n'
                "    elif mps and TORCH_2_0 and torch.backends.mps.is_available():",
            ),
        ],
    )

    text = path.read_text(encoding="utf-8")
    if "torch.xpu.manual_seed" in text:
        results.append(
            StepResult(
                "patch:torch_utils#seed",
                Status.SKIP,
                "XPU seed hook already present",
                path,
            )
        )
    else:
        new_text = text.replace(
            "torch.manual_seed(seed)",
            "torch.manual_seed(seed)\n"
            "    if hasattr(torch, 'xpu') and torch.xpu.is_available():\n"
            "        torch.xpu.manual_seed(seed)\n"
            "        torch.xpu.manual_seed_all(seed)",
        )
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            results.append(
                StepResult(
                    "patch:torch_utils#seed",
                    Status.OK,
                    "Injected XPU seed hook",
                    path,
                )
            )
        else:
            results.append(
                StepResult(
                    "patch:torch_utils#seed",
                    Status.WARN,
                    "Could not inject XPU seed hook (anchor missing)",
                    path,
                )
            )

    return results


def _patch_trainer(path: Path) -> list[StepResult]:
    results = _apply_replacements(
        path,
        [
            (
                'if self.device.type in {"cpu", "mps"}:',
                'if self.device.type in {"cpu", "mps", "xpu"}:',
            ),
            (
                'elif self.args.device in {"cpu", "mps"}:',
                'elif self.args.device in {"cpu", "mps", "xpu"}:',
            ),
        ],
    )

    text = path.read_text(encoding="utf-8")
    original = text

    text, n_amp = re.subn(
        r"with autocast\((self\.amp)\):",
        r"with autocast(\1, device=self.device.type):",
        text,
    )
    results.append(
        StepResult(
            "patch:trainer#autocast",
            Status.OK if n_amp else Status.SKIP,
            f"autocast occurrences rewritten: {n_amp}",
            path,
        )
    )

    swaps = [
        (
            "torch.cuda.empty_cache()",
            'torch.xpu.empty_cache() if self.device.type=="xpu" else torch.cuda.empty_cache()',
            "trainer#empty_cache",
        ),
        (
            "torch.cuda.memory_reserved()",
            'torch.xpu.memory_reserved() if self.device.type=="xpu" else torch.cuda.memory_reserved()',
            "trainer#memory_reserved",
        ),
        (
            "torch.cuda.get_device_properties(self.device).total_memory",
            'torch.xpu.get_device_properties(self.device).total_memory if self.device.type=="xpu" else torch.cuda.get_device_properties(self.device).total_memory',
            "trainer#total_memory",
        ),
    ]
    for old, new, label in swaps:
        if new in text:
            results.append(
                StepResult(f"patch:{label}", Status.SKIP, "Already patched", path)
            )
        elif old in text:
            text = text.replace(old, new)
            results.append(
                StepResult(f"patch:{label}", Status.OK, "CUDA call wrapped for XPU", path)
            )
        else:
            results.append(
                StepResult(
                    f"patch:{label}",
                    Status.WARN,
                    "Target CUDA call not found",
                    path,
                )
            )

    if text != original:
        path.write_text(text, encoding="utf-8")

    return results


def _patch_validator(path: Path) -> list[StepResult]:
    return _apply_replacements(
        path,
        [
            (
                'if self.device.type in {"cpu", "mps"}:',
                'if self.device.type in {"cpu", "mps", "xpu"}:',
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_patch(base: Path, on_progress: Optional[Progress] = None) -> PatchReport:
    """Apply the full XPU patch to an Ultralytics install rooted at ``base``."""
    report = PatchReport()
    files = target_files(base)

    def _emit(step: StepResult) -> None:
        report.add(step)
        if on_progress is not None:
            on_progress(step)

    for path in files.values():
        _emit(_backup(path))

    patchers = {
        "torch_utils": _patch_torch_utils,
        "trainer": _patch_trainer,
        "validator": _patch_validator,
    }
    for key, path in files.items():
        if not path.exists():
            _emit(
                StepResult(
                    name=f"patch:{path.name}",
                    status=Status.ERROR,
                    detail=f"Missing file: {path}",
                    file=path,
                )
            )
            continue
        for step in patchers[key](path):
            _emit(step)

    return report
