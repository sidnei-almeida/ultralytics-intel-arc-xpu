"""Restore logic — reinstate original Ultralytics files from ``.bak`` copies."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .env import target_files
from .patcher import Status, StepResult


@dataclass
class RestoreReport:
    steps: list[StepResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not any(s.status is Status.ERROR for s in self.steps)

    def add(self, step: StepResult) -> StepResult:
        self.steps.append(step)
        return step


Progress = Callable[[StepResult], None]


def _restore_one(path: Path, keep_backup: bool) -> StepResult:
    bak = Path(str(path) + ".bak")
    if not bak.exists():
        return StepResult(
            name=f"restore:{path.name}",
            status=Status.WARN,
            detail=f"No backup found ({bak.name})",
            file=path,
        )
    shutil.copy2(bak, path)
    if not keep_backup:
        bak.unlink(missing_ok=True)
        suffix = " (backup removed)"
    else:
        suffix = " (backup kept)"
    return StepResult(
        name=f"restore:{path.name}",
        status=Status.OK,
        detail=f"Restored from {bak.name}{suffix}",
        file=path,
    )


def run_restore(
    base: Path,
    keep_backup: bool = True,
    on_progress: Optional[Progress] = None,
) -> RestoreReport:
    """Restore every patched file under ``base`` using its ``.bak`` sibling."""
    report = RestoreReport()
    for path in target_files(base).values():
        step = _restore_one(path, keep_backup=keep_backup)
        report.add(step)
        if on_progress is not None:
            on_progress(step)
    return report
