"""Interactive command-line entry point.

Run:

    python -m ultralytics_xpu

The CLI is intentionally thin: every heavy decision lives in :mod:`patcher`,
:mod:`restorer` or :mod:`env`. This module is only responsible for rendering
and prompting.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

from rich.live import Live
from rich.prompt import Confirm, Prompt
from rich.text import Text

from . import env as env_mod
from . import ui
from .patcher import Status, StepResult, run_patch
from .restorer import run_restore

EXIT_OK = 0
EXIT_GENERIC = 1
EXIT_ABORT = 130


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pause() -> None:
    ui.console.print()
    Prompt.ask(
        Text("  ↵ Press Enter to return to the menu", style=ui.MUTED),
        default="",
        show_default=False,
    )


def _preflight(report: env_mod.EnvReport, require_xpu: bool = False) -> bool:
    if not report.in_venv:
        ui.console.print(
            ui.error_panel(
                "No active virtual environment detected.\n"
                "Activate your venv first, e.g.:  source venv/bin/activate"
            )
        )
        return False
    if report.torch_version is None:
        ui.console.print(
            ui.error_panel(
                "PyTorch is not installed inside the active environment.\n"
                "Install a PyTorch build with XPU support before patching."
            )
        )
        return False
    if report.ultralytics_path is None:
        ui.console.print(
            ui.error_panel(
                "Ultralytics is not installed inside the active environment.\n"
                "Install it with:  pip install ultralytics"
            )
        )
        return False
    if require_xpu and not report.torch_xpu_available:
        ui.console.print(
            ui.info_panel(
                "⚠ XPU not detected",
                [
                    "torch.xpu.is_available() returned False.",
                    "The patch can still be applied, but training on 'xpu'",
                    "will fail until your PyTorch build exposes an Intel XPU device.",
                ],
                border="yellow",
            )
        )
    return True


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def _stream_steps(render_title: str, iterator_factory) -> list[StepResult]:
    """Render patch/restore progress line-by-line using a Live view."""
    steps: list[StepResult] = []
    header = Text.assemble(("▸ ", ui.ARC_CYAN), (render_title, "bold white"))

    with Live(console=ui.console, refresh_per_second=24, transient=False) as live:
        body = Text()
        body.append(header)
        body.append("\n\n")
        live.update(body)

        def on_progress(step: StepResult) -> None:
            steps.append(step)
            body.append(ui.render_step(step))
            body.append("\n")
            live.update(body)
            time.sleep(0.02)

        iterator_factory(on_progress)

    return steps


def action_patch(report: env_mod.EnvReport) -> int:
    ui.console.print(ui.divider("Patch · Ultralytics → XPU"))
    if not _preflight(report, require_xpu=False):
        return EXIT_GENERIC

    assert report.ultralytics_path is not None
    base = report.ultralytics_path
    files = env_mod.target_files(base)
    file_list = "\n".join(f"  • {p}" for p in files.values())
    ui.console.print(
        ui.info_panel(
            "The following files will be patched (backups end in .bak)",
            file_list.splitlines(),
        )
    )

    if not Confirm.ask(
        Text("  Proceed with patch?", style="bold white"),
        default=True,
        console=ui.console,
    ):
        ui.console.print(Text("  Aborted by user.\n", style=ui.MUTED))
        return EXIT_OK

    def factory(cb):
        run_patch(base, on_progress=cb)

    steps = _stream_steps("Applying patch", factory)
    ui.console.print()
    ui.console.print(ui.report_summary(steps, "📦  Patch report"))

    if any(s.status is Status.ERROR for s in steps):
        return EXIT_GENERIC

    ui.console.print(
        ui.info_panel(
            "Next steps",
            [
                "1. Restart your Python kernel / shell.",
                "2. Use  device=torch.device('xpu')  in model.train().",
                "3. Consider  amp=False  for initial smoke tests.",
                "",
                "Example: see  examples/train_yolo11n_xpu.py",
            ],
        )
    )
    return EXIT_OK


def action_restore(report: env_mod.EnvReport) -> int:
    ui.console.print(ui.divider("Restore · original files"))
    if not _preflight(report):
        return EXIT_GENERIC

    assert report.ultralytics_path is not None
    base = report.ultralytics_path

    keep_backup = Confirm.ask(
        Text("  Keep .bak files after restoring?", style="bold white"),
        default=True,
        console=ui.console,
    )

    def factory(cb):
        run_restore(base, keep_backup=keep_backup, on_progress=cb)

    steps = _stream_steps("Restoring files", factory)
    ui.console.print()
    ui.console.print(ui.report_summary(steps, "↩  Restore report"))
    return EXIT_OK if all(s.status is not Status.ERROR for s in steps) else EXIT_GENERIC


def action_doctor(report: env_mod.EnvReport) -> int:
    ui.console.print(ui.divider("Doctor · environment diagnostics"))
    ui.console.print(ui.env_panel(report))
    if report.errors:
        ui.console.print(
            ui.info_panel(
                "Warnings",
                [f"• {e}" for e in report.errors],
                border="yellow",
            )
        )
    if report.ultralytics_path is not None:
        files = env_mod.target_files(report.ultralytics_path)
        lines = []
        for key, path in files.items():
            bak = path.with_suffix(path.suffix + ".bak")
            status = []
            status.append("patched:no" if not bak.exists() else "patched:yes")
            status.append(f"exists:{'yes' if path.exists() else 'no'}")
            lines.append(f"  {key:<12} {path}  ({', '.join(status)})")
        ui.console.print(
            ui.info_panel("Target files", lines, border=ui.ARC_VIOLET)
        )
    return EXIT_OK


def action_examples() -> int:
    ui.console.print(ui.divider("Examples"))
    ui.console.print(
        ui.info_panel(
            "Ready-to-run scripts",
            [
                "examples/train_yolo11n_xpu.py   — quick smoke test on coco8",
                "examples/train_custom_xpu.py    — custom dataset template",
                "examples/predict_xpu.py         — inference on a folder/image",
                "examples/benchmark_xpu.py       — throughput measurement",
                "",
                "Run any of them with:  python examples/<file>.py",
            ],
        )
    )
    return EXIT_OK


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

MENU: list[tuple[str, str, str]] = [
    ("1", "Patch Ultralytics",   "Apply the Intel Arc XPU patch (with backups)"),
    ("2", "Restore original",    "Revert files from their .bak copies"),
    ("3", "Doctor",              "Inspect the environment and patch status"),
    ("4", "Examples",            "List example training / inference scripts"),
    ("q", "Quit",                "Exit the CLI"),
]


def _show_home(report: env_mod.EnvReport) -> None:
    ui.console.clear()
    ui.console.print(ui.banner())
    ui.console.print(ui.env_panel(report))
    ui.console.print(ui.menu_panel(MENU))


def interactive_loop() -> int:
    last_exit = EXIT_OK
    while True:
        report = env_mod.collect()
        _show_home(report)

        choice = Prompt.ask(
            Text("  Select an option", style="bold white"),
            choices=[k for k, *_ in MENU],
            default="1",
            show_choices=False,
            console=ui.console,
        ).strip().lower()

        if choice == "q":
            ui.console.print(
                Text("  ◆ Bye — happy training on XPU!\n", style="bold " + ui.ARC_CYAN)
            )
            return last_exit

        if choice == "1":
            last_exit = action_patch(report)
        elif choice == "2":
            last_exit = action_restore(report)
        elif choice == "3":
            last_exit = action_doctor(report)
        elif choice == "4":
            last_exit = action_examples()
        _pause()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def _argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ultralytics-xpu",
        description="Patch Ultralytics to enable Intel Arc XPU training.",
    )
    sub = p.add_subparsers(dest="command")

    sub.add_parser("patch", help="Apply the XPU patch non-interactively")
    r = sub.add_parser("restore", help="Restore original files from backups")
    r.add_argument(
        "--delete-backups",
        action="store_true",
        help="Remove .bak files after restoring",
    )
    sub.add_parser("doctor", help="Print an environment report")
    sub.add_parser("menu", help="Launch the interactive TUI (default)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _argparser().parse_args(argv)
    try:
        if args.command in (None, "menu"):
            return interactive_loop()

        report = env_mod.collect()
        if args.command == "patch":
            return action_patch(report)
        if args.command == "restore":
            if not _preflight(report):
                return EXIT_GENERIC
            assert report.ultralytics_path is not None
            steps = _stream_steps(
                "Restoring files",
                lambda cb: run_restore(
                    report.ultralytics_path,
                    keep_backup=not args.delete_backups,
                    on_progress=cb,
                ),
            )
            ui.console.print(ui.report_summary(steps, "↩  Restore report"))
            return EXIT_OK if all(s.status is not Status.ERROR for s in steps) else EXIT_GENERIC
        if args.command == "doctor":
            return action_doctor(report)
    except KeyboardInterrupt:
        ui.console.print(Text("\n  Aborted.", style="bold red"))
        return EXIT_ABORT
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
