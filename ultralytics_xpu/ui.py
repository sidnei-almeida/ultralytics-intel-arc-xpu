"""Shared visual components built on top of ``rich``.

All colors, banners and panels the CLI renders live here so the rest of the
codebase stays focused on logic.
"""

from __future__ import annotations

from typing import Optional

from rich.align import Align
from rich.box import HEAVY, ROUNDED
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .env import EnvReport
from .patcher import Status, StepResult

# ---------------------------------------------------------------------------
# Palette — inspired by Intel Arc gradient
# ---------------------------------------------------------------------------

ARC_CYAN = "#00C7FD"
ARC_BLUE = "#0068B5"
ARC_VIOLET = "#7D4CDB"
ARC_PINK = "#E96DB7"
ACCENT = "bold " + ARC_CYAN
MUTED = "grey58"

STATUS_STYLE: dict[Status, tuple[str, str]] = {
    Status.OK: ("✔", "bold green"),
    Status.SKIP: ("•", "grey62"),
    Status.WARN: ("!", "bold yellow"),
    Status.ERROR: ("✖", "bold red"),
}


console = Console()


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

_BANNER_LINES = [
    r"  _   _ _ _               _       _   _          __  __ ____  _   _  ",
    r" | | | | | |_ _ __ __ _  | |_   _| |_(_) ___ ___ \ \/ /|  _ \| | | | ",
    r" | | | | | __| '__/ _` | | | | | | __| |/ __/ __| \  / | |_) | | | | ",
    r" | |_| | | |_| | | (_| | | | |_| | |_| | (__\__ \ /  \ |  __/| |_| | ",
    r"  \___/|_|\__|_|  \__,_| |_|\__, |\__|_|\___|___//_/\_\|_|    \___/  ",
    r"                            |___/                                    ",
]

_GRADIENT = [ARC_CYAN, "#22B8F0", "#4A9BE0", ARC_BLUE, ARC_VIOLET, ARC_PINK]


def banner(subtitle: str = "Intel Arc GPU enablement for Ultralytics YOLO") -> Panel:
    grad = Text()
    for line, color in zip(_BANNER_LINES, _GRADIENT):
        grad.append(line + "\n", style=color)

    tagline = Text.assemble(
        ("  ◆ ", ARC_CYAN),
        (subtitle, "bold white"),
        ("  ◆", ARC_CYAN),
    )

    inner = Group(grad, Align.center(tagline))
    return Panel(
        Align.center(inner),
        box=HEAVY,
        border_style=ARC_CYAN,
        padding=(1, 4),
    )


def divider(label: Optional[str] = None) -> Rule:
    return Rule(
        title=Text(label, style=ACCENT) if label else "",
        style=ARC_BLUE,
        characters="─",
    )


# ---------------------------------------------------------------------------
# Environment card
# ---------------------------------------------------------------------------

def env_panel(env: EnvReport) -> Panel:
    if env.in_venv:
        venv_value = Text.assemble(
            ("active  ", "bold green"),
            (env.venv_path or "", MUTED),
        )
    else:
        venv_value = Text("not detected", style="bold red")

    torch_value = (
        Text(env.torch_version, style="bold white")
        if env.torch_version
        else Text("not installed", style="bold red")
    )

    if env.torch_xpu_available:
        xpu_value = Text.assemble(
            ("available  ", "bold green"),
            (f"→ {env.xpu_device_name or 'XPU:0'}", "bold " + ARC_CYAN),
        )
    elif env.torch_version is None:
        xpu_value = Text("torch missing", style="bold red")
    else:
        xpu_value = Text("unavailable", style="bold yellow")

    if env.ultralytics_path:
        ul_value = Text.assemble(
            (f"{env.ultralytics_version or 'unknown'}  ", "bold white"),
            (str(env.ultralytics_path), MUTED),
        )
    else:
        ul_value = Text("not installed", style="bold red")

    table = Table.grid(padding=(0, 2), expand=False)
    table.add_column(justify="left", style=MUTED, no_wrap=True, min_width=14)
    table.add_column(justify="left", overflow="fold")

    rows: list[tuple[str, object]] = [
        ("Python",      Text(env.python_version, style="bold white")),
        ("Executable",  Text(env.executable, style=MUTED)),
        ("Virtualenv",  venv_value),
        ("PyTorch",     torch_value),
        ("Intel XPU",   xpu_value),
        ("Ultralytics", ul_value),
    ]
    for label, value in rows:
        table.add_row(label, value)

    title = Text("⚙  Environment", style=ACCENT)
    return Panel(
        table,
        title=title,
        title_align="left",
        border_style=ARC_BLUE,
        box=ROUNDED,
        padding=(1, 2),
    )


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------

def menu_panel(options: list[tuple[str, str, str]]) -> Panel:
    """Render a numbered main menu.

    Each option is a tuple ``(key, title, description)``.
    """
    table = Table.grid(padding=(0, 2), expand=True)
    table.add_column(justify="right", no_wrap=True)
    table.add_column()
    table.add_column()

    for key, title, desc in options:
        table.add_row(
            Text(f"[{key}]", style="bold " + ARC_CYAN),
            Text(title, style="bold white"),
            Text(desc, style=MUTED),
        )

    return Panel(
        table,
        title=Text("❯ What would you like to do?", style=ACCENT),
        title_align="left",
        border_style=ARC_VIOLET,
        box=ROUNDED,
        padding=(1, 2),
    )


# ---------------------------------------------------------------------------
# Step / report rendering
# ---------------------------------------------------------------------------

def render_step(step: StepResult) -> Text:
    glyph, style = STATUS_STYLE[step.status]
    line = Text()
    line.append(f"  {glyph}  ", style=style)
    line.append(f"{step.name:<32}", style="bold white")
    line.append("  ")
    line.append(step.detail, style=MUTED)
    return line


def report_summary(steps: list[StepResult], title: str) -> Panel:
    counts = {s: 0 for s in Status}
    for step in steps:
        counts[step.status] += 1

    stats = Table.grid(padding=(0, 3), expand=False)
    stats.add_column(justify="left")
    for status in Status:
        glyph, style = STATUS_STYLE[status]
        cell = Text()
        cell.append(f"{glyph} ", style=style)
        cell.append(f"{status.value:<6}", style="bold white")
        cell.append(f"{counts[status]:>3}", style="bold " + ARC_CYAN)
        stats.add_row(cell)

    overall_ok = counts[Status.ERROR] == 0
    headline = (
        Text("✓ Completed successfully", style="bold green")
        if overall_ok
        else Text("✗ Completed with errors", style="bold red")
    )

    return Panel(
        Group(headline, Text(""), stats),
        title=Text(title, style=ACCENT),
        title_align="left",
        border_style="green" if overall_ok else "red",
        box=ROUNDED,
        padding=(1, 2),
    )


# ---------------------------------------------------------------------------
# Info blocks
# ---------------------------------------------------------------------------

def info_panel(title: str, lines: list[str], border: str = ARC_CYAN) -> Panel:
    text = Text("\n".join(lines), style="white")
    return Panel(
        text,
        title=Text(title, style=ACCENT),
        title_align="left",
        border_style=border,
        box=ROUNDED,
        padding=(1, 2),
    )


def error_panel(message: str) -> Panel:
    return Panel(
        Text(message, style="bold red"),
        title=Text("✖ Error", style="bold red"),
        border_style="red",
        box=ROUNDED,
        padding=(1, 2),
    )
