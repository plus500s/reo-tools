import sys

FG_BLUE = "\x1b[34m"
FG_GREEN = "\x1b[32m"
RESET = "\x1b[0m"


def progress(pct: int, msg: str = "", *, color: bool = True, enabled: bool = True) -> None:
    """Print a progress message with a percentage and an icon."""
    if not enabled:
        return  # Skip progress output if disabled

    pct = max(0, min(100, pct))
    token = f"[{pct:03d}] "

    clr = FG_GREEN if pct == 100 else FG_BLUE
    icon = "✔" if pct == 100 else "▶"
    message = f"{clr}{pct:3d}% {icon} {msg}{RESET}" if color and sys.stdout.isatty() else f"{pct:3d}% {icon} {msg}"

    print(f"{token}{message}", flush=True)
