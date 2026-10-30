"""Wait until the machine has enough free memory before heavy work (macOS)."""

import re
import subprocess
import time


def free_percent() -> float | None:
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=30).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"free percentage:\s*(\d+)%", out)
    return float(m.group(1)) if m else None


def wait_for_memory(min_free: float = 25.0, poll_s: float = 60.0, max_wait_s: float = 3600.0) -> None:
    """Block while free memory is below `min_free` percent. No-op off macOS."""
    waited = 0.0
    while True:
        f = free_percent()
        if f is None or f >= min_free:
            return
        if waited >= max_wait_s:
            raise RuntimeError(f"free memory stayed at {f}% < {min_free}% for {max_wait_s}s")
        print(f"[memory] free {f}% < {min_free}%, waiting {poll_s}s", flush=True)
        time.sleep(poll_s)
        waited += poll_s
