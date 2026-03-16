from __future__ import annotations


def format_seconds(seconds: float) -> str:
    """Format a duration in seconds as 'Xh Ym Zs', 'Ym Zs', or 'Zs'."""
    total = int(seconds)
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {sec}s"
    if minutes > 0:
        return f"{minutes}m {sec}s"
    return f"{sec}s"
