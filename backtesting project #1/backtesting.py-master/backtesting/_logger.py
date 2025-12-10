"""Lightweight logging helpers for backtesting events.

The :class:`EventLogger` class writes timestamped messages to a log file while
avoiding duplicate consecutive entries. The default location follows the
desktop path requested by the product requirements (``~/Desktop/log``) and the
filename encodes the creation time together with an optional priority label so
log files are easy to sort and identify.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


def _default_log_dir() -> Path:
    """Return the default directory where log files should be stored."""

    return Path.home() / "Desktop" / "log"


@dataclass
class EventLogger:
    """Simple file-backed logger used to capture order lifecycle events.

    Parameters
    ----------
    priority
        Optional label used in the log filename. It represents the "highest"
        priority as defined by the consuming application (e.g. order type or
        indicator name). If omitted, ``"backtest"`` is used.
    log_dir
        Directory where log files are written. Defaults to ``~/Desktop/log``.
    name
        Explicit file name to use. When omitted, the name is composed of the
        current timestamp and the priority label.
    """

    priority: Optional[str] = None
    log_dir: Optional[Path] = None
    name: Optional[str] = None

    def __post_init__(self):
        self._log_dir = Path(self.log_dir) if self.log_dir else _default_log_dir()
        self._log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        priority = (self.priority or "backtest").replace(" ", "_")
        filename = self.name or f"{timestamp}_{priority}.log"
        self._path = self._log_dir / filename
        self._last_line: Optional[str] = None

    @property
    def path(self) -> Path:
        """Path to the current log file."""

        return self._path

    def log(self, message: str) -> str:
        """Write a message with a timestamp prefix.

        Duplicate consecutive messages are ignored to avoid double writes (for
        example, when a take-profit event triggers multiple callbacks).
        Returns the formatted log line for convenience.
        """

        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        if line == self._last_line:
            return line

        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

        self._last_line = line
        return line

    def info(self, message: str) -> str:
        """Convenience wrapper for ``log``."""

        return self.log(f"INFO: {message}")

    def error(self, message: str) -> str:
        """Write an error message."""

        return self.log(f"ERROR: {message}")
