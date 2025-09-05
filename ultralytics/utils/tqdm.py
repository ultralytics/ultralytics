# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
import sys
import time
from functools import lru_cache
from typing import IO, Any


@lru_cache(maxsize=1)
def is_noninteractive_console() -> bool:
    """Check for known non-interactive console environments."""
    return "GITHUB_ACTIONS" in os.environ or "RUNPOD_POD_ID" in os.environ


class TQDM:
    """
    Lightweight zero-dependency progress bar for Ultralytics.

    Provides clean, rich-style progress bars suitable for various environments including Weights & Biases,
    console outputs, and other logging systems. Features zero external dependencies, clean single-line output,
    rich-style progress bars with Unicode block characters, context manager support, iterator protocol support,
    and dynamic description updates.

    Attributes:
        iterable (object): Iterable to wrap with progress bar.
        desc (str): Prefix description for the progress bar.
        total (int): Expected number of iterations.
        disable (bool): Whether to disable the progress bar.
        unit (str): String for units of iteration.
        unit_scale (bool): Auto-scale units flag.
        unit_divisor (int): Divisor for unit scaling.
        leave (bool): Whether to leave the progress bar after completion.
        mininterval (float): Minimum time interval between updates.
        initial (int): Initial counter value.
        n (int): Current iteration count.
        closed (bool): Whether the progress bar is closed.
        bar_format (str): Custom bar format string.
        file (object): Output file stream.

    Methods:
        update: Update progress by n steps.
        set_description: Set or update the description.
        set_postfix: Set postfix for the progress bar.
        close: Close the progress bar and clean up.
        refresh: Refresh the progress bar display.
        clear: Clear the progress bar from display.
        write: Write a message without breaking the progress bar.

    Examples:
        Basic usage with iterator:
        >>> for i in TQDM(range(100)):
        ...     time.sleep(0.01)

        With custom description:
        >>> pbar = TQDM(range(100), desc="Processing")
        >>> for i in pbar:
        ...     pbar.set_description(f"Processing item {i}")

        Context manager usage:
        >>> with TQDM(total=100, unit="B", unit_scale=True) as pbar:
        ...     for i in range(100):
        ...         pbar.update(1)

        Manual updates:
        >>> pbar = TQDM(total=100, desc="Training")
        >>> for epoch in range(100):
        ...     # Do work
        ...     pbar.update(1)
        >>> pbar.close()
    """

    # Constants
    MIN_RATE_CALC_INTERVAL = 0.01  # Minimum time interval for rate calculation
    RATE_SMOOTHING_FACTOR = 0.3  # Factor for exponential smoothing of rates
    MAX_SMOOTHED_RATE = 1000000  # Maximum rate to apply smoothing to
    NONINTERACTIVE_MIN_INTERVAL = 60.0  # Minimum interval for non-interactive environments

    def __init__(
        self,
        iterable: Any = None,
        desc: str | None = None,
        total: int | None = None,
        leave: bool = True,
        file: IO[str] | None = None,
        mininterval: float = 0.1,
        disable: bool | None = None,
        unit: str = "it",
        unit_scale: bool = True,
        unit_divisor: int = 1000,
        bar_format: str | None = None,  # kept for API compatibility; not used for formatting
        initial: int = 0,
        **kwargs,
    ) -> None:
        """
        Initialize the TQDM progress bar with specified configuration options.

        Args:
            iterable (object, optional): Iterable to wrap with progress bar.
            desc (str, optional): Prefix description for the progress bar.
            total (int, optional): Expected number of iterations.
            leave (bool, optional): Whether to leave the progress bar after completion.
            file (object, optional): Output file stream for progress display.
            mininterval (float, optional): Minimum time interval between updates (default 0.1s, 60s in GitHub Actions).
            disable (bool, optional): Whether to disable the progress bar. Auto-detected if None.
            unit (str, optional): String for units of iteration (default "it" for items).
            unit_scale (bool, optional): Auto-scale units for bytes/data units.
            unit_divisor (int, optional): Divisor for unit scaling (default 1000).
            bar_format (str, optional): Custom bar format string.
            initial (int, optional): Initial counter value.
            **kwargs (Any): Additional keyword arguments for compatibility (ignored).

        Examples:
            >>> pbar = TQDM(range(100), desc="Processing")
            >>> with TQDM(total=1000, unit="B", unit_scale=True) as pbar:
            ...     pbar.update(1024)  # Updates by 1KB
        """
        # Disable if not verbose
        if disable is None:
            try:
                from ultralytics.utils import LOGGER, VERBOSE

                disable = not VERBOSE or LOGGER.getEffectiveLevel() > 20
            except ImportError:
                disable = False

        self.iterable = iterable
        self.desc = desc or ""
        self.total = total or (len(iterable) if hasattr(iterable, "__len__") else None) or None  # prevent total=0
        self.disable = disable
        self.unit = unit
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        self.leave = leave
        self.noninteractive = is_noninteractive_console()
        self.mininterval = max(mininterval, self.NONINTERACTIVE_MIN_INTERVAL) if self.noninteractive else mininterval
        self.initial = initial

        # Kept for API compatibility (unused for f-string formatting)
        self.bar_format = bar_format

        self.file = file or sys.stdout

        # Internal state
        self.n = self.initial
        self.last_print_n = self.initial
        self.last_print_t = time.time()
        self.start_t = time.time()
        self.last_rate = 0.0
        self.closed = False
        self.is_bytes = unit_scale and unit in ("B", "bytes")
        self.scales = (
            [(1073741824, "GB/s"), (1048576, "MB/s"), (1024, "KB/s")]
            if self.is_bytes
            else [(1e9, f"G{self.unit}/s"), (1e6, f"M{self.unit}/s"), (1e3, f"K{self.unit}/s")]
        )

        if not self.disable and self.total and not self.noninteractive:
            self._display()

    def _format_rate(self, rate: float) -> str:
        """Format rate with units."""
        if rate <= 0:
            return ""
        fallback = f"{rate:.1f}B/s" if self.is_bytes else f"{rate:.1f}{self.unit}/s"
        return next((f"{rate / t:.1f}{u}" for t, u in self.scales if rate >= t), fallback)

    def _format_num(self, num: int | float) -> str:
        """Format number with optional unit scaling."""
        if not self.unit_scale or not self.is_bytes:
            return str(num)

        for unit in ("", "K", "M", "G", "T"):
            if abs(num) < self.unit_divisor:
                return f"{num:3.1f}{unit}B" if unit else f"{num:.0f}B"
            num /= self.unit_divisor
        return f"{num:.1f}PB"

    def _format_time(self, seconds: float) -> str:
        """Format time duration."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}:{seconds % 60:02.0f}"
        else:
            h, m = int(seconds // 3600), int((seconds % 3600) // 60)
            return f"{h}:{m:02d}:{seconds % 60:02.0f}"

    def _generate_bar(self, width: int = 12) -> str:
        """Generate progress bar."""
        if self.total is None:
            return "━" * width if self.closed else "─" * width

        frac = min(1.0, self.n / self.total)
        filled = int(frac * width)
        bar = "━" * filled + "─" * (width - filled)
        if filled < width and frac * width - filled > 0.5:
            bar = f"{bar[:filled]}╸{bar[filled + 1 :]}"
        return bar

    def _should_update(self, dt: float, dn: int) -> bool:
        """Check if display should update."""
        if self.noninteractive:
            return False
        return (self.total is not None and self.n >= self.total) or (dt >= self.mininterval)

    def _display(self, final: bool = False) -> None:
        """Display progress bar."""
        if self.disable or (self.closed and not final):
            return

        current_time = time.time()
        dt = current_time - self.last_print_t
        dn = self.n - self.last_print_n

        if not final and not self._should_update(dt, dn):
            return

        # Calculate rate (avoid crazy numbers)
        if dt > self.MIN_RATE_CALC_INTERVAL:
            rate = dn / dt if dt else 0.0
            # Smooth rate for reasonable values, use raw rate for very high values
            if rate < self.MAX_SMOOTHED_RATE:
                self.last_rate = self.RATE_SMOOTHING_FACTOR * rate + (1 - self.RATE_SMOOTHING_FACTOR) * self.last_rate
                rate = self.last_rate
        else:
            rate = self.last_rate

        # At completion, use overall rate
        if self.total and self.n >= self.total:
            overall_elapsed = current_time - self.start_t
            if overall_elapsed > 0:
                rate = self.n / overall_elapsed

        # Update counters
        self.last_print_n = self.n
        self.last_print_t = current_time
        elapsed = current_time - self.start_t

        # Remaining time
        remaining_str = ""
        if self.total and 0 < self.n < self.total and elapsed > 0:
            est_rate = rate or (self.n / elapsed)
            remaining_str = f"<{self._format_time((self.total - self.n) / est_rate)}"

        # Numbers and percent
        if self.total:
            percent = (self.n / self.total) * 100
            n_str = self._format_num(self.n)
            t_str = self._format_num(self.total)
            if self.is_bytes:
                # Collapse suffix only when identical (e.g. "5.4/5.4MB")
                if n_str[-2] == t_str[-2]:
                    n_str = n_str.rstrip("KMGTPB")  # Remove unit suffix from current if different than total
        else:
            percent = 0.0
            n_str, t_str = self._format_num(self.n), "?"

        elapsed_str = self._format_time(elapsed)
        rate_str = self._format_rate(rate) or (self._format_rate(self.n / elapsed) if elapsed > 0 else "")

        bar = self._generate_bar()

        # Compose progress line via f-strings (two shapes: with/without total)
        if self.total:
            if self.is_bytes and self.n >= self.total:
                # Completed bytes: show only final size
                progress_str = f"{self.desc}: {percent:.0f}% {bar} {t_str} {rate_str} {elapsed_str}"
            else:
                progress_str = (
                    f"{self.desc}: {percent:.0f}% {bar} {n_str}/{t_str} {rate_str} {elapsed_str}{remaining_str}"
                )
        else:
            progress_str = f"{self.desc}: {bar} {n_str} {rate_str} {elapsed_str}"

        # Write to output
        try:
            if self.noninteractive:
                # In non-interactive environments, avoid carriage return which creates empty lines
                self.file.write(progress_str)
            else:
                # In interactive terminals, use carriage return and clear line for updating display
                self.file.write(f"\r\033[K{progress_str}")
            self.file.flush()
        except Exception:
            pass

    def update(self, n: int = 1) -> None:
        """Update progress by n steps."""
        if not self.disable and not self.closed:
            self.n += n
            self._display()

    def set_description(self, desc: str | None) -> None:
        """Set description."""
        self.desc = desc or ""
        if not self.disable:
            self._display()

    def set_postfix(self, **kwargs: Any) -> None:
        """Set postfix (appends to description)."""
        if kwargs:
            postfix = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            base_desc = self.desc.split(" | ")[0] if " | " in self.desc else self.desc
            self.set_description(f"{base_desc} | {postfix}")

    def close(self) -> None:
        """Close progress bar."""
        if self.closed:
            return

        self.closed = True

        if not self.disable:
            # Final display
            if self.total and self.n >= self.total:
                self.n = self.total
            self._display(final=True)

            # Cleanup
            if self.leave:
                self.file.write("\n")
            else:
                self.file.write("\r\033[K")

            try:
                self.file.flush()
            except Exception:
                pass

    def __enter__(self) -> TQDM:
        """Enter context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager and close progress bar."""
        self.close()

    def __iter__(self) -> Any:
        """Iterate over the wrapped iterable with progress updates."""
        if self.iterable is None:
            raise TypeError("'NoneType' object is not iterable")

        try:
            for item in self.iterable:
                yield item
                self.update(1)
        finally:
            self.close()

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except Exception:
            pass

    def refresh(self) -> None:
        """Refresh display."""
        if not self.disable:
            self._display()

    def clear(self) -> None:
        """Clear progress bar."""
        if not self.disable:
            try:
                self.file.write("\r\033[K")
                self.file.flush()
            except Exception:
                pass

    @staticmethod
    def write(s: str, file: IO[str] | None = None, end: str = "\n") -> None:
        """Static method to write without breaking progress bar."""
        file = file or sys.stdout
        try:
            file.write(s + end)
            file.flush()
        except Exception:
            pass


if __name__ == "__main__":
    import time

    print("1. Basic progress bar with known total:")
    for i in TQDM(range(3), desc="Known total"):
        time.sleep(0.05)

    print("\n2. Manual updates with known total:")
    pbar = TQDM(total=300, desc="Manual updates", unit="files")
    for i in range(300):
        time.sleep(0.03)
        pbar.update(1)
        if i % 10 == 9:
            pbar.set_description(f"Processing batch {i // 10 + 1}")
    pbar.close()

    print("\n3. Progress bar with unknown total:")
    pbar = TQDM(desc="Unknown total", unit="items")
    for i in range(25):
        time.sleep(0.08)
        pbar.update(1)
        if i % 5 == 4:
            pbar.set_postfix(processed=i + 1, status="OK")
    pbar.close()

    print("\n4. Context manager with unknown total:")
    with TQDM(desc="Processing stream", unit="B", unit_scale=True, unit_divisor=1024) as pbar:
        for i in range(30):
            time.sleep(0.1)
            pbar.update(1024 * 1024 * i)  # Simulate processing MB of data

    print("\n5. Iterator with unknown length:")

    def data_stream():
        """Simulate a data stream of unknown length."""
        import random

        for i in range(random.randint(10, 20)):
            yield f"data_chunk_{i}"

    for chunk in TQDM(data_stream(), desc="Stream processing", unit="chunks"):
        time.sleep(0.1)

    print("\n6. File processing simulation (unknown size):")

    def process_files():
        """Simulate processing files of unknown count."""
        return [f"file_{i}.txt" for i in range(18)]

    pbar = TQDM(desc="Scanning files", unit="files")
    files = process_files()
    for i, filename in enumerate(files):
        time.sleep(0.06)
        pbar.update(1)
        pbar.set_description(f"Processing {filename}")
    pbar.close()
