# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import sys
import time


def is_github_action_running():
    """Determine if the current environment is a GitHub Actions runner."""
    import os

    return "GITHUB_ACTIONS" in os.environ and "GITHUB_WORKFLOW" in os.environ and "RUNNER_OS" in os.environ


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

    def __init__(
        self,
        iterable=None,
        desc=None,
        total=None,
        leave=True,
        file=None,
        mininterval=0.1,
        disable=None,
        unit="it",
        unit_scale=False,
        unit_divisor=1000,
        bar_format=None,
        initial=0,
        **kwargs,  # Accept unused args for compatibility
    ):
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
            **kwargs: Additional keyword arguments for compatibility (ignored).

        Examples:
            >>> pbar = TQDM(range(100), desc="Processing")
            >>> with TQDM(total=1000, unit="B", unit_scale=True) as pbar:
            ...     pbar.update(1024)  # Updates by 1KB
        """
        if is_github_action_running():
            mininterval = max(mininterval, 60.0)

        # Auto-disable if not verbose
        if disable is None:
            try:
                from ultralytics.utils import LOGGER, VERBOSE

                disable = not VERBOSE or LOGGER.getEffectiveLevel() > 20
            except ImportError:
                disable = False

        self.iterable = iterable
        self.desc = desc or ""
        self.total = total if total is not None else (len(iterable) if hasattr(iterable, "__len__") else None)
        self.disable = disable
        self.unit = unit
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        self.leave = leave
        self.mininterval = mininterval
        self.initial = initial
        self.bar_format = bar_format or "{desc}: {percentage:3.0f}% {bar} {n_fmt}/{total_fmt} {rate_fmt} {elapsed}"
        self.file = file or sys.stdout

        # Internal state
        self.n = self.initial
        self.last_print_n = self.initial
        self.last_print_t = time.time()
        self.start_t = time.time()
        self.last_rate = 0
        self.closed = False

        # Display initial bar if we have total and not disabled
        if not self.disable and self.total is not None:
            self._display()

    def _format_rate(self, rate):
        """Format rate with proper units and reasonable precision."""
        if rate <= 0:
            return ""

        # For bytes, use appropriate units
        if self.unit in ("B", "bytes") and self.unit_scale:
            if rate < 1024:
                return f"{rate:.1f}B/s"
            elif rate < 1024**2:
                return f"{rate / 1024:.1f}KB/s"
            elif rate < 1024**3:
                return f"{rate / 1024**2:.1f}MB/s"
            else:
                return f"{rate / 1024**3:.1f}GB/s"

        # For regular items
        if rate >= 1000000:
            return f"{rate / 1000000:.1f}M{self.unit}/s"
        elif rate >= 1000:
            return f"{rate / 1000:.1f}K{self.unit}/s"
        elif rate >= 1:
            return f"{rate:.1f}{self.unit}/s"
        else:
            return f"{rate:.2f}{self.unit}/s"

    def _format_num(self, num):
        """Format number with optional unit scaling."""
        if not self.unit_scale or self.unit not in ("B", "bytes"):
            return str(num)

        for unit in ["", "K", "M", "G", "T"]:
            if abs(num) < self.unit_divisor:
                return f"{num:3.1f}{unit}" if unit else f"{num:.0f}"
            num /= self.unit_divisor
        return f"{num:.1f}P"

    def _format_time(self, seconds):
        """Format time duration."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}:{seconds % 60:02.0f}"
        else:
            h, m = int(seconds // 3600), int((seconds % 3600) // 60)
            return f"{h}:{m:02d}:{seconds % 60:02.0f}"

    def _generate_bar(self, width=10):
        """Generate progress bar."""
        if self.total is None or self.total == 0:
            return "━" * width

        frac = min(1.0, self.n / self.total)
        filled = int(frac * width)

        bar = "━" * filled + "─" * (width - filled)

        # Add partial character for smoother progress
        if filled < width and frac * width - filled > 0.5:
            bar = bar[:filled] + "╸" + bar[filled + 1 :]

        return bar

    def _should_update(self, dt, dn):
        """Check if display should update."""
        if self.n >= (self.total or float("inf")):
            return True

        # GitHub Actions: suppress initial display until mininterval
        if is_github_action_running() and self.n == self.initial and dt < self.mininterval:
            return False

        return dt >= self.mininterval and dn >= 1

    def _display(self):
        """Display progress bar."""
        if self.disable or self.closed:
            return

        current_time = time.time()
        dt = current_time - self.last_print_t
        dn = self.n - self.last_print_n

        if not self._should_update(dt, dn):
            return

        # Calculate rate (avoid crazy numbers)
        if dt > 0.01:  # Only calculate rate if enough time has passed
            rate = dn / dt
            # Cap the rate to reasonable values and smooth it
            if rate < 1000000:  # Less than 1M items/s
                self.last_rate = 0.3 * rate + 0.7 * self.last_rate
                rate = self.last_rate
            else:
                # For very high rates, use the calculated rate but don't smooth
                pass  # rate = rate (no change needed)
        else:
            rate = self.last_rate

        # At completion, calculate the overall rate if we have valid data
        if self.n >= (self.total or float("inf")) and self.total and self.total > 0:
            overall_elapsed = current_time - self.start_t
            if overall_elapsed > 0:
                overall_rate = self.n / overall_elapsed
                # Use overall rate for final display if it seems reasonable
                if overall_rate > 0:
                    rate = overall_rate

        # Update counters
        self.last_print_n = self.n
        self.last_print_t = current_time
        elapsed = current_time - self.start_t

        # Build progress components
        if self.total is not None:
            percentage = (self.n / self.total) * 100
            n_fmt = self._format_num(self.n)
            total_fmt = self._format_num(self.total)
            bar = self._generate_bar()
        else:
            percentage = 0
            n_fmt = self._format_num(self.n)
            total_fmt = "?"
            bar = ""

        elapsed_str = self._format_time(elapsed)
        rate_fmt = self._format_rate(rate)

        # Safeguard: if rate_fmt is empty at completion, use overall rate
        if not rate_fmt and self.n >= (self.total or float("inf")) and elapsed > 0:
            overall_rate = self.n / elapsed
            rate_fmt = self._format_rate(overall_rate)

        # Format progress string with rate
        progress_str = self.bar_format.format(
            desc=self.desc,
            percentage=percentage,
            bar=bar,
            n_fmt=n_fmt,
            total_fmt=total_fmt,
            rate_fmt=rate_fmt,
            elapsed=elapsed_str,
        )

        # Write to output
        try:
            if is_github_action_running():
                # GitHub Actions doesn't handle ANSI escape sequences well
                # Just use carriage return without clearing
                self.file.write(f"\r{progress_str}")
            else:
                # Clear line to avoid leftover characters, then write progress
                self.file.write(f"\r\033[K{progress_str}")
            self.file.flush()
        except Exception:
            pass

    def update(self, n=1):
        """Update progress by n steps."""
        if not self.disable and not self.closed:
            self.n += n
            self._display()

    def set_description(self, desc):
        """Set description."""
        self.desc = desc or ""
        if not self.disable:
            self._display()

    def set_postfix(self, **kwargs):
        """Set postfix (appends to description)."""
        if kwargs:
            postfix = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            base_desc = self.desc.split(" | ")[0] if " | " in self.desc else self.desc
            self.set_description(f"{base_desc} | {postfix}")

    def close(self):
        """Close progress bar."""
        if self.closed:
            return

        if not self.disable:
            # Final display
            if self.total and self.n >= self.total:
                self.n = self.total
            self._display()

            # Cleanup
            end_char = "\n" if self.leave else "\r" + " " * 100 + "\r"
            try:
                self.file.write(end_char)
                self.file.flush()
            except Exception:
                pass

        self.closed = True

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, *args):
        """Exit context manager and close progress bar."""
        self.close()

    def __iter__(self):
        """Iterate over the wrapped iterable with progress updates."""
        if self.iterable is None:
            raise TypeError("'NoneType' object is not iterable")

        try:
            for item in self.iterable:
                yield item
                self.update(1)
        finally:
            self.close()

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except Exception:
            pass

    def refresh(self):
        """Refresh display."""
        if not self.disable:
            self._display()

    def clear(self):
        """Clear progress bar."""
        if not self.disable:
            try:
                self.file.write("\r" + " " * 100 + "\r")
                self.file.flush()
            except Exception:
                pass

    @staticmethod
    def write(s, file=None, end="\n"):
        """Static method to write without breaking progress bar."""
        file = file or sys.stdout
        try:
            file.write(s + end)
            file.flush()
        except Exception:
            pass
