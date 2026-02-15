---
description: Lightweight zero-dependency progress bar for Ultralytics with rich-style displays, GitHub Actions support, and comprehensive formatting options.
keywords: TQDM, progress bar, Ultralytics, GitHub Actions, zero dependencies, rich style, download progress, training progress, Unicode blocks
---

# Reference for `ultralytics/utils/tqdm.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tqdm.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tqdm.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`TQDM`](#ultralytics.utils.tqdm.TQDM)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`TQDM._format_rate`](#ultralytics.utils.tqdm.TQDM._format_rate)
        - [`TQDM._format_num`](#ultralytics.utils.tqdm.TQDM._format_num)
        - [`TQDM._format_time`](#ultralytics.utils.tqdm.TQDM._format_time)
        - [`TQDM._generate_bar`](#ultralytics.utils.tqdm.TQDM._generate_bar)
        - [`TQDM._should_update`](#ultralytics.utils.tqdm.TQDM._should_update)
        - [`TQDM._display`](#ultralytics.utils.tqdm.TQDM._display)
        - [`TQDM.update`](#ultralytics.utils.tqdm.TQDM.update)
        - [`TQDM.set_description`](#ultralytics.utils.tqdm.TQDM.set_description)
        - [`TQDM.set_postfix`](#ultralytics.utils.tqdm.TQDM.set_postfix)
        - [`TQDM.close`](#ultralytics.utils.tqdm.TQDM.close)
        - [`TQDM.__enter__`](#ultralytics.utils.tqdm.TQDM.__enter__)
        - [`TQDM.__exit__`](#ultralytics.utils.tqdm.TQDM.__exit__)
        - [`TQDM.__iter__`](#ultralytics.utils.tqdm.TQDM.__iter__)
        - [`TQDM.__del__`](#ultralytics.utils.tqdm.TQDM.__del__)
        - [`TQDM.refresh`](#ultralytics.utils.tqdm.TQDM.refresh)
        - [`TQDM.clear`](#ultralytics.utils.tqdm.TQDM.clear)
        - [`TQDM.write`](#ultralytics.utils.tqdm.TQDM.write)

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`is_noninteractive_console`](#ultralytics.utils.tqdm.is_noninteractive_console)


## Class `ultralytics.utils.tqdm.TQDM` {#ultralytics.utils.tqdm.TQDM}

```python
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
) -> None
```

Lightweight zero-dependency progress bar for Ultralytics.

Provides clean, rich-style progress bars suitable for various environments including Weights & Biases, console outputs, and other logging systems. Features zero external dependencies, clean single-line output, rich-style progress bars with Unicode block characters, context manager support, iterator protocol support, and dynamic description updates.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `iterable` | `Any, optional` | Iterable to wrap with progress bar. | `None` |
| `desc` | `str, optional` | Prefix description for the progress bar. | `None` |
| `total` | `int, optional` | Expected number of iterations. | `None` |
| `leave` | `bool, optional` | Whether to leave the progress bar after completion. | `True` |
| `file` | `IO[str], optional` | Output file stream for progress display. | `None` |
| `mininterval` | `float, optional` | Minimum time interval between updates (default 0.1s, 60s in GitHub Actions). | `0.1` |
| `disable` | `bool, optional` | Whether to disable the progress bar. Auto-detected if None. | `None` |
| `unit` | `str, optional` | String for units of iteration (default "it" for items). | `"it"` |
| `unit_scale` | `bool, optional` | Auto-scale units for bytes/data units. | `True` |
| `unit_divisor` | `int, optional` | Divisor for unit scaling (default 1000). | `1000` |
| `bar_format` | `str, optional` | Custom bar format string. | `None` |
| `initial` | `int, optional` | Initial counter value. | `0` |
| `**kwargs` | `Any` | Additional keyword arguments for compatibility (ignored). | *required* |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `iterable` | `Any` | Iterable to wrap with progress bar. |
| `desc` | `str` | Prefix description for the progress bar. |
| `total` | `int | None` | Expected number of iterations. |
| `disable` | `bool` | Whether to disable the progress bar. |
| `unit` | `str` | String for units of iteration. |
| `unit_scale` | `bool` | Auto-scale units flag. |
| `unit_divisor` | `int` | Divisor for unit scaling. |
| `leave` | `bool` | Whether to leave the progress bar after completion. |
| `mininterval` | `float` | Minimum time interval between updates. |
| `initial` | `int` | Initial counter value. |
| `n` | `int` | Current iteration count. |
| `closed` | `bool` | Whether the progress bar is closed. |
| `bar_format` | `str | None` | Custom bar format string. |
| `file` | `IO[str]` | Output file stream. |

**Methods**

| Name | Description |
| --- | --- |
| [`__del__`](#ultralytics.utils.tqdm.TQDM.__del__) | Destructor to ensure cleanup. |
| [`__enter__`](#ultralytics.utils.tqdm.TQDM.__enter__) | Enter context manager. |
| [`__exit__`](#ultralytics.utils.tqdm.TQDM.__exit__) | Exit context manager and close progress bar. |
| [`__iter__`](#ultralytics.utils.tqdm.TQDM.__iter__) | Iterate over the wrapped iterable with progress updates. |
| [`_display`](#ultralytics.utils.tqdm.TQDM._display) | Display progress bar. |
| [`_format_num`](#ultralytics.utils.tqdm.TQDM._format_num) | Format number with optional unit scaling. |
| [`_format_rate`](#ultralytics.utils.tqdm.TQDM._format_rate) | Format rate with units, switching between it/s and s/it for readability. |
| [`_format_time`](#ultralytics.utils.tqdm.TQDM._format_time) | Format time duration. |
| [`_generate_bar`](#ultralytics.utils.tqdm.TQDM._generate_bar) | Generate progress bar. |
| [`_should_update`](#ultralytics.utils.tqdm.TQDM._should_update) | Check if display should update. |
| [`clear`](#ultralytics.utils.tqdm.TQDM.clear) | Clear progress bar. |
| [`close`](#ultralytics.utils.tqdm.TQDM.close) | Close progress bar. |
| [`refresh`](#ultralytics.utils.tqdm.TQDM.refresh) | Refresh display. |
| [`set_description`](#ultralytics.utils.tqdm.TQDM.set_description) | Set description. |
| [`set_postfix`](#ultralytics.utils.tqdm.TQDM.set_postfix) | Set postfix (appends to description). |
| [`update`](#ultralytics.utils.tqdm.TQDM.update) | Update progress by n steps. |
| [`write`](#ultralytics.utils.tqdm.TQDM.write) | Static method to write without breaking progress bar. |

**Examples**

```python
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
```

<details>
<summary>Source code in <code>ultralytics/utils/tqdm.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tqdm.py#L18-L385"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class TQDM:
    """Lightweight zero-dependency progress bar for Ultralytics.

    Provides clean, rich-style progress bars suitable for various environments including Weights & Biases, console
    outputs, and other logging systems. Features zero external dependencies, clean single-line output, rich-style
    progress bars with Unicode block characters, context manager support, iterator protocol support, and dynamic
    description updates.

    Attributes:
        iterable (Any): Iterable to wrap with progress bar.
        desc (str): Prefix description for the progress bar.
        total (int | None): Expected number of iterations.
        disable (bool): Whether to disable the progress bar.
        unit (str): String for units of iteration.
        unit_scale (bool): Auto-scale units flag.
        unit_divisor (int): Divisor for unit scaling.
        leave (bool): Whether to leave the progress bar after completion.
        mininterval (float): Minimum time interval between updates.
        initial (int): Initial counter value.
        n (int): Current iteration count.
        closed (bool): Whether the progress bar is closed.
        bar_format (str | None): Custom bar format string.
        file (IO[str]): Output file stream.

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
        """Initialize the TQDM progress bar with specified configuration options.

        Args:
            iterable (Any, optional): Iterable to wrap with progress bar.
            desc (str, optional): Prefix description for the progress bar.
            total (int, optional): Expected number of iterations.
            leave (bool, optional): Whether to leave the progress bar after completion.
            file (IO[str], optional): Output file stream for progress display.
            mininterval (float, optional): Minimum time interval between updates (default 0.1s, 60s in GitHub Actions).
            disable (bool, optional): Whether to disable the progress bar. Auto-detected if None.
            unit (str, optional): String for units of iteration (default "it" for items).
            unit_scale (bool, optional): Auto-scale units for bytes/data units.
            unit_divisor (int, optional): Divisor for unit scaling (default 1000).
            bar_format (str, optional): Custom bar format string.
            initial (int, optional): Initial counter value.
            **kwargs (Any): Additional keyword arguments for compatibility (ignored).
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
        self.is_bytes = unit_scale and unit in {"B", "bytes"}
        self.scales = (
            [(1073741824, "GB/s"), (1048576, "MB/s"), (1024, "KB/s")]
            if self.is_bytes
            else [(1e9, f"G{self.unit}/s"), (1e6, f"M{self.unit}/s"), (1e3, f"K{self.unit}/s")]
        )

        if not self.disable and self.total and not self.noninteractive:
            self._display()
```
</details>

<br>

### Method `ultralytics.utils.tqdm.TQDM.__del__` {#ultralytics.utils.tqdm.TQDM.\_\_del\_\_}

```python
def __del__(self) -> None
```

Destructor to ensure cleanup.

<details>
<summary>Source code in <code>ultralytics/utils/tqdm.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tqdm.py#L356-L361"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __del__(self) -> None:
    """Destructor to ensure cleanup."""
    try:
        self.close()
    except Exception:
        pass
```
</details>

<br>

### Method `ultralytics.utils.tqdm.TQDM.__enter__` {#ultralytics.utils.tqdm.TQDM.\_\_enter\_\_}

```python
def __enter__(self) -> TQDM
```

Enter context manager.

<details>
<summary>Source code in <code>ultralytics/utils/tqdm.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tqdm.py#L336-L338"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __enter__(self) -> TQDM:
    """Enter context manager."""
    return self
```
</details>

<br>

### Method `ultralytics.utils.tqdm.TQDM.__exit__` {#ultralytics.utils.tqdm.TQDM.\_\_exit\_\_}

```python
def __exit__(self, *args: Any) -> None
```

Exit context manager and close progress bar.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `*args` | `Any` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/tqdm.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tqdm.py#L340-L342"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __exit__(self, *args: Any) -> None:
    """Exit context manager and close progress bar."""
    self.close()
```
</details>

<br>

### Method `ultralytics.utils.tqdm.TQDM.__iter__` {#ultralytics.utils.tqdm.TQDM.\_\_iter\_\_}

```python
def __iter__(self) -> Any
```

Iterate over the wrapped iterable with progress updates.

<details>
<summary>Source code in <code>ultralytics/utils/tqdm.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tqdm.py#L344-L354"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.utils.tqdm.TQDM._display` {#ultralytics.utils.tqdm.TQDM.\_display}

```python
def _display(self, final: bool = False) -> None
```

Display progress bar.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `final` | `bool` |  | `False` |

<details>
<summary>Source code in <code>ultralytics/utils/tqdm.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tqdm.py#L211-L288"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
        if self.is_bytes and n_str[-2] == t_str[-2]:  # Collapse suffix only when identical (e.g. "5.4/5.4MB")
            n_str = n_str.rstrip("KMGTPB")
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
```
</details>

<br>

### Method `ultralytics.utils.tqdm.TQDM._format_num` {#ultralytics.utils.tqdm.TQDM.\_format\_num}

```python
def _format_num(self, num: int | float) -> str
```

Format number with optional unit scaling.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `num` | `int | float` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/tqdm.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tqdm.py#L171-L180"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _format_num(self, num: int | float) -> str:
    """Format number with optional unit scaling."""
    if not self.unit_scale or not self.is_bytes:
        return str(num)

    for unit in ("", "K", "M", "G", "T"):
        if abs(num) < self.unit_divisor:
            return f"{num:3.1f}{unit}B" if unit else f"{num:.0f}B"
        num /= self.unit_divisor
    return f"{num:.1f}PB"
```
</details>

<br>

### Method `ultralytics.utils.tqdm.TQDM._format_rate` {#ultralytics.utils.tqdm.TQDM.\_format\_rate}

```python
def _format_rate(self, rate: float) -> str
```

Format rate with units, switching between it/s and s/it for readability.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `rate` | `float` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/tqdm.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tqdm.py#L156-L169"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _format_rate(self, rate: float) -> str:
    """Format rate with units, switching between it/s and s/it for readability."""
    if rate <= 0:
        return ""

    inv_rate = 1 / rate if rate else None

    # Use s/it format when inv_rate > 1 (i.e., rate < 1 it/s) for better readability
    if inv_rate and inv_rate > 1:
        return f"{inv_rate:.1f}s/B" if self.is_bytes else f"{inv_rate:.1f}s/{self.unit}"

    # Use it/s format for fast iterations
    fallback = f"{rate:.1f}B/s" if self.is_bytes else f"{rate:.1f}{self.unit}/s"
    return next((f"{rate / t:.1f}{u}" for t, u in self.scales if rate >= t), fallback)
```
</details>

<br>

### Method `ultralytics.utils.tqdm.TQDM._format_time` {#ultralytics.utils.tqdm.TQDM.\_format\_time}

```python
def _format_time(seconds: float) -> str
```

Format time duration.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `seconds` | `float` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/tqdm.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tqdm.py#L183-L191"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def _format_time(seconds: float) -> str:
    """Format time duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}:{seconds % 60:02.0f}"
    else:
        h, m = int(seconds // 3600), int((seconds % 3600) // 60)
        return f"{h}:{m:02d}:{seconds % 60:02.0f}"
```
</details>

<br>

### Method `ultralytics.utils.tqdm.TQDM._generate_bar` {#ultralytics.utils.tqdm.TQDM.\_generate\_bar}

```python
def _generate_bar(self, width: int = 12) -> str
```

Generate progress bar.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `width` | `int` |  | `12` |

<details>
<summary>Source code in <code>ultralytics/utils/tqdm.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tqdm.py#L193-L203"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.utils.tqdm.TQDM._should_update` {#ultralytics.utils.tqdm.TQDM.\_should\_update}

```python
def _should_update(self, dt: float, dn: int) -> bool
```

Check if display should update.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dt` | `float` |  | *required* |
| `dn` | `int` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/tqdm.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tqdm.py#L205-L209"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _should_update(self, dt: float, dn: int) -> bool:
    """Check if display should update."""
    if self.noninteractive:
        return False
    return (self.total is not None and self.n >= self.total) or (dt >= self.mininterval)
```
</details>

<br>

### Method `ultralytics.utils.tqdm.TQDM.clear` {#ultralytics.utils.tqdm.TQDM.clear}

```python
def clear(self) -> None
```

Clear progress bar.

<details>
<summary>Source code in <code>ultralytics/utils/tqdm.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tqdm.py#L368-L375"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def clear(self) -> None:
    """Clear progress bar."""
    if not self.disable:
        try:
            self.file.write("\r\033[K")
            self.file.flush()
        except Exception:
            pass
```
</details>

<br>

### Method `ultralytics.utils.tqdm.TQDM.close` {#ultralytics.utils.tqdm.TQDM.close}

```python
def close(self) -> None
```

Close progress bar.

<details>
<summary>Source code in <code>ultralytics/utils/tqdm.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tqdm.py#L309-L334"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def close(self) -> None:
    """Close progress bar."""
    if self.closed:
        return

    self.closed = True

    if not self.disable:
        # Final display
        if self.total and self.n >= self.total:
            self.n = self.total
            if self.n != self.last_print_n:  # Skip if 100% already shown
                self._display(final=True)
        else:
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
```
</details>

<br>

### Method `ultralytics.utils.tqdm.TQDM.refresh` {#ultralytics.utils.tqdm.TQDM.refresh}

```python
def refresh(self) -> None
```

Refresh display.

<details>
<summary>Source code in <code>ultralytics/utils/tqdm.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tqdm.py#L363-L366"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def refresh(self) -> None:
    """Refresh display."""
    if not self.disable:
        self._display()
```
</details>

<br>

### Method `ultralytics.utils.tqdm.TQDM.set_description` {#ultralytics.utils.tqdm.TQDM.set\_description}

```python
def set_description(self, desc: str | None) -> None
```

Set description.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `desc` | `str | None` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/tqdm.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tqdm.py#L296-L300"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def set_description(self, desc: str | None) -> None:
    """Set description."""
    self.desc = desc or ""
    if not self.disable:
        self._display()
```
</details>

<br>

### Method `ultralytics.utils.tqdm.TQDM.set_postfix` {#ultralytics.utils.tqdm.TQDM.set\_postfix}

```python
def set_postfix(self, **kwargs: Any) -> None
```

Set postfix (appends to description).

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `**kwargs` | `Any` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/tqdm.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tqdm.py#L302-L307"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def set_postfix(self, **kwargs: Any) -> None:
    """Set postfix (appends to description)."""
    if kwargs:
        postfix = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        base_desc = self.desc.split(" | ")[0] if " | " in self.desc else self.desc
        self.set_description(f"{base_desc} | {postfix}")
```
</details>

<br>

### Method `ultralytics.utils.tqdm.TQDM.update` {#ultralytics.utils.tqdm.TQDM.update}

```python
def update(self, n: int = 1) -> None
```

Update progress by n steps.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `n` | `int` |  | `1` |

<details>
<summary>Source code in <code>ultralytics/utils/tqdm.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tqdm.py#L290-L294"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def update(self, n: int = 1) -> None:
    """Update progress by n steps."""
    if not self.disable and not self.closed:
        self.n += n
        self._display()
```
</details>

<br>

### Method `ultralytics.utils.tqdm.TQDM.write` {#ultralytics.utils.tqdm.TQDM.write}

```python
def write(s: str, file: IO[str] | None = None, end: str = "\n") -> None
```

Static method to write without breaking progress bar.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `s` | `str` |  | *required* |
| `file` | `IO[str] | None` |  | `None` |
| `end` | `str` |  | `"\n"` |

<details>
<summary>Source code in <code>ultralytics/utils/tqdm.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tqdm.py#L378-L385"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def write(s: str, file: IO[str] | None = None, end: str = "\n") -> None:
    """Static method to write without breaking progress bar."""
    file = file or sys.stdout
    try:
        file.write(s + end)
        file.flush()
    except Exception:
        pass
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.tqdm.is_noninteractive_console` {#ultralytics.utils.tqdm.is\_noninteractive\_console}

```python
def is_noninteractive_console() -> bool
```

Check for known non-interactive console environments.

<details>
<summary>Source code in <code>ultralytics/utils/tqdm.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tqdm.py#L13-L15"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@lru_cache(maxsize=1)
def is_noninteractive_console() -> bool:
    """Check for known non-interactive console environments."""
    return "GITHUB_ACTIONS" in os.environ or "RUNPOD_POD_ID" in os.environ
```
</details>

<br><br>
