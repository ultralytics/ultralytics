---
title: utils.cpu API Reference
description: Reference documentation for CPUInfo, a lightweight utility to get system CPU details in Ultralytics.
keywords: Ultralytics, CPUInfo, CPU, system info, hardware, utils
---

# Reference for `ultralytics/utils/cpu.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/cpu.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/cpu.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`CPUInfo`](#ultralytics.utils.cpu.CPUInfo)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`CPUInfo._sysctl`](#ultralytics.utils.cpu.CPUInfo._sysctl)
        - [`CPUInfo.name`](#ultralytics.utils.cpu.CPUInfo.name)
        - [`CPUInfo._clean`](#ultralytics.utils.cpu.CPUInfo._clean)
        - [`CPUInfo.__str__`](#ultralytics.utils.cpu.CPUInfo.__str__)


## Class `ultralytics.utils.cpu.CPUInfo` {#ultralytics.utils.cpu.CPUInfo}

```python
CPUInfo()
```

Provide cross-platform CPU brand and model information.

Query platform-specific sources to retrieve a human-readable CPU descriptor and normalize it for consistent presentation across macOS, Linux, and Windows. If platform-specific probing fails, generic platform identifiers are used to ensure a stable string is always returned.

**Methods**

| Name | Description |
| --- | --- |
| [`__str__`](#ultralytics.utils.cpu.CPUInfo.__str__) | Return the normalized CPU name. |
| [`_clean`](#ultralytics.utils.cpu.CPUInfo._clean) | Normalize and prettify a raw CPU descriptor string. |
| [`_sysctl`](#ultralytics.utils.cpu.CPUInfo._sysctl) | Read a macOS sysctl string through libc, since spawning `sysctl` costs milliseconds. |
| [`name`](#ultralytics.utils.cpu.CPUInfo.name) | Return a normalized CPU model string from platform-specific sources. |

**Examples**

```python
>>> name = CPUInfo.name()
>>> text = str(CPUInfo())
```

<details>
<summary>Source code in <code>ultralytics/utils/cpu.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/cpu.py#L12-L100">View on GitHub</a>
```python
class CPUInfo:
    """Provide cross-platform CPU brand and model information.

    Query platform-specific sources to retrieve a human-readable CPU descriptor and normalize it for consistent
    presentation across macOS, Linux, and Windows. If platform-specific probing fails, generic platform identifiers are
    used to ensure a stable string is always returned.

    Methods:
        name: Return the normalized CPU name using platform-specific sources with robust fallbacks.
        _sysctl: Read a macOS sysctl string through libc.
        _clean: Normalize and prettify common vendor brand strings and frequency patterns.
        __str__: Return the normalized CPU name for string contexts.

    Examples:
        >>> name = CPUInfo.name()
        >>> text = str(CPUInfo())
    """
```
</details>

<br>

### Method `ultralytics.utils.cpu.CPUInfo.__str__` {#ultralytics.utils.cpu.CPUInfo.\_\_str\_\_}

```python
def __str__(self) -> str
```

Return the normalized CPU name.

<details>
<summary>Source code in <code>ultralytics/utils/cpu.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/cpu.py#L98-L100">View on GitHub</a>
```python
def __str__(self) -> str:
    """Return the normalized CPU name."""
    return self.name()
```
</details>

<br>

### Method `ultralytics.utils.cpu.CPUInfo._clean` {#ultralytics.utils.cpu.CPUInfo.\_clean}

```python
def _clean(s: str) -> str
```

Normalize and prettify a raw CPU descriptor string.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `s` | `str` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/cpu.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/cpu.py#L88-L96">View on GitHub</a>
```python
@staticmethod
def _clean(s: str) -> str:
    """Normalize and prettify a raw CPU descriptor string."""
    s = re.sub(r"\s+", " ", s.strip())
    s = s.replace("(TM)", "").replace("(tm)", "").replace("(R)", "").replace("(r)", "").strip()
    if m := re.search(r"(Intel.*?i\d[\w-]*) CPU @ ([\d.]+GHz)", s, re.IGNORECASE):
        return f"{m.group(1)} {m.group(2)}"
    if m := re.search(r"(AMD.*?Ryzen.*?[\w-]*) CPU @ ([\d.]+GHz)", s, re.IGNORECASE):
        return f"{m.group(1)} {m.group(2)}"
    return s
```
</details>

<br>

### Method `ultralytics.utils.cpu.CPUInfo._sysctl` {#ultralytics.utils.cpu.CPUInfo.\_sysctl}

```python
def _sysctl(key: str) -> str
```

Read a macOS sysctl string through libc, since spawning `sysctl` costs milliseconds.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `key` | `str` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/cpu.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/cpu.py#L31-L48">View on GitHub</a>
```python
@staticmethod
def _sysctl(key: str) -> str:
    """Read a macOS sysctl string through libc, since spawning `sysctl` costs milliseconds."""
    libc = ctypes.CDLL(None)
    libc.sysctlbyname.restype = ctypes.c_int
    libc.sysctlbyname.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    name, n = key.encode(), ctypes.c_size_t(0)
    if libc.sysctlbyname(name, None, ctypes.byref(n), None, 0) or not n.value:  # size probe
        return ""
    buf = ctypes.create_string_buffer(n.value)
    if libc.sysctlbyname(name, buf, ctypes.byref(n), None, 0):
        return ""
    return buf.value.decode(errors="ignore").strip()
```
</details>

<br>

### Method `ultralytics.utils.cpu.CPUInfo.name` {#ultralytics.utils.cpu.CPUInfo.name}

```python
def name() -> str
```

Return a normalized CPU model string from platform-specific sources.

<details>
<summary>Source code in <code>ultralytics/utils/cpu.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/cpu.py#L51-L85">View on GitHub</a>
```python
@staticmethod
def name() -> str:
    """Return a normalized CPU model string from platform-specific sources."""
    try:
        if sys.platform == "darwin":
            # Query macOS sysctl for the CPU brand string
            s = CPUInfo._sysctl("machdep.cpu.brand_string")
            if s:
                return CPUInfo._clean(s)
        elif sys.platform.startswith("linux"):
            # Parse /proc/cpuinfo for the first "model name" entry, streaming since a many-core host
            # repeats a full block per logical CPU and only the first is needed
            p = Path("/proc/cpuinfo")
            if p.exists():
                with p.open(errors="ignore") as f:
                    for line in f:
                        if "model name" in line:
                            return CPUInfo._clean(line.split(":", 1)[1])
        elif sys.platform.startswith("win"):
            try:
                import winreg as wr

                with wr.OpenKey(wr.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                    val, _ = wr.QueryValueEx(k, "ProcessorNameString")
                    if val:
                        return CPUInfo._clean(val)
            except Exception:
                # Fall through to generic platform fallbacks on Windows registry access failure
                pass
        # Generic platform fallbacks
        s = platform.processor() or getattr(platform.uname(), "processor", "") or platform.machine()
        return CPUInfo._clean(s or "Unknown CPU")
    except Exception:
        # Ensure a string is always returned even on unexpected failures
        s = platform.processor() or platform.machine() or ""
        return CPUInfo._clean(s or "Unknown CPU")
```
</details>

<br><br>
