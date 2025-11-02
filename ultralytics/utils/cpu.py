# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import platform
import re
import subprocess
import sys
from pathlib import Path


class CPUInfo:
    """Provide cross-platform CPU brand and model information.

    Query platform-specific sources to retrieve a human-readable CPU descriptor and normalize it for consistent
    presentation across macOS, Linux, and Windows. If platform-specific probing fails, generic platform identifiers are
    used to ensure a stable string is always returned.

    Methods:
        name: Return the normalized CPU name using platform-specific sources with robust fallbacks.
        _clean: Normalize and prettify common vendor brand strings and frequency patterns.
        __str__: Return the normalized CPU name for string contexts.

    Examples:
        >>> CPUInfo.name()
        'Apple M4 Pro'
        >>> str(CPUInfo())
        'Intel Core i7-9750H 2.60GHz'
    """

    @staticmethod
    def name() -> str:
        """Return a normalized CPU model string from platform-specific sources."""
        try:
            if sys.platform == "darwin":
                # Query macOS sysctl for the CPU brand string
                s = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
                ).stdout.strip()
                if s:
                    return CPUInfo._clean(s)
            elif sys.platform.startswith("linux"):
                # Parse /proc/cpuinfo for the first "model name" entry
                p = Path("/proc/cpuinfo")
                if p.exists():
                    for line in p.read_text(errors="ignore").splitlines():
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

    @staticmethod
    def _clean(s: str) -> str:
        """Normalize and prettify a raw CPU descriptor string."""
        s = re.sub(r"\s+", " ", s.strip())
        s = s.replace("(TM)", "").replace("(tm)", "").replace("(R)", "").replace("(r)", "").strip()
        # Normalize common Intel pattern to 'Model Freq'
        m = re.search(r"(Intel.*?i\d[\w-]*) CPU @ ([\d.]+GHz)", s, re.I)
        if m:
            return f"{m.group(1)} {m.group(2)}"
        # Normalize common AMD Ryzen pattern to 'Model Freq'
        m = re.search(r"(AMD.*?Ryzen.*?[\w-]*) CPU @ ([\d.]+GHz)", s, re.I)
        if m:
            return f"{m.group(1)} {m.group(2)}"
        return s

    def __str__(self) -> str:
        """Return the normalized CPU name."""
        return self.name()


if __name__ == "__main__":
    print(CPUInfo.name())
