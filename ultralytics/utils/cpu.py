# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import gc
import platform
import re
import subprocess
import sys
import time
from pathlib import Path


class CPUInfo:
    """Return a concise, human-readable CPU name (uses sysctl on macOS)."""

    @staticmethod
    def name() -> str:
        try:
            if sys.platform == "darwin":
                s = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
                ).stdout.strip()
                if s:
                    return CPUInfo._clean(s)
            elif sys.platform.startswith("linux"):
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
                    pass
            s = platform.processor() or getattr(platform.uname(), "processor", "") or platform.machine()
            return CPUInfo._clean(s or "Unknown CPU")
        except Exception:
            s = platform.processor() or platform.machine() or ""
            return CPUInfo._clean(s or "Unknown CPU")

    @staticmethod
    def _clean(s: str) -> str:
        s = re.sub(r"\s+", " ", s.strip())
        s = s.replace("(TM)", "™").replace("(tm)", "™").replace("(R)", "®").replace("(r)", "®")
        m = re.search(r"(Intel.*?i\d[\w-]*) CPU @ ([\d.]+GHz)", s, re.I)
        if m:
            return f"{m.group(1)} @ {m.group(2)}"
        m = re.search(r"(AMD.*?Ryzen.*?[\w-]*) CPU @ ([\d.]+GHz)", s, re.I)
        if m:
            return f"{m.group(1)} @ {m.group(2)}"
        return s

    def __str__(self) -> str:
        return self.name()


if __name__ == "__main__":
    print(CPUInfo.name())
