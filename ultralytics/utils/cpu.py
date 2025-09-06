from __future__ import annotations
from pathlib import Path
import platform, re, sys, subprocess, time, gc

class CPUInfo:
    """Return a concise, human-readable CPU name (uses sysctl on macOS)."""

    @staticmethod
    def name() -> str:
        try:
            if sys.platform == "darwin":
                s = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
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

    @staticmethod
    def benchmark(iters: int = 200) -> dict:
        """Compare call-time vs py-cpuinfo; returns {'cpuinfo':s,'ours':s,'py_s':f,'our_s':f,'speedup':x}."""

        def _time(fn, n: int) -> float:
            gc_old = gc.isenabled(); gc.disable()
            try:
                t0 = time.perf_counter()
                for _ in range(n):
                    fn()
                return time.perf_counter() - t0
            finally:
                if gc_old: gc.enable()

        # warmups (avoid one-time costs skewing)
        _ = CPUInfo.name()
        try:
            import cpuinfo as _ci
            def _py(): return _ci.get_cpu_info().get("brand_raw", "")
            _ = _py()
        except Exception:
            _ci = None
            def _py(): return ""

        our_name = CPUInfo.name()
        py_name  = (_ci.get_cpu_info().get("brand_raw", "") if _ci else "")

        our_time = _time(CPUInfo.name, iters)
        py_time  = _time((_ci.get_cpu_info if _ci else CPUInfo.name), iters)  # if missing, compare to ourselves

        # If py-cpuinfo exists, measure brand_raw extraction cost (cached dict access is negligible)
        if _ci:
            def _py_brand(): return _ci.get_cpu_info().get("brand_raw", "")
            py_time = _time(_py_brand, iters)

        speedup = (py_time / our_time) if our_time > 0 else float("inf")
        return {"cpuinfo": our_name, "py_cpuinfo": py_name, "our_s": our_time, "py_s": py_time, "speedup": speedup}

    def __str__(self) -> str:
        return self.name()


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="CPU name and optional benchmark")
    ap.add_argument("--bench", action="store_true", help="benchmark vs py-cpuinfo if available")
    ap.add_argument("-n", "--iters", type=int, default=200, help="iterations for benchmark")
    args = ap.parse_args()

    if True: #args.bench:
        print(json.dumps(CPUInfo.benchmark(args.iters), indent=2))
    else:
        print(CPUInfo.name())
