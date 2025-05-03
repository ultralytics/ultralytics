# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license


from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements


class GPUInfo:
    """
    Manages NVIDIA GPU information via pynvml with robust error handling.

    Provides methods to query detailed GPU statistics (utilization, memory, temp, power) and select the most idle
    GPUs based on configurable criteria. It safely handles the absence or initialization failure of the pynvml
    library by logging warnings and disabling related features, preventing application crashes.

    Includes fallback logic using `torch.cuda` for basic device counting if NVML is unavailable during GPU
    selection. Manages NVML initialization and shutdown internally.

    Attributes:
        pynvml (module | None): The `pynvml` module if successfully imported and initialized, otherwise `None`.
        nvml_available (bool): Indicates if `pynvml` is ready for use. True if import and `nvmlInit()` succeeded,
            False otherwise.
        gpu_stats (list[dict]): A list of dictionaries, each holding stats for one GPU. Populated on initialization
            and by `refresh_stats()`. Keys include: 'index', 'name', 'utilization' (%), 'memory_used' (MiB),
            'memory_total' (MiB), 'memory_free' (MiB), 'temperature' (C), 'power_draw' (W),
            'power_limit' (W or 'N/A'). Empty if NVML is unavailable or queries fail.
    """

    def __init__(self):
        """Initializes GPUInfo, attempting to import and initialize pynvml."""
        self.pynvml = None
        self.nvml_available = False
        self.gpu_stats = []

        try:
            check_requirements("pynvml>=12.0.0")
            self.pynvml = __import__("pynvml")
            self.pynvml.nvmlInit()
            self.nvml_available = True
            self.refresh_stats()
        except Exception as e:
            LOGGER.warning(f"Failed to initialize pynvml, GPU stats disabled: {e}")

    def __del__(self):
        """Ensures NVML is shut down when the object is garbage collected."""
        self.shutdown()

    def shutdown(self):
        """Shuts down NVML if it was initialized."""
        if self.nvml_available and self.pynvml:
            try:
                self.pynvml.nvmlShutdown()
            except Exception:
                pass
            self.nvml_available = False

    def refresh_stats(self):
        """Refreshes the internal gpu_stats list by querying NVML."""
        self.gpu_stats = []
        if not self.nvml_available or not self.pynvml:
            return

        try:
            device_count = self.pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                self.gpu_stats.append(self._get_device_stats(i))
        except Exception as e:
            LOGGER.warning(f"Error during device query: {e}")
            self.gpu_stats = []

    def _get_device_stats(self, index):
        """Gets stats for a single GPU device."""
        handle = self.pynvml.nvmlDeviceGetHandleByIndex(index)
        memory = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)

        def safe_get(func, *args, default=-1, divisor=1):
            try:
                val = func(*args)
                return val // divisor if divisor != 1 and isinstance(val, (int, float)) else val
            except Exception:
                return default

        temp_type = getattr(self.pynvml, "NVML_TEMPERATURE_GPU", -1)

        return {
            "index": index,
            "name": self.pynvml.nvmlDeviceGetName(handle),
            "utilization": util.gpu if util else -1,
            "memory_used": memory.used >> 20 if memory else -1,
            "memory_total": memory.total >> 20 if memory else -1,
            "memory_free": memory.free >> 20 if memory else -1,
            "temperature": safe_get(self.pynvml.nvmlDeviceGetTemperature, handle, temp_type),
            "power_draw": safe_get(self.pynvml.nvmlDeviceGetPowerUsage, handle, divisor=1000),
            "power_limit": safe_get(self.pynvml.nvmlDeviceGetEnforcedPowerLimit, handle, divisor=1000),
        }

    def print_status(self):
        """Prints GPU status in a compact table format using current stats."""
        self.refresh_stats()
        if not self.gpu_stats:
            LOGGER.warning("No GPU stats available.")
            return

        stats = self.gpu_stats
        name_len = max(len(gpu.get("name", "N/A")) for gpu in stats)
        hdr = f"{'Idx':<3} {'Name':<{name_len}} {'Util':>6} {'Mem (MiB)':>15} {'Temp':>5} {'Pwr (W)':>10}"
        LOGGER.info(f"\n--- GPU Status ---\n{hdr}\n{'-' * len(hdr)}")

        for gpu in stats:
            u = f"{gpu['utilization']:>5}%" if gpu["utilization"] >= 0 else " N/A "
            m = f"{gpu['memory_used']:>6}/{gpu['memory_total']:<6}" if gpu["memory_used"] >= 0 else " N/A / N/A "
            t = f"{gpu['temperature']}C" if gpu["temperature"] >= 0 else " N/A "
            p = f"{gpu['power_draw']:>3}/{gpu['power_limit']:<3}" if gpu["power_draw"] >= 0 else " N/A "

            LOGGER.info(f"{gpu.get('index'):<3d} {gpu.get('name', 'N/A'):<{name_len}} {u:>6} {m:>15} {t:>5} {p:>10}")

        LOGGER.info(f"{'-' * len(hdr)}\n")

    def select_idle_gpu(self, count=1, min_memory_mb=0):
        """
        Selects the 'count' most idle GPUs based on utilization and free memory.

        Args:
            count (int): The number of idle GPUs to select. Defaults to 1.
            min_memory_mb (int): Minimum free memory required (MiB). Defaults to 0.

        Returns:
            (list[int]): Indices of the selected GPUs, sorted by idleness.

        Notes:
             Returns fewer than 'count' if not enough qualify or exist.
             Returns basic CUDA indices if NVML fails. Empty list if no GPUs found.
        """
        LOGGER.info(f"Searching for {count} idle GPUs with >= {min_memory_mb} MiB free memory...")

        if count <= 0:
            return []

        self.refresh_stats()
        if not self.gpu_stats:
            LOGGER.warning("NVML stats unavailable.")
            return []

        # Filter and sort eligible GPUs
        eligible_gpus = [
            gpu
            for gpu in self.gpu_stats
            if gpu.get("memory_free", -1) >= min_memory_mb and gpu.get("utilization", -1) != -1
        ]
        eligible_gpus.sort(key=lambda x: (x.get("utilization", 101), -x.get("memory_free", 0)))

        # Select top 'count' indices
        selected = [gpu["index"] for gpu in eligible_gpus[:count]]

        if selected:
            LOGGER.info(f"Selected idle CUDA devices {selected}")
        else:
            LOGGER.warning(f"No GPUs met criteria (Util != -1, Free Mem >= {min_memory_mb} MiB).")

        return selected


if __name__ == "__main__":
    required_free_mem = 2048  # Require 2GB free VRAM
    num_gpus_to_select = 1

    gpu_info = GPUInfo()
    gpu_info.print_status()

    selected = gpu_info.select_idle_gpu(count=num_gpus_to_select, min_memory_mb=required_free_mem)
    if selected:
        print(f"\n==> Using selected GPU indices: {selected}")
        devices = [f"cuda:{idx}" for idx in selected]
        print(f"    Target devices: {devices}")
