# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import LOGGER

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
            import pynvml

            self.pynvml = pynvml
            self.pynvml.nvmlInit()
            self.nvml_available = True
            self.refresh_stats()
        except ImportError:
            LOGGER.warning("nvidia-ml-py (pynvml) not found. GPU stats features will be disabled.")
        except pynvml.NVMLError as error:
            LOGGER.warning(f"Failed to initialize NVML: {error}. GPU stats features will be disabled.")
            self.pynvml = None
        except Exception as e:
            LOGGER.warning(f"An unexpected error occurred during pynvml initialization: {e}")
            self.pynvml = None

    def __del__(self):
        """Ensures NVML is shut down when the object is garbage collected."""
        self.shutdown()

    def shutdown(self):
        """Shuts down NVML if it was initialized."""
        if self.nvml_available and self.pynvml:
            try:
                self.pynvml.nvmlShutdown()
            except self.pynvml.NVMLError:
                pass
            self.nvml_available = False

    def refresh_stats(self):
        """Refreshes the internal gpu_stats list by querying NVML."""
        if not self.nvml_available or not self.pynvml:
            self.gpu_stats = []
            return

        self.gpu_stats = []
        try:
            device_count = self.pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                memory = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)

                def safe_get(func, *args, default=-1, divisor=1):
                    try:
                        val = getattr(self.pynvml, func.__name__)(*args)
                        if divisor != 1 and isinstance(val, (int, float)):
                            return val // divisor
                        return val
                    except (self.pynvml.NVMLError, AttributeError):
                        return default

                temp_type = getattr(self.pynvml, "NVML_TEMPERATURE_GPU", -1)  # More concise getattr

                self.gpu_stats.append(
                    {
                        "index": i,
                        "name": self.pynvml.nvmlDeviceGetName(handle),
                        "utilization": util.gpu if util else -1,
                        "memory_used": memory.used >> 20 if memory else -1,
                        "memory_total": memory.total >> 20 if memory else -1,
                        "memory_free": memory.free >> 20 if memory else -1,
                        "temperature": safe_get(self.pynvml.nvmlDeviceGetTemperature, handle, temp_type),
                        "power_draw": safe_get(self.pynvml.nvmlDeviceGetPowerUsage, handle, divisor=1000),
                        "power_limit": safe_get(
                            self.pynvml.nvmlDeviceGetEnforcedPowerLimit, handle, divisor=1000, default="N/A"
                        ),
                    }
                )
        except self.pynvml.NVMLError as error:
            LOGGER.warning(f"NVML error during device query: {error}")
            self.gpu_stats = []
        except Exception as e:
            LOGGER.warning(f"Unexpected error during device query: {e}")
            self.gpu_stats = []

    def print_status(self):
        """Prints GPU status in a compact table format using current stats."""
        self.refresh_stats()
        if not self.gpu_stats:
            LOGGER.warning("\nNo GPU stats available.")
            return

        stats = self.gpu_stats
        max_name_len = max(len(gpu.get("name", "N/A")) for gpu in stats) if stats else 10
        hdr = f"{'Idx':<3} {'Name':<{max_name_len}} {'Util':>6} {'Mem (MiB)':>15} {'Temp':>5} {'Pwr (W)':>10}"
        LOGGER.info(f"\n--- GPU Status ---\n{hdr}\n{'-' * len(hdr)}")
        for gpu in stats:
            u = f"{gpu.get('utilization', -1):>5}%" if gpu.get("utilization", -1) != -1 else " N/A "
            m = (
                f"{gpu.get('memory_used', -1):>6}/{gpu.get('memory_total', -1):<6}"
                if gpu.get("memory_used", -1) != -1
                else " N/A / N/A "
            )
            t = f"{gpu.get('temperature', -1)}C" if gpu.get("temperature", -1) != -1 else " N/A "
            p = (
                f"{gpu.get('power_draw', -1):>3}/{str(gpu.get('power_limit', 'N/A')):<3}"
                if gpu.get("power_draw", -1) != -1
                else " N/A "
            )
            LOGGER.info(f"{gpu.get('index', '?'):<3d} {gpu.get('name', 'N/A'):<{max_name_len}} {u:>6} {m:>15} {t:>5} {p:>10}")
        LOGGER.info(f"{'-' * len(hdr)}\n")

    def select_idle_gpu(self, count=1, min_memory_mb=0):
        """
        Selects the 'count' most idle GPUs based on utilization and free memory.

        Args:
            count (int): The number of idle GPUs to select. Defaults to 1.
            min_memory_mb (int): Minimum free memory required (MiB). Defaults to 0.

        Returns:
            (list[int]): Indices of the selected GPUs, sorted by idleness.
                         Returns fewer than 'count' if not enough qualify or exist.
                         Returns basic CUDA indices if NVML fails. Empty list if no GPUs found.
        """
        LOGGER.info(f"Searching for {count} idle GPUs with >= {min_memory_mb} MiB free memory...")

        if count <= 0:
            return []
        self.refresh_stats()

        # Fallback if NVML failed or no stats
        if not self.nvml_available or not self.gpu_stats:
            LOGGER.warning("NVML stats unavailable.")
            return []

        # --- NVML Selection Logic ---
        # Filter GPUs meeting memory requirement and having valid utilization
        eligible_gpus = [
            gpu
            for gpu in self.gpu_stats
            if gpu.get("memory_free", -1) >= min_memory_mb and gpu.get("utilization", -1) != -1
        ]

        # Sort by utilization (asc), then free memory (desc)
        eligible_gpus.sort(key=lambda x: (x.get("utilization", 101), -x.get("memory_free", 0)))

        # Select the top 'count' indices
        selected = [gpu["index"] for gpu in eligible_gpus[:count]]

        if selected:
            LOGGER.info(f"Selected GPU indices based on idleness: {selected} (requested {count})")
        elif self.gpu_stats:
            LOGGER.warning(f"No GPUs met the criteria (Util != -1, Free Mem >= {min_memory_mb} MiB).")
        else:
            LOGGER.warning("No GPUs detected by NVML.")

        return selected


if __name__ == "__main__":
    required_free_mem = 2048  # Require 2GB free VRAM
    num_gpus_to_select = 1  # <<< Number of GPUs to select

    gpu_info = GPUInfo()
    gpu_info.print_status()

    selected_indices = gpu_info.select_idle_gpu(count=num_gpus_to_select, min_memory_mb=required_free_mem)

    if selected_indices:
        print(f"\n==> Using selected GPU indices: {selected_indices}")
        devices = [f"cuda:{idx}" for idx in selected_indices]
        print(f"    Target devices: {devices}")
    else:
        print(f"\n==> Failed to select {num_gpus_to_select} suitable GPUs.")

