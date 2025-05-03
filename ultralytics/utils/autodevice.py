# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import sys

import torch

from ultralytics.utils import LOGGER


class GPUInfo:
    """
    Provides information about system GPUs using pynvml, selecting the most idle GPUs.

    Attributes:
        pynvml (module | None): The imported pynvml module, or None if unavailable.
        nvml_available (bool): True if pynvml was successfully imported and initialized.
        gpu_stats (list): A list of dictionaries, each containing stats for a GPU.
                             Refreshed by calling refresh_stats().
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
            print("\nNo GPU stats available.")
            try:
                if torch.cuda.is_available():
                    print(f"PyTorch reports {torch.cuda.device_count()} CUDA devices.")
                else:
                    print("PyTorch reports no CUDA devices.")
            except Exception:
                pass  # Ignore torch check errors silently
            return

        stats = self.gpu_stats
        max_name_len = max(len(gpu.get("name", "N/A")) for gpu in stats) if stats else 10
        hdr = f"{'Idx':<3} {'Name':<{max_name_len}} {'Util':>6} {'Mem (MiB)':>15} {'Temp':>5} {'Pwr (W)':>10}"
        print(f"\n--- GPU Status ---\n{hdr}\n{'-' * len(hdr)}")
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
            print(f"{gpu.get('index', '?'):<3d} {gpu.get('name', 'N/A'):<{max_name_len}} {u:>6} {m:>15} {t:>5} {p:>10}")
        print(f"{'-' * len(hdr)}\n")

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
        if count <= 0:
            return []
        self.refresh_stats()

        # Fallback if NVML failed or no stats
        if not self.nvml_available or not self.gpu_stats:
            LOGGER.warning("NVML stats unavailable. Falling back to basic CUDA device check.")
            try:
                if torch.cuda.is_available():
                    num_devs = torch.cuda.device_count()
                    return list(range(min(count, num_devs)))  # Return first 'count' available devices
            except Exception:
                pass  # Ignore torch check errors
            return []  # No NVML and no CUDA info

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
        selected_indices = [gpu["index"] for gpu in eligible_gpus[:count]]

        # Optional: Print selection summary (can be removed for absolute minimum)
        if selected_indices:
            print(f"Selected GPU indices based on idleness: {selected_indices} (requested {count})")
        elif self.gpu_stats:  # Only warn if GPUs existed but didn't meet criteria
            print(f"Info: No GPUs met the criteria (Util != -1, Free Mem >= {min_memory_mb} MiB).")
        else:  # Should be covered by fallback, but as a safeguard
            print("Info: No GPUs detected by NVML.")

        return selected_indices


# Main execution example (remains the same)
if __name__ == "__main__":
    required_free_mem = 2048  # Require 2GB free VRAM
    num_gpus_to_select = 1  # <<< Number of GPUs to select

    print("Initializing GPUInfo...")
    gpu_info = GPUInfo()
    gpu_info.print_status()

    print(f"Attempting to select {num_gpus_to_select} idle GPUs with >= {required_free_mem} MiB free memory...")
    selected_indices = gpu_info.select_idle_gpu(count=num_gpus_to_select, min_memory_mb=required_free_mem)

    if selected_indices:
        print(f"\n==> Using selected GPU indices: {selected_indices}")
        devices = [f"cuda:{idx}" for idx in selected_indices]
        print(f"    Target devices: {devices}")
        # Example: Test assignment on the first selected device
        try:
            print(f"    Testing assignment on {devices[0]}...")
            t = torch.zeros(1).to(devices[0])
            print(f"    Successfully created tensor on {devices[0]}.")
            del t
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    Error testing device {devices[0]}: {e}", file=sys.stderr)
    else:
        print(f"\n==> Failed to select {num_gpus_to_select} suitable GPUs.")

    print("\nScript finished.")
