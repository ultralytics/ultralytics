# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import sys
import torch
import warnings # Use warnings module for better control

class GPUInfo:
    """
    Provides information about system GPUs using pynvml, selecting the most idle GPU.

    Attributes:
        pynvml (module | None): The imported pynvml module, or None if unavailable.
        nvml_available (bool): True if pynvml was successfully imported and initialized.
        gpu_stats (list): A list of dictionaries, each containing stats for a GPU.
    """

    def __init__(self):
        """Initializes GPUInfo, attempting to import and initialize pynvml."""
        self.pynvml = None
        self.nvml_available = False
        self.gpu_stats = []
        try:
            # Import pynvml here, making it optional
            import pynvml
            self.pynvml = pynvml
            self.pynvml.nvmlInit() # Initialize NVML
            self.nvml_available = True
            self.refresh_stats() # Get initial stats
            # Note: We keep NVML initialized until explicitly shut down or object deletion

        except ImportError:
            warnings.warn("nvidia-ml-py (pynvml) not found. GPU stats features will be disabled.", ImportWarning)
        except self.pynvml.NVMLError as error:
            warnings.warn(f"Failed to initialize NVML: {error}. GPU stats features will be disabled.", RuntimeWarning)
            self.pynvml = None # Ensure pynvml is None if init failed
        except Exception as e:
            warnings.warn(f"An unexpected error occurred during pynvml initialization: {e}", RuntimeWarning)
            self.pynvml = None

    def __del__(self):
        """Ensures NVML is shut down when the object is garbage collected."""
        self.shutdown()

    def shutdown(self):
        """Shuts down NVML if it was initialized."""
        if self.nvml_available and self.pynvml:
            try:
                self.pynvml.nvmlShutdown()
            except self.pynvml.NVMLError as error:
                # Ignore shutdown errors, might happen if already shut down elsewhere
                pass
            self.nvml_available = False # Mark as shut down

    def refresh_stats(self):
        """Refreshes the internal gpu_stats list by querying NVML."""
        if not self.nvml_available or not self.pynvml:
            # No NVML, clear stats
            self.gpu_stats = []
            return

        self.gpu_stats = []
        try:
            device_count = self.pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                memory = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)

                # Helper to safely get optional stats directly using self.pynvml
                def safe_get(func, *args, default=-1, divisor=1):
                    try:
                        # Need to call via self.pynvml stored reference
                        val = getattr(self.pynvml, func.__name__)(*args)
                        return val // divisor
                    except self.pynvml.NVMLError:
                        return default

                self.gpu_stats.append({
                    'index': i,
                    'name': self.pynvml.nvmlDeviceGetName(handle),
                    'utilization': util.gpu,
                    'memory_used': memory.used >> 20,
                    'memory_total': memory.total >> 20,
                    'memory_free': memory.free >> 20,
                    'temperature': safe_get(self.pynvml.nvmlDeviceGetTemperature, handle, self.pynvml.NVML_TEMPERATURE_GPU),
                    'power_draw': safe_get(self.pynvml.nvmlDeviceGetPowerUsage, handle, divisor=1000),
                    'power_limit': safe_get(self.pynvml.nvmlDeviceGetEnforcedPowerLimit, handle, divisor=1000, default="N/A"),
                })
        except self.pynvml.NVMLError as error:
            warnings.warn(f"NVML error during device query: {error}", RuntimeWarning)
            self.gpu_stats = [] # Clear stats on error


    def print_status(self):
        """Prints GPU status in a compact table format using current stats."""
        if not self.gpu_stats: # Checks if list is empty (covers NVML unavailable or query errors)
            print("No GPU stats available (NVML may be missing, failed to initialize, or no GPUs found).")
            return

        stats = self.gpu_stats # Use the stored stats
        max_name_len = max(len(gpu['name']) for gpu in stats) if stats else 10
        header = f"{'Idx':<3} {'Name':<{max_name_len}} {'Util':>6} {'Mem (MiB)':>15} {'Temp':>5} {'Pwr (W)':>10}"
        print(f"\n--- GPU Status ---")
        print(header)
        print("-" * len(header))

        for gpu in stats:
            mem_str = f"{gpu['memory_used']:>6}/{gpu['memory_total']:<6}"
            temp_str = f"{gpu['temperature']}C" if gpu['temperature'] != -1 else "N/A"
            pwr_str = f"{gpu['power_draw']:>3}/{str(gpu['power_limit']):<3}" if gpu['power_draw'] != -1 else "N/A"
            print(f"{gpu['index']:<3d} {gpu['name']:<{max_name_len}} {gpu['utilization']:>5}% "
                  f"{mem_str:>15} {temp_str:>5} {pwr_str:>10}")
        print("-" * len(header))
        print("")

    def select_idle_gpu(self, min_memory_mb=2048):
        """
        Selects the most idle GPU based on utilization and free memory from current stats.

        Args:
            min_memory_mb (int): Minimum free memory required in MiB.

        Returns:
            int: The index of the most idle GPU meeting the criteria,
                 or 0 if NVML unavailable but CUDA devices exist (basic fallback),
                 or -1 if no suitable GPU found or CUDA unavailable.
        """
        if not self.gpu_stats:
            warnings.warn("Cannot select GPU based on NVML stats. Falling back to basic check.", RuntimeWarning)
            # Simple fallback check using torch
            try:
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    warnings.warn("NVML unavailable, selecting default GPU index 0 as fallback.", RuntimeWarning)
                    return 0 # Default to first GPU if CUDA is available
            except Exception as e: # Catch potential errors during torch check
                 warnings.warn(f"Error during PyTorch CUDA check: {e}", RuntimeWarning)
            return -1 # No NVML and no CUDA devices detected by torch

        stats = self.gpu_stats # Use stored stats
        # Filter GPUs meeting memory requirement and having valid utilization
        eligible_gpus = [gpu for gpu in stats if gpu.get('memory_free', 0) >= min_memory_mb and gpu.get('utilization') is not None]

        # If none meet memory criteria, consider all GPUs with valid utilization as fallback
        if not eligible_gpus:
            print(f"Info: No GPUs found with >= {min_memory_mb} MiB free. Considering all GPUs based on utilization.")
            eligible_gpus = [gpu for gpu in stats if gpu.get('utilization') is not None]

        if not eligible_gpus:
            print("Error: No suitable GPUs found for selection (no valid utilization reported).")
            return -1 # No GPUs available or none with valid utilization

        # Sort by utilization (asc), then free memory (desc)
        eligible_gpus.sort(key=lambda x: (x['utilization'], -x['memory_free']))

        selected_gpu = eligible_gpus[0]
        print(f"Selected GPU {selected_gpu['index']} ({selected_gpu['name']}) - Util: {selected_gpu['utilization']}%, Free Mem: {selected_gpu['memory_free']} MiB")
        return selected_gpu['index']

# Main execution example
if __name__ == "__main__":
    required_free_mem = 4096  # Require 4GB free VRAM

    # 1. Create GPUInfo instance (imports pynvml, initializes, gets stats)
    gpu_info = GPUInfo()

    # 2. Print status (uses stored stats)
    gpu_info.print_status()

    # Optional: Refresh stats if needed after some time/operations
    # gpu_info.refresh_stats()
    # gpu_info.print_status()

    # 3. Select GPU (uses stored stats)
    selected_index = gpu_info.select_idle_gpu(min_memory_mb=required_free_mem)

    # 4. Assign & Test (optional)
    if selected_index != -1:
        device = f"cuda:{selected_index}"
        print(f"\n==> Assigning task to: {device}")
        # Optional: Test assignment
        try:
            t = torch.zeros(1).to(device)
            print(f"Successfully tested device {device}.")
            del t
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error testing device {device}: {e}", file=sys.stderr)
    else:
        print("\n==> No suitable GPU found. Consider CPU or check GPU availability/status.")
        device = "cpu" # Example fallback

    # 5. Clean up NVML explicitly if needed before script ends,
    # otherwise __del__ will handle it eventually.
    # gpu_info.shutdown()
