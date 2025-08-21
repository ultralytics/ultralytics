# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import psutil
import pynvml


class SystemMetrics:
    """System metrics collector for CPU, disk, memory, network, and GPU monitoring."""

    def __init__(self, sample_interval: float = 15.0, disk_paths: Optional[List[Union[str, Path]]] = None):
        """Initialize system metrics collector with configurable sampling interval and disk paths."""
        self.sample_interval = sample_interval
        self.disk_paths = disk_paths or ["/"]
        self.monitoring = False
        
        # Initialize GPU monitoring
        self._gpu_initialized = self._init_nvidia_gpu()
        self._apple_gpu_available = platform.system() == "Darwin" and "arm64" in platform.machine()
        
        # Reset counters for cumulative metrics
        self.initial_disk_io = psutil.disk_io_counters()
        self.initial_network_io = psutil.net_io_counters()

    def _init_nvidia_gpu(self) -> bool:
        """Initialize NVIDIA GPU monitoring."""
        try:
            pynvml.nvmlInit()
            return True
        except Exception:
            return False

    def get_cpu_metrics(self) -> Dict[str, float]:
        """Get CPU usage metrics."""
        process = psutil.Process()
        return {
            "cpu": process.cpu_percent() / psutil.cpu_count(),
            "proc.cpu.threads": process.num_threads()
        }

    def get_disk_metrics(self) -> Dict[str, float]:
        """Get disk usage and I/O metrics."""
        metrics = {}
        
        # Disk usage for specified paths
        for path in self.disk_paths:
            try:
                usage = psutil.disk_usage(str(path))
                path_key = str(path).replace("/", "_").strip("_") or "root"
                metrics[f"disk.{path_key}.usagePercent"] = (usage.used / usage.total) * 100
                metrics[f"disk.{path_key}.usageGB"] = usage.used / (1024**3)
            except Exception:
                continue
        
        # Disk I/O
        if self.initial_disk_io and (current_io := psutil.disk_io_counters()):
            metrics["disk.in"] = (current_io.read_bytes - self.initial_disk_io.read_bytes) / (1024**2)
            metrics["disk.out"] = (current_io.write_bytes - self.initial_disk_io.write_bytes) / (1024**2)
        
        return metrics

    def get_memory_metrics(self) -> Dict[str, float]:
        """Get memory usage metrics."""
        process = psutil.Process()
        memory = psutil.virtual_memory()
        
        return {
            "proc.memory.rssMB": process.memory_info().rss / (1024**2),
            "proc.memory.percent": process.memory_percent(),
            "memory_percent": memory.percent,
            "proc.memory.availableMB": memory.available / (1024**2)
        }

    def get_network_metrics(self) -> Dict[str, float]:
        """Get network I/O metrics."""
        if not self.initial_network_io or not (current_io := psutil.net_io_counters()):
            return {}
        
        return {
            "network.sent": current_io.bytes_sent - self.initial_network_io.bytes_sent,
            "network.recv": current_io.bytes_recv - self.initial_network_io.bytes_recv
        }

    def get_nvidia_gpu_metrics(self) -> Dict[str, float]:
        """Get NVIDIA GPU metrics using pynvml."""
        if not self._gpu_initialized:
            return {}
        
        metrics = {}
        try:
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_percent = (mem_info.used / mem_info.total) * 100
                metrics.update({
                    f"gpu.{i}.memory": mem_percent,
                    f"gpu.{i}.memoryAllocated": mem_percent,
                    f"gpu.{i}.memoryAllocatedBytes": mem_info.used,
                })
                
                # GPU utilization and temperature
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                metrics.update({
                    f"gpu.{i}.gpu": util.gpu,
                    f"gpu.{i}.temp": temp,
                })
                
                # Power metrics
                try:
                    power_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
                    metrics.update({
                        f"gpu.{i}.powerWatts": power_watts,
                        f"gpu.{i}.powerPercent": (power_watts / power_limit) * 100,
                    })
                except Exception:
                    pass
                
                # Clock speeds
                try:
                    metrics.update({
                        f"gpu.{i}.smClock": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
                        f"gpu.{i}.memoryClock": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM),
                        f"gpu.{i}.graphicsClock": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS),
                    })
                except Exception:
                    pass
                
                # Memory errors
                try:
                    metrics.update({
                        f"gpu.{i}.correctedMemoryErrors": pynvml.nvmlDeviceGetMemoryErrorCounter(
                            handle, pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED, 
                            pynvml.NVML_VOLATILE_ECC, pynvml.NVML_MEMORY_LOCATION_DEVICE_MEMORY
                        ),
                        f"gpu.{i}.unCorrectedMemoryErrors": pynvml.nvmlDeviceGetMemoryErrorCounter(
                            handle, pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, 
                            pynvml.NVML_VOLATILE_ECC, pynvml.NVML_MEMORY_LOCATION_DEVICE_MEMORY
                        ),
                    })
                except Exception:
                    pass
                
                # Encoder utilization
                try:
                    encoder_util = pynvml.nvmlDeviceGetEncoderUtilization(handle)
                    metrics[f"gpu.{i}.encoderUtilization"] = encoder_util[0]
                except Exception:
                    pass
                    
        except Exception:
            pass
        
        return metrics

    def get_apple_gpu_metrics(self) -> Dict[str, float]:
        """Get Apple ARM Mac GPU metrics using powermetrics."""
        if not self._apple_gpu_available:
            return {}
        
        try:
            # Basic implementation - would need detailed powermetrics parsing for full functionality
            result = subprocess.run(
                ["sudo", "powermetrics", "--samplers", "gpu_power", "-n", "1", "-f", "plist"],
                capture_output=True, text=True, timeout=5
            )
            
            # Placeholder metrics - would need plist parsing for actual values
            return {
                "gpu.0.gpu": 0.0,
                "gpu.0.memoryAllocated": 0.0,
                "gpu.0.temp": 0.0,
                "gpu.0.powerWatts": 0.0,
                "gpu.0.powerPercent": 0.0
            }
            
        except Exception:
            return {}

    def get_all_metrics(self) -> Dict[str, float]:
        """Get all available system metrics."""
        return {
            **self.get_cpu_metrics(),
            **self.get_disk_metrics(),
            **self.get_memory_metrics(),
            **self.get_network_metrics(),
            **self.get_nvidia_gpu_metrics(),
            **self.get_apple_gpu_metrics()
        }

    def start_monitoring(self, callback=None):
        """Start continuous monitoring with optional callback for metrics."""
        def monitor():
            while self.monitoring:
                if callback:
                    callback(self.get_all_metrics())
                time.sleep(self.sample_interval)
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)


def get_system_info() -> Dict[str, Union[str, float, int]]:
    """Get basic system information."""
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "total_memory_gb": psutil.virtual_memory().total / (1024**3),
    }
