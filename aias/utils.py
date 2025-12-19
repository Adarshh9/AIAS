"""
AIAS Utilities
Helper functions and common utilities
"""

import os
import sys
import time
import psutil
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def check_gpu_availability() -> Dict[str, Any]:
    """
    Check GPU availability and VRAM
    
    Returns:
        Dictionary with GPU information
    """
    result = {
        "cuda_available": False,
        "gpu_count": 0,
        "gpus": [],
        "recommended_model": "Qwen/Qwen2-VL-2B-Instruct"
    }
    
    try:
        import torch
        
        result["cuda_available"] = torch.cuda.is_available()
        
        if result["cuda_available"]:
            result["gpu_count"] = torch.cuda.device_count()
            
            for i in range(result["gpu_count"]):
                props = torch.cuda.get_device_properties(i)
                total_vram = props.total_memory / (1024 ** 3)  # GB
                
                gpu_info = {
                    "index": i,
                    "name": props.name,
                    "total_vram_gb": round(total_vram, 1),
                    "compute_capability": f"{props.major}.{props.minor}"
                }
                result["gpus"].append(gpu_info)
            
            # Recommend model based on VRAM
            max_vram = max(g["total_vram_gb"] for g in result["gpus"]) if result["gpus"] else 0
            
            if max_vram >= 24:
                result["recommended_model"] = "Qwen/Qwen2-VL-7B-Instruct"
            elif max_vram >= 16:
                result["recommended_model"] = "Qwen/Qwen2-VL-7B-Instruct"
            elif max_vram >= 8:
                result["recommended_model"] = "Qwen/Qwen2-VL-2B-Instruct"
            else:
                result["recommended_model"] = "Qwen/Qwen2-VL-2B-Instruct"
                result["warning"] = "Limited VRAM. Performance may be slow."
    
    except ImportError:
        result["error"] = "PyTorch not installed"
    except Exception as e:
        result["error"] = str(e)
    
    return result


def check_audio_devices() -> Dict[str, Any]:
    """
    Check available audio input devices
    
    Returns:
        Dictionary with audio device information
    """
    result = {
        "available": False,
        "devices": [],
        "default_device": None
    }
    
    try:
        import pyaudio
        
        pa = pyaudio.PyAudio()
        result["available"] = True
        
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            
            if info["maxInputChannels"] > 0:
                device = {
                    "index": i,
                    "name": info["name"],
                    "channels": info["maxInputChannels"],
                    "sample_rate": int(info["defaultSampleRate"])
                }
                result["devices"].append(device)
                
                if info.get("index") == pa.get_default_input_device_info().get("index"):
                    result["default_device"] = device
        
        pa.terminate()
        
    except ImportError:
        result["error"] = "PyAudio not installed"
    except Exception as e:
        result["error"] = str(e)
    
    return result


def check_dependencies() -> Dict[str, bool]:
    """
    Check if required dependencies are installed
    
    Returns:
        Dictionary mapping package name to availability
    """
    packages = {
        # Core
        "torch": "torch",
        "transformers": "transformers",
        "PIL": "pillow",
        
        # Audio
        "pyaudio": "pyaudio",
        "faster_whisper": "faster-whisper",
        "pvporcupine": "pvporcupine",
        "webrtcvad": "webrtcvad",
        
        # Screen
        "mss": "mss",
        
        # Utilities
        "numpy": "numpy",
        "yaml": "pyyaml"
    }
    
    result = {}
    
    for module_name, package_name in packages.items():
        try:
            __import__(module_name)
            result[package_name] = True
        except ImportError:
            result[package_name] = False
    
    return result


def get_system_info() -> Dict[str, Any]:
    """
    Get system information
    
    Returns:
        Dictionary with system information
    """
    return {
        "platform": sys.platform,
        "python_version": sys.version,
        "cpu_count": psutil.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024 ** 3), 1)
    }


def format_time(seconds: float) -> str:
    """Format seconds as human-readable string"""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"


@contextmanager
def timer(name: str = "Operation"):
    """Context manager for timing operations"""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info(f"{name} completed in {format_time(elapsed)}")


class PerformanceMonitor:
    """Monitor system performance during operation"""
    
    def __init__(self):
        self._start_time = None
        self._metrics = []
    
    def start(self):
        """Start monitoring"""
        self._start_time = time.time()
        self._metrics = []
    
    def record(self, name: str, value: float):
        """Record a metric"""
        self._metrics.append({
            "name": name,
            "value": value,
            "timestamp": time.time() - (self._start_time or time.time())
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self._metrics:
            return {}
        
        # Group by name
        by_name = {}
        for m in self._metrics:
            name = m["name"]
            if name not in by_name:
                by_name[name] = []
            by_name[name].append(m["value"])
        
        summary = {}
        for name, values in by_name.items():
            summary[name] = {
                "count": len(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values)
            }
        
        return summary


def ensure_directory(path: str) -> Path:
    """Ensure a directory exists"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_cache_dir() -> Path:
    """Get AIAS cache directory"""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    
    cache_dir = base / "aias"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_data_dir() -> Path:
    """Get AIAS data directory"""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    
    data_dir = base / "aias"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
