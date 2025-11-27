import json
import psutil
import time
import platform

def record_telemetry(output_path: str):
    """Collects and stores CPU, memory, and swap stats."""
    try:
        snapshot = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system": platform.system(),
            "platform": platform.platform(),
            "cpu_usage_percent": psutil.cpu_percent(interval=1),
            "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "ram_percent": psutil.virtual_memory().percent,
            "swap_percent": psutil.swap_memory().percent,
            "ram_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
            "swap_used_gb": round(psutil.swap_memory().used / (1024**3), 2),
            "process_count": len(psutil.pids())
        }
        with open(output_path, "w") as f:
            json.dump(snapshot, f, indent=2)
        print(f"Telemetry saved → {output_path}")
    except Exception as e:
        print(f"Telemetry collection failed: {e}")
