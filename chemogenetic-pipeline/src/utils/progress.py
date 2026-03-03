"""
Pipeline Progress Tracking
Writes JSON status files that the Streamlit dashboard can read for live updates.
"""

import os
import json
import time
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROGRESS_DIR = os.path.join(PROJECT_ROOT, "data", "progress")


def _ensure_dir():
    os.makedirs(PROGRESS_DIR, exist_ok=True)


def update_module_status(module_num, status, step=None, detail=None,
                         progress=None, total=None, metrics=None):
    """Update progress status for a module.

    Args:
        module_num: Module number (4-8)
        status: One of "idle", "running", "completed", "failed"
        step: Current step description (e.g., "Downloading ChEMBL M3 data")
        detail: Extra detail text
        progress: Current progress count
        total: Total items to process
        metrics: Dict of key metrics to display
    """
    _ensure_dir()

    info = {
        "module": module_num,
        "status": status,
        "step": step,
        "detail": detail,
        "progress": progress,
        "total": total,
        "metrics": metrics or {},
        "updated_at": datetime.now().isoformat(),
        "timestamp": time.time(),
    }

    path = os.path.join(PROGRESS_DIR, f"module{module_num}.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2)


def get_module_status(module_num):
    """Read current status for a module."""
    path = os.path.join(PROGRESS_DIR, f"module{module_num}.json")
    if not os.path.exists(path):
        return {"module": module_num, "status": "idle", "step": None}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"module": module_num, "status": "idle", "step": None}


def get_all_status():
    """Read status for all modules."""
    statuses = {}
    for mod in [1, 2, 4, 5, 6, 7, 8]:
        statuses[mod] = get_module_status(mod)
    return statuses


def clear_module_status(module_num):
    """Clear status file for a module."""
    path = os.path.join(PROGRESS_DIR, f"module{module_num}.json")
    if os.path.exists(path):
        os.remove(path)
