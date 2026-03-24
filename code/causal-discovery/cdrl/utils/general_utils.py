"""
Miscellaneous utility functions: memory reporting, list chunking, time formatting,
and a JSON encoder for NumPy types.
"""

import os
import time
from datetime import datetime
import json
import numpy as np

# strftime/strptime format used consistently across experiment logs and file names.
date_format = "%Y-%m-%d-%H-%M-%S"


def get_memory_usage_str():
    """Get the virtual memory usage of the current process in megabytes as a string."""
    import psutil
    mb_used = psutil.Process(os.getpid()).memory_info().vms / 1024 ** 2
    return f"Process memory usage: {mb_used} MBs."


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_current_time_millis():
    """Return epoch time in milliseconds."""
    return int(time.time() * 1000)


def print_time_from(dt):
    """Print the start time, end time, and elapsed seconds relative to the given datetime."""
    started_str = dt.strftime(date_format)
    print(f"started at {started_str}")

    experiment_ended_datetime = datetime.now()
    ended_str = experiment_ended_datetime.strftime(date_format)
    print(f"ended at {ended_str}")
    print(f"took {(experiment_ended_datetime - dt).total_seconds(): .3f} seconds.")


class NpEncoder(json.JSONEncoder):
    """
    JSON encoder that handles NumPy scalar and array types.

    Pass this as the `cls` argument to `json.dumps` or `json.dump` whenever
    the object being serialised may contain numpy integers, floats, or arrays:
        json.dumps(result, cls=NpEncoder)
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
