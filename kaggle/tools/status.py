"""Status of every project kernel, one line each.   python kaggle/tools/status.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_cli import KERNELS, status, last_pushed_version   # noqa: E402

# Include the last recorded version so status output is useful when deciding whether to push.
for name, k in KERNELS.items():
    v = last_pushed_version(name)
    print("%-16s %-40s %-10s %s" % (name, k["slug"], status(k["slug"]), ("last push v%d" % v) if v else ""))
