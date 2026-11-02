"""Push a project kernel safely: idle check (all kernels) -> push -> confirm -> pull-back diff.

    python kaggle/tools/push.py endpoint | shipped_quality | multifile | bakeoff
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_cli import KERNELS, push   # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else ""
# Reject typos before querying Kaggle or starting a billable kernel session.
if name not in KERNELS:
    sys.exit("usage: push.py {%s}" % "|".join(KERNELS))
# push() checks that all project kernels are idle and compares the pulled code byte-for-byte.
v = push(name, force="--force" in sys.argv)
print("pushed %s as version %d; Kaggle copy verified identical" % (KERNELS[name]["slug"], v))
