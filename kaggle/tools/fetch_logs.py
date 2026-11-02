"""Save a kernel's latest Kaggle log and top-level outputs into kaggle/logs/<kind>/<kernel>/<date>_v<N>/.

    python kaggle/tools/fetch_logs.py endpoint [label]
    python kaggle/tools/fetch_logs.py endpoint --into kaggle/logs/endpoint/<kernel>/<existing run>
Never downloads bin/; refuses to finish if any file came down empty (the Windows charmap crash).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_cli import KERNELS, fetch_logs, status   # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else ""
# A missing or misspelled kernel must not create a misleading empty run directory.
if name not in KERNELS:
    sys.exit("usage: fetch_logs.py {%s} [label]" % "|".join(KERNELS))
s = status(KERNELS[name]["slug"])
# Kaggle can stream partial output while a kernel is still running; make that explicit.
if s in ("RUNNING", "QUEUED"):
    print("note: %s is %s - this is a partial log" % (name, s))
into = sys.argv[sys.argv.index("--into") + 1] if "--into" in sys.argv else None
label = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else ""
d, files = fetch_logs(name, label, into)
print("saved %d file(s) to %s: %s" % (len(files), d, ", ".join(files)))
