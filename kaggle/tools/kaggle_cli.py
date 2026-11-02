"""Shared Kaggle helpers for every kernel in this project. The other kaggle/tools/*.py use these.

Each rule here was learned the hard way:
- The CLI crashes on Windows when it writes a log with non-ASCII text ('charmap' codec) and leaves
  an EMPTY file behind. Every call runs with PYTHONUTF8=1.
- A push does NOT replace a running kernel: it runs a second session and doubles GPU quota use. The
  CLI also has no cancel. So nothing is pushed while ANY project kernel is RUNNING or QUEUED.
- A push that does not print "successfully pushed" did not happen (a 409 once lost a set of fixes),
  and the pushed code is pulled back and compared before we trust it.
- `kernels output` without a filter downloads the kernel's bin/ (164 MB of CUDA libraries per run).
  Only top-level logs and results are fetched.
"""
import datetime, io, json, os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
KAGGLE_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(KAGGLE_DIR)
LOGS = os.path.join(KAGGLE_DIR, "logs")

# Centralize kernel metadata so push, status, and log collection cannot disagree on slugs or paths.
KERNELS = {
    "endpoint":        {"slug": "lacrimatct/qwen3-8-27b-live-endpoint", "kind": "endpoint",
                        "push": os.path.join(KAGGLE_DIR, "endpoint", "push")},
    "shipped_quality": {"slug": "lacrimatct/qwen3-8-shipped-quality-eval", "kind": "evals",
                        "push": os.path.join(KAGGLE_DIR, "evals", "shipped_quality", "push")},
    "multifile":       {"slug": "lacrimatct/qwen3-8-multifile-eval", "kind": "evals",
                        "push": os.path.join(KAGGLE_DIR, "evals", "multifile", "push")},
    "bakeoff":         {"slug": "lacrimatct/qwen3-8-model-bakeoff", "kind": "bakeoff",
                        "push": os.path.join(KAGGLE_DIR, "evals", "bakeoff", "push")},
}
BUSY = ("RUNNING", "QUEUED")


def cli(*args, timeout=600):
    """Run the Kaggle CLI in UTF-8 mode. Returns (returncode, combined output without warnings)."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    p = subprocess.run([sys.executable, "-m", "kaggle"] + list(args), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env, timeout=timeout)
    out = "\n".join(l for l in (p.stdout + p.stderr).splitlines()
                    if l.strip() and "Warning" not in l and "warnings.warn" not in l)
    return p.returncode, out


def status(slug):
    """'COMPLETE' / 'RUNNING' / 'QUEUED' / 'ERROR' / 'CANCEL...' or 'NOT_FOUND' (never pushed)."""
    rc, out = cli("kernels", "status", slug, timeout=120)
    m = re.search(r"KernelWorkerStatus\.([A-Z_]+)", out)
    if m:
        return m.group(1)
    return "NOT_FOUND" if re.search(r"404|not found|Permission .* denied", out, re.I) else "UNKNOWN: " + out[-120:]


def busy_kernels():
    # A push starts another session instead of replacing a running one, so all kernels are guarded.
    return {k: s for k, s in ((k, status(v["slug"])) for k, v in KERNELS.items()) if s in BUSY}


def code_file(push_dir):
    # kernel-metadata.json is authoritative because each generated kernel uses a different filename.
    meta = json.load(io.open(os.path.join(push_dir, "kernel-metadata.json"), encoding="utf-8"))
    return os.path.join(push_dir, meta["code_file"]), meta


def push(name, force=False):
    """Idle check -> push -> require the success line -> pull back -> byte-compare. Returns the version."""
    k = KERNELS[name]
    busy = busy_kernels()
    if busy and not force:
        raise SystemExit("REFUSED: kernel(s) still busy %s - a push would run a second GPU session "
                         "(the CLI cannot cancel). Wait for COMPLETE." % busy)
    rc, out = cli("kernels", "push", "-p", k["push"], timeout=600)
    m = re.search(r"Kernel version (\d+) successfully pushed", out)
    if not m:
        raise SystemExit("PUSH NOT CONFIRMED (no 'successfully pushed' line):\n" + out[-600:])
    version = int(m.group(1))
    # Pull the published artifact back before recording success; this catches stale or partial pushes.
    src, meta = code_file(k["push"])
    pull = tempfile.mkdtemp()
    try:
        cli("kernels", "pull", k["slug"], "-p", pull, timeout=300)
        pulled = os.path.join(pull, os.path.basename(src))
        norm = lambda p: io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")
        if not os.path.exists(pulled) or norm(pulled) != norm(src):
            raise SystemExit("PULL-BACK MISMATCH: Kaggle holds different code than %s (v%d)" % (src, version))
    finally:
        shutil.rmtree(pull, ignore_errors=True)
    record(name, {"event": "push", "version": version, "code": os.path.relpath(src, ROOT)})
    return version


def record(name, entry):
    """Append to logs/<kind>/<kernel>/history.jsonl - the one place a kernel's pushes are listed."""
    k = KERNELS[name]
    d = os.path.join(LOGS, k["kind"], k["slug"].split("/")[1])
    os.makedirs(d, exist_ok=True)
    entry = dict(entry, at=datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
    io.open(os.path.join(d, "history.jsonl"), "a", encoding="utf-8").write(json.dumps(entry) + "\n")


def last_pushed_version(name):
    k = KERNELS[name]
    p = os.path.join(LOGS, k["kind"], k["slug"].split("/")[1], "history.jsonl")
    if not os.path.exists(p):
        return None
    pushes = [json.loads(l) for l in io.open(p, encoding="utf-8") if '"push"' in l]
    return pushes[-1]["version"] if pushes else None


def run_dir(name, label=""):
    """logs/<kind>/<kernel>/<date>_v<N>[_label]/ for the kernel's latest pushed version."""
    k = KERNELS[name]
    v = last_pushed_version(name)
    stem = datetime.date.today().isoformat() + ("_v%d" % v if v else "") + ("_" + label if label else "")
    d = os.path.join(LOGS, k["kind"], k["slug"].split("/")[1], stem)
    os.makedirs(d, exist_ok=True)
    return d


def fetch_logs(name, label="", into=None):
    """The kernel's latest log + top-level outputs (logs, json, txt - never bin/) into its run folder
    (or into an existing run folder, to backfill a run that was logged before these tools existed)."""
    k = KERNELS[name]
    d = into or run_dir(name, label)
    os.makedirs(d, exist_ok=True)
    rc, out = cli("kernels", "output", k["slug"], "-p", d, "--file-pattern", r"^[^/\\]+\.(log|json|txt)$",
                  timeout=900)
    # Restrict downloads to human-readable outputs; the kernel's CUDA bin/ is large and not diagnostic.
    kernel_log = os.path.join(d, k["slug"].split("/")[1] + ".log")
    if os.path.exists(kernel_log):
        os.replace(kernel_log, os.path.join(d, "kernel.log"))
    got = sorted(os.listdir(d))
    empty = [f for f in got if os.path.getsize(os.path.join(d, f)) == 0]
    if empty:
        raise SystemExit("EMPTY file(s) %s in %s - the download failed part-way" % (empty, d))
    record(name, {"event": "fetch_logs", "dir": os.path.relpath(d, ROOT), "files": got})
    return d, got
