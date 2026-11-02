"""Follow the live-endpoint kernel's log until it announces the endpoint.

`kaggle kernels output` returns nothing while a kernel is RUNNING - only `kernels logs -f`
streams. Prints the lines that decide go/no-go (context chosen, whether tensor split and MTP
actually engaged, speed) and saves the URL next to the local n8n secrets, never in the project.

Everything it reads is also saved to the run's folder, kaggle/logs/endpoint/<kernel>/<date>_v<N>/:
  boot.log    the go/no-go lines (what this script prints)
  stream.log  the whole log as streamed, deduplicated - `logs -f` replays from the start on reconnect
"""
import os, re, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_cli import KERNELS, run_dir   # noqa: E402

KERNEL = KERNELS["endpoint"]["slug"]
HOME = os.path.join(os.environ["LOCALAPPDATA"], "selfheal-n8n")
KEEP = ("GPU:", "got ", "TRYING ctx", "failed at ctx", "booted.", "SELECTED", "tensor split engaged",
        "MTP engaged", "WARNING", "URL      :", "SPEED", "Context", "Traceback", "Error", "ABORT", "DOWNLOADING MODEL")
# Keep a complete stream for diagnostics, but print only lines that determine deployment health.
OUT = run_dir("endpoint")
boot = open(os.path.join(OUT, "boot.log"), "a", encoding="utf-8")
stream = open(os.path.join(OUT, "stream.log"), "a", encoding="utf-8")
seen = set()
print("saving to", OUT, flush=True)
deadline = time.time() + 45 * 60
env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
while time.time() < deadline:
    # Reconnect because the Kaggle logs endpoint may close between polling intervals.
    p = subprocess.Popen([sys.executable, "-m", "kaggle", "kernels", "logs", KERNEL, "-f"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", env=env)
    for line in p.stdout:
        s = line.rstrip()
        if s in seen:
            continue
        seen.add(s)
        stream.write(s + "\n"); stream.flush()
        if any(k in s for k in KEEP):
            print(s[:200], flush=True)
            boot.write(s + "\n"); boot.flush()
        m = re.search(r"URL\s+:\s+(https://[-a-z0-9]+\.trycloudflare\.com)", s)
        if m:
            open(os.path.join(HOME, "qwen_url.txt"), "w").write(m.group(1))
            print("\nENDPOINT URL saved", flush=True); p.kill(); sys.exit(0)
    p.kill(); time.sleep(20)
print("no endpoint within the window"); sys.exit(1)
