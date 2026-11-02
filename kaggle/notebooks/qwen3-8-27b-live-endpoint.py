# %% [code]
"""Live Qwen3.8-27B endpoint on Kaggle 2xT4 - fast path (no compile).

Deploy method borrowed from linsxav/qwopus3-6-35b-a3b-1: the CUDA llama-server is built ONCE
in linsxav/llama-server-builder and attached here via kernel_sources, so startup is an unzip
plus LD_LIBRARY_PATH instead of ~40 minutes of cmake on 4 slow vCPUs.

Model: chosen in kaggle/endpoint/models.json; kaggle/tools/build_endpoint.py writes the chosen
entry into MODEL_SPEC below and produces kaggle/endpoint/push/. The default is unsloth
Qwen3.8-27B UD-Q4_K_XL (17.6 GB), the measured production model.

Speed: with -sm tensor + MTP n=2 the default serves ~24-30 tok/s on 2xT4 (measured live; it was
~7-9 tok/s before those two levers). This script benchmarks itself on boot so the number for
whatever model it serves is measured, not assumed.
"""
import json, os, re, shutil, subprocess, time, urllib.request, datetime, glob

NL = chr(10)
OUT = "/kaggle/working"
BIN = OUT + "/bin"
API_KEY = "selfheal-e2e-key"
os.makedirs(OUT, exist_ok=True)


def sh(cmd, timeout=None):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def banner(m):
    print(NL + "=" * 78 + NL + m + NL + "=" * 78, flush=True)


_, gpu, _ = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
vram = 0
for r in gpu.strip().split(NL):
    try: vram += int(r.split(",")[1].strip().split()[0])
    except Exception: pass
print("GPU:", gpu.strip().replace(NL, " | "), "| total", vram, "MiB", flush=True)
if vram < 24000:
    print("ABORT: need 2xT4"); raise SystemExit(0)

# ---------- 1. prebuilt llama-server (no compile) ----------
banner("EXTRACTING PREBUILT llama-server")
cands = glob.glob("/kaggle/input/**/llama-server-cuda.zip", recursive=True)
print("zip candidates:", cands, flush=True)
if not cands:
    print("!! prebuilt zip not attached - falling back to compile is NOT done here.", flush=True)
    print("   contents of /kaggle/input:", flush=True)
    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files[:20]:
            print("   ", os.path.join(root, f), flush=True)
        break
    raise SystemExit("attach kernel_sources = linsxav/llama-server-builder")

if os.path.exists(BIN):
    shutil.rmtree(BIN)
os.makedirs(BIN, exist_ok=True)
t0 = time.time()
shutil.unpack_archive(cands[0], BIN)
SERVER = os.path.join(BIN, "llama-server")
os.chmod(SERVER, 0o755)
for so in glob.glob(BIN + "/*.so*"):
    os.chmod(so, 0o755)
ENV_PREFIX = "LD_LIBRARY_PATH=%s:$LD_LIBRARY_PATH " % BIN
rc, o, e = sh(ENV_PREFIX + SERVER + " --help 2>&1 | head -3")
print("extracted in %.1fs; --help rc=%s" % (time.time() - t0, rc), flush=True)
print((o or e)[:300], flush=True)
assert rc == 0, "prebuilt binary does not run"

# ---------- 2. model ----------
# Filled by kaggle/tools/build_endpoint.py from kaggle/endpoint/models.json - edit the JSON, not this line.
MODEL_SPEC = {"key": "unsloth-ud-q4kxl", "repo": "unsloth/Qwen3.8-27B-GGUF", "pattern": "UD-Q4_K_XL\\.gguf$", "fallbacks": [["unsloth/Qwen3.8-27B-GGUF", "UD-Q4_K_M\\.gguf$"], ["unsloth/Qwen3.8-27B-GGUF", "UD-Q6_K\\.gguf$"]]}
REPO, QUANT = MODEL_SPEC["repo"], MODEL_SPEC["pattern"]
FALLBACKS = [tuple(f) for f in MODEL_SPEC["fallbacks"]]

from huggingface_hub import HfApi, hf_hub_download
API = HfApi()
MODELS = "/kaggle/tmp/models"
os.makedirs(MODELS, exist_ok=True)


def fetch(repo, rx):
    try:
        files = [f for f in API.list_repo_files(repo) if f.lower().endswith(".gguf")]
    except Exception as ex:
        print("  repo err", ex, flush=True); return None, None
    hits = sorted([f for f in files if re.search(rx, f, re.I) and "mmproj" not in f.lower()])
    if not hits:
        print("  no match /%s/ in %s -> %s" % (rx, repo, [os.path.basename(x) for x in files][:10]), flush=True)
        return None, None
    print("  downloading", os.path.basename(hits[0]), flush=True)
    t = time.time()
    p = hf_hub_download(repo, hits[0], local_dir=MODELS)
    print("  got %.1f GB in %.0fs" % (os.path.getsize(p) / 1e9, time.time() - t), flush=True)
    return p, repo


banner("DOWNLOADING MODEL %s (%s)" % (MODEL_SPEC["key"], REPO))
MODEL, USED_REPO = fetch(REPO, QUANT)
if not MODEL:
    for r, rx in FALLBACKS:
        MODEL, USED_REPO = fetch(r, rx)
        if MODEL: break
assert MODEL, "no model downloaded"
GB = os.path.getsize(MODEL) / 1e9


def wait_health(port=8000, timeout=900):
    t = time.time()
    while time.time() - t < timeout:
        try:
            if urllib.request.urlopen("http://localhost:%d/health" % port, timeout=5).status == 200:
                return True
        except Exception:
            if not sh("pgrep -f llama-server")[1].strip():
                return False
            time.sleep(5)
    return False


def vram_per_gpu():
    _, o, _ = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")
    try: return [int(x) for x in o.split()]
    except Exception: return []


def vram_used():
    return sum(vram_per_gpu()) or -1


def launch(ctx, port=8000, timeout=900):
    sh("pkill -f llama-server; sleep 4")
    # Measured winner (SPEED_LEVERS_EVAL.md / SPEED_QUALITY_EVAL.md): 12.35 -> 28.02 tok/s.
    #   -sm tensor  splits every tensor so BOTH T4s stream weights concurrently. Layer split
    #               (the old default) runs them sequentially - 320 GB/s, not 640.
    #   f16 KV      NOT a preference: tensor mode REFUSES to start with quantized KV.
    #   draft-mtp   speculative decode off the MTP head already inside unsloth's GGUF. Measured
    #               acceptance 0.965, mean 2.9 tokens per verify pass.
    #   --parallel 1  speculative decode is a single-stream optimisation; the gain is gone by 4.
    #
    # `--parallel 1` is a deliberate PRODUCTION choice, not an inherited benchmark flag. Two
    # failing pushes close together mean n8n issues a second heal while one is in flight; with
    # -np 1 that request QUEUES rather than running concurrently. Accepted on purpose: heals are
    # rare, and a queued second heal is better than halving draft acceptance on both. If heals
    # ever become frequent, raise -np/--parallel together and expect the MTP gain to shrink.
    cmd = (ENV_PREFIX + "nohup " + SERVER +
           " -m '%s' --n-gpu-layers 999 --ctx-size %d --flash-attn on "
           "-sm tensor --cache-type-k f16 --cache-type-v f16 "
           "--spec-type draft-mtp --spec-draft-n-max 2 --parallel 1 "
           "-b 2048 -ub 512 -t 4 -tb 4 "
           # Caps thinking, which the workflow enables on RETRIES only. Measured: 1024 healed
           # 18/18 hard calls, worst eval call 78 s, but a live retry ran 126 s and hit the tunnel's 120 s timeout (524).
           "--reasoning-budget 1024 "
           "--host 0.0.0.0 --port %d -np 1 --jinja --api-key %s > %s/server.log 2>&1 &"
           % (MODEL, ctx, port, API_KEY, OUT))
    os.system(cmd)
    return wait_health(port, timeout=timeout)


def bench(port=8000, n=3, max_tokens=256):
    """Real generation speed, reported by the server itself."""
    got = []
    for i in range(n):
        body = json.dumps({"model": "local", "max_tokens": max_tokens, "temperature": 0.7,
                           "stream": False,
                           "messages": [{"role": "user",
                                         "content": "Write a short technical explanation of how "
                                                    "continuous integration pipelines detect broken builds."}]}).encode()
        try:
            req = urllib.request.Request("http://localhost:%d/v1/chat/completions" % port, body,
                                         {"Content-Type": "application/json",
                                          "Authorization": "Bearer " + API_KEY})
            j = json.loads(urllib.request.urlopen(req, timeout=600).read())
        except Exception as ex:
            got.append({"error": str(ex)[:120]}); continue
        tm = j.get("timings", {})
        # accept_rate is the PROOF that MTP engaged. Do NOT grep the log for "draft-mtp": that
        # string is absent on b10709 even while speculation is demonstrably running, and a log
        # needle has already produced two false negatives in this project. A non-null
        # draft_n means the draft model actually produced tokens and had them accepted.
        dn, da = tm.get("draft_n"), tm.get("draft_n_accepted")
        got.append({"gen_tps": tm.get("predicted_per_second"),
                    "prompt_tps": tm.get("prompt_per_second"),
                    "tokens": j.get("usage", {}).get("completion_tokens"),
                    "draft_n": dn,
                    "accept_rate": round((da or 0) / dn, 3) if dn else None})
        print("    run %d: %.1f tok/s (%s tokens)" % (i + 1, got[-1].get("gen_tps") or 0,
                                                      got[-1].get("tokens")), flush=True)
    g = [x["gen_tps"] for x in got if x.get("gen_tps")]
    acc = [x["accept_rate"] for x in got if x.get("accept_rate") is not None]
    per = vram_per_gpu()
    # Tensor split gives IDENTICAL per-GPU memory because every tensor is divided; layer split
    # cannot. That asymmetry is how a silent fallback to layer mode would be caught.
    sym = bool(len(per) == 2 and abs(per[0] - per[1]) <= 64)
    return {"runs": got, "avg": round(sum(g) / len(g), 1) if g else None,
            "best": round(max(g), 1) if g else None,
            "mtp_engaged": bool(acc),
            "accept_rate": round(sum(acc) / len(acc), 3) if acc else None,
            "vram_per_gpu": per, "tensor_split_engaged": sym}


# ---------- 3. context ladder: biggest that boots AND answers ----------
# f16 KV (forced by tensor mode) costs ~2x the q8_0 cache this ladder was tuned for, so the
# floor is extended: better to serve a smaller context than to fail to deploy. Real prompts
# measured 740-835 tokens, so even 8192 has ~5x headroom; over-length prompts are REJECTED
# (exceed_context_size_error), never silently truncated.
# MEASURED on the 2026-09-22 go-live: with tensor split + f16 KV, 262144 and 196608 do not
# fail fast - tensor mode disables llama.cpp's auto-fit, so an oversized ctx HANGS for the
# full health timeout (15 min each, ~30 min of GPU wasted). 131072 booted at 13,287 MiB per
# card in under a minute. Start there; the lower rungs are insurance, not expected.
LADDER = [131072, 98304, 65536, 32768, 16384, 8192]
RUNG_TIMEOUT = 300   # a rung that fits boots in <1 min; don't wait 15 min for one that hangs
chosen = None
for ctx in LADDER:
    banner("TRYING ctx = %d" % ctx)
    if not launch(ctx, timeout=RUNG_TIMEOUT):
        _, tail, _ = sh("tail -5 %s/server.log" % OUT)
        print("  failed at ctx=%d:%s%s" % (ctx, NL, tail[-600:]), flush=True)
        continue
    used = vram_used()
    print("  booted. VRAM %d / %d MiB. benchmarking..." % (used, vram), flush=True)
    b = bench()
    if not b["avg"]:
        print("  booted but requests failed", flush=True); continue
    chosen = {"ctx": ctx, "vram_used_mib": used, "bench": b}
    banner("SELECTED ctx=%d | VRAM %d/%d MiB | %s tok/s avg (best %s)"
           % (ctx, used, vram, b["avg"], b["best"]))
    print("  tensor split engaged: %s  (per-GPU %s, symmetric == engaged)"
          % (b["tensor_split_engaged"], b["vram_per_gpu"]), flush=True)
    print("  MTP engaged         : %s  (accept_rate %s)"
          % (b["mtp_engaged"], b["accept_rate"]), flush=True)
    if not (b["tensor_split_engaged"] and b["mtp_engaged"]):
        print("  WARNING: a lever did not engage - this endpoint will be slower than measured. "
              "Serving anyway, but the 28 tok/s figure does not apply.", flush=True)
    break
assert chosen, "no context size worked"

# ---------- 4. tunnel ----------
banner("STARTING CLOUDFLARE TUNNEL")
sh("wget -qO /kaggle/tmp/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/"
   "download/cloudflared-linux-amd64 && chmod +x /kaggle/tmp/cloudflared")
os.system("nohup /kaggle/tmp/cloudflared tunnel --url http://localhost:8000 --no-autoupdate "
          "> %s/tunnel.log 2>&1 &" % OUT)
url = None
for _ in range(45):
    time.sleep(4)
    try: log = open(OUT + "/tunnel.log").read()
    except Exception: continue
    m = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", log)
    if m: url = m.group(0); break

json.dump({"url": url, "api_key": API_KEY, "repo": USED_REPO,
           "file": os.path.basename(MODEL), "model_gb": round(GB, 1),
           "ctx": chosen["ctx"], "vram_used_mib": chosen["vram_used_mib"],
           "tensor_split_engaged": chosen["bench"]["tensor_split_engaged"],
           "mtp_engaged": chosen["bench"]["mtp_engaged"],
           "accept_rate": chosen["bench"]["accept_rate"],
           "vram_total_mib": vram, "gen_tps_avg": chosen["bench"]["avg"],
           "gen_tps_best": chosen["bench"]["best"]},
          open(OUT + "/endpoint.json", "w"), indent=1)

# Bounded to the minimum that fits deploy + one confirmation drive. Earlier a 45-min
# window was spent for no gain; the model call itself is already proven five times.
# Live window after the endpoint comes up. 0.3 h covered one confirmation drive; a go-live
# needs n8n import + 5 sequential heals (~2-3 min each incl. restore pushes) with margin.
HOURS = 1.5   # round 3: 2 multi-attempt retry cases + 5 controls; the CLI cannot cancel, so not more
deadline = time.time() + HOURS * 3600


def announce():
    banner("ENDPOINT LIVE")
    print("URL      : %s" % url, flush=True)
    print("API key  : %s" % API_KEY, flush=True)
    print("Model    : %s (%.1f GB, UD-Q4_K_XL)" % (os.path.basename(MODEL), GB), flush=True)
    print("Context  : %d tokens" % chosen["ctx"], flush=True)
    print("VRAM     : %d / %d MiB" % (chosen["vram_used_mib"], vram), flush=True)
    print("SPEED    : %s tok/s avg, %s tok/s best" % (chosen["bench"]["avg"], chosen["bench"]["best"]), flush=True)
    print(NL + "curl %s/v1/models -H 'Authorization: Bearer %s'" % (url, API_KEY), flush=True)
    print(NL + "curl %s/v1/chat/completions -H 'Authorization: Bearer %s' "
          "-H 'Content-Type: application/json' -d '{\"model\":\"local\",\"messages\":"
          "[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":200}'" % (url, API_KEY), flush=True)


announce()
n = 0
while time.time() < deadline:
    time.sleep(60); n += 1
    if not sh("pgrep -f llama-server")[1].strip():
        print("!! server died, restarting", flush=True)
        launch(chosen["ctx"])
    if n % 10 == 0:
        announce()
    else:
        print("[%3d min] vram=%d MiB URL=%s" % (n, vram_used(), url), flush=True)
print("window elapsed", flush=True)
