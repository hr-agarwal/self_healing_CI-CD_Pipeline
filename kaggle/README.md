# Kaggle infrastructure and experiment archive

This directory contains everything that runs on Kaggle for the self-healing CI/CD pipeline:

- the production **Qwen3.8-27B** endpoint on two NVIDIA Tesla T4 GPUs;
- generated evaluation kernels for model quality, multifile repair, and model bake-offs;
- the Windows-safe tools used to build, push, monitor, and download kernels; and
- the logs and result artifacts produced by those runs.

The endpoint exposes an OpenAI-compatible API through a temporary Cloudflare tunnel. It is a
development and evaluation service, not a durable production host. Kaggle sessions expire, GPU
quotas apply, and the tunnel URL changes whenever the endpoint restarts.

## Architecture

```text
GitHub Actions failure
        |
        v
n8n workflow -> prompt construction -> Kaggle Qwen3.8 endpoint
        ^                                      |
        |                                      v
        +---------- streamed model response / repair result
```

The endpoint downloads a configured GGUF model, starts a prebuilt CUDA `llama-server`, selects the
largest context size that boots and answers, benchmarks the live server, and exposes port `8000`
through Cloudflare Tunnel. The n8n workflow uses the resulting URL and API key from the local
runtime state under `%LOCALAPPDATA%\selfheal-n8n\`.

## Directory layout

```text
kaggle/
  endpoint/
    deploy_src.py        maintained endpoint source template
    models.json           model registry; `default` selects production
    push/                 generated Kaggle upload (`kernel-metadata.json` + `.py`)
  notebooks/
    qwen3-8-27b-live-endpoint.py      exact source pulled from the public kernel
    qwen3-8-27b-live-endpoint.ipynb   local notebook wrapper for that source
    kernel-metadata.json              downloaded Kaggle metadata
  evals/
    shipped_quality/push/              log-quality and reasoning evaluation kernel
    multifile/push/                    cross-file repair evaluation kernel
    bakeoff/push/                      model speed and repair-quality comparison kernel
  tools/
    kaggle_cli.py        shared CLI, status, push, pull-back, and log helpers
    build_endpoint.py    generate endpoint/push for a registry model
    push.py              guarded push with pull-back verification
    status.py            status and last pushed version for every project kernel
    watch_deploy.py      stream endpoint boot output and save its URL
    fetch_logs.py        download selected logs and top-level outputs
  logs/                  archived kernel output grouped by kind, kernel, and run
```

Generated files under `endpoint/push/` and `evals/*/push/` should not be edited manually. Change the
maintained source or builder, regenerate the kernel, and then verify the generated artifact.

## Prerequisites

### Local machine

- Windows PowerShell with Python 3.10+;
- repository dependencies installed as described in `docs/SETUP.md`;
- the Kaggle CLI installed and authenticated with a Kaggle API token; and
- access to the project root so all commands below resolve `kaggle/...` paths correctly.

The Kaggle CLI normally reads `%USERPROFILE%\.kaggle\kaggle.json`. Keep that credential outside the
repository. On Windows, the project tools set `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` to prevent
non-ASCII kernel output from becoming an empty log file.

### Kaggle kernel

The live endpoint requires:

- machine shape `NvidiaTeslaT4` with two GPUs;
- internet access enabled;
- the attached `linsxav/llama-server-builder` kernel source containing the prebuilt CUDA server; and
- permission to download the selected Hugging Face GGUF model.

The endpoint intentionally exits when the assigned GPU memory is below the two-T4 threshold instead
of downloading a model that cannot run.

## Public notebook snapshot

The public endpoint notebook is [`lacrimatct/qwen3-8-27b-live-endpoint`](https://www.kaggle.com/code/lacrimact/qwen3-8-27b-live-endpoint).
The checked-in snapshot is separate from the maintained deployment source:

- `notebooks/qwen3-8-27b-live-endpoint.py` is the exact script downloaded from Kaggle.
- `notebooks/qwen3-8-27b-live-endpoint.ipynb` provides a local notebook view and executes that sibling script.
- `notebooks/kernel-metadata.json` records the public kernel ID, title, machine shape, internet setting,
  and attached kernel source.
- `endpoint/deploy_src.py` is the source used when generating future endpoint pushes.

Refresh the public snapshot from the project root:

```powershell
$env:PYTHONUTF8="1"; $env:PYTHONIOENCODING="utf-8"
python -m kaggle kernels pull lacrimatct/qwen3-8-27b-live-endpoint -p kaggle/notebooks -m
```

Kaggle returns this script kernel as `.py`, not as a native `.ipynb`; the local wrapper avoids
duplicating the implementation while still providing a notebook artifact for inspection.

## Deploy the live endpoint

Run these commands from the repository root:

```powershell
python kaggle/tools/status.py
python kaggle/tools/build_endpoint.py --list
python kaggle/tools/build_endpoint.py                         # use models.json default
python kaggle/tools/build_endpoint.py <model-key>             # optional registry override
python kaggle/tools/push.py endpoint
python kaggle/tools/watch_deploy.py
```

What each step does:

1. `status.py` confirms that no project kernel is already `RUNNING` or `QUEUED`.
2. `build_endpoint.py` substitutes the selected registry entry into `deploy_src.py`, compiles the
   result, and writes the metadata-selected file under `endpoint/push/`.
3. `push.py` refuses unsafe concurrent pushes, requires Kaggle's successful-push confirmation, pulls
   the published file back, and compares it with the local upload before recording the version.
4. `watch_deploy.py` streams the boot, context ladder, benchmark, tensor/MTP checks, and public URL.
   It stores the complete stream and a filtered boot log under `kaggle/logs/` and writes the URL to
   `%LOCALAPPDATA%\selfheal-n8n\qwen_url.txt`.

After the endpoint is reachable, start the n8n integration from the project documentation:

```powershell
python tools/golive/n8n_import.py
python tools/golive/up.py
```

Stop the local integration after the run:

```powershell
python tools/golive/down.py
```

Save endpoint outputs after a session:

```powershell
python kaggle/tools/fetch_logs.py endpoint
```

For a stopped or still-running kernel, an optional label or existing destination can be supplied:

```powershell
python kaggle/tools/fetch_logs.py endpoint round-name
python kaggle/tools/fetch_logs.py endpoint --into kaggle/logs/endpoint/<existing-run>
```

## Evaluation kernels

The evaluation kernels are generated artifacts and use the same production prompt, extraction, and
test behavior wherever possible:

- **Shipped quality** compares cases with no log, with the real CI log, and with thinking enabled on
  retry cases.
- **Multifile** tests repairs where the failing code, changed file, and fixable file are not always the
  same file.
- **Bake-off** compares registered model builds using control gates, hard cases, multifile cases,
  throughput, MTP acceptance, tensor-split engagement, and thinking leakage.

Local scorer and builder checks do not require a GPU:

```powershell
python eval/test_bakeoff_eval.py
python eval/test_multifile_eval.py
python eval/test_qualityspeed.py
python eval/test_shipped_eval.py
```

The actual GPU kernels are pushed with the same guarded helper, using their registered names:

```powershell
python kaggle/tools/push.py shipped_quality
python kaggle/tools/push.py multifile
python kaggle/tools/push.py bakeoff
```

## Operational rules

- **Never push while any project kernel is `RUNNING` or `QUEUED`.** Kaggle starts another session
  instead of replacing the current one, and GPU quota can be consumed twice. `push.py` enforces this
  unless `--force` is explicitly supplied.
- **Treat a push as successful only after pull-back verification.** A Kaggle response alone is not
  enough; the published source must match the local generated upload.
- **Do not download unfiltered kernel output.** Kaggle output can include large CUDA libraries.
  `fetch_logs.py` requests only top-level `.log`, `.json`, and `.txt` files.
- **Judge startup by measured effects.** Symmetric per-GPU VRAM indicates tensor split engagement;
  a non-null MTP acceptance rate indicates that draft tokens were actually accepted.
- **Expect queue delays.** T4×2 kernels can remain queued for hours, and a free Cloudflare tunnel
  can terminate silent requests after approximately 120 seconds.
- **Stop sessions from the Kaggle website.** The CLI cannot reliably cancel an active kernel. Use
  *View Active Events* → the session menu → *Stop Session*, then fetch the resulting logs.

## Troubleshooting

### The endpoint aborts before model download

Check that the kernel uses `NvidiaTeslaT4` with two GPUs. The boot output must show at least the
required total VRAM. Also confirm that the `llama-server-cuda.zip` source is attached.

### A context rung fails or hangs

The endpoint tries descending context sizes and records the last server output in `server.log`.
Tensor split disables some automatic fitting behavior, so an oversized rung can consume the complete
health timeout before the next rung is attempted. Inspect the selected context, per-GPU VRAM, and
`server.log` in the run directory.

### The endpoint URL changed

This is expected after a new Cloudflare quick-tunnel session. Run `watch_deploy.py` again and update
the n8n `QWEN_BASE_URL` variable or the local runtime state used by the go-live tools.

### Logs are empty or contain encoding errors

Use the project tools rather than calling the Kaggle CLI directly, and ensure the Windows UTF-8
environment variables are set. `kaggle_cli.py` configures them automatically.

### The healer should not call the model

The CI workflow classifies infrastructure failures separately from code failures. Confirm that the
failure kind is `code`, the n8n workflow is imported from the current generated JSON, and the endpoint
URL/API credential are current.

## Results and related documentation

- `docs/results/GOLIVE_RESULTS.md` — live endpoint rounds and repair outcomes.
- `docs/results/SHIPPED_QUALITY_EVAL.md` — shipped prompt/log/thinking quality results.
- `docs/results/MULTIFILE_EVAL.md` — cross-file repair results.
- `docs/research/` — hardware, model identity, speed, and quantization measurements.
- `docs/SETUP.md` — complete repository and n8n setup instructions.
- `README.md` — project overview, end-to-end testing, and go-live workflow.

## Security and lifecycle notes

Do not commit Kaggle API credentials, Hugging Face tokens, tunnel secrets, or live endpoint keys.
Runtime secrets belong under `%LOCALAPPDATA%\selfheal-n8n\` or the appropriate external credential
store. The endpoint is suitable for experiments and demonstrations; use a durable authenticated host
and a named tunnel for an always-on deployment.
