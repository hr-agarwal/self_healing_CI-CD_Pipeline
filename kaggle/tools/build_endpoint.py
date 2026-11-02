"""Write the endpoint kernel for a chosen model: kaggle/endpoint/push/<code_file>.

    python kaggle/tools/build_endpoint.py                 the models.json default
    python kaggle/tools/build_endpoint.py jumbo-fusion-747
    python kaggle/tools/build_endpoint.py --list

deploy_src.py carries one MODEL_SPEC line; this replaces it with the chosen models.json entry, so
the pushed kernel is a single self-contained file (Kaggle runs only the code file, it cannot read
models.json). With the default model the output is byte-identical to deploy_src.py.
"""
import io, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_cli import KAGGLE_DIR, code_file, KERNELS   # noqa: E402

# The registry is the source of truth for model selection; the pushed kernel remains self-contained.
REG = json.load(io.open(os.path.join(KAGGLE_DIR, "endpoint", "models.json"), encoding="utf-8"))
SRC = os.path.join(KAGGLE_DIR, "endpoint", "deploy_src.py")

# Listing is intentionally side-effect free so it is safe to use before a push.
if "--list" in sys.argv:
    for k, m in REG["models"].items():
        print("%s %-26s %5.1f GB  %s" % ("*" if k == REG["default"] else " ", k, m["gb"], m["note"][:80]))
    sys.exit(0)
key = next((a for a in sys.argv[1:] if not a.startswith("-")), REG["default"])
if key not in REG["models"]:
    sys.exit("unknown model %r - see --list" % key)
m = REG["models"][key]
spec = {"key": key, "repo": m["repo"], "pattern": m["pattern"], "fallbacks": m["fallbacks"]}
# Replace exactly one generated slot and compile before writing the Kaggle upload file.
src = io.open(SRC, encoding="utf-8", newline="").read()
line = re.compile(r"^MODEL_SPEC = \{.*\}$", re.M)
assert len(line.findall(src)) == 1, "deploy_src.py must carry exactly one MODEL_SPEC line"
out = line.sub(lambda _: "MODEL_SPEC = " + json.dumps(spec), src)
compile(out, "endpoint", "exec")                       # never write a kernel that does not parse
# The pull-back verification in kaggle_cli.py depends on this being the metadata-selected code file.
dst, _ = code_file(KERNELS["endpoint"]["push"])
io.open(dst, "w", encoding="utf-8", newline="").write(out)
print("wrote %s for %s (%s, %.1f GB)%s" % (os.path.relpath(dst, os.path.dirname(KAGGLE_DIR)), key, m["repo"],
                                          m["gb"], "  == deploy_src.py" if out == src else ""))
