# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Monte Carlo fuzzing of the `yolo` CLI to find bugs outside the finite test matrix.

Runs randomized/mutated `yolo` commands in subprocesses, classifies every outcome (pass, expected cfg error,
environment skip, network flake, hang, crash, bug candidate), confirms bug candidates by replaying them, and emits
per-shard JSONL trial logs plus a findings file. A stdlib-only `report` subcommand aggregates shard findings in CI,
dedupes them against existing GitHub issues by stable signature, and files at most `--max-issues` new issues per run.

Subcommands:
    fuzz    Run a budgeted fuzzing loop (imports ultralytics).
    repro   Replay one exact command several times and print its classification (imports ultralytics).
    report  Aggregate shard findings and file GitHub issues via `gh` (stdlib only, no ultralytics import).

Usage:
    python .github/scripts/fuzz.py fuzz --budget-minutes 300 --seed 123 --personality chaos --out fuzz-out
    python .github/scripts/fuzz.py repro "train detect model=yolo26n.pt data=coco8.yaml epochs=abc imgsz=32"
    python .github/scripts/fuzz.py report --in fuzz-out --max-issues 3 --dry-run
"""

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PureWindowsPath

# Per-mode subprocess timeouts (seconds) on Linux CPU runners, ~6-10x headroom over observed norms
# Windows runners are ~2x slower (interpreter startup, filesystem), so all timeouts scale there
TIMEOUT_SCALE = 2 if os.name == "nt" else 1
MODE_TIMEOUTS = {"train": 360, "val": 180, "predict": 180, "track": 240, "export": 480}
CONFIRM_TIMEOUT = 180  # shorter secondary timeout when confirming hangs (never re-pay the full timeout)
MAX_HANG_CONFIRMS = 5  # cap hang confirmations per shard so one pathological class can't eat the budget
MIN_FREE_GB = 5  # stop fuzzing gracefully below this much free disk
CANARY_FAIL_FRACTION = 0.2  # >20% unmutated known-good corpus failures marks the shard infra_failed

MODES = ["train", "val", "predict", "track", "export"]  # benchmark swallows exceptions; solutions deferred
PERSONALITIES = {  # mode-selection weights only; mutation kernel and classifier are identical across shards
    "train": {"train": 0.7, "val": 0.075, "predict": 0.075, "track": 0.075, "export": 0.075},
    "export": {"train": 0.075, "val": 0.075, "predict": 0.075, "track": 0.075, "export": 0.7},
    "predict-val": {"train": 0.05, "val": 0.35, "predict": 0.35, "track": 0.2, "export": 0.05},
    "chaos": {m: 0.2 for m in MODES},
}
STRATEGY_WEIGHTS = [("invalid", 0.4), ("combo", 0.4), ("source", 0.2)]

# Cost/hazard keys pinned to clamped known-good values, never mutated (`time` is training duration in HOURS)
NEVER_MUTATE = frozenset(
    {
        "model",
        "data",
        "epochs",
        "imgsz",
        "batch",
        "workers",
        "device",
        "source",
        "project",
        "name",
        "time",
        "resume",
        "cfg",
        "tracker",
        "show",
        "mode",
        "task",
        "format",
    }
)
CLAMPS = {
    "train": "imgsz=32 epochs=1 batch=4 workers=2 cache=disk",
    "val": "imgsz=32",
    "predict": "imgsz=32",
    "track": "imgsz=160",
    "export": "imgsz=32",
}
EXPORT_POOL = ["torchscript", "onnx", "openvino"]  # CPU-friendly formats installed on every shard

# Additional pretrained families with ordinary CLI contracts. Keep modes narrow to avoid prompt-only training paths.
ALTERNATE_CORPUS = (
    ("detect", "rtdetr-l.pt", "coco8.yaml", {"val", "predict", "export"}),
    ("detect", "yolov8s-worldv2.pt", "coco8.yaml", {"predict", "export"}),
    ("segment", "yoloe-11s-seg-pf.pt", "coco8-seg.yaml", {"predict", "export"}),
)

# Controlled variations for cost-sensitive keys excluded from arbitrary mutation.
SAFE_BOUNDARIES = {
    "train": ["imgsz=48", "imgsz=64", "batch=1", "batch=2", "workers=0", "workers=1"],
    "val": ["imgsz=48", "imgsz=64", "batch=1", "batch=2", "workers=0", "workers=1"],
    "predict": ["imgsz=48", "imgsz=64"],
    "track": ["imgsz=128", "imgsz=192", "vid_stride=2"],
    "export": ["imgsz=48", "imgsz=64", "batch=2"],
}

# Probe pools: "valid" values are supported inputs (deep failures are T1 bugs); "invalid" values are ones the
# cfg layer SHOULD reject — by current checks or by missing range checks — so deep failures are T2 validation
# gaps. Labels express that intent, not what check_cfg happens to accept today (e.g. it passes negative ints).
ENUM_POOLS = {
    "optimizer": {
        "valid": ["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "auto", "sgd"],  # case is canonicalized
        "invalid": ["Ranger", "", "none"],
    },
    "split": {"valid": ["val", "test", "train"], "invalid": ["trainval", "", "0.5"]},
    "cache": {"valid": ["True", "False", "ram", "disk"], "invalid": ["gpu", "1.5"]},
    "compile": {"valid": ["True", "False", "default", "reduce-overhead", "max-autotune"], "invalid": ["turbo"]},
    "auto_augment": {"valid": ["randaugment", "autoaugment", "augmix"], "invalid": ["randaug", ""]},
    "copy_paste_mode": {"valid": ["flip", "mixup"], "invalid": ["paste", ""]},
    "quantize": {"valid": ["fp16", "w8a8", "none"], "invalid": ["half", "int8_dynamic", "int4"]},
}
PROBES = {  # boundary and wrong-type probes per typed key family (values are CLI strings)
    "fraction": {"valid": ["0.0", "1.0", "0.5"], "invalid": ["-0.1", "1.5", "half", "True", "none"]},
    "int": {"valid": ["0", "1", "7"], "invalid": ["-1", "3.5", "ten", "none"]},
    "bool": {"valid": ["True", "False"], "invalid": ["yes", "1", "none"]},
    "float": {"valid": ["0.0", "0.1", "10"], "invalid": ["-5", "big", "none"]},
}
CHAOS_PROBES = ["[]", "[1,2]", "{}", "🚀", "1e309", "nan", "-0"]  # chaos shard extras for any key, all invalid

# Valid-but-rare combinations (mode, extra args) — where the T1 semantic bugs live
COMBO_POOL = [
    ("train", "rect=True"),
    ("train", "single_cls=True"),
    ("train", "cos_lr=True close_mosaic=0"),
    ("train", "fraction=0.5"),
    ("train", "freeze=10"),
    ("train", "multi_scale=0.5"),
    ("train", "optimizer=NAdam warmup_epochs=0"),
    ("train", "deterministic=False seed=7"),
    ("train", "overlap_mask=False mask_ratio=1"),
    ("train", "mosaic=0 mixup=1.0 cutmix=1.0"),
    ("train", "copy_paste=1.0 copy_paste_mode=mixup"),
    ("train", "hsv_h=1.0 hsv_s=0.0 degrees=180 perspective=0.001"),
    ("val", "save_json=True"),
    ("val", "split=train"),
    ("val", "end2end=True max_det=5"),
    ("val", "agnostic_nms=True conf=0.9 iou=0.1"),
    ("predict", "save_txt=True save_conf=True save_crop=True"),
    ("predict", "visualize=True"),
    ("predict", "augment=True"),
    ("predict", "classes=0"),
    ("predict", "classes=[0,2]"),
    ("predict", "retina_masks=True"),
    ("predict", "line_width=1 show_labels=False show_conf=False"),
    ("predict", "vid_stride=2 stream_buffer=True"),
    ("track", "tracker=bytetrack.yaml"),
    ("track", "tracker=botsort.yaml"),
    ("track", "tracker=fasttrack.yaml"),
    ("track", "save_txt=True save_conf=True"),
    ("track", "vid_stride=2 stream_buffer=True"),
    ("export", "dynamic=True"),
    ("export", "nms=True"),
    ("export", "simplify=False"),
    ("export", "opset=12"),
    ("export", "quantize=fp16"),
    ("export", "end2end=True max_det=10"),
]

# Oracle: expected clean errors are these types raised from the validation layers (modules or exact frames)
EXPECTED_TYPES = {"SyntaxError", "ValueError", "TypeError", "AssertionError", "FileNotFoundError"}
EXPECTED_MODULES = (
    "ultralytics/cfg/__init__.py",
    "ultralytics/utils/checks.py",
    "ultralytics/data/utils.py",
    "ultralytics/data/augment.py:classify_augmentations",
    "ultralytics/engine/exporter.py:validate_args",  # exporter's intentional per-format argument validation
    "ultralytics/engine/exporter.py:__call__",  # intentional compat asserts; per-format bugs raise in deeper frames
)
NETWORK_MARKERS = (  # specific download/network signatures only; bare ConnectionError is raised for local sources too
    "urlopen error",
    "Read timed out",
    "Download failure",
    "HTTPError",
    "requests.exceptions",
    "name resolution",
    "getaddrinfo",
    "RemoteDisconnected",
    "SSLError",
)


def load_universe():
    """Lazily import the arg universe from the ultralytics package (kept out of module scope for `report`)."""
    from ultralytics.cfg import (
        CFG_BOOL_KEYS,
        CFG_FLOAT_KEYS,
        CFG_FRACTION_KEYS,
        CFG_INT_KEYS,
        TASK2DATA,
        TASK2MODEL,
        TASKS,
    )
    from ultralytics.utils import ASSETS, DEFAULT_CFG_DICT, LOGGER

    return {
        "tasks": sorted(TASKS),
        "task2model": TASK2MODEL,
        "task2data": TASK2DATA,
        "defaults": DEFAULT_CFG_DICT,
        "fraction_keys": sorted(CFG_FRACTION_KEYS - NEVER_MUTATE),
        "int_keys": sorted(CFG_INT_KEYS - NEVER_MUTATE),
        "bool_keys": sorted(CFG_BOOL_KEYS - NEVER_MUTATE),
        "float_keys": sorted(CFG_FLOAT_KEYS - NEVER_MUTATE - CFG_FRACTION_KEYS),
        "enum_keys": sorted(set(ENUM_POOLS) - NEVER_MUTATE),
        "source": str(ASSETS / "bus.jpg"),
        "export_pool": [*EXPORT_POOL, *(["coreml"] if importlib.util.find_spec("coremltools") else [])],
        "logger": LOGGER,
    }


def precache_assets(uni):
    """Download the corpus weights and datasets once into the shared caches (no-op when already present)."""
    from ultralytics.data.utils import check_cls_dataset, check_det_dataset
    from ultralytics.utils import WEIGHTS_DIR
    from ultralytics.utils.downloads import attempt_download_asset

    for task in uni["tasks"]:
        attempt_download_asset(WEIGHTS_DIR / uni["task2model"][task])
        data = uni["task2data"][task]
        check_cls_dataset(data) if str(data).startswith("imagenet") else check_det_dataset(data, autodownload=True)
    for _task, model, _data, _modes in ALTERNATE_CORPUS:
        attempt_download_asset(WEIGHTS_DIR / model)

    prepare_sources(uni)


def prepare_sources(uni):
    """Create cached valid and malformed media sources used by predict and track trials."""
    from ultralytics.utils import ASSETS, ASSETS_URL, WEIGHTS_DIR
    from ultralytics.utils.downloads import safe_download

    source_dir = WEIGHTS_DIR.parent / "fuzz-sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    unicode_image = source_dir / "path with spaces 🚀.jpg"
    shutil.copy2(ASSETS / "bus.jpg", unicode_image)
    empty_image, corrupt_image, empty_dir = source_dir / "empty.jpg", source_dir / "corrupt.jpg", source_dir / "empty"
    empty_image.touch()
    corrupt_image.write_bytes(b"not an image")
    empty_dir.mkdir(exist_ok=True)
    video = source_dir / "decelera_portrait_min.mov"
    safe_download(f"{ASSETS_URL}/{video.name}", file=video)
    uni["sources"] = {
        "predict": [
            (str(ASSETS / "bus.jpg"), True),
            (str(ASSETS / "zidane.jpg"), True),
            (str(ASSETS), True),
            (str(unicode_image), True),
            (str(video), True),
            (str(empty_image), False),
            (str(corrupt_image), False),
            (str(empty_dir), False),
        ],
        "track": [
            (str(video), True),
            (str(unicode_image), True),
            (str(empty_image), False),
            (str(corrupt_image), False),
            (str(empty_dir), False),
        ],
    }
    uni["video"] = str(video)


def strip_defaults(pairs, defaults):
    """Drop k=v args whose value equals the package default: argv only ever carries changed args."""
    return [a for a in pairs if a.partition("=")[2].lower() != str(defaults.get(a.partition("=")[0])).lower()]


def build_corpus(uni):
    """Build the known-good seed corpus: every task x mode with clamped fast knobs and explicit local source."""
    corpus = []
    for task in uni["tasks"]:
        # Bare weight names keep issue repro commands portable; they resolve against the precached weights_dir
        model, data = uni["task2model"][task], uni["task2data"][task]
        for mode in MODES:
            if mode == "track" and task in {"classify", "semantic"}:
                continue
            argv = [mode, task, f"model={model}", *strip_defaults(CLAMPS[mode].split(), uni["defaults"])]
            if mode in {"train", "val"}:
                argv.append(f"data={data}")
            elif mode == "predict":
                argv.append(f"source={uni['source']}")
            elif mode == "track":
                argv.append(f"source={uni['video']}")
            corpus.append({"mode": mode, "task": task, "argv": argv})  # export: default torchscript stays implicit
    for task, model, data, modes in ALTERNATE_CORPUS:
        for mode in modes:
            argv = [mode, task, f"model={model}", *strip_defaults(CLAMPS[mode].split(), uni["defaults"])]
            if mode == "val":
                argv.append(f"data={data}")
            elif mode == "predict":
                argv.append(f"source={uni['source']}")
            corpus.append({"mode": mode, "task": task, "argv": argv})
    return corpus


def sample_trial(rng, uni, corpus, personality):
    """Sample one trial with canonical arguments from invalid, valid-combination, or source strategies."""
    weights = PERSONALITIES[personality]
    mode = rng.choices(MODES, weights=[weights[m] for m in MODES])[0]
    base = rng.choice([c for c in corpus if c["mode"] == mode])
    argv, mutated = list(base["argv"]), []
    strategies = STRATEGY_WEIGHTS if mode in {"predict", "track"} else STRATEGY_WEIGHTS[:2]
    strategy = rng.choices([s for s, _ in strategies], weights=[w for _, w in strategies])[0]

    validity = {}  # key -> is its EFFECTIVE value supported: stripped args contribute nothing, duplicates last-win

    def mutate(pairs, valid=True):
        """Append the non-default k=v pairs to argv and record their keys as mutated."""
        for a in pairs:
            key, _, value = a.partition("=")
            argv[:] = [x for x in argv if x.partition("=")[0] != key]  # canonical last-value-wins CLI semantics
            if value.lower() != str(uni["defaults"].get(key)).lower():
                argv.append(a)
            if key not in mutated:
                mutated.append(key)
            validity[key] = valid

    def mutate_combos(max_groups=4):
        """Combine compatible mode-specific argument groups without repeating keys."""
        options = [c.split() for m, c in COMBO_POOL if m == mode]
        rng.shuffle(options)
        used, target = set(), rng.randint(1, max_groups)
        for combo in options:
            keys = {a.partition("=")[0] for a in combo}
            if keys.isdisjoint(used):
                mutate(combo)
                used.update(keys)
                target -= 1
            if not target:
                break

    if mode == "export":  # fuzz the format from the installable pool; the default torchscript stays implicit
        mutate([f"format={rng.choice(uni['export_pool'])}"])
    if strategy == "combo":
        mutate_combos()
        mutate([rng.choice(SAFE_BOUNDARIES[mode])])
    elif strategy == "invalid":
        n_keys = rng.randint(1, 4 if personality == "chaos" else 3)
        for _ in range(n_keys):
            key, value, valid = sample_mutation(rng, uni, chaos=personality == "chaos")
            mutate([f"{key}={value}"], valid=valid)
    else:
        source, valid = rng.choice(uni["sources"][mode])
        mutate([f"source={source}"], valid=valid)
        mutate_combos(max_groups=2)
        mutate([rng.choice(SAFE_BOUNDARIES[mode])])
    return {
        "mode": mode,
        "task": base["task"],
        "argv": argv,
        "strategy": strategy,
        "mutated": mutated,
        "valid_input": all(validity.values()),
    }


def sample_mutation(rng, uni, chaos=False):
    """Pick one fuzzable key, a probe value, and whether that value is documented-valid for the key."""
    family = rng.choices(["enum", "fraction", "int", "bool", "float"], weights=[4, 2, 2, 1, 1])[0]
    if family == "enum":
        key = rng.choice(uni["enum_keys"])
        pool = ENUM_POOLS[key]
    else:
        key = rng.choice(uni[f"{family}_keys"])
        pool = PROBES[family]
    if family in {"fraction", "int", "float"} and rng.random() < 0.5:
        valid = rng.random() < 0.5
        if family == "fraction":
            value = f"{rng.uniform(0.000001, 0.999999):.8g}" if valid else f"{rng.uniform(1.000001, 4):.8g}"
        elif family == "int":
            value = str(rng.randint(1, 4096)) if valid else f"{rng.randint(0, 4096)}.5"
        else:
            value = f"{rng.uniform(0, 180):.8g}" if valid else rng.choice(["nan", "1e309"])
        return key, value, valid
    value = rng.choice(pool["valid"] + pool["invalid"] + (CHAOS_PROBES if chaos else []))
    return key, value, value in pool["valid"] and not (family == "fraction" and value == "0.0")


def run_trial(trial, timeout=None):
    """Execute one trial in an isolated tmp workdir; outputs go to a per-trial `project=`, assets stay shared."""
    mode = trial["mode"]
    timeout = (timeout or MODE_TIMEOUTS[mode]) * TIMEOUT_SCALE
    workdir = Path(tempfile.mkdtemp(prefix="fuzz-trial-"))
    argv = list(trial["argv"])
    if mode == "export":  # exports write beside the model file: copy the weight in so shared weights_dir stays clean
        src = Path(next(a for a in argv if a.startswith("model=")).split("=", 1)[1])
        if not src.exists():  # bare weight name: resolve via the package downloader (no-op on the precached cache)
            from ultralytics.utils import WEIGHTS_DIR
            from ultralytics.utils.downloads import attempt_download_asset

            src = Path(attempt_download_asset(WEIGHTS_DIR / src.name))
        local = workdir / src.name
        shutil.copy2(src, local)
        argv = [a if not a.startswith("model=") else f"model={local}" for a in argv]
    else:
        argv.append(f"project={workdir / 'runs'}")
    from ultralytics.cfg import _YOLO_CLI_COMMAND  # same invocation trainer/tuner use to respawn the CLI

    cmd = [*_YOLO_CLI_COMMAND, *argv]
    env = {**os.environ, "YOLO_AUTOINSTALL": "false", "PYTHONFAULTHANDLER": "1"}
    t0 = time.perf_counter()
    # Own session/group so a timeout kills the whole tree (dataloader workers, export converter subprocesses)
    group = (
        {"start_new_session": True} if os.name == "posix" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    )
    proc = subprocess.Popen(
        cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, **group
    )
    try:
        _, stderr = proc.communicate(timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        _, stderr = proc.communicate()
        rc, stderr = "timeout", stderr or ""
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return rc, stderr, round(time.perf_counter() - t0, 1)


def parse_traceback(stderr):
    """Parse the last traceback block from stderr into (exception type, [ultralytics frames as path:function])."""
    blocks = stderr.split("Traceback (most recent call last):")
    if len(blocks) < 2:
        return None, []
    block = blocks[-1]
    frames = []
    for path, func in re.findall(r'File "([^"]+)", line \d+, in (\S+)', block):
        if "ultralytics" in path:
            norm = path.replace("\\", "/")
            frames.append(f"{norm[norm.rindex('ultralytics/') :]}:{func}")
    exc = None
    for line in reversed(block.strip().splitlines()):
        m = re.match(r"^([A-Za-z_][\w.]*(?:Error|Exception|Warning|Interrupt|Exit|SyntaxError))\b", line.strip())
        if m:
            exc = m.group(1).split(".")[-1]
            break
    return exc, frames


def make_signature(exc, frames, mode, task):
    """Build a stable dedup signature: exception type + deepest ultralytics frame + its caller (no line numbers)."""
    deepest = frames[-1] if frames else f"{mode}:{task}"
    caller = frames[-2] if len(frames) > 1 else ""
    human = f"{exc or 'Unknown'} in {deepest}"
    return hashlib.sha256(f"{exc}|{deepest}|{caller}".encode()).hexdigest()[:12], human


def classify(trial, rc, stderr):
    """Classify one trial outcome into pass/expected/env-skip/flake/timeout/crash/bug-candidate."""
    if rc == 0:
        return "pass", None, None
    if rc == "timeout":
        keys = set(trial.get("mutated", []))  # exact mutated k=v pairs: distinct hangs get distinct signatures
        mutated = "|".join(sorted(a for a in trial["argv"] if a.partition("=")[0] in keys)) or "baseline"
        sig = hashlib.sha256(f"Timeout|{trial['mode']}|{trial['task']}|{mutated}".encode()).hexdigest()[:12]
        return "timeout", sig, f"Timeout in yolo {trial['mode']} ({trial['task']})"
    exc, frames = parse_traceback(stderr)
    if isinstance(rc, int) and rc < 0:
        sig, human = make_signature(f"Signal{-rc}", frames, trial["mode"], trial["task"])
        return "crash", sig, human
    missing = re.search(r"No module named '(\w+)", stderr)
    if missing and not importlib.util.find_spec(missing.group(1)):  # module genuinely absent: optional-dep skip
        return "env-skip", None, None
    if any(marker in stderr for marker in NETWORK_MARKERS):
        return "flake", None, None
    if trial.get("mutated") and (
        # Intentional unsupported-choice errors are expected; abstract "not implemented" gaps keep their signatures
        (exc == "NotImplementedError" and re.search(r"not supported|(?:doesn't|does not) support", stderr))
        or (exc == "NotImplementedError" and "not found in list of available optimizers" in stderr)
        or (exc == "ValueError" and "Expected `mode` to be `flip` or `mixup`" in stderr)
        or (exc in EXPECTED_TYPES and frames and frames[-1].startswith(EXPECTED_MODULES))
    ):
        return "expected", None, None  # clean validation errors are expected only for trials we actually mutated
    sig, human = make_signature(exc, frames, trial["mode"], trial["task"])
    return "bug-candidate", sig, human


def stderr_tail(stderr, lines=30):
    """Return the last meaningful lines of stderr for logs and issue bodies."""
    return "\n".join(stderr.strip().splitlines()[-lines:])


def cmd_fuzz(args):
    """Run the budgeted fuzzing loop and write trials JSONL + findings JSON for the report job."""
    uni = load_universe()
    log = uni["logger"]
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trials_path = out / f"trials-{args.personality}.jsonl"
    log.info(f"[fuzz] personality={args.personality} seed={args.seed} budget={args.budget_minutes}min")
    from ultralytics.utils.checks import collect_system_info

    environment = {k: str(v) for k, v in collect_system_info().items()}  # before fuzzing: fail fast, never lose trials
    precache_assets(uni)
    corpus = build_corpus(uni)

    deadline = time.time() + args.budget_minutes * 60
    counters = {k: 0 for k in ("pass", "expected", "env-skip", "flake", "timeout", "crash", "bug-candidate")}
    findings, seen, canary_results, hang_confirms, n = {}, set(), [], 0, 0

    def execute(trial, canary=False):
        nonlocal n, hang_confirms
        rc, stderr, duration = run_trial(trial, timeout=args.debug_timeout)
        outcome, sig, human = classify(trial, rc, stderr)
        if outcome == "flake":  # network flakes get one retry, then are discarded
            rc, stderr, duration = run_trial(trial, timeout=args.debug_timeout)
            outcome, sig, human = classify(trial, rc, stderr)
        confirmed = False
        tier = "T2" if outcome == "bug-candidate" and not trial.get("valid_input", True) else "T1"

        def finding():
            """Build the finding record from this occurrence's fresh trial, outcome, and traceback."""
            return {
                "signature": sig,
                "title": human,
                "tier": tier,
                "outcome": outcome,
                "mode": trial["mode"],
                "task": trial["task"],
                "strategy": trial.get("strategy", "corpus"),
                "command": "yolo " + shlex.join(trial["argv"]),
                "stderr_tail": stderr_tail(stderr),
                "duration_s": duration,
            }

        if sig and sig in findings and tier == "T1" and findings[sig]["tier"] == "T2":
            # a valid-input occurrence proves a confirmed validation-gap signature is a real bug: confirm, then upgrade
            rc2, stderr2, _ = run_trial(trial, timeout=args.debug_timeout)
            outcome2, sig2, _ = classify(trial, rc2, stderr2)
            if outcome2 == outcome and sig2 == sig:
                findings[sig] = finding()
        if sig and sig not in seen:  # confirm only the first occurrence of each signature
            seen.add(sig)
            if outcome == "timeout":
                if hang_confirms < MAX_HANG_CONFIRMS:
                    hang_confirms += 1
                    rc2, _, _d2 = run_trial(trial, timeout=CONFIRM_TIMEOUT)
                    confirmed = rc2 == "timeout"
                    if not confirmed:
                        outcome = "pass" if rc2 == 0 else outcome  # completed quickly on replay: not a hang
            else:
                rc2, stderr2, _ = run_trial(trial, timeout=args.debug_timeout)
                outcome2, sig2, _ = classify(trial, rc2, stderr2)
                confirmed = outcome2 == outcome and sig2 == sig  # same failure, not merely the same class
            if not confirmed:  # flaky/capped confirmations may still be real failures: let later occurrences retry
                seen.discard(sig)
            if confirmed:
                findings[sig] = finding()
        counters[outcome] += 1
        if canary:
            canary_results.append(outcome == "pass")
        n += 1
        with trials_path.open("a") as f:
            record = {
                "n": n,
                "outcome": outcome,
                "duration_s": duration,
                "signature": sig,
                "canary": canary,
                **{k: trial[k] for k in ("mode", "task", "argv")},
                "strategy": trial.get("strategy", "corpus"),
            }
            f.write(json.dumps(record) + "\n")
        changed = " ".join(a for a in trial["argv"] if a.partition("=")[0] in set(trial.get("mutated", [])))
        log.info(f"[fuzz] #{n} {outcome:>13} {duration:6.1f}s  yolo {trial['mode']} {trial['task']} {changed}".rstrip())

    executed, duplicate_samples = set(), 0
    for base in corpus:  # canaries first: unmutated corpus must pass or the environment itself is broken
        if time.time() > deadline or (args.max_trials and n >= args.max_trials):
            break
        executed.add(tuple(base["argv"]))
        execute(dict(base), canary=True)
    while time.time() < deadline and (not args.max_trials or n < args.max_trials):
        if shutil.disk_usage(tempfile.gettempdir()).free < MIN_FREE_GB * 1024**3:
            log.warning(f"[fuzz] stopping early: <{MIN_FREE_GB}GB free disk")
            break
        trial = sample_trial(rng, uni, corpus, args.personality)
        command = tuple(trial["argv"])
        if command in executed:
            duplicate_samples += 1
            continue
        executed.add(command)
        execute(trial)

    infra_failed = bool(canary_results) and (canary_results.count(False) / len(canary_results)) > CANARY_FAIL_FRACTION
    summary = {
        "personality": args.personality,
        "seed": args.seed,
        "trials": n,
        "unique_commands": len(executed),
        "duplicate_samples": duplicate_samples,
        "counters": counters,
        "infra_failed": infra_failed,
        "findings": sorted(findings.values(), key=lambda x: (x["tier"], x["signature"])),
        "environment": environment,
    }
    (out / f"findings-{args.personality}.json").write_text(json.dumps(summary, indent=2))
    log.info(
        f"[fuzz] done: {n} trials {counters}, {len(findings)} confirmed unique findings, infra_failed={infra_failed}"
    )


def cmd_repro(args):
    """Replay one exact command several times through the classifier and print the verdict."""
    uni = load_universe()
    log = uni["logger"]
    argv = shlex.split(args.command)
    if argv and argv[0] == "yolo":  # issue bodies quote full `yolo ...` commands; accept them verbatim
        argv = argv[1:]
    mode = next((a for a in argv if a in MODES), "predict")

    def portable(arg):
        """Remap runner-local absolute model/source paths from issue commands to this machine's copies."""
        k, _, v = arg.partition("=")
        if k in {"model", "source"} and ("/" in v or "\\" in v) and not Path(v).exists():
            from ultralytics.utils import ASSETS, WEIGHTS_DIR

            # PureWindowsPath splits on both separators, so Windows-origin issue commands remap on any OS.
            if k == "model":
                candidates = [WEIGHTS_DIR / PureWindowsPath(v).name]
            else:
                prepare_sources(uni)
                candidates = [ASSETS / PureWindowsPath(v).name]
                candidates.extend(Path(p) for p, _valid in uni["sources"][mode])
            if local := next((p for p in candidates if p.name == PureWindowsPath(v).name and p.exists()), None):
                return f"{k}={local}"
        return arg

    argv = [portable(a) for a in argv]
    task = next((a for a in argv if a in uni["tasks"]), "detect")
    trial = {"mode": mode, "task": task, "argv": argv, "mutated": ["repro"]}  # replayed commands were fuzz-mutated
    outcomes = []
    for i in range(args.runs):
        rc, stderr, duration = run_trial(trial, timeout=args.debug_timeout)
        outcome, _sig, human = classify(trial, rc, stderr)
        outcomes.append(outcome)
        log.info(f"[repro] run {i + 1}/{args.runs}: {outcome} ({duration}s) {human or ''}")
        if outcome not in {"pass", "expected"}:
            log.info(stderr_tail(stderr))
    reproduces = all(o == outcomes[0] for o in outcomes) and outcomes[0] not in {"pass", "expected"}
    log.info(f"[repro] verdict: {'REPRODUCES as ' + outcomes[0] if reproduces else 'does not reproduce'}")
    sys.exit(1 if reproduces else 0)


def gh(*cli_args, dry_run=False):
    """Run a `gh` CLI command and return stdout (prints instead when dry_run)."""
    if dry_run:
        print(f"[dry-run] gh {' '.join(cli_args)}")
        return ""
    return subprocess.run(["gh", *cli_args], capture_output=True, text=True, check=True).stdout


def cmd_report(args):
    """Aggregate shard findings, dedup against existing issues by signature, and file at most --max-issues issues."""
    in_dir = Path(args.in_dir)
    shards = [json.loads(p.read_text()) for p in sorted(in_dir.glob("findings-*.json"))]
    if not shards:
        print("[report] no findings files found")
        return
    run_url = f"https://github.com/{args.repo}/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}"
    findings, counters, flagged, unique_commands, duplicate_samples = {}, {}, [], 0, 0
    for shard in shards:
        for k, v in shard["counters"].items():
            counters[k] = counters.get(k, 0) + v
        if shard["infra_failed"]:  # warn only: a regression tripping the canaries IS the finding, never discard it
            flagged.append(shard["personality"])
        unique_commands += shard.get("unique_commands", shard["trials"])
        duplicate_samples += shard.get("duplicate_samples", 0)
        for f in shard["findings"]:
            prev = findings.get(f["signature"])
            if not prev or (prev["tier"] == "T2" and f["tier"] == "T1"):  # prefer the T1 view of a shared signature
                findings[f["signature"]] = {**f, "environment": shard["environment"], "seed": shard["seed"]}

    existing = {}
    if not args.dry_run:
        gh(
            "label",
            "create",
            "fuzz",
            "--description",
            "Found by the scheduled Fuzz workflow",
            "--color",
            "8B5CF6",
            "--repo",
            args.repo,
            "--force",
        )
        issues = json.loads(
            gh(
                "issue",
                "list",
                "--repo",
                args.repo,
                "--label",
                "fuzz",
                "--state",
                "all",
                "--limit",
                "500",
                "--json",
                "number,state,title,body",
            )
            or "[]"
        )
    umbrella = None
    if not args.dry_run:
        umbrella = next((i for i in issues if i["title"].startswith("Fuzz: CLI validation gaps")), None)
        for issue in issues:
            if umbrella and issue["number"] == umbrella["number"]:
                continue  # umbrella signatures are collected below with setdefault so standalone issues always win
            for sig in re.findall(r"fuzz-signature: (\w+)", issue.get("body") or ""):
                existing[sig] = issue
    if umbrella:  # T2 signatures from prior runs live in the umbrella body and comments
        view = json.loads(gh("issue", "view", str(umbrella["number"]), "--repo", args.repo, "--json", "comments"))
        for body in [umbrella.get("body") or ""] + [c.get("body") or "" for c in view.get("comments", [])]:
            for sig in re.findall(r"fuzz-signature: (\w+)", body):
                existing.setdefault(sig, umbrella)  # a standalone issue for the same signature takes precedence

    def only_umbrella(s):
        """True when a signature is known only as a T2 umbrella entry, so a T1 sighting still deserves its own issue."""
        return umbrella and s in existing and existing[s]["number"] == umbrella["number"]

    new_t1 = [f for s, f in findings.items() if f["tier"] == "T1" and (s not in existing or only_umbrella(s))]
    new_t2 = [f for s, f in findings.items() if f["tier"] == "T2" and s not in existing]
    regressions = [(f, existing[s]) for s, f in findings.items() if s in existing and existing[s]["state"] == "CLOSED"]

    created = 0
    for f in new_t1:
        if created >= args.max_issues:
            print(f"[report] issue cap ({args.max_issues}) reached; {len(new_t1) - created} T1 findings deferred")
            break
        body = issue_body(f, run_url)
        title = f"yolo {f['mode']}: {f['title']}"
        print(f"[report] filing T1 issue: {title}")
        gh(
            "issue",
            "create",
            "--repo",
            args.repo,
            "--title",
            title,
            "--body",
            body,
            "--label",
            "bug,fuzz",
            dry_run=args.dry_run,
        )
        created += 1

    if new_t2:  # T2 validation gaps roll up into one umbrella issue; only its creation counts against the cap
        lines = [f"- `{f['command']}` → {f['title']} `<!-- fuzz-signature: {f['signature']} -->`" for f in new_t2]
        comment = f"New CLI validation gaps found by [fuzzing]({run_url}):\n\n" + "\n".join(lines)
        if umbrella and umbrella["state"] == "OPEN":
            gh(
                "issue",
                "comment",
                str(umbrella["number"]),
                "--repo",
                args.repo,
                "--body",
                comment,
                dry_run=args.dry_run,
            )
        elif umbrella:  # a closed umbrella means T2 reporting was deliberately opted out: summary only
            print(f"[report] umbrella issue #{umbrella['number']} is closed; {len(new_t2)} T2 gaps in summary only")
        elif created < args.max_issues:
            gh(
                "issue",
                "create",
                "--repo",
                args.repo,
                "--title",
                "Fuzz: CLI validation gaps (rolling)",
                "--body",
                "Deep tracebacks from invalid CLI input, found by scheduled fuzzing. Each should raise "
                "a clean error from the cfg layer instead.\n\n" + comment + "\n<!-- fuzz-signature: umbrella -->",
                "--label",
                "bug,fuzz",
                dry_run=args.dry_run,
            )
            created += 1

    by_issue = {}
    for f, issue in regressions:
        if umbrella and issue["number"] == umbrella["number"]:
            continue  # known T2 signatures in a closed umbrella are dedup state, not per-bug regression signals
        by_issue.setdefault(issue["number"], []).append(f)
    for number, group in by_issue.items():  # one comment per issue per run, and never twice for the same signature
        commented = ""
        if not args.dry_run:
            view = json.loads(gh("issue", "view", str(number), "--repo", args.repo, "--json", "comments") or "{}")
            commented = " ".join(c.get("body") or "" for c in view.get("comments", []))
        fresh = [f for f in group if f"fuzz-regression: {f['signature']}" not in commented]
        if not fresh:
            continue
        blocks = "\n\n".join(f"```bash\n{f['command']}\n```\n<!-- fuzz-regression: {f['signature']} -->" for f in fresh)
        gh(
            "issue",
            "comment",
            str(number),
            "--repo",
            args.repo,
            "--body",
            f"Reproduced again by [fuzzing]({run_url}) after this issue was closed — possible regression.\n\n{blocks}",
            dry_run=args.dry_run,
        )

    total = sum(counters.values())
    table = ["| Outcome | Count |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sorted(counters.items())]
    summary = (
        f"## Fuzz — {total} trials\n\n"
        + "\n".join(table)
        + f"\n\nExploration: {unique_commands} unique commands · {duplicate_samples} duplicate samples skipped"
        + f"\n\nNew issue threads created: {created} (cap {args.max_issues})"
        + (f" · ⚠️ shards with >20% canary failures: {', '.join(flagged)}" if flagged else "")
    )
    if step_summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(step_summary).write_text(summary)
    if gh_output := os.environ.get("GITHUB_OUTPUT"):
        with Path(gh_output).open("a") as f:
            f.write(f"new_issues={created}\n")
    print(summary)


def issue_body(f, run_url):
    """Format the GitHub issue body for one confirmed T1 finding."""
    env = "\n".join(f"{k}: {v}" for k, v in f["environment"].items())
    return f"""Automated fuzzing found a reproducible failure (confirmed 2/2 runs).

### Reproduce

```bash
{f["command"]}
```

### Details

- Outcome: `{f["outcome"]}` · strategy: `{f["strategy"]}` · task: `{f["task"]}` · seed: `{f["seed"]}`
- Run: {run_url}

### Traceback (tail)

```
{f["stderr_tail"]}
```

### Environment

```
{env}
```

<!-- fuzz-signature: {f["signature"]} -->
"""


def main():
    """Parse arguments and dispatch to the fuzz/repro/report subcommands."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fuzz", help="run a budgeted fuzzing loop")
    p.add_argument("--budget-minutes", type=float, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--personality", choices=sorted(PERSONALITIES), default="chaos")
    p.add_argument("--out", default="fuzz-out")
    p.add_argument("--max-trials", type=int, default=0, help="optional hard trial cap (smoke tests)")
    p.add_argument("--debug-timeout", type=float, default=None, help="override all trial timeouts (test hang path)")

    p = sub.add_parser("repro", help="replay one exact command and classify it")
    p.add_argument("command", help='yolo args, e.g. "train detect model=yolo26n.pt data=coco8.yaml epochs=abc"')
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--debug-timeout", type=float, default=None)

    p = sub.add_parser("report", help="aggregate shard findings and file GitHub issues (stdlib only)")
    p.add_argument("--in", dest="in_dir", default="fuzz-out")
    p.add_argument("--max-issues", type=int, default=3, help="hard cap on `gh issue create` calls per run")
    p.add_argument("--repo", default="ultralytics/ultralytics")
    p.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    {"fuzz": cmd_fuzz, "repro": cmd_repro, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
