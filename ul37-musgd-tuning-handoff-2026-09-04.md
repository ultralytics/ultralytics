# UL37 YOLO27n MuSGD tuning handoff

Updated: 2026-09-04. Canonical tracking is [Portal issue 3849](https://github.com/ultralytics/portal/issues/3849). Benchmark results are in [yolo27-benchmarks issue 10](https://github.com/ultralytics/yolo27-benchmarks/issues/10).

## Objective

Tune YOLO27n on the revised oil-storage UL37 basket with a fixed MuSGD optimizer. Start from the best prior AdamW candidate, iteration 432, while keeping the prior campaign's batch and memory-format choices fixed so optimizer is the principal search change.

This is a new objective. Never insert the historical AdamW fitness into the new MongoDB collection.

## Current state

The Vast instance is running five persistent tuning workers:

```text
vast-gpu0
vast-gpu4
vast-gpu5
vast-gpu6
vast-gpu7
```

GPUs 1–3 are finishing the YOLO26m/l/x benchmark jobs. When each benchmark has finished, its result JSON contains a full wall time, and its process has exited, add that GPU to tuning. The intended steady state is eight Vast workers, one per GPU.

The cancelled YOLO27s/m/l/x checkpoints are held under:

```text
/workspace/ul37-yolo27/held-yolo27-scale-checkpoints
```

Do not move them back into the launch path or restart the scale queue.

Vast instance: `48672632`, with 8× RTX PRO 6000 Blackwell Server Edition GPUs. Access it through Vast → Applications → Jupyter Terminal. The endpoint and token are ephemeral and must not be committed.

## Immutable campaign contract

| Item                            | Value                                                              |
| ------------------------------- | ------------------------------------------------------------------ |
| Model                           | YOLO27n                                                            |
| Model SHA-256                   | `cf3efad73ea0fe44ab89c2889eb87e45e6a9daad20129fc62bcbcecb6218bedc` |
| Source                          | `exp27-clean`                                                      |
| Source commit                   | `e7b3673925e2fc013c2c39515ad8cfea445f83b6`                         |
| Basket                          | Revised oil-storage UL37                                           |
| Manifest SHA-256                | `b1fb5f8f098078a2d495d0e33c65967b56f057e85ab89bdc89afe82dfa7bde0d` |
| Sorted dataset-identity SHA-256 | `4852269ebfc9dd677e4d0d26aa0738e9303b645893602749da5f6ae1588210ba` |
| Oil NDJSON SHA-256              | `075b5971e44af73635a87e7cd8436b95a6a0c3ffb395e9383b5124d21f8eccaf` |
| Optimizer                       | Explicit `MuSGD`                                                   |
| Batch                           | `128`                                                              |
| Channels last                   | `True`                                                             |
| MongoDB database                | `Tuner`                                                            |
| MongoDB collection              | `tune-yolo27n-ul37-oil-v1-b128-musgd`                              |
| Shared target                   | 500 completed results plus valid in-flight overshoot               |
| Search dimensions               | 25                                                                 |
| Seed                            | Historical AdamW iteration 432 plus `warmup_momentum=0.8`          |

Every worker must also use:

```text
epochs=100
patience=100
imgsz=640
fraction=[1000, 1.0, 0]
deterministic=True
plots=False
save=False
val=False
```

The manifest must contain exactly 37 unique existing YAML files, exactly one `airbus-oil-storage-detection` entry, and no `airbus-aircraft-detection` entry. `ULTRALYTICS_API_KEY` must be absent; all datasets are local.

A failed, missing, or non-finite dataset result contributes zero fitness and remains represented in the 37-dataset result map. Never drop a dataset or change the denominator.

## Seed

Iteration 432 scored `0.4729362162162163` in the historical aircraft-basket AdamW campaign. That fitness is provenance only and is not comparable to the oil-storage objective.

```yaml
lr0: 0.0017
lrf: 0.02486
momentum: 0.75337
weight_decay: 0.00079
warmup_epochs: 4.09896
warmup_momentum: 0.8
box: 9.35812
cls: 1.39302
cls_pw: 0.64446
dfl: 0.89109
hsv_h: 0.0113
hsv_s: 0.66035
hsv_v: 0.5454
degrees: 0.0
translate: 0.08624
scale: 0.453
shear: 0.05067
perspective: 0.00006
flipud: 0.15953
fliplr: 0.22928
bgr: 0.06683
mosaic: 0.72661
mixup: 0.33404
cutmix: 0.46818
close_mosaic: 0
```

`warmup_momentum` was absent and inert in the AdamW search because AdamW parameter groups expose `betas`, not a literal `momentum` field. MuSGD parameter groups do expose `momentum`, so restore this dimension and seed it at the repository default of `0.8`.

## Search space

Use exactly these 25 dimensions. Keep detection-inert `copy_paste` excluded.

```python
TUNE_SPACE = {
    "lr0": (1e-5, 1e-2),
    "lrf": (0.01, 1.0),
    "momentum": (0.7, 0.98, 0.3),
    "weight_decay": (0.0, 0.001),
    "warmup_epochs": (0.0, 5.0),
    "warmup_momentum": (0.0, 0.95),
    "box": (1.0, 20.0),
    "cls": (0.1, 4.0),
    "cls_pw": (0.0, 1.0),
    "dfl": (0.4, 12.0),
    "hsv_h": (0.0, 0.1),
    "hsv_s": (0.0, 0.9),
    "hsv_v": (0.0, 0.9),
    "degrees": (0.0, 45.0),
    "translate": (0.0, 0.9),
    "scale": (0.0, 0.95),
    "shear": (0.0, 10.0),
    "perspective": (0.0, 0.001),
    "flipud": (0.0, 1.0),
    "fliplr": (0.0, 1.0),
    "bgr": (0.0, 1.0),
    "mosaic": (0.0, 1.0),
    "mixup": (0.0, 1.0),
    "cutmix": (0.0, 1.0),
    "close_mosaic": (0.0, 10.0),
}
```

The optimizer is fixed and is not part of the search space.

## Authoritative Vast paths

```text
Python:       /workspace/ul37-yolo27/.venv/bin/python
Source:       /workspace/ul37-yolo27/source-exp27-clean
Model:        /workspace/ul37-yolo27/yolo27n.pt
Campaign:     /workspace/ul37-oil-experiments
Manifest:     /workspace/ul37-oil-experiments/ul37-local-datasets-oil-v1.txt
Worker:       /workspace/ul37-oil-experiments/run_ul37_musgd_tuner_worker.py
Launcher:     /workspace/ul37-oil-experiments/launch_ul37_musgd_workers.py
Worker state: /workspace/ul37-oil-experiments/musgd-workers.json
Logs:         /workspace/ul37-oil-experiments/logs-tune-musgd
Runs:         /workspace/ul37-oil-experiments/runs-tune-musgd
```

Artifact identities:

```text
3a2d2b1cf6186fc5891f9b4f45d418c7f02cd5399b0999db47545b862a22af76  run_ul37_musgd_tuner_worker.py
f4894e47745985bd1b33c8f7b0910a2f94bb7726d5e2f84cdd9dfa385819f4b3  launch_ul37_musgd_workers.py
```

The worker accepts these path overrides for another machine:

```text
UL37_ROOT
UL37_CAMPAIGN
UL37_MANIFEST
UL37_MANIFEST_SHA256
UL37_MODEL
ULTRALYTICS_SOURCE
```

Do not change the collection, model hash, source commit, seed, search space, or training arguments.

## Add the remaining Vast GPUs

First prove that the YOLO26 process on a GPU has exited, its summary exists, and the GPU is idle. The existing launcher rejects GPUs using more than 100 MiB and rejects a duplicate live worker.

Obtain `MONGODB_URI` from the local Assistant checkout at `../assistant/assistant/mongodb/tune_hyps.py`. Never print it, paste it into an issue, commit it, or place it on a command line. Upload it through the Jupyter Contents API as the temporary file below:

```text
/workspace/ul37-oil-experiments/.mongodb-uri-once
```

Then launch only the newly idle GPU indices:

```bash
/workspace/ul37-yolo27/.venv/bin/python \
  /workspace/ul37-oil-experiments/launch_ul37_musgd_workers.py 1 2 3
```

The launcher passes the credential only in each process environment and deletes the temporary file in a `finally` block. Verify the file is absent after launch, all new PIDs are live, logs have reached `Starting iteration`, and `nvidia-smi` shows one training process on each requested GPU.

Do not rerun the launcher for an already-live GPU.

## Add a worker on another machine

1. Use Python 3.14 and the exact `exp27-clean` source commit above. Set `PYTHONPATH` and `ULTRALYTICS_SOURCE` to that checkout; verify `ultralytics.__file__` resolves inside it.
2. Copy the exact worker artifact from Vast and verify its SHA-256.
3. Copy `yolo27n.pt` and verify its SHA-256.
4. Materialize or copy all revised UL37 datasets locally. Build a 37-line manifest. If absolute paths differ from Vast, compute its SHA-256, pass that value as `UL37_MANIFEST_SHA256`, and record it in issue 3849 before launch. The worker independently requires the same 37 sorted dataset identities via the path-independent identity hash above.
5. Select a globally unique worker name such as `g14` or `provider-host-gpu0`. Never reuse a live worker name.
6. Securely load the same MongoDB URI into the process environment. Do not store or log it.
7. Launch one persistent process per GPU:

```bash
mkdir -p "$UL37_CAMPAIGN/logs-tune-musgd"
nohup env \
  UL37_WORKER="$WORKER_NAME" \
  UL37_GPU="$GPU_INDEX" \
  UL37_ROOT="$UL37_ROOT" \
  UL37_CAMPAIGN="$UL37_CAMPAIGN" \
  UL37_MANIFEST="$UL37_MANIFEST" \
  UL37_MANIFEST_SHA256="$UL37_MANIFEST_SHA256" \
  UL37_MODEL="$UL37_MODEL" \
  ULTRALYTICS_SOURCE="$EXP27_SOURCE" \
  PYTHONPATH="$EXP27_SOURCE" \
  MONGODB_URI="$MONGODB_URI" \
  "$PYTHON314" "$UL37_CAMPAIGN/run_ul37_musgd_tuner_worker.py" \
  >> "$UL37_CAMPAIGN/logs-tune-musgd/$WORKER_NAME.log" 2>&1 < /dev/null &
```

Unset the shell credential immediately after launch. Confirm exactly one outer worker per assigned GPU, no traceback, and active GPU compute. A worker runs repeated generations until the shared collection reaches 500; `iterations=500` is not a per-worker target.

## MongoDB checks

Before adding workers, read the collection without modifying it:

- completed result iterations are unique and contiguous;
- each completed document has aggregate fitness, 25 hyperparameters, exactly 37 dataset entries, and save directories;
- there are no zero/NaN aggregate-fitness documents;
- the best iteration and fitness are recorded;
- the collection has not already reached 500.

The `_id="defaults"` document is Tuner coordination state, not a completed result. Count only documents containing `fitness`.

After each completed generation, ensure failures remain explicit in `failed_datasets` and have zero metric/fitness entries in `datasets`.

## Monitoring and issue updates

Check every one to two hours:

1. All expected outer workers and current child trainers are live.
2. Every assigned GPU is making progress; brief zero-utilization samples during validation or dataset transitions are normal.
3. Logs contain no traceback, OOM, repeated dataset failure, or MongoDB write warning.
4. Mongo iterations remain unique/contiguous and documents satisfy the 37-dataset/25-key contract.
5. Restart only a genuinely dead or stalled worker and never launch a duplicate.

Update Portal issue 3849 whenever workers are added or removed, on every meaningful failure/restart, and when a new incumbent appears. Include worker names, source/model/manifest hashes, collection count, best iteration/fitness, and any failed datasets. Never include credentials or ephemeral access tokens.
