"""Compare assignment backends on cached MOT16-04 person detections.

For scientific benchmarking, see https://github.com/raphaelreme/pylapy instead.

Install: pip install "lapcpp>=1.2.0" scipy
Run from this checkout: python .github/scripts/benchmark-tracking.py
Optional baselines: pip install lap==0.5.14, then run with --lap for both cost_limit and post-filtered LAP.
Post-filtering solves a different objective and can produce different matches.

The compressed JSON contains YOLO26n detections, not images or ground-truth IDs.
MOT16-04 source attribution and CC BY-NC-SA 3.0 terms are in the replay metadata.
Replay detection quality (imgsz=1280, conf=0.1): precision 72.1%, recall 67.0%, F1 69.4%.
Measured once against MOT16-04 gt.txt with per-frame IoU >= 0.5 matching and MOT distractor filtering;
these are detection diagnostics, not mAP or tracking scores, and are not recomputed by this script.
Costs use adjacent-frame IoU and current-frame confidence, with threshold 0.8.
This times assignment only. Most full-tracker time is spent elsewhere (motion updates,
distance calculations, track management), so solver speedups do not translate directly to tracking speedups.

Example output from Windows 11, Python 3.11.18, Intel i9-13900KS (timings vary by machine):
    python benchmark-tracking.py --lap
    MOT16-04: 1050 frames, 1049 nonempty adjacent-frame assignments; threshold-aware objective parity passed
    Median time per assignment call; 3 rounds, 6 replays/round; includes wrapper overhead, excludes cost construction and inference
    lap-post filters after solving (different objective); matches differ from SciPy on 14/1049 matrices

    Backend    Version       Time (us)  Relative speed
    lapcpp     1.2.0              5.55           3.47x
    scipy      1.17.1            19.23           1.00x
    lap        0.5.14            32.16           0.60x
    lap-post   0.5.14            19.83           0.97x
"""

import argparse
import gzip
import json
import sys
from functools import partial
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np


def main():
    """Print median assignment time for each backend on the same precomputed costs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lap", action="store_true", help="Include lap==0.5.14 with cost_limit and with post-filtering"
    )
    args = parser.parse_args()

    import lapcpp
    import scipy

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ultralytics.trackers.utils import matching
    from ultralytics.utils import ops
    from ultralytics.utils.metrics import bbox_ioa

    backends = {"lapcpp": matching.linear_assignment, "scipy": partial(matching.linear_assignment, use_lap=False)}
    versions = {"lapcpp": lapcpp.__version__, "scipy": scipy.__version__}
    if args.lap:
        import lap

        assert lap.__version__ == "0.5.14", "Install lap==0.5.14 (lapx also uses the lap import name)"

        def lap_assignment(cost, thresh, post=False):
            """Run LAP with NumPy array outputs and skip the unused total cost."""
            # return_cost=False saves time by skipping the unused total-cost sum; assignments are unchanged.
            x, y = lap.lapjv(cost, extend_cost=True, cost_limit=np.inf if post else thresh, return_cost=False)
            valid = x >= 0
            if post:
                rows = np.flatnonzero(valid)
                rejected = rows[cost[rows, x[rows]] > thresh]
                y[x[rejected]] = -1
                valid[rejected] = False
            matches = np.column_stack((np.flatnonzero(valid), x[valid]))
            return matches, np.flatnonzero(~valid), np.flatnonzero(y < 0)

        backends["lap"] = lap_assignment
        backends["lap-post"] = partial(lap_assignment, post=True)
        versions["lap"] = versions["lap-post"] = lap.__version__

    with gzip.open(Path(__file__).with_name("MOT16-04.json.gz"), "rt", encoding="utf-8") as f:
        replay = json.load(f)
    frames = [np.asarray(frame, dtype=np.float32).reshape(-1, 5) for frame in replay["frames"]]
    costs = [
        1 - bbox_ioa(previous[:, :4], current[:, :4], iou=True) * current[:, 4]
        for previous, current in zip(frames, frames[1:])
        if len(previous) and len(current)
    ]
    ops.linear_sum_assignment(costs[0])
    assert ops._assignment_solver is scipy.optimize.linear_sum_assignment, "SciPy backend was not selected"

    # Validate equivalent objectives and warm every backend before measuring; equal-cost ties may differ.
    post_differences = 0
    for cost in costs:
        pairs = {name: solve(cost, 0.8)[0] for name, solve in backends.items()}
        objectives = [
            cost[p[:, 0], p[:, 1]].sum(dtype=np.float64) - 0.8 * len(p)
            for name, p in pairs.items()
            if name != "lap-post"
        ]
        np.testing.assert_allclose(objectives, objectives[0], rtol=1e-6, atol=1e-6)
        if args.lap:
            post_differences += not np.array_equal(pairs["lap-post"], pairs["scipy"])

    timings = {name: [] for name in backends}
    order = list(backends)
    rounds, replays = 3, 6
    for _ in range(rounds):
        for name in order:
            solve = backends[name]
            start = perf_counter()
            for _ in range(replays):
                for cost in costs:
                    solve(cost, 0.8)
            timings[name].append((perf_counter() - start) * 1e6 / (replays * len(costs)))
        order = order[1:] + order[:1]

    print(
        f"MOT16-04: {len(frames)} frames, {len(costs)} nonempty adjacent-frame assignments; "
        "threshold-aware objective parity passed"
    )
    print(
        f"Median time per assignment call; {rounds} rounds, {replays} replays/round; "
        "includes wrapper overhead, excludes cost construction and inference"
    )
    if args.lap:
        print(
            f"lap-post filters after solving (different objective); matches differ from SciPy on {post_differences}/{len(costs)} matrices"
        )
    baseline = median(timings["scipy"])
    print()
    print(f"{'Backend':<10} {'Version':<12} {'Time (us)':>10} {'Relative speed':>15}")
    for name, values in timings.items():
        elapsed = median(values)
        print(f"{name:<10} {versions[name]:<12} {elapsed:>10.2f} {baseline / elapsed:>14.2f}x")


if __name__ == "__main__":
    main()
