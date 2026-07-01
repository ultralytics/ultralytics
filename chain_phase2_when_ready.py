"""Fail-closed auto-starter: launch a ul33 phase-2 once its pretrain checkpoint finishes.

Bridges an overnight classification pretrain to its downstream multi_det_finetune eval without a
premature launch. Every poll re-checks a set of AND-ed gates and only fires when all pass; a
half-written checkpoint, a crashed trainer, or a busy GPU hold the launch instead of triggering it.

Launch gates (all must hold, fail-closed):
    1. no `.phase2_chained` sentinel next to the pretrain weights (fires exactly once, idempotent)
    2. results.csv has >= total_epochs data rows (cheap pre-check that training reached the last epoch)
    3. best.pt epoch == -1 (strip_optimizer stamps this only at a successful final_eval; a live or
       crashed trainer keeps epoch >= 0, so this is the authoritative "finished" signal, and on a
       direct-to-NFS run it also proves the finalized weights already landed on NFS)
    4. the target GPU has no compute process (pretrain released it and no one else grabbed it)

A dead trainer with epoch >= 0 is a crash, not a completion: the poller writes `.phase2_bailed`,
notifies, and stops rather than launching on stale weights. A finished pretrain whose GPU was taken
by someone else holds (notify-once) rather than launching elsewhere.

Usage:
    python chain_phase2_when_ready.py --ce-dir <pretrain_run_dir> --gpu 2 \
        --name phase2-33det-frz-inet-x-fastvit-lr1p5e3 --datasets ul33.txt \
        --total-epochs 114 --freeze 9 --lr 0.0015
    # dry-run one gate evaluation against live cluster state, never launches:
    python chain_phase2_when_ready.py --ce-dir ... --gpu 2 --name ... --check-once
"""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from ultralytics.utils.patches import torch_load

REPO = Path(__file__).resolve().parent
POLL_SEC = 300
MAX_HOURS = 48


def stamp() -> str:
    """Return a UTC timestamp for log lines (cluster clock is UTC, +8h to GMT+3)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def say(msg: str) -> None:
    """Print a timestamped poller log line."""
    print(f"[chain {stamp()}] {msg}", flush=True)


def slack(msg: str) -> None:
    """Post a Slack line via the webhook in $SLACK_WEBHOOK_URL or ~/.slack_webhook, else log only."""
    wf = Path.home() / ".slack_webhook"
    url = os.environ.get("SLACK_WEBHOOK_URL") or (wf.read_text().strip() if wf.exists() else "")
    if url:
        subprocess.run(
            ["curl", "-s", "-X", "POST", "-H", "Content-type: application/json", "--data", json.dumps({"text": msg}), url],
            check=False,
        )
    else:
        say(f"(no slack webhook) SLACK: {msg}")


def csv_rows(csv_path: Path) -> int:
    """Count data rows (lines minus header) in a results.csv, -1 if absent."""
    if not csv_path.exists():
        return -1
    return max(0, sum(1 for _ in csv_path.open()) - 1)


def ckpt_epoch(weights: Path) -> int | None:
    """Return the checkpoint epoch field, or None if the file cannot be read yet."""
    if not weights.exists():
        return None
    return torch_load(weights, map_location="cpu").get("epoch")


def gate_wait(alive: bool, detail: str) -> str:
    """Return a wait verdict when the trainer is alive, else a maybe-crash verdict."""
    return f"wait {detail} trainer_alive=True" if alive else f"maybe-crash {detail} trainer_dead"


def trainer_alive(token: str) -> bool:
    """Return True if a pretrain process matching token is running (excluding this poller)."""
    out = subprocess.run(["pgrep", "-af", token], capture_output=True, text=True).stdout
    me = os.getpid()
    for line in out.splitlines():
        pid = int(line.split(None, 1)[0])
        if pid == me or "chain_phase2_when_ready" in line:
            continue
        return True
    return False


def gpu_pids(gpu: int) -> list[str]:
    """Return compute-process pids on the given GPU (empty means idle)."""
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader", "-i", str(gpu)],
        capture_output=True,
        text=True,
    ).stdout
    return [p.strip() for p in out.splitlines() if p.strip()]


def launch(args, best: Path) -> None:
    """Sleep any launch jitter, then start phase-2 in its own tmux session."""
    if args.launch_jitter:
        say(f"gate passed, sleeping {args.launch_jitter}s launch-jitter to stagger off a sibling chain's wandb second")
        time.sleep(args.launch_jitter)
    cmd = (
        f"cd {REPO} && source .venv/bin/activate && "
        f"python run_enc_distill_phase2.py {args.gpu} {best} multi_det_finetune {args.name} "
        f"--datasets {args.datasets} --freeze {args.freeze} --lr {args.lr} "
        f"2>&1 | tee -a /data/shared-datasets/fatih-runs/logs/{args.name}.log"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", args.name, cmd], check=True)
    time.sleep(8)
    started = trainer_alive("run_enc_distill_phase2.py")
    say(f"launched tmux={args.name} phase2_proc_alive={started}")
    slack(
        f":rocket: auto-started {args.name} on g{args.gpu} "
        f"({'proc up' if started else 'WARN proc not seen, check tmux'}) | {cmd}"
    )


def evaluate(args) -> str:
    """Evaluate all gates once and return a status verdict (launch / wait / crash)."""
    best = Path(args.ce_dir) / "weights" / "best.pt"
    rows = csv_rows(Path(args.ce_dir) / "results.csv")
    if rows < args.total_epochs:  # cheap short-circuit before the heavy ckpt load
        return gate_wait(trainer_alive(args.token), f"rows={rows}/{args.total_epochs}")
    epoch = ckpt_epoch(best)
    if epoch != -1:
        return gate_wait(trainer_alive(args.token), f"rows={rows} epoch={epoch} finalizing")
    pids = gpu_pids(args.gpu)
    if pids:
        return f"wait-gpu epoch=-1 g{args.gpu}_busy={pids}"
    return f"launch epoch=-1 rows={rows} g{args.gpu}_idle"


def main() -> None:
    """Poll the pretrain checkpoint and launch phase-2 the moment every gate passes."""
    p = argparse.ArgumentParser()
    p.add_argument("--ce-dir", required=True, help="Pretrain run dir holding weights/best.pt + results.csv")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--name", required=True, help="Phase-2 run/tmux/log name")
    p.add_argument("--datasets", default="ul33.txt")
    p.add_argument("--total-epochs", type=int, required=True)
    p.add_argument("--freeze", type=int, default=9)
    p.add_argument("--lr", default="0.0015")
    p.add_argument("--token", default="", help="pgrep token for the pretrain process (default: ce-dir basename)")
    p.add_argument("--launch-jitter", type=int, default=0, help="Seconds to sleep after the gate passes before launching (stagger sibling chains off the same wandb second)")
    p.add_argument("--check-once", action="store_true", help="Evaluate gates once, print status, never launch")
    args = p.parse_args()
    args.token = args.token or Path(args.ce_dir).name

    sentinel = Path(args.ce_dir) / ".phase2_chained"
    bailed = Path(args.ce_dir) / ".phase2_bailed"

    if args.check_once:
        say(f"CHECK-ONCE verdict: {evaluate(args)}  (sentinel={sentinel.exists()} bailed={bailed.exists()})")
        return

    if sentinel.exists():
        say(f"sentinel {sentinel} exists, already fired, exiting")
        return

    say(f"polling {args.ce_dir} every {POLL_SEC}s -> launch {args.name} on g{args.gpu} (token={args.token!r})")
    slack(f":hourglass_flowing_sand: chain poller up for {args.name}: waiting on {Path(args.ce_dir).name} to finish, then g{args.gpu}")
    deadline = time.time() + MAX_HOURS * 3600
    crash_strikes = 0
    warned_busy = False

    while time.time() < deadline:
        verdict = evaluate(args)
        say(verdict)

        if verdict.startswith("launch"):
            try:
                sentinel.open("x").write(f"{stamp()} pid={os.getpid()}\n")  # atomic fire-once guard
            except FileExistsError:
                say("sentinel appeared concurrently, another poller fired, exiting")
                return
            launch(args, Path(args.ce_dir) / "weights" / "best.pt")
            return

        if verdict.startswith("maybe-crash"):
            crash_strikes += 1
            if crash_strikes >= 2:  # debounce a transient pgrep miss before bailing
                bailed.write_text(f"{stamp()} {verdict}\n")
                slack(f":warning: chain poller BAILED for {args.name}: {verdict} (pretrain looks crashed, not launching)")
                say("bailed on suspected crash, exiting")
                return
        else:
            crash_strikes = 0

        if verdict.startswith("wait-gpu") and not warned_busy:
            slack(f":warning: {Path(args.ce_dir).name} finished but g{args.gpu} is taken by someone else, holding: {verdict}")
            warned_busy = True

        time.sleep(POLL_SEC)

    slack(f":alarm_clock: chain poller for {args.name} hit the {MAX_HOURS}h cap without launching, giving up")
    say("max runtime reached, exiting")


if __name__ == "__main__":
    main()
