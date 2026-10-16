"""Download verified data, then run the validated robustness fine-tune on this Mac.

Keeps status in training/local_work/local_status.json and preserves the web app's
current checkpoint. Run using the prepared WeSep virtualenv.
"""
import fcntl
from hashlib import md5
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "training/local_work"
CHECKPOINT = ROOT / "evaluation/wesep_eval/checkpoints/pretrained/retrained_finetune_research"
RUN_NAME = "finetune_noise_robust_research"
ARCHIVES = {"train-clean-100": "2a93770f6d5c6c964bc36631d331a522",
            "dev-clean": "42e2234ba48799c1f50f24a7926300a1"}
child = None


def status(stage, **details):
    record = {"stage": stage, "pid": os.getpid(), "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "run": str(WORK / "runs" / RUN_NAME), **details}
    tmp = WORK / "local_status.json.tmp"
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(WORK / "local_status.json")
    print(json.dumps(record), flush=True)


def digest(path):
    result = md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def execute(command, progress=None, env=None):
    global child
    child = subprocess.Popen(command, env=env)
    while child.poll() is None:
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            if progress:
                progress()
    code = child.returncode
    child = None
    if code:
        raise RuntimeError(f"Command exited {code}: {command[0]}")


def interrupted(signum, frame):
    if child is not None:
        child.terminate()
    raise KeyboardInterrupt


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    lock = (WORK / "local_robust.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("A local robustness job is already running")
    signal.signal(signal.SIGTERM, interrupted)
    smoke = WORK / "runs" / f"smoke_{RUN_NAME}"
    if not (smoke / "DONE").exists() or not (smoke / "export" / f"retrained_{RUN_NAME}" / "avg_model.pt").exists():
        raise RuntimeError("Complete the 20-step smoke check before starting this job")
    try:
        for name, expected in ARCHIVES.items():
            if (WORK / "data/LibriSpeech" / name).is_dir():
                continue
            archive = WORK / "data/archives" / f"{name}.tar.gz"
            archive.parent.mkdir(parents=True, exist_ok=True)
            if not archive.exists() or digest(archive) != expected:
                status("downloading", dataset=name, bytes_downloaded=archive.stat().st_size if archive.exists() else 0)
                execute(["curl", "--silent", "--show-error", "--location", "--fail", "--retry", "3",
                         "--continue-at", "-", "--output", str(archive),
                         f"https://www.openslr.org/resources/12/{name}.tar.gz"],
                        progress=lambda: status("downloading", dataset=name, bytes_downloaded=archive.stat().st_size))
            if digest(archive) != expected:
                raise RuntimeError(f"MD5 mismatch: {archive}; full training was not started")
            status("download_verified", dataset=name)
        env = dict(os.environ)
        env.update(TSE_PROFILE="noise_robust", TSE_MODE="finetune_research", TSE_RUN_NAME=RUN_NAME,
                   TSE_SMOKE="0", TSE_WORK=str(WORK), TSE_LIBRISPEECH=str(WORK / "data/LibriSpeech"),
                   TSE_INIT_CHECKPOINT=str(CHECKPOINT), TSE_INSTALL="0", TSE_DEVICE="mps",
                   MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
        status("preparing_and_training", steps=5000, batch=2, checkpoint=str(CHECKPOINT))
        execute([sys.executable, str(ROOT / "training/wesep_retrain.py")], env=env)
        status("completed", export=str(WORK / "runs" / RUN_NAME / "export" / f"retrained_{RUN_NAME}"))
    except KeyboardInterrupt:
        status("stopped")
        raise
    except Exception as error:
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
