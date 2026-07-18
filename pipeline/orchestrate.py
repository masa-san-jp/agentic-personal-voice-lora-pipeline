#!/usr/bin/env python3
"""Voice LoRA training orchestrator.

Reads versions.yaml, runs each version sequentially. Each version inherits the
previous version's final_adapter via `resume_from: previous`, so what you get is
one continuous training schedule expressed as N discrete LoRA "versions" that
gradually grow the corpus, anneal the learning rate, or change other knobs.

State machine per version:
  pending -> running -> done   (happy path)
            -> failed -> retry (up to MAX_RETRIES)
            -> abandon         (exit 2)

The orchestrator is resumable: on restart, it reads state.json and skips
versions already marked `done`. Designed to run under systemd with
Restart=on-failure (see systemd/voice-lora.service).

Paths are configurable via env:
  VOICE_LORA_ROOT      base install dir (default: ~/voice-lora)
  VOICE_LORA_VENV_PY   python binary with ML deps (default: $VOICE_LORA_ROOT/.venv/bin/python3)
  VOICE_LORA_RUNS      output checkpoints root (default: ~/runs)
"""
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("VOICE_LORA_ROOT", Path.home() / "voice-lora"))
PIPELINE = ROOT / "pipeline"
VERSIONS_YAML = PIPELINE / "versions.yaml"
STATE_JSON = ROOT / "state.json"
LOGS = ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

TRAIN_SCRIPT = PIPELINE / "train_bf16_lora.py"
BUILD_CORPUS = PIPELINE / "build_corpus.py"
COMPARE_SCRIPT = PIPELINE / "compare_checkpoints.py"
DATA_DIR = ROOT / "data"
RUNS_DIR = Path(os.environ.get("VOICE_LORA_RUNS", Path.home() / "runs"))

VENV_PY = Path(os.environ.get(
    "VOICE_LORA_VENV_PY",
    str(ROOT / ".venv" / "bin" / "python3"),
))

# Output dir naming: <RUNS_DIR>/<RUN_PREFIX>-<vid>/
RUN_PREFIX = os.environ.get("VOICE_LORA_RUN_PREFIX", "voice-lora")

MAX_RETRIES = 3
SLEEP_BETWEEN = 30

# Exit code the trainer uses when it stops early for a graceful pause (SIGUSR1
# from the memory watcher, or a manual `systemctl --user kill -s SIGUSR1`).
# A pause is resumable, not a failure: we wait for memory to recover, then
# re-run the same version, which auto-resumes from its last checkpoint. Pauses
# do NOT consume the per-version retry budget.
PAUSE_EXIT_CODE = 3
SLEEP_AFTER_PAUSE = int(os.environ.get("VOICE_LORA_PAUSE_SLEEP", "120"))


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def load_state():
    if STATE_JSON.exists():
        return json.loads(STATE_JSON.read_text())
    return {"versions": {}, "history": [], "started": now_iso()}


def save_state(s):
    STATE_JSON.write_text(json.dumps(s, indent=2, ensure_ascii=False))


def log_event(state, vid, event, **kw):
    entry = {"ts": now_iso(), "version": vid, "event": event, **kw}
    state["history"].append(entry)
    save_state(state)
    print(f"[{entry['ts']}] {vid}: {event} {kw}")


def resolve_resume(spec, prior_versions):
    """If spec is 'previous', return the previous version's final_adapter path."""
    if not spec:
        return None
    if spec == "previous":
        for prev in reversed(prior_versions):
            adapter = RUNS_DIR / f"{RUN_PREFIX}-{prev}" / "final_adapter"
            if adapter.exists():
                return str(adapter)
        return None
    return str(spec)


def ensure_corpus(corpus_name, log_fp):
    """Ensure data/corpus_{name}.jsonl exists; build it via build_corpus.py if not."""
    path = DATA_DIR / f"corpus_{corpus_name}.jsonl"
    if path.exists() and path.stat().st_size > 0:
        log_fp.write(f"corpus {corpus_name}: already exists ({path.stat().st_size:,} bytes)\n")
        return path
    log_fp.write(f"corpus {corpus_name}: building via build_corpus.py...\n")
    log_fp.flush()
    proc = subprocess.run(
        [str(VENV_PY), str(BUILD_CORPUS), "--name", corpus_name, "--out", str(path)],
        stdout=log_fp, stderr=subprocess.STDOUT, timeout=3600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"corpus build failed (exit {proc.returncode})")
    if not path.exists():
        raise RuntimeError(f"corpus {corpus_name} not produced by build_corpus.py")
    return path


def run_version(vid, vspec, defaults, prior_done):
    """Build corpus + train one version.

    Returns one of:
      "done"   — trained to completion and final_adapter saved.
      "paused" — trainer stepped aside for memory (exit 3); resume, no retry cost.
      "failed" — real failure; counts against the retry budget.
    """
    log_path = LOGS / f"{vid}.log"
    cfg = {**defaults, **vspec}

    output_dir = RUNS_DIR / f"{RUN_PREFIX}-{vid}"
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_from = resolve_resume(vspec.get("resume_from"), prior_done)

    with open(log_path, "a") as log_fp:
        log_fp.write(f"\n{'='*60}\n=== {vid} start {now_iso()} ===\n")
        log_fp.write(f"description: {vspec.get('description')}\n")
        log_fp.write(f"corpus: {cfg['corpus']}\n")
        log_fp.write(f"resume_from: {resume_from}\n")
        log_fp.write(f"output: {output_dir}\n")
        log_fp.flush()

        corpus_path = ensure_corpus(cfg["corpus"], log_fp)

        cmd = [
            str(VENV_PY), str(TRAIN_SCRIPT),
            "--data", str(corpus_path),
            "--model", cfg["base_model"],
            "--output", str(output_dir),
            "--rank", str(cfg["rank"]),
            "--alpha", str(cfg["alpha"]),
            "--dropout", str(cfg["dropout"]),
            "--lr", str(cfg["lr"]),
            "--epochs", str(cfg["epochs"]),
            "--seq_len", str(cfg["seq_len"]),
            "--batch", str(cfg["batch"]),
            "--grad_accum", str(cfg["grad_accum"]),
        ]
        if resume_from:
            cmd += ["--resume_from", resume_from]
        if cfg.get("qlora"):
            cmd += ["--qlora"]

        log_fp.write(f"cmd: {' '.join(shlex.quote(c) for c in cmd)}\n\n")
        log_fp.flush()

        env = os.environ.copy()
        env.setdefault("HF_HOME", str(Path.home() / "hf_cache"))
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

        start = time.time()
        proc = subprocess.run(cmd, stdout=log_fp, stderr=subprocess.STDOUT, env=env)
        elapsed = time.time() - start
        log_fp.write(f"\n=== {vid} exit={proc.returncode} elapsed={elapsed:.1f}s ===\n")

        if proc.returncode == PAUSE_EXIT_CODE:
            log_fp.write(f"{vid} paused (graceful checkpoint) — will resume from "
                         f"latest checkpoint after memory recovers\n")
            return "paused"

        if proc.returncode != 0:
            return "failed"

        # Verify final_adapter saved
        if not (output_dir / "final_adapter" / "adapter_config.json").exists():
            log_fp.write(f"WARN: final_adapter missing for {vid}\n")
            return "failed"

        # Optionally run compare_checkpoints (non-fatal)
        if COMPARE_SCRIPT.exists():
            log_fp.write(f"\n=== running compare_checkpoints for {vid} ===\n")
            try:
                subprocess.run(
                    [str(VENV_PY), str(COMPARE_SCRIPT),
                     "--run_dir", str(output_dir),
                     "--base", cfg["base_model"]],
                    stdout=log_fp, stderr=subprocess.STDOUT, timeout=3600,
                )
            except Exception as e:
                log_fp.write(f"compare_checkpoints failed (non-fatal): {e}\n")

        return "done"


def main():
    # The memory watcher pauses training with `systemctl --user kill -s SIGUSR1`,
    # which delivers SIGUSR1 to every process in the unit's cgroup — including
    # this orchestrator. Ignore it here so only the trainer child acts on it;
    # otherwise the default SIGUSR1 action would terminate the orchestrator.
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)

    if not VERSIONS_YAML.exists():
        print(f"ERROR: {VERSIONS_YAML} not found. Copy versions.example.yaml first.",
              file=sys.stderr)
        sys.exit(1)

    cfg = yaml.safe_load(VERSIONS_YAML.read_text())
    defaults = cfg["defaults"]
    versions = cfg["versions"]  # dict of vid -> spec

    state = load_state()
    log_event(state, "_orchestrator_", "boot")

    prior_done = []
    for vid in sorted(versions):  # lexical sort: v06, v07, ..., v20
        vstate = state["versions"].setdefault(vid, {"status": "pending", "retries": 0})

        if vstate["status"] == "done":
            log_event(state, vid, "skip_done")
            prior_done.append(vid)
            continue

        while vstate["retries"] < MAX_RETRIES:
            vstate["status"] = "running"
            vstate["started"] = now_iso()
            save_state(state)
            log_event(state, vid, "start", retry=vstate["retries"])

            try:
                result = run_version(vid, versions[vid], defaults, prior_done)
            except Exception as e:
                result = "failed"
                log_event(state, vid, "exception", error=str(e))

            if result == "done":
                vstate["status"] = "done"
                vstate["finished"] = now_iso()
                save_state(state)
                log_event(state, vid, "done")
                prior_done.append(vid)
                break
            elif result == "paused":
                # Graceful memory pause: not a failure. Wait for memory to
                # recover, then re-run the same version (auto-resumes from its
                # last checkpoint). Does not consume the retry budget.
                vstate["status"] = "paused"
                vstate["pauses"] = vstate.get("pauses", 0) + 1
                save_state(state)
                log_event(state, vid, "paused", pauses=vstate["pauses"])
                time.sleep(SLEEP_AFTER_PAUSE)
                continue
            else:
                vstate["retries"] += 1
                vstate["status"] = "failed"
                save_state(state)
                log_event(state, vid, "fail", retries=vstate["retries"])
                time.sleep(SLEEP_BETWEEN)

        if vstate["status"] != "done":
            log_event(state, vid, "abandon",
                      reason=f"exceeded {MAX_RETRIES} retries")
            print(f"ABANDONING {vid} — halting orchestrator", file=sys.stderr)
            sys.exit(2)

    log_event(state, "_orchestrator_", "all_done")
    print("All versions completed.")


if __name__ == "__main__":
    main()
