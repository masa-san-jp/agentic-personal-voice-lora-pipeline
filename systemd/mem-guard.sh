#!/usr/bin/env bash
#
# mem-guard.sh — nudge training to checkpoint-and-pause BEFORE the machine OOMs.
#
# Run once per minute by voice-lora-memguard.timer. When free RAM drops below a
# threshold, send SIGUSR1 to the training unit: the trainer finishes its current
# step, writes a checkpoint, and exits (code 3). The orchestrator then waits for
# memory to recover and resumes from that checkpoint. This is the "step aside
# gracefully" backstop that runs a beat ahead of earlyoom / the kernel OOM
# killer, so you resume from the last checkpoint instead of a hard crash.
#
# It is a safety net, not the primary control: the primary controls are the
# systemd MemoryMax cap and earlyoom (see docs/stability.md). Tune the threshold
# to fire before those do.
#
# Config (env):
#   VOICE_LORA_MEM_MIN_KB   pause when MemAvailable < this (default 4 GiB)
#   VOICE_LORA_SERVICE      user unit to signal (default voice-lora.service)
set -euo pipefail

THRESHOLD_KB="${VOICE_LORA_MEM_MIN_KB:-4194304}"   # 4 GiB
SERVICE="${VOICE_LORA_SERVICE:-voice-lora.service}"

avail=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
if [ -z "${avail:-}" ]; then
    exit 0   # can't read memory; do nothing rather than guess
fi

if [ "$avail" -lt "$THRESHOLD_KB" ]; then
    # Only signal if the unit is actually running a step to interrupt.
    if systemctl --user is-active --quiet "$SERVICE"; then
        logger -t voice-lora-memguard \
            "MemAvailable ${avail}KB < ${THRESHOLD_KB}KB — SIGUSR1 -> ${SERVICE} (graceful checkpoint+pause)"
        systemctl --user kill -s SIGUSR1 "$SERVICE"
    fi
fi
