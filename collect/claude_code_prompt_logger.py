#!/usr/bin/env python3
"""claude_code_prompt_logger.py — append each Claude Code prompt to a growing
voice corpus, live, as you type it.

This is the "keep accumulating" companion to claude_code_history_to_jsonl.py.
Wire it as a Claude Code `UserPromptSubmit` hook and every instruction you send
is appended (one JSON line) to $VOICE_LORA_ROOT/raw/claude_code_live.jsonl, with
obvious PII masked. build_corpus.py can then read that file like any other raw
source.

Hook setup (opt-in — edit your Claude Code settings.json):

  {
    "hooks": {
      "UserPromptSubmit": [
        { "hooks": [ { "type": "command",
          "command": "python3 ~/voice-lora/collect/claude_code_prompt_logger.py" } ] }
      ]
    }
  }

Design rules this script must obey (do not change lightly):
  * It reads the hook payload as JSON on stdin: {"prompt": "...", ...}.
  * It prints NOTHING to stdout. For UserPromptSubmit, stdout on exit 0 is
    injected into the model's context — so any output here would pollute your
    prompt. All logging goes to the file only.
  * It never fails the prompt: every error is swallowed and it exits 0.

The log lives under raw/, which is gitignored — it never leaves your machine.
"""
import json
import os
import re
import sys

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
LONGNUM = re.compile(r"\b\d[\d\s-]{8,}\d\b")

MIN_CHARS = int(os.environ.get("VOICE_LORA_PROMPT_MIN_CHARS", "15"))


def redact(t):
    t = EMAIL.sub("<EMAIL>", t)
    t = LONGNUM.sub("<NUM>", t)
    return t


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return  # no/garbled stdin — do nothing, exit 0

    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return
    # Skip bare slash-commands — they aren't voice.
    if re.match(r"^/[A-Za-z0-9_-]+\s*$", prompt):
        return
    if len(prompt) < MIN_CHARS:
        return

    root = os.environ.get("VOICE_LORA_ROOT", os.path.expanduser("~/voice-lora"))
    out = os.path.join(root, "raw", "claude_code_live.jsonl")
    rec = {"text": redact(prompt), "source": "claude_code.live"}
    ts = payload.get("timestamp") or payload.get("session_id")
    if ts:
        rec["ts"] = str(ts)

    try:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        return  # never block a prompt over a logging failure


if __name__ == "__main__":
    main()
    sys.exit(0)  # UserPromptSubmit: exit 0, no stdout
