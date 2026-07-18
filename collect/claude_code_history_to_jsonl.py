#!/usr/bin/env python3
"""claude_code_history_to_jsonl.py — harvest YOUR instructions to Claude Code
into voice-LoRA JSONL.

The prompts you type at Claude Code are text *you* wrote, in your own voice
(imperative, terse, technical) — a source just like email or chat. Claude Code
stores every session as a transcript under ~/.claude/projects/<proj>/<id>.jsonl,
one JSON event per line. The human turns are `type=="user"` messages whose
content is a plain string; tool results are also `type=="user"` but their
content is a list of tool_result blocks — those are NOT yours, so we drop them.

We keep only your text, per session in timestamp order, strip harness artifacts
(system reminders, slash-command wrappers, fenced code blocks by default),
chunk on paragraph boundaries into [MIN,MAX]-char windows, dedup by the first
512 chars, and light-redact obvious PII (emails / long digit runs). Re-running
picks up new sessions, so this "accumulates" as you keep using Claude Code.

Usage:
  # zero-arg: reads ~/.claude/projects, writes $VOICE_LORA_ROOT/raw/claude_code.jsonl
  claude_code_history_to_jsonl.py
  # explicit:
  claude_code_history_to_jsonl.py --in ~/.claude/projects --out raw/claude_code.jsonl
  # only one project's sessions:
  claude_code_history_to_jsonl.py --project-substr agentic-personal-voice
"""
import argparse
import glob
import hashlib
import json
import os
import re

MIN_CHARS = 400
MAX_CHARS = 16000

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
LONGNUM = re.compile(r"\b\d[\d\s-]{8,}\d\b")  # phone / long id-ish runs

# Harness-injected wrappers that are not the human's prose.
_ARTIFACT_TAGS = ("command-name", "command-message", "command-args",
                  "local-command-stdout", "bash-input", "bash-stdout", "bash-stderr")


def redact(t):
    t = EMAIL.sub("<EMAIL>", t)
    t = LONGNUM.sub("<NUM>", t)
    return t


def extract_user_text(o):
    """Return the human-typed text of a transcript line, or None if it isn't one.

    Keep: type=='user' with string content, or a list containing text blocks.
    Drop: meta/sidechain lines, and tool_result payloads (content is a list of
    tool_result blocks — that's tool output, not something you wrote).
    """
    if o.get("type") != "user" or o.get("isMeta") or o.get("isSidechain"):
        return None
    content = (o.get("message") or {}).get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = [b.get("text", "") for b in content
                 if isinstance(b, dict) and b.get("type") == "text"]
        if not parts:  # e.g. a tool_result-only turn
            return None
        text = "\n\n".join(p for p in parts if p.strip())
    else:
        return None
    return text.strip() or None


def clean(text, keep_code=False):
    """Strip harness artifacts and (optionally) code so only voice prose remains."""
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.S)
    # Whole message is a harness wrapper (slash-command expansion, command stdout).
    if re.match(r"^\s*<(" + "|".join(_ARTIFACT_TAGS) + r")>", text):
        return ""
    text = re.sub(r"<(command-[a-z]+|local-command-stdout)>.*?</\1>", "", text, flags=re.S)
    if text.lstrip().startswith("[Request interrupted"):
        return ""
    text = re.sub(r"\[Image #\d+\]", "", text)
    if not keep_code:
        text = re.sub(r"```.*?```", "", text, flags=re.S)  # fenced code blocks aren't voice
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    # A bare slash-command line ("/review") carries no voice.
    if re.match(r"^/[A-Za-z0-9_-]+\s*$", text):
        return ""
    return text


def chunk(text):
    """Accumulate paragraphs into [MIN,MAX]-char windows on \\n\\n boundaries."""
    out, buf = [], ""
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if not para:
            continue
        if len(buf) + len(para) + 2 > MAX_CHARS and buf:
            out.append(buf)
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        out.append(buf)
    return [c for c in out if len(c) >= MIN_CHARS]


def session_texts(path, keep_code):
    """All human turns in one transcript file, in timestamp order."""
    turns = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw = extract_user_text(o)
            if not raw:
                continue
            cleaned = clean(raw, keep_code=keep_code)
            if cleaned:
                turns.append((o.get("timestamp") or "", i, cleaned))
    turns.sort(key=lambda x: (x[0], x[1]))
    return [t for _, _, t in turns]


def main():
    default_root = os.environ.get("VOICE_LORA_ROOT", os.path.expanduser("~/voice-lora"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir",
                    default=os.path.expanduser("~/.claude/projects"),
                    help="Claude Code transcript dir (or a single .jsonl). "
                         "Default: ~/.claude/projects")
    ap.add_argument("--out", default=os.path.join(default_root, "raw", "claude_code.jsonl"),
                    help="Output JSONL. Default: $VOICE_LORA_ROOT/raw/claude_code.jsonl")
    ap.add_argument("--source", default="claude_code")
    ap.add_argument("--project-substr", default=None,
                    help="Only include transcript paths containing this substring.")
    ap.add_argument("--keep-code", action="store_true",
                    help="Keep fenced code blocks (dropped by default — not voice).")
    args = ap.parse_args()

    if os.path.isdir(args.indir):
        files = sorted(glob.glob(os.path.join(args.indir, "**", "*.jsonl"), recursive=True))
    elif os.path.isfile(args.indir):
        files = [args.indir]
    else:
        files = []
    if args.project_substr:
        files = [f for f in files if args.project_substr in f]

    seen, written, sessions, turns_total = set(), 0, 0, 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fo:
        for path in files:
            texts = session_texts(path, keep_code=args.keep_code)
            if not texts:
                continue
            sessions += 1
            turns_total += len(texts)
            for ch in chunk("\n\n".join(texts)):
                ch = redact(ch)
                key = hashlib.sha256(ch[:512].encode("utf-8")).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                fo.write(json.dumps({"text": ch, "source": args.source,
                                     "id": key[:12]}, ensure_ascii=False) + "\n")
                written += 1

    chars = sum(len(json.loads(l)["text"]) for l in open(args.out, encoding="utf-8")) if written else 0
    print(f"transcripts={len(files)} sessions_with_text={sessions} human_turns={turns_total} "
          f"chunks_written={written} total_chars={chars} approx_tokens={chars // 2}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
