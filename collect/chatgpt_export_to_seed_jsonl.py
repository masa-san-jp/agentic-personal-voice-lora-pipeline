#!/usr/bin/env python3
"""chatgpt_export_to_seed_jsonl.py — extract the USER's own writing from a
ChatGPT data export into voice-LoRA seed JSONL.

A voice model of a person must learn *their* text, not the assistant's. So we
keep only role=="user", content_type=="text" messages, join each conversation's
user turns, chunk on paragraph boundaries into [MIN,MAX] char windows, dedup by
the first 512 chars, and light-redact obvious PII (emails / long digit runs).

Usage:
  chatgpt_export_to_seed_jsonl.py --in <export_dir> --out <seed.jsonl> [--source chatgpt.masa]
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


def redact(t):
    t = EMAIL.sub("<EMAIL>", t)
    t = LONGNUM.sub("<NUM>", t)
    return t


def user_texts(conv):
    """User message texts in create_time order (fallback: mapping order)."""
    nodes = []
    for node in (conv.get("mapping") or {}).values():
        m = node.get("message") or {}
        if (m.get("author") or {}).get("role") != "user":
            continue
        c = m.get("content") or {}
        if c.get("content_type") != "text":
            continue
        parts = [p for p in (c.get("parts") or []) if isinstance(p, str) and p.strip()]
        if not parts:
            continue
        nodes.append((m.get("create_time") or 0, "\n\n".join(parts).strip()))
    nodes.sort(key=lambda x: x[0])
    return [t for _, t in nodes]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="chatgpt.masa")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.indir, "conversations-*.json"))) \
        or sorted(glob.glob(os.path.join(args.indir, "conversations.json")))
    seen, written, convs, usermsgs = set(), 0, 0, 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fo:
        for f in files:
            for conv in json.load(open(f, encoding="utf-8")):
                convs += 1
                texts = user_texts(conv)
                usermsgs += len(texts)
                joined = "\n\n".join(texts)
                for ch in chunk(joined):
                    ch = redact(ch)
                    key = hashlib.sha256(ch[:512].encode("utf-8")).hexdigest()
                    if key in seen:
                        continue
                    seen.add(key)
                    fo.write(json.dumps({"text": ch, "source": args.source,
                                         "id": key[:12]}, ensure_ascii=False) + "\n")
                    written += 1
    chars = sum(len(json.loads(l)["text"]) for l in open(args.out, encoding="utf-8"))
    print(f"conversations={convs} user_msgs={usermsgs} chunks_written={written} "
          f"total_chars={chars} approx_tokens={chars//2}")


if __name__ == "__main__":
    main()
