#!/usr/bin/env python3
"""voice_bench.py — repeatable voice-fidelity benchmark for a voice-LoRA.

Protocol (per Masa's idea): write a fixed-length article on a fixed set of
themes, then measure how close the voice is to the target persona's REAL
writing. Run it on base vs each adapter version to watch the voice lock in.

Three scores per generation, vs the real corpus:
  1. stylometric distance  — deterministic feature vector (丁寧語率 / 平均文長 /
     呼称頻度 / 絵文字密度 / ひらがな率). 0 = identical profile to the corpus.
  2. local-LLM judge (0-100) — gpt-oss compares the text to real samples.
  3. (human blind — done separately by Masa.)

Usage:
  voice_bench.py --base <model> --adapter <dir|-> --corpus <jsonl> --out <md>
"""
import argparse
import json
import re
import subprocess
import sys
import urllib.request

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 汎用テーマ例（誰でも使える書き出しお題）。自分の対象に合うものに --themes-file で差し替える。
THEMES = [
    "最近読んだ本や記事の感想",
    "休日の過ごし方について",
    "仕事や活動で大切にしていること",
    "はじめて何かに挑戦したときのこと",
    "10年後の自分に伝えたいこと",
    "毎日続けている習慣について",
    "好きな場所とその理由",
    "最近の失敗とそこから学んだこと",
    "誰かに伝えたい考えや気づき",
    "自分の考え方が変わった経験",
]
TARGET_CHARS = 400
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF☀-➿❤]")


def style_profile(text):
    t = text.strip()
    n = max(len(t), 1)
    sents = [s for s in re.split(r"[。！？\n]", t) if s.strip()]
    hira = sum(1 for c in t if "぀" <= c <= "ゟ")
    return {
        "polite": (t.count("です") + t.count("ます") + t.count("ですっ")) / n * 1000,
        "plain": (len(re.findall(r"(だ|である|だよ|だね)", t))) / n * 1000,
        "sent_len": sum(len(s) for s in sents) / max(len(sents), 1),
        "address": (t.count("さん") + t.count("くん") + t.count("ちゃん")) / n * 1000,
        "emoji": len(EMOJI.findall(t)) / n * 1000,
        "hira": hira / n * 100,
    }


def profile_from_corpus(path, limit=400):
    agg = {}
    rows = [json.loads(l)["text"] for l in open(path, encoding="utf-8")][:limit]
    profs = [style_profile(r) for r in rows]
    for k in profs[0]:
        agg[k] = sum(p[k] for p in profs) / len(profs)
    return agg, rows


def distance(a, ref):
    # normalized abs diff per feature, averaged (scale by ref magnitude)
    ds = []
    for k in ref:
        scale = abs(ref[k]) + 1e-6
        ds.append(abs(a[k] - ref[k]) / scale)
    return sum(ds) / len(ds)


def judge(text, samples):
    ref = "\n---\n".join(s[:300] for s in samples[:3])
    prompt = (f"以下は著者Aの実際の文章サンプルです:\n{ref}\n\n"
              f"次の文章が、著者Aの声（口調・語彙・リズム・雰囲気）にどれだけ似ているかを"
              f"0〜100で採点し、JSONのみで {{\"score\": <int>, \"why\": \"<一言>\"}} を返して:\n{text[:600]}")
    payload = {"model": "gpt-oss:20b", "messages": [{"role": "user", "content": prompt}],
               "stream": False, "think": "low", "options": {"num_predict": 300, "temperature": 0.2}}
    try:
        req = urllib.request.Request("http://localhost:11434/api/chat",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            c = json.loads(r.read())["message"]["content"]
        m = re.search(r'\{.*\}', c, re.S)
        return json.loads(m.group(0)) if m else {"score": None, "why": "parse-fail"}
    except Exception as e:
        return {"score": None, "why": f"judge-error {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default="-")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--themes-file", default=None,
                    help="1行1テーマのファイル（未指定なら組込みの汎用テーマ例）")
    args = ap.parse_args()

    global THEMES
    if args.themes_file:
        THEMES = [t.strip() for t in open(args.themes_file, encoding="utf-8") if t.strip()]

    ref_prof, samples = profile_from_corpus(args.corpus)
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="auto")
    label = "base"
    if args.adapter != "-":
        model = PeftModel.from_pretrained(model, args.adapter)
        label = args.adapter.rstrip("/").split("/")[-2] if "/" in args.adapter else "adapter"
    model.eval()

    rows, dists, scores = [], [], []
    for theme in THEMES:
        # a sentence-opener prompt (not a bare markdown heading) — reliably
        # induces prose continuation instead of empty/meta output
        prompt = f"{theme}について、"
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=int(TARGET_CHARS * 1.2), do_sample=True,
                                 temperature=0.8, top_p=0.9, repetition_penalty=1.15,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()
        d = distance(style_profile(gen), ref_prof)
        j = judge(gen, samples)
        dists.append(d)
        if isinstance(j.get("score"), (int, float)):
            scores.append(j["score"])
        rows.append((theme, gen, d, j))

    avg_d = sum(dists) / len(dists)
    avg_s = sum(scores) / len(scores) if scores else None
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# voice-bench: {label}\n\n")
        f.write(f"**声の距離(文体・低いほど本物に近い): {avg_d:.3f}** / "
                f"**LLM審査(0-100・高いほど似てる): {avg_s}**\n\n")
        for theme, gen, d, j in rows:
            f.write(f"## {theme}\n距離 {d:.3f} / 審査 {j.get('score')}（{j.get('why','')}）\n\n> {gen[:400]}\n\n")
    print(f"{label}: voice_distance={avg_d:.3f} judge={avg_s}")


if __name__ == "__main__":
    main()
