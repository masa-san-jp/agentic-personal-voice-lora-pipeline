#!/usr/bin/env python3
"""make_blind_eval.py — build a self-contained human blind-eval page from voice_bench outputs.

voice_bench.py produces two machine metrics per adapter version; the third metric
(human blind judgment) was always "done separately". This turns those bench
markdown files into ONE static HTML page where a human picks, per theme and with
the conditions hidden, the sample closest to the real target voice — then sees
their blind choice next to the machine metrics. This closes the pipeline:
  build_corpus -> orchestrate(train) -> voice_bench (per version) -> make_blind_eval -> (report/blog)

Input = the .md files voice_bench.py wrote (one per condition, worst->best order).
Themes must match; pass the same --themes-file you gave voice_bench (or rely on the
built-in generic themes). No external assets, no network — open the HTML anywhere.

Usage:
  make_blind_eval.py \
    --bench base=runs/voice-bench-10-base.md \
            mini=runs/voice-bench-10-adapter.md \
            pen=runs/voice-bench-10-pen.md \
    --labels "素モデル (base)=base" "浅い学習 (25step)=mini" "反復学習版 (100step)=pen" \
    --best pen --target "自分" --themes-file themes.txt \
    --out blind-eval.html
"""
import argparse, json, re, pathlib, sys

# 汎用テーマ例。voice_bench.py と同じものを使うこと（--themes-file で差し替え可）。
BUILTIN_THEMES = [
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

# deterministic blinded orders (spread the "best" across A/B/C so position gives no hint)
PERMS = [
    ["b", "c", "a"], ["c", "a", "b"], ["a", "b", "c"], ["c", "b", "a"], ["b", "a", "c"],
    ["a", "c", "b"], ["c", "a", "b"], ["b", "c", "a"], ["a", "b", "c"], ["c", "b", "a"],
]


def parse_bench(path, themes):
    """Extract per-theme generated text + per-theme dist/judge, anchored on the known
    theme list (base outputs embed '## 短答式問題' etc., so a naive '## ' split breaks)."""
    t = pathlib.Path(path).read_text(encoding="utf-8")
    # header line: "...距離(...): 0.738** / **...審査(0-100...): 69.4" — anchor on the colon
    # before each number so the "0-100" scale text is not mis-captured.
    hd = re.search(r"距離[^:：]*[:：]\s*([\d.]+)", t)
    hj = re.search(r"審査[^:：]*[:：]\s*([\d.]+|None)", t)
    overall = {"dist": hd.group(1) if hd else "?", "judge": hj.group(1) if hj else "?"}
    out = {}
    for th in themes:
        start = t.find("## " + th)
        if start < 0:
            out[th] = {"text": "", "dist": "?", "judge": "?"}
            continue
        end = len(t)
        for th2 in themes:
            p = t.find("## " + th2, start + 3)
            if 0 < p < end:
                end = p
        block = t[start:end]
        m = re.search(r"距離\s*([\d.]+)\s*/\s*審査\s*([^（(\n]+)", block)
        gi = block.find("\n> ")
        out[th] = {
            "text": block[gi + 3:].strip() if gi >= 0 else "",
            "dist": m.group(1) if m else "?",
            "judge": (m.group(2).strip() if m else "?"),
        }
    return out, overall


def build(items, cond_meta, best_key, target, title):
    DATA_JSON = json.dumps(items, ensure_ascii=False)
    META_JSON = json.dumps(cond_meta, ensure_ascii=False)
    tmpl = pathlib.Path(__file__).with_name("blind_eval_template.html").read_text(encoding="utf-8")
    return (tmpl
            .replace("__TITLE_TAG__", title)
            .replace("__DATA__", DATA_JSON)
            .replace("__META__", META_JSON)
            .replace("__BEST__", json.dumps(best_key))
            .replace("__TARGET__", json.dumps(target))
            .replace("__TITLE__", json.dumps(title)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="+", required=True,
                    help="key=path per condition, e.g. base=... mini=... pen=... (worst->best)")
    ap.add_argument("--labels", nargs="+", default=[],
                    help='display "label=key" per condition, e.g. "反復学習版=pen"')
    ap.add_argument("--steps", nargs="+", default=[],
                    help='optional "steps=key", e.g. "学習100step=pen"')
    ap.add_argument("--best", required=True, help="key of the condition machine ranks best")
    ap.add_argument("--target", default="本物", help="name of the real voice being matched")
    ap.add_argument("--title", default="文体ブラインド判定")
    ap.add_argument("--themes-file", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    themes = ([l.strip() for l in open(args.themes_file, encoding="utf-8") if l.strip()]
              if args.themes_file else BUILTIN_THEMES)

    benches = {}
    order = []
    for spec in args.bench:
        k, _, p = spec.partition("=")
        benches[k], ov = parse_bench(p, themes)
        order.append(k)
        benches[k + "__overall"] = ov

    labels = {kv.split("=")[1]: kv.split("=")[0] for kv in args.labels}
    steps = {kv.split("=")[1]: kv.split("=")[0] for kv in args.steps}
    cond_meta = {}
    for k in order:
        cond_meta[k] = {"label": labels.get(k, k), "step": steps.get(k, ""),
                        "dist": benches[k + "__overall"]["dist"],
                        "judge": benches[k + "__overall"]["judge"]}

    items = []
    for i, th in enumerate(themes):
        perm = PERMS[i % len(PERMS)]
        slots = []
        for pos in perm:  # 'a'/'b'/'c' -> order[0..2]
            k = order["abc".index(pos)]
            s = benches[k].get(th, {})
            slots.append({"cond": k, "text": s.get("text", ""),
                          "dist": s.get("dist", "?"), "judge": str(s.get("judge", "?"))})
        items.append({"theme": th, "slots": slots})

    html = build(items, cond_meta, args.best, args.target, args.title)
    pathlib.Path(args.out).write_text(html, encoding="utf-8")
    print(f"wrote {args.out} ({len(html)} bytes) — {len(items)} themes x {len(order)} conditions")


if __name__ == "__main__":
    main()
