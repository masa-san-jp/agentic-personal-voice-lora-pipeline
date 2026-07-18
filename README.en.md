# agentic-personal-voice-lora-pipeline (English)

> 日本語版は [README.md](README.md) を読んでください。
> Japanese is the primary README; this is the English mirror.

> **About this fork.** This repository is a fork of Yoichi Ochiai's
> [personal-voice-lora-pipeline](https://github.com/ochyai/personal-voice-lora-pipeline) (MIT).
> The original corpus/training/compare machinery is inherited as-is; the design,
> implementation, and examples (`ochiai-v20`) are all his. What this fork adds on
> top is an **evaluation-to-writeup layer**: (1) a machine benchmark with two
> metrics — a deterministic stylometric distance and a local-LLM similarity judge
> (`pipeline/voice_bench.py`); (2) a **human blind test** — a self-contained,
> offline HTML that hides which adapter is which and asks a person to pick the
> closest sample (`pipeline/make_blind_eval.py`); and (3) [`AGENTS.md`](AGENTS.md),
> a runbook so a coding agent can drive the whole flow end-to-end, with the
> human-required steps clearly marked. Same MIT license as upstream.

> A pipeline for training a LoRA adapter that **writes like a specific person**
> from their accumulated text. Built and tuned on an NVIDIA GB10 (DGX Spark)
> for a ~100M-token Japanese-essayist corpus, and used in the Ochiai-juku
> workshop where each participant trains a voice model on **their own** data.

This repo packages the practical scaffolding: orchestrator, corpus builder,
training script (bf16 **or** 4-bit QLoRA), evaluation harness, and systemd
unit. The strategy it encodes — **corpus growth, then penetration** — is what
made the difference between "model knows the topics" and "model writes in this
voice."

---

## What this is for

You have ~50-200M tokens of text written by one person (books, articles,
emails, chat logs, talk transcripts). You want a model that, given a prompt
the person has never seen, produces something that reads like them — same
sentence rhythm, same idiosyncratic turns, same way of opening an essay. Not
a chatbot; a continuation engine in their voice.

This is **continued pre-training as a LoRA adapter** on top of a strong base
model (Llama-3.1-Swallow-8B for Japanese, swap for your language). LoRA, not
full finetune, because the base's general knowledge stays intact while the
adapter encodes the voice. r=64, alpha=128 across all linear modules gives a
~670MB adapter — small enough to ship, large enough to carry the style.

It is **not** instruction tuning, **not** RLHF, **not** dialog SFT. Those are
later steps if you want a chatbot. The objective here is one thing: pure
continuation in voice.

---

## Why "corpus growth, then penetration"

The trap on a first attempt is to train a single epoch on a fixed corpus and
call it done. The model reliably reproduces the subject's *vocabulary* but
writes it inside a textbook frame ("This concept refers to…"). It hasn't
internalized the voice; it's *translating into* the voice from a generic frame.

Two phases work better:

1. **Growth phase** (2-3 versions). Start with a curated seed corpus. Each
   next version adds another bundle of source material — emails, chat logs,
   talk transcripts, manuscripts. Each version is one epoch, inheriting the
   previous version's adapter. Vocabulary and topic priors lock in fast.

2. **Penetration phase** (8-12 versions). Freeze the corpus at its largest.
   Iterate epochs at decaying learning rate (5e-5 → 3e-6). This is where
   style locks in. Around epoch 4-6 the model stops opening with "Let me
   explain…" and starts using the subject's actual signature moves.

The schedule is in [`pipeline/versions.example.yaml`](pipeline/versions.example.yaml)
(15-version, big-GPU) and [`pipeline/versions.mini.example.yaml`](pipeline/versions.mini.example.yaml)
(10-version, small-GPU/QLoRA).

---

## Two hardware paths

| | Big-GPU (bf16) | Small-GPU / Colab (QLoRA) |
|---|---|---|
| VRAM | ≥24GB (GB10 128GB, A100, RTX 4090) | 12-16GB (T4, RTX 3060/4060) |
| Trainer flag | (default) | `--qlora` (`pip install bitsandbytes`) |
| Schedule | `versions.example.yaml` (15 ver) | `versions.mini.example.yaml` (10 ver) |
| Speed | reference | ~30% slower |
| Result | identical adapter shape | identical adapter shape |

Reference timing (GB10, Llama-3.1-Swallow-8B, bf16, seq_len 4096, ~100M-token
corpus): ~25s/step, ~23h/epoch, ~14 days for the full 15-version schedule.

---

## Step 0: build your own corpus

The training pipeline assumes you already have your text in `~/voice-lora/raw/`.
Workshop participants start one step earlier — assembling their own personal
archive (the generic version of Ochiai's `homo-convivium`). See
[`collect/README.md`](collect/README.md) for per-source export guides and
converter scripts (`files_to_seed_jsonl.py`, `twitter_archive_to_jsonl.py`,
`gmail_mbox_to_jsonl.py`) that produce the `raw/*.jsonl` files build_corpus expects.

## Quick start (big-GPU)

```bash
git clone https://github.com/masa-san-jp/agentic-personal-voice-lora-pipeline.git ~/voice-lora
cd ~/voice-lora
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers peft datasets pyyaml python-docx
# small GPU also: pip install bitsandbytes

mkdir -p ~/models
hf download tokyotech-llm/Llama-3.1-Swallow-8B-v0.5 --local-dir ~/models/Llama-3.1-Swallow-8B-v0.5

# put your raw data into ~/voice-lora/raw/, then:
cp pipeline/versions.example.yaml pipeline/versions.yaml       # or versions.mini.example.yaml
cp pipeline/eval_prompts.example.yaml pipeline/eval_prompts.yaml
# edit build_corpus.py BUILD_PLAN + eval_prompts.yaml for your subject

cp systemd/voice-lora.service ~/.config/systemd/user/
systemctl --user daemon-reload
loginctl enable-linger $USER
systemctl --user enable --now voice-lora.service
journalctl --user -u voice-lora.service -f
```

See [docs/setup.md](docs/setup.md) for the full install, and
[docs/workshop-ja.md](docs/workshop-ja.md) for the step-by-step workshop guide.

**Before running unattended:** a multi-day run on a home machine can be taken
down by OOM, which is hard to recover from remotely. [docs/stability.md](docs/stability.md)
covers keeping the box alive (cgroup memory caps, swap, earlyoom) and resuming
without progress loss (per-checkpoint auto-resume + a graceful checkpoint-and-pause
signal), including the unified-memory caveats specific to DGX Spark (GB10).

---

## What it gets you, what it doesn't

Gets you: the model writes *from inside* the voice, not *toward* it. Off-domain
generalization. Style that holds without explicit anchoring. A deployable
artifact (a ~670MB adapter you can run in Ollama / vLLM).

Doesn't get you: factual loyalty (it will confidently make up "WES
(Water-Ethanol-Solvent)" in a semiconductor essay), tool use, multi-turn
coherence. Those are post-training steps that compose on top.

---

## License

Code & docs: **MIT** (see [LICENSE](LICENSE)). The generated text under
`examples/` is model output trained on Yoichi Ochiai's writings — included for
reference/education only; don't redistribute it as your own or present it as
authentic Ochiai statements.
