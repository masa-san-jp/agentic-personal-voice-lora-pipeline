# Setup on a fresh GPU host

> 日本語の通し手順は [workshop-ja.md](workshop-ja.md) にあります。この setup.md は
> インストールの細部とトラブル対応のリファレンスです。

Reference target: NVIDIA GB10 (DGX Spark), Ubuntu 24.04, CUDA 13. Substitute
paths as needed.

**Two paths.** Big-GPU (≥24GB) trains in native bf16 — nothing extra to install.
Small-GPU / Colab (12-16GB) uses 4-bit QLoRA: `pip install bitsandbytes`, copy
`versions.mini.example.yaml` (it sets `qlora: true`), and you're done. The
orchestrator passes `--qlora` to the trainer automatically when the YAML has
`qlora: true`. See the QLoRA section at the bottom for details.

## 1. System prerequisites

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git poppler-utils
# poppler-utils for pdftotext if you extract PDFs in build_corpus.py
```

## 2. Repo + venv

```bash
git clone <this-repo-url> ~/voice-lora
cd ~/voice-lora
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers peft datasets pyyaml python-docx
```

If you're on a GB10 or other Blackwell GPU, you want a CUDA 12.8+
PyTorch build (`pip install --index-url https://download.pytorch.org/whl/cu128 torch`).

## 3. Base model

```bash
mkdir -p ~/models
pip install huggingface-hub
hf download tokyotech-llm/Llama-3.1-Swallow-8B-v0.5 \
    --local-dir ~/models/Llama-3.1-Swallow-8B-v0.5
```

For non-Japanese projects, use `meta-llama/Llama-3.1-8B-Instruct` or any
8B-class causal LM. Update `base_model` in `pipeline/versions.yaml`.

## 4. Drop raw data into `~/voice-lora/raw/`

The shape `build_corpus.py` expects by default:

```
~/voice-lora/raw/
├── seed_corpus.jsonl          # your initial curated corpus
├── email.jsonl                # mbox → JSONL with is_author_sent flag
├── slack_team_a.jsonl         # Slack export, one msg per line
├── interviews/*.txt           # long-form transcripts
├── presentations/*.md         # talk notes
└── manuscripts/*.docx         # book drafts etc
```

Edit `BUILD_PLAN` in `pipeline/build_corpus.py` to match your actual
layout. The source-loader functions (`add_authored_jsonl`,
`add_text_glob`, `add_slack_export`, `add_docx_dir`) are designed to be
copied and adapted.

## 5. Customize the schedule and prompts

```bash
cp pipeline/versions.example.yaml pipeline/versions.yaml
cp pipeline/eval_prompts.example.yaml pipeline/eval_prompts.yaml
```

Edit `pipeline/eval_prompts.yaml` with prompts about your subject (see
the comments in the example file for the three diagnostic categories).

Adjust `pipeline/versions.yaml` if needed: base_model path, lr schedule,
how many growth steps before penetration. The example has 3 growth +
12 penetration; trim to 2+8 for a one-week run.

## 6. systemd user service

```bash
cp systemd/voice-lora.service ~/.config/systemd/user/
systemctl --user daemon-reload
loginctl enable-linger $USER          # service survives logout
systemctl --user enable voice-lora.service
systemctl --user start voice-lora.service

# verify
systemctl --user status voice-lora.service
journalctl --user -u voice-lora.service -f
```

**Keeping it alive while you're away.** A multi-day run can be killed by OOM,
which is hard to recover from remotely. Before running unattended, set up the
memory caps, swap, earlyoom, and the graceful checkpoint-and-pause watcher in
[stability.md](stability.md). The `voice-lora.service` unit ships with the
memory-limit lines pre-written (commented) so you only size them to your host.
On GB10 (unified memory) this matters more, not less — read the GB10 section.

## 7. (Optional) Watch from another machine

If the GPU box is on Tailscale, you can monitor from anywhere:

```bash
ssh gpu-host "tail -f ~/voice-lora/logs/v06.log"
ssh gpu-host "cat ~/voice-lora/state.json | jq"
ssh gpu-host "nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw --format=csv"
```

## VRAM-constrained path (QLoRA) — Colab / small GPU

QLoRA is built in. You don't edit any code — `train_bf16_lora.py` already
takes a `--qlora` flag that loads the base model in 4-bit (nf4).

```bash
pip install bitsandbytes
cp pipeline/versions.mini.example.yaml pipeline/versions.yaml   # has qlora: true
```

`versions.mini.example.yaml` sets `qlora: true` in `defaults`, and the
orchestrator forwards `--qlora` to the trainer for every version. A 10-version
schedule (2 growth + 8 penetration) at `seq_len: 2048` fits ~12-16GB VRAM.

To run a single version by hand (e.g. on Colab without systemd):

```bash
python pipeline/train_bf16_lora.py \
    --data data/corpus_v6.jsonl \
    --model ~/models/Llama-3.1-Swallow-8B-v0.5 \
    --output ~/runs/voice-lora-v01 \
    --qlora --seq_len 2048 --grad_accum 16 --lr 5e-5 --epochs 1
```

Expect ~30% slower than bf16 native, but the adapter output is the same shape.
If you still OOM, drop `--seq_len` to 1024 first.

## Common failures

- **OOM at step 0**: gradient_checkpointing is off, or seq_len too high.
  Verify `gradient_checkpointing=True` and try `seq_len=2048` first.
- **Loss goes to NaN**: lr too high for the chosen base model. Halve lr.
- **Service starts but training hangs at "Loading weights"**: HF cache
  permission issue. Check that `HF_HOME` points to a writable dir.
- **`compare_checkpoints.py` OOMs**: it loads all checkpoints as named
  adapters. Reduce `--last_n` to 2-3.
- **Adapter inherits don't take effect**: orchestrator's `resume_from:
  previous` looks for `<output>/<prefix>-<prev_vid>/final_adapter`. If
  that's missing (previous version didn't save it), the resume falls
  back to `null` and the model trains from base. Check `logs/<vid>.log`
  for the resolved `resume_from:` line.
- **Whole machine freezes / reboots mid-run (not just the process)**: that's
  RAM OOM, not VRAM. Confirm with `dmesg | grep -i "out of memory"` and apply
  the memory caps / swap / earlyoom in [stability.md](stability.md). On GB10
  the GPU and CPU share one memory pool, so VRAM pressure and RAM pressure are
  the same thing — see the GB10 section there.
- **A version restarts from step 0 after an interruption**: it shouldn't — the
  trainer auto-resumes from the newest `checkpoint-*` in its output dir. If it
  didn't, either `--no_auto_resume` was passed, or the last checkpoint lacks
  `trainer_state.json` (interrupted mid-save; it falls back to the prior one).
