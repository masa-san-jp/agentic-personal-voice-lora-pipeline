#!/usr/bin/env python3
"""LoRA training for causal LMs on a single GPU — bf16 native or 4-bit QLoRA.

Designed for the NVIDIA GB10 (DGX Spark, 128GB unified) but works on any
GPU with enough VRAM for an 8B-class model in bf16 + LoRA + gradient
checkpointing (≈ 25GB for the configuration here). Native bf16 by default —
no quantization, no bitsandbytes.

If your GPU is smaller than ~24GB (e.g. a 16GB Colab T4, an RTX 3060/4060),
pass `--qlora`. The base model is then loaded in 4-bit (nf4) and the LoRA
adapter trains on top of it. Same adapter shape, same output, ~12-16GB VRAM,
~30% slower. Requires `pip install bitsandbytes`.

Continuous-friendly: when invoked with `--resume_from <adapter_dir>`, loads
the LoRA weights from that dir before starting a fresh training run on this
corpus. This is how the orchestrator chains versions: each step inherits
the previous step's adapter and trains for one more (epoch-on-corpus) pass.

Crash-friendly: within a single version, the HF Trainer checkpoints every
`save_steps`. If this run is interrupted (OOM kill, reboot, graceful pause)
and re-invoked with the same `--output`, it auto-resumes from the latest
`checkpoint-*` in that dir — restoring optimizer/scheduler/step, not just
weights — so an interrupted version loses at most `save_steps` steps instead
of restarting from zero. Disable with `--no_auto_resume`.

Pause-friendly: on SIGUSR1 the trainer finishes the current step, writes a
checkpoint, and exits with code 3 ("paused, resume me"). This is how a
memory watcher (see docs/stability.md) asks training to step aside *before*
the kernel OOM-kills it, without losing progress. The orchestrator treats
exit 3 as a resumable pause, not a failure.
"""
import argparse
import json
import os
import re
import signal
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer, TrainerCallback, TrainingArguments,
)

# Exit code the trainer uses when it stops early because of a graceful-pause
# signal. The orchestrator interprets this specifically (resume, don't count
# as a retry). Anything else non-zero is a real failure.
PAUSE_EXIT_CODE = 3

# Set by the SIGUSR1 handler; polled by GracefulStopCallback at each step end.
_STOP_REQUESTED = False


def _request_graceful_stop(signum, _frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(
        f"\n[signal {signum}] graceful stop requested — will checkpoint at the "
        f"end of the current step and exit {PAUSE_EXIT_CODE} for resume.",
        flush=True,
    )


class GracefulStopCallback(TrainerCallback):
    """Turn a pending SIGUSR1 into a clean checkpoint-and-stop at a step boundary.

    Stopping between steps (rather than mid-backward) guarantees the saved
    checkpoint is consistent, so the resumed run continues from a valid state.
    """

    def on_step_end(self, args, state, control, **kwargs):
        if _STOP_REQUESTED:
            control.should_save = True
            control.should_training_stop = True
        return control


def find_last_checkpoint(output_dir):
    """Return the newest valid `checkpoint-<step>` dir in output_dir, or None."""
    root = Path(output_dir)
    if not root.is_dir():
        return None
    candidates = []
    for d in root.glob("checkpoint-*"):
        m = re.match(r"checkpoint-(\d+)$", d.name)
        # A checkpoint is only resumable if the trainer state was written.
        if m and d.is_dir() and (d / "trainer_state.json").exists():
            candidates.append((int(m.group(1)), d))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_records", type=int, default=None)
    ap.add_argument("--qlora", action="store_true",
                    help="Load the base model in 4-bit (nf4) for small GPUs (<24GB). "
                         "Requires bitsandbytes. ~30%% slower, same adapter output.")
    ap.add_argument("--resume_from", default=None,
                    help="Path to adapter to resume from")
    ap.add_argument("--no_auto_resume", action="store_true",
                    help="Do not auto-resume from the latest checkpoint in "
                         "--output. By default an interrupted run continues "
                         "from its last checkpoint instead of restarting.")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"Loading tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    load_kwargs = dict(
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",  # built-in scaled dot product attention
    )
    if args.qlora:
        print(f"Loading model (4-bit QLoRA): {args.model}")
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        print(f"Loading model (bf16): {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

    if args.qlora:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)

    if args.resume_from and Path(args.resume_from).exists():
        print(f"Resuming from adapter: {args.resume_from}")
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
    else:
        print("Creating fresh LoRA adapter")
        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    print(f"Loading data: {args.data}")
    records = load_jsonl(args.data)
    if args.max_records:
        records = records[: args.max_records]
    print(f"Records: {len(records):,}")

    ds = Dataset.from_list([{"text": r["text"]} for r in records])

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.seq_len, padding=False)

    ds = ds.map(tokenize, batched=True, remove_columns=["text"],
                num_proc=4, desc="Tokenizing")
    total_tokens = sum(len(x) for x in ds["input_ids"])
    print(f"Total tokens: {total_tokens:,}")

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=5,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=ds, data_collator=collator)
    trainer.add_callback(GracefulStopCallback())

    # SIGUSR1 = "checkpoint and step aside" (memory watcher / manual pause).
    # See docs/stability.md. We deliberately do NOT trap SIGTERM here so a
    # normal `systemctl stop` still tears the process down promptly.
    signal.signal(signal.SIGUSR1, _request_graceful_stop)

    # Crash recovery: resume this version from its own latest checkpoint if one
    # exists. Distinct from --resume_from (which inherits the *previous*
    # version's adapter); here we continue an interrupted run of *this* version,
    # restoring optimizer + lr schedule + step, so we lose at most save_steps.
    resume_ckpt = None
    if not args.no_auto_resume:
        resume_ckpt = find_last_checkpoint(args.output)
        if resume_ckpt is not None:
            print(f"Auto-resuming from checkpoint: {resume_ckpt}")

    print("Starting training...")
    trainer.train(resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None)

    if _STOP_REQUESTED:
        # Graceful pause: checkpoint is on disk, but the version is NOT done.
        # Exit with the pause code so the orchestrator resumes rather than
        # treating a partial adapter as a finished version.
        print(f"Graceful stop complete: latest checkpoint saved under "
              f"{args.output}. Exiting {PAUSE_EXIT_CODE} for orchestrated resume.")
        sys.exit(PAUSE_EXIT_CODE)

    out = Path(args.output)
    trainer.model.save_pretrained(out / "final_adapter")
    tok.save_pretrained(out / "final_adapter")
    print(f"Saved adapter: {out / 'final_adapter'}")


if __name__ == "__main__":
    main()
