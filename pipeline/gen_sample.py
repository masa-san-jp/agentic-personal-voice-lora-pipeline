#!/usr/bin/env python3
"""gen_sample.py — generate voice samples from a trained LoRA adapter.
Loads base + adapter, continues each prompt. Voice LoRA = continuation, not chat.
"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = sys.argv[1]
ADAPTER = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
if ADAPTER:
    model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

# 継続生成のための書き出し例（voice LoRA はチャットでなく「続きを書く」）。自分用に書き換える。
PROMPTS = [
    "今日はいい天気で、",
    "最近考えているのは、",
    "毎日続けていて思うのは、",
]
for p in PROMPTS:
    ids = tok(p, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=140, do_sample=True,
                             temperature=0.8, top_p=0.9, repetition_penalty=1.15,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=True)
    print("PROMPT:", p)
    print("OUT:", text.replace("\n", " ").strip())
    print("----")
