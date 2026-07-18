#!/usr/bin/env python3
"""音声・動画ファイルを、ローカル文字起こしで voice コーパス（JSONL）に変換する.

講演・対談・ボイスメモ・YouTube などの音声/動画から、**自分が話した文字起こし**を
取り出して {"text": ..., "source": ...} の JSONL にします。文字起こしは
**ローカルの Whisper**（faster-whisper 優先、無ければ openai-whisper）で行い、
クラウド STT には一切送りません（プライバシー方針＝ネット隔離に合わせる）。

  # 1話者（独白・自分の講演・ボイスメモ）: 全文をそのまま採用
  python collect/media_transcribe_to_jsonl.py --in ~/talks --out ~/voice-lora/raw/media.jsonl

  # 複数話者（対談・インタビュー）: 話者分離して自分の分だけ残す
  python collect/media_transcribe_to_jsonl.py --in interview.mp4 --out raw/media.jsonl \
      --diarize --speaker SPEAKER_01 --hf-token <HF_TOKEN>

出力は他コレクターと同じ形式。400字未満の断片と重複（先頭512字ハッシュ）は捨て、
明白な PII（メール/長い数字列）はマスクします。

必要なもの:
  - ffmpeg（音声デコード。apt install ffmpeg / brew install ffmpeg）
  - 文字起こし: pip install faster-whisper   （または pip install -U openai-whisper）
  - 話者分離(任意): pip install "pyannote.audio"  ＋ gated モデル
      pyannote/speaker-diarization-3.1 の HF トークン（--hf-token / 環境変数 HF_TOKEN）
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

MIN_CHARS = 400
MAX_CHARS = 16000

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv", ".ts"}
MEDIA_EXTS = AUDIO_EXTS | VIDEO_EXTS

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
LONGNUM = re.compile(r"\b\d[\d\s-]{8,}\d\b")


def redact(t):
    t = EMAIL.sub("<EMAIL>", t)
    t = LONGNUM.sub("<NUM>", t)
    return t


def to_chunks(text):
    """Speech has no paragraph breaks, so accumulate whole sentences into
    [MIN,MAX]-char windows, splitting on sentence-end punctuation (JA + EN)."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    sents = [s for s in re.split(r"(?<=[。．！？!?])\s*", text) if s.strip()]
    out, buf = [], ""
    for s in sents:
        s = s.strip()
        if len(buf) + len(s) + 1 > MAX_CHARS and buf:
            out.append(buf)
            buf = s
        else:
            buf = f"{buf} {s}" if buf else s
    if buf:
        out.append(buf)
    return [c for c in out if len(c) >= MIN_CHARS]


def find_media(indir):
    p = Path(indir).expanduser()
    if p.is_file():
        return [p] if p.suffix.lower() in MEDIA_EXTS else []
    return sorted(q for q in p.rglob("*")
                  if q.is_file() and q.suffix.lower() in MEDIA_EXTS)


# ------------------------------------------------------------------
# Transcription backends (lazy imports — heavy, optional deps)
# ------------------------------------------------------------------

def transcribe(path, model_name, lang, device, compute_type):
    """Return [{'start','end','text'}] via local Whisper. faster-whisper first."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return _transcribe_openai(path, model_name, lang)
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(str(path), language=lang, vad_filter=True)
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]


def _transcribe_openai(path, model_name, lang):
    try:
        import whisper
    except ImportError:
        sys.exit("文字起こしエンジンがありません。次のどちらかを入れてください:\n"
                 "  pip install faster-whisper      (推奨・速い)\n"
                 "  pip install -U openai-whisper\n"
                 "どちらも ffmpeg が必要です（apt install ffmpeg / brew install ffmpeg）。")
    # openai-whisper uses names like tiny/base/small/medium/large-v3
    name = model_name if model_name not in {"large-v2", "large-v1"} else "large"
    model = whisper.load_model(name)
    res = model.transcribe(str(path), language=lang)
    return [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in res.get("segments", [])]


def diarize(path, hf_token, device):
    """Return [(start, end, speaker_label)] via pyannote (gated model)."""
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        sys.exit("--diarize には pyannote.audio が必要です:\n"
                 '  pip install "pyannote.audio"\n'
                 "さらに gated モデル pyannote/speaker-diarization-3.1 への同意と HF トークン"
                 "（--hf-token か環境変数 HF_TOKEN）が要ります。")
    if not hf_token:
        sys.exit("--diarize には HF トークンが必要です（--hf-token か環境変数 HF_TOKEN）。"
                 "huggingface.co で pyannote/speaker-diarization-3.1 の利用規約に同意してから発行。")
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token=hf_token)
    if device:
        import torch
        pipe.to(torch.device(device))
    ann = pipe(str(path))
    return [(turn.start, turn.end, spk) for turn, _, spk in ann.itertracks(yield_label=True)]


# ------------------------------------------------------------------
# Speaker assignment (pure — unit-tested)
# ------------------------------------------------------------------

def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speakers(segments, diar_turns):
    """Tag each transcript segment with the diarization speaker it overlaps most."""
    for seg in segments:
        best, best_spk = 0.0, None
        for (s, e, spk) in diar_turns:
            ov = _overlap(seg["start"], seg["end"], s, e)
            if ov > best:
                best, best_spk = ov, spk
        seg["speaker"] = best_spk
    return segments


def select_speaker(segments, speaker):
    """Keep only the target speaker's segments. If speaker is None, auto-pick the
    one with the most total speaking time (the dominant voice) and keep only it."""
    if speaker is None:
        dur = {}
        for seg in segments:
            spk = seg.get("speaker")
            if spk is not None:
                dur[spk] = dur.get(spk, 0.0) + (seg["end"] - seg["start"])
        if not dur:
            return segments
        speaker = max(dur, key=dur.get)
        print(f"  自動選択: 最も長く話している話者 {speaker} "
              f"({dur[speaker]:.0f}s) を採用。違う場合は --speaker で指定してください。")
    return [seg for seg in segments if seg.get("speaker") == speaker]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True,
                    help="音声/動画ファイル、またはそれらの入ったフォルダ")
    ap.add_argument("--out", required=True, help="出力 JSONL")
    ap.add_argument("--source", default="media", help="source タグ")
    ap.add_argument("--model", default="large-v3",
                    help="Whisper モデル名（tiny/base/small/medium/large-v3）。既定 large-v3")
    ap.add_argument("--lang", default=None, help="言語コード（例: ja）。既定は自動判定")
    ap.add_argument("--device", default="auto", help="cpu / cuda / auto。既定 auto")
    ap.add_argument("--compute-type", default="default",
                    help="faster-whisper の compute_type（例: int8, float16）")
    ap.add_argument("--diarize", action="store_true",
                    help="複数話者を分離して1人だけ残す（対談/インタビュー用）")
    ap.add_argument("--speaker", default=None,
                    help="残す話者ラベル（例: SPEAKER_01）。未指定なら最長話者を自動採用")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                    help="pyannote gated モデル用 HF トークン（環境変数 HF_TOKEN でも可）")
    ap.add_argument("--keep-code", action="store_true", help="（未使用・互換のため）")
    args = ap.parse_args()

    files = find_media(args.indir)
    if not files:
        sys.exit(f"対象メディアが見つかりません: {args.indir}\n"
                 f"対応拡張子: {', '.join(sorted(MEDIA_EXTS))}")

    device = None if args.device == "auto" else args.device
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    seen, written, total_chars, done = set(), 0, 0, 0
    with open(out, "w", encoding="utf-8") as fo:
        for path in files:
            print(f"[{done + 1}/{len(files)}] transcribing {path.name} ...")
            try:
                segments = transcribe(path, args.model, args.lang, device, args.compute_type)
            except SystemExit:
                raise
            except Exception as e:
                print(f"  ! {path.name}: 文字起こし失敗 ({e})", file=sys.stderr)
                continue
            done += 1

            if args.diarize:
                turns = diarize(path, args.hf_token, device)
                segments = assign_speakers(segments, turns)
                segments = select_speaker(segments, args.speaker)

            text = " ".join(seg["text"] for seg in segments if seg.get("text"))
            for ch in to_chunks(text):
                ch = redact(ch)
                key = hashlib.sha256(ch[:512].encode("utf-8", "ignore")).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                fo.write(json.dumps({"text": ch, "source": args.source, "id": key[:12]},
                                    ensure_ascii=False) + "\n")
                written += 1
                total_chars += len(ch)

    print(f"書き出し: {out}  files={done}/{len(files)} 件={written:,} "
          f"字={total_chars:,} ≈ {total_chars // 2:,} tok(JA)")
    if written == 0:
        print("  ! 0件でした。--diarize/--speaker の指定、または音声の中身を確認してください。",
              file=sys.stderr)


if __name__ == "__main__":
    main()
