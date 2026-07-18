# CLAUDE.md

Claude Code / コーディングエージェントが最初に読むファイル。このリポジトリを
**素早く正しく理解して作業する**ための地図と、越えてはいけない線をまとめる。

- 手順を「実行」したい（学習→評価→記事化を回す）→ **[AGENTS.md](AGENTS.md)** が実行順・コマンド・人間依頼点つきの runbook。
- 人間向けの全体解説（やさしい日本語）→ [README.md](README.md) ／ English: [README.en.md](README.en.md)。

---

## これは何か

**特定の一人の文体で「続きを書く」LoRA アダプタ**を、その人の書いたテキスト
（本・記事・メール・チャット・講演の文字起こし）から学習するパイプライン。
チャットボットでも instruction tuning でもない。目的は一つ、**voice の継続生成**。

土台はフォーク。原本＝落合陽一 [personal-voice-lora-pipeline](https://github.com/ochyai/personal-voice-lora-pipeline)（MIT）の
コーパス構築・学習・比較。このフォークが足したのは **評価（機械2指標＋人間ブラインド）→ 記事化 → エージェント自走** の層。

## 中心となる考え方（これを外すと全部ずれる）

**「増やす → 染み込ませる」**（corpus growth → penetration）。
1. **Growth（2〜3版）**: 種コーパスから始め、版ごとに素材（メール/チャット/講演…）を足す。1版＝1エポック。前版のアダプタを引き継ぐ。→ 語彙とトピックが入る。
2. **Penetration（8〜12版）**: コーパスを固定し、lr を下げながら反復。→ **文体が定着する**。

理由は [docs/corpus-strategy.md](docs/corpus-strategy.md)、実運用の教訓は [docs/lessons.md](docs/lessons.md)。

## パイプライン（データの流れ）

```
collect/*            raw/*.jsonl を作る（各ソース → 自分の文章だけ抽出）
   │
build_corpus.py      raw/ → data/corpus_<name>.jsonl（BUILD_PLAN で組み立て。要カスタム）
   │
orchestrate.py       versions.yaml を上から順に学習。state.json で再開可。systemd 常駐
   │  └─ train_bf16_lora.py   1版=HF Trainer で学習。bf16 / --qlora(4bit)。各版 final_adapter を保存
   │        resume_from=前版アダプタ引き継ぎ / checkpoint 自動再開（クラッシュ復帰）
   ▼
runs/voice-lora-<版>/final_adapter/   ← 成果物（~670MB の LoRA）
   │
compare_checkpoints.py   各版で同じお題を生成 → 比較レポート.md（学習後・自動）
voice_bench.py           固定10お題で採点（文体距離＋ローカルLLM審査）機械2指標
make_blind_eval.py       bench出力 → 人間ブラインド判定の自己完結HTML（第3指標）
   ▼
article-compose（Agent スキル）  レポート/ブログを書く
```

## 主要ファイル

| ファイル | 役割 |
|---|---|
| `pipeline/orchestrate.py` | 司令塔。versions.yaml を順に学習。落ちても `state.json` から再開。exit 3=一時停止(再開・リトライ不消費) / exit 2=abandon |
| `pipeline/train_bf16_lora.py` | 学習本体（HF `Trainer`）。`--resume_from`=前版引き継ぎ、`--output` 内 checkpoint から自動再開、SIGUSR1 で安全停止 |
| `pipeline/build_corpus.py` | コーパス作成**テンプレ**。末尾 `BUILD_PLAN` と source-loader を対象データに合わせる |
| `pipeline/compare_checkpoints.py` | 各 checkpoint を同一プロンプトで生成し比較 md を出す |
| `pipeline/voice_bench.py` | 固定お題ベンチ。文体距離（決定的）＋ `gpt-oss` LLM審査。Ollama 必要 |
| `pipeline/make_blind_eval.py` | bench の md → 人間A/B/C判定の静的HTML（ネット不要） |
| `pipeline/gen_sample.py` | 単発の手動生成（base + adapter で続きを書く） |
| `systemd/voice-lora.service` | ユーザー systemd ユニット。`Restart=on-failure`＋メモリ上限の雛形 |
| `systemd/mem-guard.sh` ＋ `*-memguard.{service,timer}` | メモリ監視。落ちる前に SIGUSR1 で安全停止させる backstop |
| `AGENTS.md` | エージェント実行 runbook（[AGENT]自走 / ⚑HUMAN依頼を明記） |

## よく使うコマンド

```bash
# セットアップ（詳細は docs/setup.md）
python3 -m venv .venv && . .venv/bin/activate
pip install torch transformers peft datasets pyyaml python-docx   # 小型GPUは +bitsandbytes

# 学習スケジュール/評価お題は .example をコピーして使う（実体は .gitignore 済み）
cp pipeline/versions.example.yaml pipeline/versions.yaml          # 小型GPU: versions.mini.example.yaml
cp pipeline/eval_prompts.example.yaml pipeline/eval_prompts.yaml

# 学習を回す
python3 pipeline/orchestrate.py                                   # 単発
systemctl --user start voice-lora.service                         # 常駐（推奨）
journalctl --user -u voice-lora.service -f                        # ログ

# 評価（版ごとに1回）
python3 pipeline/voice_bench.py --base ~/models/<base> --adapter runs/<版>/final_adapter/ --corpus <jsonl> --out runs/bench-<版>.md
python3 pipeline/make_blind_eval.py --bench base=... mini=... pen=... --best pen --out blind-eval.html
```

## 設定・パス（環境変数）

`orchestrate.py` はパスを env で受ける（既定値）:
`VOICE_LORA_ROOT`(~/voice-lora) / `VOICE_LORA_RUNS`(~/runs) / `VOICE_LORA_VENV_PY` /
`VOICE_LORA_RUN_PREFIX`(voice-lora) / `VOICE_LORA_PAUSE_SLEEP`(120)。
学習は `~/voice-lora/{raw,data,runs,logs}` 前提。ベースモデルは `~/models/<base>`。

## 安定運用（長時間学習で落とさない／落ちても再開）

長時間の無人運用で怖いのは RAM OOM でマシンごと落ちること。対策一式は
**[docs/stability.md](docs/stability.md)**（cgroup 上限・swap・earlyoom・安全停止→自動再開、
および **DGX Spark GB10 の統合メモリ特有の注意**）。要点だけ:
- 中断（OOM kill/再起動）は各版の最新 checkpoint から自動再開。失うのは最大 `save_steps` 分。
- `mem-guard` が落ちる前に SIGUSR1 → trainer は step 境界で checkpoint 保存し exit 3 → orchestrator が待って再開（**リトライ不消費**）。
- マシンが丸ごと落ちる（`dmesg | grep -i "out of memory"`）のは学習設定でなく OS 側の問題。

## 絶対にやらないこと（プライバシーの線）

- **`raw/ data/ runs/ models/ hf_cache/` を commit / push しない**。個人データ・学習済み
  LoRA が入る。`.gitignore` 済み。公開してよいのは**コードと出力例だけ**。
- `versions.yaml` / `eval_prompts.yaml` / `*_ochiai.*` / `WORKLOG-ja.md` も gitignore 済み
  （実体は個人設定）。編集するのは **`.example` 版**、または新規に自分用を作る。
- 事実を検証せず「できた」と書かない。記事の数値は bench 出力からの**転記のみ**。
- 生成モデルは**事実を平気で捏造する**（voice 用途にはOK、調査アシスタントには不可）。

## いつ人間に聞くか

個人データ/GPU/gated モデル認証が無い・学習が3回リトライで失敗・人間ブラインド判定・
記事承認・不可逆操作の前。詳細は [AGENTS.md](AGENTS.md) の停止条件。
