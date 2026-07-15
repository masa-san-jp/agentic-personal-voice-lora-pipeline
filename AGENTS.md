# AGENTS.md — コーディングエージェント向け実行手順

このリポジトリを読んだエージェント（Claude Code 等）が、**voice-LoRA を頭から最後まで自分で回せる**ようにするための実行手順書。

各ステップに次のタグを付ける：

- **[AGENT]** — エージェントが自分で実行する。
- **[⚑ HUMAN]** — 人間の作業・判断・権限が要る。ここに来たら**止まって、何をしてほしいかを1メッセージで具体的に人へ伝え、返事を待つ**。黙って飛ばさない・勝手に代行しない。

やさしい日本語の全体解説は [README.md](README.md)、当日運用は [docs/workshop-ja.md](docs/workshop-ja.md)。本ファイルは「実行順・コマンド・人間依頼点」だけを機械的にまとめたもの。

---

## 0. 最初に前提を点検する — [AGENT]

次を確認し、欠けていれば対応する ⚑ HUMAN ステップへ誘導する。

```bash
nvidia-smi                      # GPU があるか
ls ~/models/                    # ベースモデルがあるか（無ければ 2）
ls raw/ data/ 2>/dev/null       # 個人コーパスがあるか（無ければ 1）
test -f pipeline/versions.yaml  # 学習スケジュールがあるか（無ければ 4）
```

---

## 1. 個人データを集める — [⚑ HUMAN]（→ 変換は [AGENT]）

**なぜ人間**：メール・DM・非公開原稿などの個人データは自動収集できず、外部にも出せない。人間が `raw/` に置く。
**人へ依頼する文例**：「note / X / Gmail などから自分の文章を集めて `raw/` に入れてください。集め方は `collect/README.md`、変換スクリプトは `collect/` にあります」。
データが置かれたら [AGENT] が変換を実行：

```bash
python3 collect/files_to_seed_jsonl.py <入力フォルダ> raw/seed.jsonl   # txt/md/docx/pdf → 種コーパス
```

## 2. 環境構築 — [AGENT]

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install torch transformers peft datasets accelerate bitsandbytes pyyaml   # 詳細は docs/setup.md
```

## 3. ベースモデル取得 — [AGENT]（gated なら [⚑ HUMAN] 認証）

```bash
huggingface-cli download <base-model> --local-dir ~/models/<base>
```

**401/403（gated モデル）で失敗したら止まる** → [⚑ HUMAN] に「`huggingface-cli login` でトークンを通してください（対話認証）」と依頼して待つ。

## 4. コーパスと学習スケジュールを用意する — [AGENT]（BUILD_PLAN の中身は要人間確認）

```bash
# コーパスの組み立て方は build_corpus.py 末尾の BUILD_PLAN を対象データに合わせる（初回は人間に方針確認）
python3 pipeline/build_corpus.py --name v8 --out raw/v8.jsonl
cp pipeline/versions.mini.example.yaml pipeline/versions.yaml   # base_model / corpus キーを実体に合わせて編集
```

「増やす→染み込ませる（＝同じデータの反復学習）」がなぜ効くかは [docs/corpus-strategy.md](docs/corpus-strategy.md)。

## 5. 学習を回す — [AGENT] が起動（GPU 時間は [⚑ HUMAN] 依存）

```bash
python3 pipeline/orchestrate.py        # versions.yaml を順に学習。落ちても state.json で再開（最大3回/版）
# 常駐で回すなら: systemctl --user start voice-lora.service
```

- 完了物：`runs/voice-lora-<版>/final_adapter/`（浅い版＝checkpoint少 / 反復版＝checkpoint多）。
- **GPU が他で塞がっている / 3回リトライしても失敗** → 止まって [⚑ HUMAN] に状況を伝える。

## 6. 機械ベンチ（指標1・2）— [AGENT]

素モデルと各アダプタで固定10お題×400字を生成し、文体距離＋LLM審査を出す。**版ごとに1回ずつ**：

```bash
python3 pipeline/voice_bench.py --base ~/models/<base> --adapter -                              --corpus raw/seed.jsonl --themes-file themes.txt --out runs/bench-base.md
python3 pipeline/voice_bench.py --base ~/models/<base> --adapter runs/<浅い版>/final_adapter/   --corpus raw/seed.jsonl --themes-file themes.txt --out runs/bench-mini.md
python3 pipeline/voice_bench.py --base ~/models/<base> --adapter runs/<反復版>/final_adapter/   --corpus raw/seed.jsonl --themes-file themes.txt --out runs/bench-pen.md
```

`--themes-file` は任意（省略すると内蔵の10お題を使う）。別人格を測るなら、その人に合う10お題を1行1件で `themes.txt` に用意して渡す。`voice_bench.py` と `make_blind_eval.py` で**同じ themes を使う**こと。
（LLM審査はローカル `gpt-oss:20b` を `http://localhost:11434` で呼ぶ。Ollama が要る。）

## 7. 人間ブラインド判定（指標3）— [AGENT] が生成、[⚑ HUMAN] が判定

bench 出力3本から、条件を伏せた A/B/C 判定の**自己完結HTML**を生成：

```bash
python3 pipeline/make_blind_eval.py \
  --bench base=runs/bench-base.md mini=runs/bench-mini.md pen=runs/bench-pen.md \
  --labels "素モデル (base)=base" "浅い学習 (25step)=mini" "反復学習版 (100step)=pen" \
  --steps "学習 0 step=base" "学習 25 step=mini" "学習 100 step=pen" \
  --best pen --target "<対象の名前>" --themes-file themes.txt --out blind-eval.html
```

**そのあと [⚑ HUMAN] に依頼して待つ**：「`blind-eval.html` を開いて、各お題で本物にいちばん近い1本を選び、最後に出る各版の票数（例：反復学習版◯／浅い◯／素◯）を教えてください」。
人が判定するのがこのステップの本質で、自動化はしない。返ってきた票を記録する。
判定者は仮説を知らない独立した複数人ほど強い。1人だけなら結論は「兆候」どまり、と正直に書く。

## 8. 評価レポート／ブログを書く — [AGENT] が執筆、[⚑ HUMAN] が承認

素材＝`runs/bench-*.md`（機械2指標）＋人間ブラインドの票。これで2本書く：

- **技術レポート**：客観・である調。本文（背景/課題/手法/結果/考察/限界）＋付録（実験仕様設計・実験手順・実測データ台帳）。
- **ブログ**：人格の一人称。独自でよいのは「誰が書いているか」だけ。

守ること：数値は bench 出力からの**転記のみ**（記憶で書かない）／生成文の引用は**逐語**／テーマ名や引用など**実データは改変しない**／**言葉は初見の人でも定義なしで分かる一般語彙**（自作の造語・比喩的な言い換えを避け、必要な概念は初出で一般語の定義を併記）。

**着手前に [⚑ HUMAN] へ**：中心主張・章立て・分量を1メッセージで見せて GO をもらってから清書する（承認ゲート）。
（Agent 実行環境に `article-compose` スキルがあればそれに従う。無くても本節の要件で書ける。）

---

## いつ止まって人間に聞くか（[AGENT] の停止条件）

1. 個人データ / GPU / gated モデル認証が無い
2. 学習が3回リトライしても失敗する
3. 人間ブラインドの判定（7）と、記事の承認（8）
4. 不可逆操作（削除・上書き・force push・費用発生・本番変更）の前
5. `raw/ data/ runs/ models/` をコミット/push しそうになった時（**絶対にしない**）

## 絶対にしないこと

- `raw/ data/ runs/ models/`（個人データ・学習済み LoRA）をコミット/push（`.gitignore` 済み。公開してよいのはコードと出力例だけ）。
- 事実を検証せずに「できた」と書く。数値は bench 出力からの転記のみ。
