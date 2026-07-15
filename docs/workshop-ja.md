# ワークショップ手順 — 自分の声のLLMを作る

落合塾ワークショップ用の、いちばん丁寧な手順書です。
**ゴールは「自分のテキストで、自分の文体で書くモデル（LoRA）を1つ作る」**こと。
専門用語はなるべく避け、コピペで進められるようにしています。

困ったら最後の「うまくいかないとき」を見てください。

---

## 0. はじめに：今日やること（全体像）

1. **集める** … 自分が書いた文章をかき集める（本・記事・メール・チャット・講演）。
2. **整える** … それを「1行＝1かたまり」の形（JSONL）にそろえる。
3. **増やす** … 種データから始めて、素材を足していく（2〜3回）。
4. **染み込ませる** … 同じデータを、学習率を下げながら何度も読み返す（8〜12回）。
5. **見る** … できた文章を比較レポートで確認する。

3〜4は機械が自動でやります。あなたが手を動かすのは主に **1・2** です。
「**コーパス（=集めたテキスト）こそが作品**」だと思ってください。モデルの良し悪しの9割はここで決まります。

---

## 1. 自分のGPUコースを選ぶ

| | A. 自前GPU（bf16） | B. 小型GPU / Colab（QLoRA） |
|---|---|---|
| 目安 | VRAM 24GB以上 | VRAM 12〜16GB |
| 例 | DGX Spark, A100, RTX 4090 | Colab T4, RTX 3060/4060 |
| 追加で入れる物 | なし | `bitsandbytes` |
| 使う設定ファイル | `versions.example.yaml` | `versions.mini.example.yaml` |

**わからなければ B から**始めて大丈夫です。できあがるモデルは同じ形です。

---

## 2. 道具をそろえる

```bash
git clone https://github.com/masa-san-jp/agentic-personal-voice-lora-pipeline.git ~/voice-lora
cd ~/voice-lora
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers peft datasets pyyaml python-docx
pip install bitsandbytes        # ← Bコースの人だけ
```

ベースモデル（土台）を落とします。日本語ならこれ：

```bash
mkdir -p ~/models && pip install huggingface-hub
hf download tokyotech-llm/Llama-3.1-Swallow-8B-v0.5 \
    --local-dir ~/models/Llama-3.1-Swallow-8B-v0.5
```

---

## 3. 自分のテキストを集めて `raw/` に入れる（＝自分の homo-convivium を作る）

ここが今日いちばん大事な作業です。落合は個人アーカイブ `homo-convivium` を持っていますが、
**あなたは自分専用のそれを今から作ります。** 集め方と変換スクリプトは
[`../collect/README.md`](../collect/README.md) に全部まとまっています。まずそれを開いてください。

ざっくり言うと、note・ブログ・書籍・X・メールなどを書き出して、付属スクリプトで
`~/voice-lora/raw/` に決まった形（JSONL）で並べます。例：

```bash
# 長い文章（note/ブログ/原稿/文字起こし）をフォルダに集めて種コーパスへ
python collect/files_to_seed_jsonl.py --in ~/my-writings --out ~/voice-lora/raw/seed_corpus.jsonl
# X アーカイブから自分のツイートだけ
python collect/twitter_archive_to_jsonl.py --tweets ~/twitter-archive/data/tweets.js --out ~/voice-lora/raw/twitter.jsonl
# Gmail（Takeout）から自分が送ったメールだけ
python collect/gmail_mbox_to_jsonl.py --mbox ~/Takeout/Mail/all.mbox --me you@example.com --out ~/voice-lora/raw/email.jsonl
```

最終的に `~/voice-lora/raw/` がこうなっていればOKです：

```
~/voice-lora/raw/
├── seed_corpus.jsonl     ← まずはこれ1つでもOK（自分の文章を1行ずつ）
├── email.jsonl           ← 自分が送ったメールだけ
├── interviews/*.txt      ← 対談・取材の文字起こし
├── presentations/*.md    ← 講演メモ・原稿
└── manuscripts/*.docx    ← 本やレポートの下書き
```

### 種データ `seed_corpus.jsonl` の作り方（最小）

1行に1かたまり。こんな形です（`text` だけあれば動きます）:

```json
{"text": "ここにあなたが書いた文章のかたまり。段落いくつか分くらい。"}
{"text": "次のかたまり。"}
```

**コツ**
- 短すぎる断片（あいさつ、「了解です」など）は入れない。400字未満は自動で捨てられます。
- 他人の発言は入れない。チャットやメールは「**自分が書いた分だけ**」に絞る。
- まずは種だけで一度通してみて、後から素材を足すのがおすすめです。

> 個人情報が心配な人へ：`raw/` と `data/` は git に上がりません（最初から除外済み）。
> 学習はネットから切り離したGPUの中だけで完結させましょう。

---

## 4. 「どう食べさせるか」を書く（build_corpus.py）

`pipeline/build_corpus.py` の下のほうにある **`BUILD_PLAN`** を、自分のデータに合わせて直します。
読み込み関数（`add_existing_jsonl` / `add_authored_jsonl` / `add_text_glob` / `add_docx_dir` / `add_slack_export`）が
そろっているので、使うものをコピーして並べるだけです。

最小なら、`build_v6` が種ファイルを読むだけになっていればOKです:

```python
def build_v6(records):
    add_existing_jsonl(records, RAW / "seed_corpus.jsonl", source_tag="Seed")
```

素材を足したくなったら `build_v7` / `build_v8` に追記していきます（前の段を呼んでから足す）。

> 当日は、落合自身がアーカイブ（homo-convivium）をどう `build_corpus` に繋いだかの
> 実例ファイルを手元でお見せします（中身の本文は配布しません）。形の参考にしてください。

---

## 5. 設定ファイルを用意する

```bash
# Aコース（自前GPU）:
cp pipeline/versions.example.yaml pipeline/versions.yaml
# Bコース（小型GPU / Colab）:
cp pipeline/versions.mini.example.yaml pipeline/versions.yaml

cp pipeline/eval_prompts.example.yaml pipeline/eval_prompts.yaml
```

`pipeline/eval_prompts.yaml` を開いて、**自分について問う6つのお題**に書き換えます。

- **known（2つ）**: 自分がよく書くテーマ・自分が作った言葉。
- **generalize（2つ）**: 自分があまり書かない話題（ちゃんと文体が移ったかの試験）。
- **style（2つ）**: 書き出しの癖（深夜の独白、学生への助言、など）。

ファイル内のコメントに、それぞれの書き方の見本があります。

---

## 6. 学習を走らせる

長時間まわるので、systemd に任せて「寝てる間も続く」状態にします。

```bash
cp systemd/voice-lora.service ~/.config/systemd/user/
systemctl --user daemon-reload
loginctl enable-linger $USER
systemctl --user enable --now voice-lora.service
```

**Colab など systemd が無い環境**では、直接まわしてもOKです（ただしセッションが切れると止まります）:

```bash
python pipeline/orchestrate.py
```

止まっても `state.json` に進み具合が残るので、もう一度実行すれば**続きから**再開します。

---

## 7. 見守る

```bash
journalctl --user -u voice-lora.service -f     # 全体ログ（systemdのとき）
tail -f ~/voice-lora/logs/v06.log              # 各版のログ
cat ~/voice-lora/state.json                    # どこまで進んだか
```

GPUの**温度**も時々見ます（使用率は常に99%なので健康診断になりません）:

```bash
nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv
```

---

## 8. できた文章を読む（ここが楽しい）

各版が終わるたびに、比較レポートが自動で出ます:

```
~/runs/voice-lora-<版>/voice_compare_report_<日時>.md
```

**上から下へ**読んでください。同じお題に対して、
土台モデル → 各チェックポイント、と進むにつれて、
教科書みたいな文から、あなたの文体へ変わっていきます。
見本は [`../examples/voice_compare_report_example.md`](../examples/voice_compare_report_example.md)。

- 「known は良いが style が固い」→ まだ早い。浸透の回数を増やす。
- 「style は良いが generalize がダメ」→ コーパスが狭い。素材の種類を増やす。

---

## 9. できたモデルを使う

`~/runs/voice-lora-<版>/final_adapter/`（約670MB）が、あなたの声のLoRAです。
ベースモデルと一緒に読み込めば、あなたの文体で書けます。
Ollama や vLLM に載せれば、ふだん使いの自分モデルになります。

---

## どれくらい時間がかかる？

- Aコース（GB10 / 8B / 約1億トークン）：1エポック ≈ 23時間、全15段で **約2週間**。
- Bコース（小型GPU / QLoRA）：1.3倍くらいゆっくり。10段の軽量スケジュールに縮めてあります。

ワークショップ当日は**最後まで回しきりません**。
仕組みを通しで体験し、**自分のコーパスを完成させて学習を“開始”するところまで**を目標にします。
あとは各自のマシンで回しつづけて、後日できあがりを見ます。

---

## うまくいかないとき

| 症状 | 対処 |
|---|---|
| 開始直後に **OOM（メモリ不足）** | `versions.yaml` の `seq_len` を 2048 に下げる。Bコースなら `qlora: true` を確認 |
| **loss が NaN** になる | 学習率が高すぎ。その版の `lr` を半分に |
| `bitsandbytes` のエラー | Bコースは `pip install bitsandbytes`。GPUとCUDAの対応も確認 |
| 「Loading weights」で固まる | `HF_HOME` が書き込めるフォルダを指しているか確認 |
| 比較レポートが OOM | `compare_checkpoints.py` の `--last_n` を 2〜3 に減らす |
| 前の版を引き継がない | ログの `resume_from:` 行を確認。前版の `final_adapter` が保存されているか |
| 生成が**学習文をそのまま引用**する | コーパスが小さすぎ（丸暗記）。素材を増やすか、エポックを減らす |

さらに詳しくは [setup.md](setup.md) と [lessons.md](lessons.md) を読んでください。

---

## チェックリスト（今日の到達点）

- [ ] venv を作り、必要なライブラリを入れた（Bコースは bitsandbytes も）
- [ ] ベースモデルをダウンロードした
- [ ] `raw/` に自分のテキストを入れた（他人の発言は除いた）
- [ ] `build_corpus.py` の `BUILD_PLAN` を自分用に直した
- [ ] `versions.yaml`（A or B）と `eval_prompts.yaml` を用意した
- [ ] 学習を開始し、ログに loss が出ているのを確認した
- [ ] 1つ目の比較レポートを読んだ
