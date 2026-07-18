# agentic-personal-voice-lora-pipeline

> **自分のデータで、自分の「声」で書くAIを作るキット。**
> あなたが今までに書いてきた文章（本・記事・メール・チャット・講演の文字起こし）を
> 食べさせて、あなたの文体で書きつづける小さなモデル（LoRA）を育てます。

English: [README.en.md](README.en.md) ／ ワークショップ当日の手順: [docs/workshop-ja.md](docs/workshop-ja.md)

このリポジトリは、落合塾のワークショップ用の配布キットです。
**参加者ひとりひとりが、自分のテキストを使って、自分の声のモデルを作る**ことを目的にしています。
落合陽一のモデル `ochiai-v20` も、まさにこの同じ手順で作りました。`examples/` にその出力例が入っています（あなたも同じものが作れます）。

---

## このフォークについて

このリポジトリは、落合陽一さんが公開した [personal-voice-lora-pipeline](https://github.com/ochyai/personal-voice-lora-pipeline)（MIT ライセンス）の**フォーク**です。
原本のコーパス構築・学習・比較レポートの仕組みはそのまま引き継いでいます。原本の設計・実装・作例（`examples/` の `ochiai-v20` など）はすべて落合陽一さんによるものです。

このフォークが原本に**独自に追加した**のは、「学習して終わり」にせず、**どれだけ本物の文体に近づいたかを測り、その結果を記事にする**までを一本の流れにする層です。追加したのは次の3つ：

- **評価ベンチ（機械2指標）** — [`pipeline/voice_bench.py`](pipeline/voice_bench.py)
  固定お題で各版に文章を書かせ、2つの物差しで採点する。①**文体距離**（LLMを使わない決定的な計算・低いほど本物に近い）②**LLM審査**（ローカルLLMが0〜100で類似度を採点）。性質の違う2枚を重ねて、片方だけの改善に騙されないようにする。
- **人間ブラインド判定（第3指標）** — [`pipeline/make_blind_eval.py`](pipeline/make_blind_eval.py) ＋ [`pipeline/blind_eval_template.html`](pipeline/blind_eval_template.html)
  ベンチ出力から、どの版かを伏せて A/B/C で並べ、人が「本物にいちばん近い1本」を選ぶ**ネット接続不要の静的HTML**を生成する。機械指標だけでは測れない「人が読んでどう感じるか」を確かめるため。生成された1枚のHTMLを開いて全問選ぶと、各版の正体と機械指標が並んで出る。
- **エージェント自走の手順書** — [`AGENTS.md`](AGENTS.md)
  コーディングエージェント（Claude Code 等）がこのリポジトリを読めば、環境構築→学習→評価→人間テスト→記事化までを自分で実行できるように、各ステップに「**エージェントが自分でやる／人間が必要**」を明記した実行手順。人間が要る所（個人データの提供・GPU認証・人間ブラインド判定・記事の承認）だけを明示して止まる。

棲み分け：**原本＝コーパス→学習まで／このフォーク＝そこに 評価（機械2指標＋人間ブラインド）・記事化の手順・エージェント自走 を足した版**。ライセンスは原本と同じ MIT（コード）。

---

## これは何をするもの？

ふつうのチャットボットを作るのではありません。作るのは「**あなたの文体で続きを書く生成エンジン**」です。

- 見たことのないお題を渡しても、**あなたっぽいリズム・言い回し・書き出し**で文章を続けてくれる。
- 中身は、強いベースモデル（日本語なら `Llama-3.1-Swallow-8B`）の上に、
  あなたの文体だけを覚えさせた **LoRA という小さな追加パーツ（約670MB）** です。
- ベースの一般知識はそのまま残るので、文体だけを上から薄く塗るイメージです。

やらないこと: 命令に従う調整（instruction tuning）・対話の調教・RLHF。それらは後から足せます。
ここでの目的はただ一つ、「**あなたの声で書きつづける**」ことです。

---

## いちばん大事な考え方：「増やす → 染み込ませる」

最初にみんなが失敗するのは、**1回だけ学習させて終わりにする**ことです。
それだと、モデルはあなたの「単語」は覚えるのに、教科書みたいな文体で書いてしまいます
（「〜とは、〜のことである」）。声が染みていないのです。

うまくいくのは、2つの段階に分けるやり方です。

1. **増やす段階（コーパス成長, 2〜3回）**
   まず手で集めた「種テキスト」から始める。次の回ごとに、メール・チャット・講演など
   別の素材を足していく。1回＝1エポック。前の回の成果を引き継いで続ける。
   → 語彙とトピックは、ここで速く入ります。

2. **染み込ませる段階（浸透, 8〜12回）**
   テキストはこれ以上増やさず固定。学習率（lr）を少しずつ下げながら、何度も読み返させる。
   → ここで**文体が定着**します。4〜6回目あたりで、書き出しが「説明しましょう」ではなく
   あなた自身の癖（短い文、感覚的な入り、自問）に変わってきます。

くわしい理由は [docs/corpus-strategy.md](docs/corpus-strategy.md) に書きました。

---

## 必要なもの

- **あなたのテキスト**：目安は 50〜200M トークン（日本語で約1億〜4億字）。
  少なくても動きますが、30M字より少ないと“丸暗記”しがちです（[docs/corpus-strategy.md](docs/corpus-strategy.md) 参照）。
- **GPU**：下の2コースから選びます。

### GPUは2コースあります

| | A. 自前GPUコース（bf16） | B. 小型GPU / Colab コース（QLoRA） |
|---|---|---|
| VRAM | 24GB以上（DGX Spark, A100, RTX 4090 など） | 12〜16GB（Colab T4, RTX 3060/4060 など） |
| 学習スクリプトの指定 | そのまま | `--qlora` を付ける（`pip install bitsandbytes` 必要） |
| 使うスケジュール | `versions.example.yaml`（15段） | `versions.mini.example.yaml`（10段・軽量） |
| 速さ | 基準 | だいたい3割ゆっくり |
| できあがり | 同じ形のLoRA | 同じ形のLoRA |

どちらでも**できあがるモデルの形は同じ**です。手元のGPUに合わせて選んでください。

---

## まず：自分のデータを集める（あなたの homo-convivium を作る）

学習の前に、**自分が書いた文章を集めて `raw/` に整える**必要があります。
落合はこの個人アーカイブを `homo-convivium` と呼んでいますが、**参加者は自分専用のそれを作るところから**始めます。
note・ブログ・書籍・X・メールなどからの集め方と、変換スクリプトを
[`collect/`](collect/README.md) にまとめました。**最初にここを読んでください。**

```bash
# 例：note・ブログ・原稿などをフォルダに集めて → 種コーパスに変換
python collect/files_to_seed_jsonl.py --in ~/my-writings --out ~/voice-lora/raw/seed_corpus.jsonl
```

データがそろったら、下の手順に進みます。

> **コーディングエージェント（Claude Code 等）で回すなら** → [AGENTS.md](AGENTS.md) を読めば、学習からレポート/ブログまで自走できる。人間の作業が要る所（個人データ提供・GPU認証・人間ブラインド判定・記事承認）は ⚑ で明示してある。

## はじめかた（手順）

> もっと丁寧な当日用の手順は [docs/workshop-ja.md](docs/workshop-ja.md) にあります。
> インストールの細部は [docs/setup.md](docs/setup.md) を見てください。

### 1. リポジトリを置いて、道具を入れる

```bash
git clone https://github.com/masa-san-jp/agentic-personal-voice-lora-pipeline.git ~/voice-lora
cd ~/voice-lora
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers peft datasets pyyaml python-docx
# B. 小型GPUコースの人は、これも:
pip install bitsandbytes
```

### 2. ベースモデルをダウンロードする

```bash
mkdir -p ~/models
pip install huggingface-hub
hf download tokyotech-llm/Llama-3.1-Swallow-8B-v0.5 \
    --local-dir ~/models/Llama-3.1-Swallow-8B-v0.5
```

日本語以外なら `meta-llama/Llama-3.1-8B-Instruct` など、その言語が得意な8Bクラスを選びます。

### 3. 自分のテキストを `raw/` に入れる

```
~/voice-lora/raw/
├── seed_corpus.jsonl     ← 最初の種（手で集めた自分の文章）
├── email.jsonl           ← メール（自分が送ったものだけ）
├── slack_team_a.jsonl    ← チャットの書き出し
├── interviews/*.txt      ← 対談・インタビューの文字起こし
├── presentations/*.md    ← 講演メモ・原稿
└── manuscripts/*.docx    ← 本の下書きなど
```

`raw/` と `data/` は **git に上がりません**（個人情報なので最初から除外済み）。

### 4. 「どのデータをどう食べさせるか」を書く

`pipeline/build_corpus.py` の下のほうにある **`BUILD_PLAN`** を、自分のデータの置き場所に合わせて書き換えます。
よくある形（メールJSONL・Slack書き出し・docx・テキストの束）を読み込む関数が
最初から入っているので、必要なものをコピーして直すだけです。

> 落合の実例（homo-convivium のアーカイブを `build_corpus` にどう繋いだか）は、
> ワークショップで手元のサンプルとしてお見せします。中身（本文）は配布しません。

### 5. スケジュールと評価プロンプトを用意する

```bash
# A. 自前GPU:
cp pipeline/versions.example.yaml pipeline/versions.yaml
# B. 小型GPU / Colab:
cp pipeline/versions.mini.example.yaml pipeline/versions.yaml

cp pipeline/eval_prompts.example.yaml pipeline/eval_prompts.yaml
```

`eval_prompts.yaml` には、**自分について問う6つのお題**を書きます（自分の代表的な概念・苦手な話題・書き出しの癖、の3種類）。
ファイル内のコメントに書き方の例があります。

### 6. 学習を走らせる（止まっても自動で続く）

長時間の学習なので、systemd に任せて「寝てる間も回る」状態にします。

```bash
cp systemd/voice-lora.service ~/.config/systemd/user/
systemctl --user daemon-reload
loginctl enable-linger $USER          # ログアウトしても続く
systemctl --user enable --now voice-lora.service
```

途中でクラッシュしても、`state.json` に進み具合が残っているので、
**終わった版はとばして、失敗した版だけ最大3回までやり直し**ます。
さらに各版は途中のチェックポイントから自動で続きを再開するので、再起動しても
失うのは最大200ステップ分だけです。

> **外出先に出る前に**：長時間学習で自宅マシンが OOM で落ちると遠隔復旧が難しい。
> 「落とさない／落ちても自動で続きから再開する」ための設定（メモリ上限・swap・
> earlyoom・安全停止→自動再開）は [docs/stability.md](docs/stability.md) にまとめました。
> DGX Spark (GB10) の統合メモリ特有の注意もここにあります。**外に出る運用をする前に読んでください。**

### 7. 見守る

```bash
journalctl --user -u voice-lora.service -f     # 全体ログ
tail -f ~/voice-lora/logs/v06.log              # 各版のログ
cat ~/voice-lora/state.json                    # 進み具合
```

各版が終わるたびに、同じお題を全チェックポイントで生成した
**比較レポート（マークダウン）**が自動で出ます。上から下へ読むと、
教科書っぽい文 → あなたの文体、へ変わっていくのが見えます。
実例は [`examples/voice_compare_report_example.md`](examples/voice_compare_report_example.md)。

---

## 評価パイプライン（機械2指標 → 人間ブラインド → レポート）

学習した各版を「本物の文体にどれだけ近いか」で測り、最後は**人の目**でも確かめて、
記事にするところまでを一本の流れにしています。次のモデルでもそのまま回せます。

### A. 固定お題ベンチ（機械2指標）— `pipeline/voice_bench.py`

同じ10お題で各400字を生成し、2つの物差しで採点します。

- **文体距離**（LLMを使わない決定的な計算・低いほど本物に近い）：丁寧語率・平均文長・呼称頻度・絵文字密度・ひらがな率などの差。
- **LLM審査**（ローカル gpt-oss が0〜100で採点・高いほど似ている）。

```bash
# 素モデル / 各アダプタ ごとに1回ずつ回す（--adapter - で素モデル）
python3 pipeline/voice_bench.py --base ~/models/<base> --adapter - \
  --corpus ~/voice-lora/raw/seed.jsonl --themes-file themes.txt --out runs/bench-base.md
python3 pipeline/voice_bench.py --base ~/models/<base> --adapter runs/<浅い版>/final_adapter/ \
  --corpus ... --out runs/bench-mini.md
python3 pipeline/voice_bench.py --base ~/models/<base> --adapter runs/<反復版>/final_adapter/ \
  --corpus ... --out runs/bench-pen.md
```

### B. 人間ブラインド判定（第3指標）— `pipeline/make_blind_eval.py`

機械の点は当てにならない場面がある（採点係が揺れる・表層だけ合う等）ので、
**どの版かを伏せて A/B/C で並べ、人が「本物にいちばん近い1本」を選ぶ**静的ページを作ります。
ネット接続なしで開ける1枚のHTMLで、全問答えると各サンプルの正体と機械指標が並んで出ます。

```bash
python3 pipeline/make_blind_eval.py \
  --bench base=runs/bench-base.md mini=runs/bench-mini.md pen=runs/bench-pen.md \
  --labels "素モデル (base)=base" "浅い学習 (25step)=mini" "反復学習版 (100step)=pen" \
  --steps "学習 0 step=base" "学習 25 step=mini" "学習 100 step=pen" \
  --best pen --target "自分の名前" --themes-file themes.txt --out blind-eval.html
```

`--bench` は「読者に見せる順（弱い→強い）」でキーを渡します。`--best` は機械が最良とした版のキー。
判定者は複数の独立した人（仮説を知らない人）を入れるほど強くなります。1人だけなら「兆候」どまり。

### C. 評価レポート／ブログ — `article-compose` スキル

A・B の結果（`runs/bench-*.md` と人間判定の票）を素材に、
記事は Agent 側の `article-compose` スキルで書きます（技術レポート＝客観・である調／ブログ＝人格の一人称）。
数値はベンチ出力からの転記のみ・生成文の引用は逐語・**言葉は初見でも分かる一般語彙**、が守るべき点。

> 全体の順番：`build_corpus` → `orchestrate`(学習) → `voice_bench`(各版) → `make_blind_eval`(人間テスト) → `article-compose`(レポート/ブログ)

---

## できたモデルの使い方

学習が終わると `~/runs/voice-lora-<版>/final_adapter/` に LoRA（約670MB）ができます。
これをベースモデルと一緒に読み込めば、あなたの声で書けます。
Ollama や vLLM に載せて、ふだん使いのモデルにもできます（落合の `ochiai-v20` がこの形です）。

---

## できること・できないこと

**できること**：お題の外側でも文体が崩れない。明示の指示なしでもスタイルが保つ。
配って動かせる成果物（LoRA）が手に入る。

**できないこと**：事実の正確さは保証されません
（半導体の話で「WES（Water-Ethanol-Solvent）」のような**実在しない用語を堂々と作る**ことがあります）。
道具の利用や、長い対話の一貫性もここでは扱いません。それらは後から足す工程です。

→ **下書きの量産機としては優秀。事実を書く調査アシスタントとしては使わないこと。**

---

## プライバシーの線引き（だいじ）

コーパスにはメール・非公開チャット・内部資料が入ります。
**学習はネットから隔離したGPUの中で完結させ、コーパスや学習済みLoRAをGitHubに上げない**こと。
このリポジトリの `.gitignore` は最初からそう作ってあります（`raw/ data/ runs/ models/` は除外）。
公開していいのは「パイプラインのコード」と「出力の例」だけ、と決めておくと安全です。

さらに `build_corpus.py` は、コーパスを書き出す唯一の出口（`write_jsonl`）で
**全ソース一律に PII マスクをかけます**：メールアドレス→`<EMAIL>`、郵便番号→`<POSTAL>`、
電話・カード番号など長い数字列→`<NUM>`。モデルが機械的な個人情報を丸暗記して吐くのを防ぎます。
ただし**氏名や自由記述の住所は正規表現では確実に消せない**（消しすぎると文体を壊す）ので、
そこは上の「隔離」で守る前提です。マスクはプレースホルダ置換なので voice への影響は最小です。

---

## 中身の一覧

```
agentic-personal-voice-lora-pipeline/
├── collect/                       # ★ステップ0：自分のデータを集める（あなたのhomo-conviviumを作る）
│   ├── README.md                  #   集め方ガイド（note/X/Gmail…→ raw/ へ）
│   ├── files_to_seed_jsonl.py     #   txt/md/docx/pdf フォルダ → 種コーパス
│   ├── twitter_archive_to_jsonl.py#   X公式アーカイブ → 自分のツイートだけ
│   ├── gmail_mbox_to_jsonl.py     #   Google Takeout mbox → 自分が送ったメールだけ
│   ├── chatgpt_export_to_seed_jsonl.py #   ChatGPTエクスポート → 自分の発話だけ
│   ├── media_transcribe_to_jsonl.py    #   音声/動画 → ローカルWhisper文字起こし（話者分離対応）
│   ├── claude_code_history_to_jsonl.py #   Claude Code履歴 → 自分の指示だけ（過去分）
│   └── claude_code_prompt_logger.py    #   〃 これから分をフックで継続蓄積
├── pipeline/
│   ├── orchestrate.py             # 司令塔。versions.yaml を読んで順番に学習。落ちても再開
│   ├── build_corpus.py            # コーパス作成のテンプレ。BUILD_PLAN を自分用に書き換える
│   ├── train_bf16_lora.py         # 学習本体。bf16 と --qlora（4bit）の両対応
│   ├── compare_checkpoints.py     # 評価。同じお題を各版で生成して比較レポートを出す
│   ├── voice_bench.py             # 固定10お題ベンチ。文体距離＋LLM審査の機械2指標を出す
│   ├── make_blind_eval.py         # bench出力→人間ブラインド判定の静的HTML（第3指標）
│   ├── blind_eval_template.html   #   ↑が使うテンプレ（自己完結・ネット不要）
│   ├── eval_prompts.example.yaml  # 評価お題6本（known / generalize / style）
│   ├── versions.example.yaml      # 15段スケジュール（自前GPU向け）
│   └── versions.mini.example.yaml # 10段の軽量スケジュール（小型GPU / QLoRA向け）
├── systemd/
│   ├── voice-lora.service         # ユーザーsystemdユニット。落ちても自動再起動
│   ├── mem-guard.sh               # メモリ監視ワッチャー。落ちる前に安全停止を指示
│   ├── voice-lora-memguard.service#   ↑を毎分走らせる oneshot
│   └── voice-lora-memguard.timer  #   ↑のタイマー（毎分）
├── examples/voice_compare_report_example.md  # 実物の出力例（ochiai-v20）。上から下へ読む
├── docs/
│   ├── workshop-ja.md             # ★ワークショップ当日の手順（優しい日本語）
│   ├── setup.md                   # まっさらなGPUへの導入手順
│   ├── stability.md               # ★落とさない／落ちても自動復帰（OOM対策・GB10注意）
│   ├── corpus-strategy.md         # 「増やす→染み込ませる」がなぜ効くか
│   └── lessons.md                 # うまくいったこと・失敗したこと・落とし穴
├── AGENTS.md                      # ★エージェント向け実行手順（[AGENT]自走 / ⚑HUMAN依頼を明示）
├── LICENSE                        # MIT（コード）。例文は参考用・著作権あり
├── README.md / README.en.md
└── .gitignore
```

---

## ライセンス

- **コードとドキュメント：MIT**（[LICENSE](LICENSE)）。自由に使い、改変し、配布できます。
- ただし `examples/` の生成テキストは、落合陽一の著作群を学習させたモデルの出力です。
  **参考・教育目的でのみ**同梱しています。そのまま自分の文章として再配布したり、
  落合陽一本人の発言であるかのように出したりしないでください。
