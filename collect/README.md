# ステップ0：自分のデータを集める（＝あなたの homo-convivium を作る）

学習を始める前に、まず **自分が書いてきた文章を一か所に集めて、決まった形に整える** 必要があります。
落合の場合はこの「個人テキストアーカイブ」を **homo-convivium** と呼んでいます
（17年分のツイート・2,000本のnote・書籍原稿・講演…を1つに集約したもの）。

**あなたも、自分専用の homo-convivium を作るところから始めます。** ここがいちばん大事な作業です。
モデルの良し悪しの9割は「どんな文章を集めたか」で決まります。

集めて整えたファイルは、`~/voice-lora/raw/` に置きます。
そこまでできたら、本体の手順（[../docs/workshop-ja.md](../docs/workshop-ja.md) の手順4以降）に進みます。

---

## まず全体像

ゴールはこれだけです。`~/voice-lora/raw/` に、自分の文章を入れた JSONL を並べる：

```
~/voice-lora/raw/
├── seed_corpus.jsonl   ← note・ブログ・書籍・原稿など「長い文章」をまとめた種
├── twitter.jsonl       ← Xの自分のツイート（あれば）
└── email.jsonl         ← 自分が送ったメール（あれば）
```

JSONL は「1行＝1かたまりの文章」のファイルです。中身はこんな形：

```json
{"text": "あなたが書いた文章のかたまり。段落いくつか分。"}
```

下のスクリプトを使えば、各サービスの書き出しから自動でこの形にできます。

---

## どこから集められる？（ソース別）

| 集めるもの | どこで書き出す | 使うスクリプト |
|---|---|---|
| note / ブログ / 書籍下書き / 講演原稿 / 対談の文字起こし | テキストや Word/PDF をフォルダに集める | `files_to_seed_jsonl.py` |
| X（旧Twitter） | 設定 → アカウント → データのアーカイブをダウンロード | `twitter_archive_to_jsonl.py` |
| Gmail | [takeout.google.com](https://takeout.google.com) で「メール」を選ぶ | `gmail_mbox_to_jsonl.py` |
| Slack / Discord | 各ワークスペースのエクスポート（JSON） | `../pipeline/build_corpus.py` の `add_slack_export` を参照 |
| YouTube / 講演の音声 | 字幕や文字起こしを .txt にして上のフォルダへ | `files_to_seed_jsonl.py` |
| **Claude Code への指示** | 自動で貯まる（`~/.claude/projects/` の履歴） | `claude_code_history_to_jsonl.py` ／ 継続蓄積は `claude_code_prompt_logger.py` |

**まずは note やブログなど「長い文章」だけで十分**です。ツイートやメールは後から足せます。
長文ほど“声”の情報が濃く、短い断片（ツイート・チャット）は薄いです。

---

## 使い方

### 1. 長い文章 → 種コーパス（いちばん基本）

note の記事、ブログ、書籍の下書き、講演原稿、対談の文字起こしなどを、
ひとつのフォルダ（例：`~/my-writings/`）に `.txt` `.md` `.docx` `.pdf` のまま放り込みます。サブフォルダOK。

```bash
python collect/files_to_seed_jsonl.py \
    --in ~/my-writings \
    --out ~/voice-lora/raw/seed_corpus.jsonl
```

> note を全部落としたい場合：note には公式の一括書き出しが無いので、各記事を開いて
> 本文をコピペし `.txt` で保存するか、自分のページを保存して `.md`/`.html`→`.txt` にします。
> 落合は専用スクリプトでまとめて取得しましたが、まずは主要な記事を手で集めるだけでも始められます。

### 2. X（旧Twitter）→ 自分のツイートだけ

X の「データのアーカイブ」をダウンロードして展開し、中の `data/tweets.js` を渡します。

```bash
python collect/twitter_archive_to_jsonl.py \
    --tweets ~/twitter-archive/data/tweets.js \
    --out ~/voice-lora/raw/twitter.jsonl
```

リツイートと他人への返信は自動で除外し、自分の地の文だけ残します。

### 3. Gmail → 自分が送ったメールだけ

Google Takeout で受け取った `.mbox` と、自分のアドレスを渡します。

```bash
python collect/gmail_mbox_to_jsonl.py \
    --mbox ~/Takeout/Mail/all.mbox \
    --me you@example.com \
    --out ~/voice-lora/raw/email.jsonl
```

### 4. Claude Code への指示 → 自分の声として蓄積

Claude Code に打ち込む指示も、**あなたが自分の言葉で書いたテキスト**です（命令調・簡潔・技術的）。
Claude Code は各セッションを `~/.claude/projects/<プロジェクト>/<セッションID>.jsonl` に記録していて、
その中の「人間が打った turn」だけを抜き出せます（ツールの出力やアシスタントの発話は除外）。

**(a) 過去分をまとめて回収（バックフィル）**

```bash
# 引数なしで ~/.claude/projects を読み、$VOICE_LORA_ROOT/raw/claude_code.jsonl を書き出す
python collect/claude_code_history_to_jsonl.py
# 特定プロジェクトだけに絞るなら:
python collect/claude_code_history_to_jsonl.py --project-substr my-repo
```

セッション単位で時系列に人間の指示だけを結合し、システムreminder・スラッシュコマンド・
コードブロック（既定で除去。残すなら `--keep-code`）を落として整えます。**再実行すれば新しい
セッションも拾う**ので、使い続けるほど貯まります。

**(b) これから分を自動で貯める（ライブ・任意）**

Claude Code の `UserPromptSubmit` フックに登録すると、打った指示が1行ずつ
`raw/claude_code_live.jsonl` に追記されていきます。Claude Code の `settings.json` に：

```json
{
  "hooks": {
    "UserPromptSubmit": [
      { "hooks": [ { "type": "command",
        "command": "python3 ~/voice-lora/collect/claude_code_prompt_logger.py" } ] }
    ]
  }
}
```

ログは `raw/` の中（git 除外済み）に留まり、外には出ません。スラッシュコマンドや極端に短い指示は
自動でスキップ、PII（メール・電話等）はマスクされます。フックは何も標準出力に出さず、必ず正常終了
するので、プロンプトの挙動を邪魔しません。

生成した `claude_code.jsonl` / `claude_code_live.jsonl` は、`build_corpus.py` の `BUILD_PLAN`
（`build_v7` にコメントで例を用意）で `add_existing_jsonl(..., source_tag="Personal.claude_code")`
として取り込めます。

---

## 集めたあとは

`build_corpus.py` の `BUILD_PLAN` が、この `raw/` のファイルを読み込みます。
配布版の既定では `seed_corpus.jsonl`・`email.jsonl`・各フォルダを読むようになっているので、
**上のスクリプトでファイル名をそろえておけば、ほぼそのまま動きます。**

→ 続きは [../docs/workshop-ja.md](../docs/workshop-ja.md) の「手順4：どう食べさせるかを書く」へ。

---

## だいじな約束（プライバシー）

- 集めるのは **自分が書いた文章だけ**。チャットやメールは「自分が送った分」に絞ります
  （上のスクリプトはそうしています）。他人の発言を自分の声として学習させない。
- `raw/` と `data/` は **git に上がりません**（最初から除外済み）。
- 学習は **ネットから切り離したGPUの中**で完結させ、集めたデータや学習済みモデルを
  公開リポジトリに置かないこと。あなたの homo-convivium は、あなたの手元だけに。
