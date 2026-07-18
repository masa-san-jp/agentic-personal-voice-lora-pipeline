# 学習中にマシンを落とさない／落ちても自動復帰する

> LoRA 学習を長時間まわす（本キットなら 15 版・約2週間）と、いちばん怖いのは
> **学習中に自宅マシンが落ちて、外出先から復旧できない**こと。原因はほぼ
> メモリ枯渇（OOM）です。監視して気づいても間に合わないので、**OS 側で「制限」と
> 「保護」をかけ、さらに「安全に一時停止 → 自動再開」を組む**のが正解です。
>
> このキットは HuggingFace `Trainer` を `pipeline/orchestrate.py` が systemd の下で
> まわす構成なので、その前提で必要な仕組みを組み込んであります。この文書はその
> 全体像と設定手順です。

英語で運用手順の全体を読みたい場合は [setup.md](setup.md)、設計判断の背景は
[lessons.md](lessons.md) の "Operational" 節も参照してください。

---

## まず原因を切り分ける

落ち方には2種類あり、対策が違います。

| 症状 | 原因 | 対策 |
|---|---|---|
| プロセスだけ死ぬ（`CUDA out of memory`） | VRAM 不足 = 学習設定の問題 | batch/seq_len を下げる・`--qlora`・gradient checkpointing（本キットは既定でON） |
| **マシンごと固まる／落ちる** | RAM 不足 = OS ごと OOM | 本文書の cgroup 制限・swap・earlyoom・安全停止 |

判別はこれ一発：

```bash
dmesg | grep -i "out of memory"     # "Killed process ..." があれば RAM 側
```

### DGX Spark (GB10) は切り分けが特殊

GB10 は Grace CPU + Blackwell GPU が**同じ物理メモリ（統合メモリ, LPDDR5x 128GB）を
共有**します。つまり「VRAM 不足」と「RAM 不足」が**同一プールの奪い合い**になり、
上の x86＋独立VRAM 機の切り分けがそのままは効きません。学習プロセスと OS／エージェント
（Claude Code 等）が同じメモリを食い合うので、**cgroup による総量制限が通常機以上に効く**
（GPU 確保分もこの制限内で管理されるため）——これが GB10 で本文書の対策がとくに重要な理由です。

---

## 3段構えの設計

```
第1層  OOM を未然に防ぐ   … cgroup 制限 + earlyoom（最重要）
第2層  即死を回避する     … swap でハングまでの猶予を作る
第3層  それでも固まったら … カーネル自動再起動 + watchdog + 電源
────────────────────────────────────────────────
横断    安全に一時停止 → 自動再開（本キット組み込み済み）
```

第1〜3層は OS 側の一般的な保険。**横断**の「安全停止→自動再開」は、殺す前に
チェックポイントを取ってから退く仕組みで、このリポジトリのコードに組み込んであります。

---

## 第1層: OOM を未然に防ぐ（最重要）

### 1-a. 学習ユニットにメモリ上限をかける（cgroup）

`systemd/voice-lora.service` に上限の指定箇所をコメントで用意してあります。ホストの
総メモリに合わせてコメントを外してください（GB10 128GB の例）。

```ini
# systemd/voice-lora.service（[Service] 節）
MemoryHigh=104G      # ソフト上限：超えると強めに回収がかかる
MemoryMax=112G       # ハード上限：これを超えたらこのユニット内で OOM kill
MemorySwapMax=16G
OOMScoreAdjust=500   # OOM 時、sshd より先にこのユニットを殺させる（＝戻る手段を残す）
```

`orchestrate.py` と学習の子プロセスは同じ cgroup（同じ unit slice）にいるので、この上限は
**学習の子プロセスまで含めて**効きます。GPU 確保分も統合メモリ上ではこの枠内。上限値は
「総メモリ × 0.8〜0.9」を目安に、OS とエージェントの分を残して逆算します。

反映：

```bash
cp systemd/voice-lora.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user restart voice-lora.service
systemctl --user show voice-lora.service -p MemoryMax -p MemoryHigh   # 効いているか確認
```

### 1-b. earlyoom で「殺す相手」を選ぶ

上限に達する前に、空きメモリが閾値を切った時点で**最も食っているプロセスを先回りで
kill**します。sshd/systemd は守り、学習プロセスを優先的に殺す設定にします。

```bash
sudo apt install earlyoom
```

```bash
# /etc/default/earlyoom
# sshd/systemd と、GB10 の nvidia 系デーモンは絶対に殺さない。python/accelerate を優先的に殺す。
EARLYOOM_ARGS="-r 60 --avoid '(^|/)(sshd|systemd|nvidia|nv)' --prefer '(^|/)(python|accelerate)'"
```

```bash
sudo systemctl enable --now earlyoom
```

> **なぜ earlyoom も要るのか**：cgroup 上限（1-a）はこのユニット内で殺してくれますが、
> エージェントや他プロセスが枠外で暴れた場合の**システム全体の保険**が earlyoom です。
> GB10 の統合メモリでは両方あると安心。

---

## 第2層: swap で即死を回避

swap なし運用だと、OOM がそのまま即フリーズに直結し、earlyoom や安全停止が
**動く時間すら残りません**。16〜32GB の swapfile を置いて猶予を作ります。

```bash
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab   # 再起動後も有効化
```

swap は「速く動かすため」ではなく「**落ちる前に一拍おく**ため」。学習が swap に落ちて
遅くなったら、それは安全停止（下記）が発火すべきサインでもあります。

---

## 第3層: それでも固まった時の自動復帰

外出先からは物理操作ができないので、**固まったら勝手に再起動して学習を再開**させます。

```bash
# カーネルパニック時に 10 秒で自動再起動
echo 'kernel.panic = 10' | sudo tee /etc/sysctl.d/99-panic.conf
sudo sysctl --system

# ハングアップ検知で強制再起動（ハードウェア watchdog）
sudo apt install watchdog
sudo systemctl enable --now watchdog
```

- BIOS/UEFI で「**AC 電源復帰後に自動起動**」を有効化しておく。
- スマートプラグがあれば、遠隔で物理的な電源断→再投入も可能（最後の手段）。
- 再起動後は systemd の `loginctl enable-linger` で `voice-lora.service` が自動起動し、
  `state.json` から**未完の版を last checkpoint から再開**します（下記）。だから
  再起動しても失うのは最大 `save_steps` ステップ分だけです。

---

## 横断: 安全に一時停止 → 自動再開（本キット組み込み済み）

「殺す」のではなく「**チェックポイントを保存して行儀よく退く → メモリが戻ったら自動で
続きから**」を、このリポジトリに実装済みです。3つの部品が噛み合っています。

### 部品1: 学習側 — クラッシュ復帰と SIGUSR1 一時停止（`train_bf16_lora.py`）

- **クラッシュ復帰**：各版は `--output` に `save_steps`（既定200）ごとチェックポイントを
  書きます。中断後に**同じ `--output` で再実行すると、最新の `checkpoint-*` から
  自動再開**します（optimizer・lr スケジューラ・step まで復元）。既定でON。切るなら
  `--no_auto_resume`。
  （`--resume_from` は「前の版のアダプタを引き継ぐ」別物。こちらは「**同じ版の中断を
  続きから**」です。）
- **SIGUSR1 で安全停止**：`SIGUSR1` を受けると、**現在の step を終えてから**
  チェックポイントを書き、終了コード `3`（＝「一時停止した、再開してほしい」）で抜けます。
  step 境界で止めるので、保存中の破損が起きません。

### 部品2: オーケストレータ — 「一時停止」と「失敗」を区別（`orchestrate.py`）

- 終了コード `3` を**失敗ではなく一時停止**として扱い、`SLEEP_AFTER_PAUSE`（既定120秒、
  `VOICE_LORA_PAUSE_SLEEP` で変更可）だけ待ってから**同じ版を再実行**（→ 部品1の自動再開で
  続きから）。**リトライ回数を消費しません**（メモリ都合の一時停止で `MAX_RETRIES` を
  使い切って abandon しないため）。
- 監視ワッチャーの `systemctl kill` は cgroup 内**全プロセス**に SIGUSR1 を配るため、
  オーケストレータ自身は SIGUSR1 を**無視**（`SIG_IGN`）にしてあります。反応するのは
  学習の子プロセスだけ。

### 部品3: メモリ監視ワッチャー — 落ちる前に「退いて」と言う（`systemd/mem-guard.sh` ＋ timer）

毎分、空きメモリ（`MemAvailable`）を見て、閾値を切ったら **kill ではなく SIGUSR1** を
送ります。earlyoom やカーネル OOM が動くより**一拍早く**発火させるのが狙いです。

```bash
# 導入
cp systemd/voice-lora-memguard.service systemd/voice-lora-memguard.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now voice-lora-memguard.timer

# 閾値の調整（既定4GiB）。1 step のピーク確保量より大きめにして、上限に達する前に退かせる。
systemctl --user edit voice-lora-memguard.service
#   [Service]
#   Environment=VOICE_LORA_MEM_MIN_KB=8388608     # 8 GiB で退避
```

> **層の役割分担**：一次制御は cgroup 上限（1-a）と earlyoom（1-b）。ワッチャーは
> その一歩手前で「**進捗を失わずに退く**」ための保険です。閾値は earlyoom より先に
> 発火するよう、少し高めに置きます。

### 外出先からの手動制御（副産物）

同じ SIGUSR1 の仕組みで、遠隔から安全に止めて／再開できます。

```bash
ssh gpu-host 'systemctl --user kill -s SIGUSR1 voice-lora.service'  # 安全に一時停止（checkpoint 保存）
# → オーケストレータが自動で待って再開。完全に止めたいなら:
ssh gpu-host 'systemctl --user stop voice-lora.service'             # 停止
ssh gpu-host 'systemctl --user start voice-lora.service'            # 再開（未完の版を続きから）
```

---

## 全体の流れ（一時停止→再開）

```
mem-guard.timer（毎分）
   └─ MemAvailable < 閾値？
        └─ systemctl --user kill -s SIGUSR1 voice-lora.service
             ├─ orchestrate.py … SIGUSR1 を無視（SIG_IGN）
             └─ train ……… 今の step を終える → checkpoint-N 保存 → exit 3
                  └─ orchestrate.py … exit 3 を「一時停止」と判定
                       └─ 120秒待つ（メモリ回復）→ 同じ版を再実行
                            └─ train … checkpoint-N を自動検出して続きから
```

クラッシュ／再起動の場合も末尾は同じで、systemd が `voice-lora.service` を起こし、
`state.json` の未完の版が last checkpoint から再開します。

---

## なぜ「監視アプリの自作」ではないのか

「Claude Code がどれだけメモリを使うか監視するアプリを作る」という発想は遠回りです。

- 原因はほぼ OOM で、**監視して気づいても間に合わない**（対策は制限と保護）。
- Claude Code は常駐監視エージェントではなく、応答の合間は状態が見えず、子プロセスの
  メモリを**リアルタイムに制御できない**。メモリ制御は構造的に OS の仕事です。
- エージェントに向くのは、**この文書のような設定ファイルの作成・ログ解析・切り分け**の
  ほう。役割分担として、制御は OS（cgroup/earlyoom/systemd）、設定と診断はエージェント、が正解。

---

## チェックリスト

- [ ] `dmesg | grep -i "out of memory"` で RAM 側か VRAM 側かを確認した
- [ ] `voice-lora.service` の `MemoryMax`/`MemoryHigh`/`OOMScoreAdjust` をホストに合わせて有効化した
- [ ] earlyoom を入れ、`--avoid` に sshd/systemd/nvidia、`--prefer` に python/accelerate を設定した
- [ ] 16〜32GB の swap を用意し `/etc/fstab` に登録した
- [ ] `kernel.panic=10` ＋ watchdog ＋ BIOS 自動起動を設定した
- [ ] `voice-lora-memguard.timer` を有効化し、閾値を 1 step のピークより大きめにした
- [ ] `loginctl enable-linger $USER` で再起動後も自動再開することを確認した
- [ ]（GB10）上限値を統合メモリ総量から逆算し、OS＋エージェントの分を残した
