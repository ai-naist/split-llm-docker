# split-llm-docker

LLM（Causal LM）の推論を **クライアント側（前半レイヤ）** と **サーバー側（後半レイヤ）** に分割して実行する、最小構成の Docker サンプルです。

- クライアント: Embedding + 0〜(split_layer-1) 層を実行し、中間表現（hidden states）をサーバーへ送信
- サーバー: split_layer〜最終層 + lm_head を実行し、次トークンの logits を返却
- クライアント: logits から次トークンを選び、1 トークンずつ生成（デモ用）

## 前提

- Docker / Docker Compose が使えること
- 初回はモデル重みがローカルキャッシュに必要（このプロジェクトは **ローカルのみで読む設定** になっています）

> 注意: 通信は TCP + pickle を使っています。**信頼できないネットワーク/相手に対しては絶対に公開しない**でください（デモ・研究用途想定）。

## クイックスタート

### 1)（初回のみ）モデルをキャッシュに用意

このプロジェクトのロード処理は `local_files_only=True` のため、先に Hugging Face のスナップショットをキャッシュへ入れておく必要があります。

例（Qwen2.5 0.5B を取得）:

```bash
docker compose run --rm server python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"
```

- `docker-compose.yml` で `./models` をコンテナ内の `/root/.cache/huggingface` にマウントしているため、ダウンロード内容はホスト側 `models/` に永続化されます。
- 対象モデルは `model.safetensors` もしくは `model.safetensors.index.json` を含む必要があります。

### 2) サーバー起動

```bash
docker compose up -d server
```

ログ確認:

```bash
docker compose logs -f server
```

### 3) クライアントから実行

```bash
docker compose up -d client
docker compose exec client python client.py --prompt "日本の首都はどこですか？"
```

または `env/client.env` の `PROMPT` を使って実行します（`--prompt` が優先）。

終了:

```bash
docker compose down
```

## 設定

設定は以下の優先順で決まります。

1. 環境変数（`env/*.env` や `docker compose` の環境）
2. `settings.json`
3. `settings.py` 内のデフォルト

### 主なパラメータ

- `MODEL_NAME`: 例 `Qwen/Qwen2.5-0.5B-Instruct`
- `SPLIT_LAYER`: 分割境界レイヤ（`1 <= split_layer <= num_layers-1`）
- `MODEL_DTYPE`: `float32` / `float16` / `bfloat16` など
- `PORT`: サーバー待受ポート（クライアントも同値を参照）
- `SERVER_BIND_HOST`: サーバー bind アドレス（通常 `0.0.0.0`）
- `SERVER_HOST`: クライアントから見たサーバーホスト（Compose 内なら `server`）
- `MAX_NEW_TOKENS`: 生成トークン数
- `USE_CHAT_TEMPLATE`: `true/false`（モデルの chat template を使う）
- `SYSTEM_PROMPT`: chat template 使用時の system 文
- `SETTINGS_PATH`: 設定 JSON のパス（既定: `/app/settings.json`）

## 仕組み（ざっくり）

1. クライアントが入力をトークナイズし、Embedding と前半レイヤを実行
2. 中間表現 `hidden_states` と、Attention / Position / Cache の情報をソケット送信
3. サーバーが後半レイヤを実行して logits を計算し、クライアントへ返す
4. クライアントが argmax で次トークンを選び、(1) に戻る

## 工夫点（この実装のポイント）

- **重みの部分ロード**: `model_loader.py` で safetensors のキーを見て必要なテンソルだけ読み込み、クライアント/サーバーそれぞれの常駐メモリを抑えています（`accelerate.init_empty_weights` + `set_module_tensor_to_device`）。
- **シャーディング対応**: `model.safetensors.index.json` があるモデルでは shard ごとに必要キーだけ読む実装にして、I/O を無駄に増やしません。
- **TCP のフレーミング**: pickle をそのまま `send` せず、先頭に 4byte 長を付ける length-prefix 方式にして、TCP のメッセージ境界問題を回避しています。
- **KV cache を分割して維持**: `DynamicCache` と `cache_position` を用い、トークン生成を 1 ステップずつ回してもキャッシュが破綻しにくい形にしています。
- **RoPE 位置埋め込みの取り回し**: Qwen2.5 系で必要になる `position_embeddings=(cos, sin)` をクライアント/サーバー双方の layer 呼び出しに渡す形に揃え、分割推論時の位置情報の整合性を確保しています。
- **設定の上書きしやすさ**: JSON 設定 + 環境変数のマージにより、Compose からの上書きとローカル実験の両方を簡単にしています。

## ファイル構成

- `client.py`: 前半レイヤ実行 + サーバー通信 + 逐次生成
- `server.py`: 後半レイヤ実行 + logits 返却
- `model_loader.py`: safetensors から必要部分だけロード
- `settings.py` / `settings.json`: 設定読み込み（環境変数で上書き可）
- `docker-compose.yml`: client/server の 2 サービス、モデルキャッシュを `./models` に永続化

## よくあるハマり

- **モデルが見つからない / 取得できない**: 本体コードは `local_files_only=True` です。上の「モデルをキャッシュに用意」を先に実行してください。
- **別モデルに変えたら落ちる**: `*.safetensors` が無いモデルだと `model_loader.py` が失敗します（この実装は safetensors 前提）。
- **速度が遅い**: この Dockerfile は CPU 版の `torch` を入れています。GPU を使う場合はベースイメージ/torch の入れ方を環境に合わせて変更してください。
