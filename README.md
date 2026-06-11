# RAG Demo

LLM単体回答とRAG回答を比較するための、シンプルなPythonデモです。  
加えて、`malicious_policy.md` を混ぜることで、**プロンプトインジェクション耐性の基本デモ**もできます。

## 特徴

- LLM only と RAG の比較
- OpenAI Python SDK の `responses.create()` を利用
- 検索チャンクのスコア表示
- 基本的なプロンプトインジェクション検知
- `--inject` による悪性チャンク混入デモ
- 日本語回答

## 必要条件

- Python 3.10 以上
- OpenAI API key
- `rag.py` または `rag` モジュール
  - `LocalVectorIndex`
  - `Chunk`
  - `simple_chunk`
- コーパス用ディレクトリ

## セットアップ

### 1. 仮想環境作成

#### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. パッケージインストール

```bash
pip install -r requirements.txt
```

### 3. `.env` 作成

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

## ディレクトリ例

```text
.
├── demo.py
├── requirements.txt
├── README.md
├── rag.py
└── data
    └── corpus
        ├── dartmouth.txt
        ├── ai_history.md
        └── malicious_policy.md
```

## 実行

### デフォルト実行

```bash
python demo.py
```

### 質問を指定

```bash
python demo.py --question "ダートマス会議（1956）の要点を3点で。根拠も示して。"
```

### 上位検索件数を指定

```bash
python demo.py --top-k 3
```

### 注入デモ

```bash
python demo.py --inject
```

### コーパスディレクトリ指定

```bash
python demo.py --corpus-dir data/corpus
```

## `--inject` について

`--inject` を付けると、`data/corpus/malicious_policy.md` の先頭チャンクを取得結果に追加します。  
これにより、RAGの文脈へ悪意ある命令文が紛れ込んだケースを再現できます。

このデモでは以下の対策を入れています。

- CONTEXT を **untrusted data** として扱う
- 文書中の命令に従わないよう system prompt で制約
- suspicious な文を簡易検知
- source boundary (`[SOURCE_START] ... [SOURCE_END]`) を明示

ただし、これは**教育用の基本対策**であり、完全な防御ではありません。

## 注意

- `rag.py` はこのリポジトリに含まれている前提です。
- `LocalVectorIndex`, `Chunk`, `simple_chunk` の実装は別途必要です。
- OpenAI API の利用料金が発生する場合があります。

## 今後の改善案

- citation の事後検証
- suspicious chunk の除外ポリシー
- rerank の導入
- JSON schema での出力制御
- 単体テスト追加
