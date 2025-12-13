# Zunkey_GPT_Search

DuckDuckGo検索→ページ抽出→（任意でローカルLLM要約）→DB保存→RAG検索できるローカル調査ツール。

## Requirements
- Python 3.10+
- Windows推奨
- （任意）llama-cpp-python + gguf モデル

## Install
pip install -r requirements.txt

## Run
python .\Zunkey_GPT_Search.py

## Notes
- settings.json / sqlite / log は実行フォルダに生成されます
- gguf は同梱しません。各自で配置してください
