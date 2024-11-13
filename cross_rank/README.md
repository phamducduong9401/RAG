---
license: mit
datasets:
- hotchpotch/JQaRA
- shunk031/JGLUE
- miracl/miracl
- castorini/mr-tydi
- unicamp-dl/mmarco
language:
- ja
library_name: sentence-transformers
---

## hotchpotch/japanese-reranker-cross-encoder-large-v1

日本語で学習させた Reranker (CrossEncoder) シリーズです。

| モデル名                                                                                                                                | layers | hidden_size |
| ----------------------------------------------------------------------------------------------------------------------------------- | ------ | ----------- |
| [hotchpotch/japanese-reranker-cross-encoder-xsmall-v1](https://huggingface.co/hotchpotch/japanese-reranker-cross-encoder-xsmall-v1) | 6      | 384         |
| [hotchpotch/japanese-reranker-cross-encoder-small-v1](https://huggingface.co/hotchpotch/japanese-reranker-cross-encoder-small-v1)   | 12     | 384         |
| [hotchpotch/japanese-reranker-cross-encoder-base-v1](https://huggingface.co/hotchpotch/japanese-reranker-cross-encoder-base-v1)     | 12     | 768         |
| [hotchpotch/japanese-reranker-cross-encoder-large-v1](https://huggingface.co/hotchpotch/japanese-reranker-cross-encoder-large-v1)   | 24     | 1024        |
| [hotchpotch/japanese-bge-reranker-v2-m3-v1](https://huggingface.co/hotchpotch/japanese-bge-reranker-v2-m3-v1)                       | 24     | 1024        |

Reranker についてや、技術レポート・評価等は以下を参考ください。

- [日本語最高性能のRerankerをリリース / そもそも Reranker とは?](https://secon.dev/entry/2024/04/02/070000-japanese-reranker-release/)
- [日本語 Reranker 作成のテクニカルレポート](https://secon.dev/entry/2024/04/02/080000-japanese-reranker-tech-report/)


## 使い方

### SentenceTransformers

```python
from sentence_transformers import CrossEncoder
import torch

MODEL_NAME = "hotchpotch/japanese-reranker-cross-encoder-large-v1"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CrossEncoder(MODEL_NAME, max_length=512, device=device)
if device == "cuda":
    model.model.half()
query = "感動的な映画について"
passages = [
    "深いテーマを持ちながらも、観る人の心を揺さぶる名作。登場人物の心情描写が秀逸で、ラストは涙なしでは見られない。",
    "重要なメッセージ性は評価できるが、暗い話が続くので気分が落ち込んでしまった。もう少し明るい要素があればよかった。",
    "どうにもリアリティに欠ける展開が気になった。もっと深みのある人間ドラマが見たかった。",
    "アクションシーンが楽しすぎる。見ていて飽きない。ストーリーはシンプルだが、それが逆に良い。",
]
scores = model.predict([(query, passage) for passage in passages])
```

## HuggingFace transformers

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn import Sigmoid

MODEL_NAME = "hotchpotch/japanese-reranker-cross-encoder-large-v1"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

if device == "cuda":
    model.half()

query = "感動的な映画について"
passages = [
    "深いテーマを持ちながらも、観る人の心を揺さぶる名作。登場人物の心情描写が秀逸で、ラストは涙なしでは見られない。",
    "重要なメッセージ性は評価できるが、暗い話が続くので気分が落ち込んでしまった。もう少し明るい要素があればよかった。",
    "どうにもリアリティに欠ける展開が気になった。もっと深みのある人間ドラマが見たかった。",
    "アクションシーンが楽しすぎる。見ていて飽きない。ストーリーはシンプルだが、それが逆に良い。",
]
inputs = tokenizer(
    [(query, passage) for passage in passages],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)
inputs = {k: v.to(device) for k, v in inputs.items()}
logits = model(**inputs).logits
activation = Sigmoid()
scores = activation(logits).squeeze().tolist()
```

## 評価結果

| Model Name                                                                                                               | [JQaRA](https://huggingface.co/datasets/hotchpotch/JQaRA) | [JaCWIR](https://huggingface.co/datasets/hotchpotch/JaCWIR) | [MIRACL](https://huggingface.co/datasets/miracl/miracl) | [JSQuAD](https://github.com/yahoojapan/JGLUE) |
| ------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------- | --------------------------------------------- |
| [japanese-reranker-cross-encoder-xsmall-v1](https://huggingface.co/hotchpotch/japanese-reranker-cross-encoder-xsmall-v1) | 0.6136                                                    | 0.9376                                                      | 0.7411                                                  | 0.9602                                        |
| [japanese-reranker-cross-encoder-small-v1](https://huggingface.co/hotchpotch/japanese-reranker-cross-encoder-small-v1)   | 0.6247                                                    | 0.939                                                       | 0.7776                                                  | 0.9604                                        |
| [japanese-reranker-cross-encoder-base-v1](https://huggingface.co/hotchpotch/japanese-reranker-cross-encoder-base-v1)     | 0.6711                                                    | 0.9337                                                      | 0.818                                                   | 0.9708                                        |
| [japanese-reranker-cross-encoder-large-v1](https://huggingface.co/hotchpotch/japanese-reranker-cross-encoder-large-v1)   | 0.7099                                                    | 0.9364                                                      | 0.8406                                                  | 0.9773                                        |
| [japanese-bge-reranker-v2-m3-v1](https://huggingface.co/hotchpotch/japanese-bge-reranker-v2-m3-v1)                       | 0.6918                                                    | 0.9372                                                      | 0.8423                                                  | 0.9624                                        |
| [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)                                                     | 0.673                                                     | 0.9343                                                      | 0.8374                                                  | 0.9599                                        |
| [bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)                                                     | 0.4718                                                    | 0.7332                                                      | 0.7666                                                  | 0.7081                                        |
| [bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)                                                       | 0.2445                                                    | 0.4905                                                      | 0.6792                                                  | 0.5757                                        |
| [cross-encoder-mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/corrius/cross-encoder-mmarco-mMiniLMv2-L12-H384-v1)  | 0.5588                                                    | 0.9211                                                      | 0.7158                                                  | 0.932                                         |
| [shioriha-large-reranker](https://huggingface.co/cl-nagoya/shioriha-large-reranker)                                      | 0.5775                                                    | 0.8458                                                      | 0.8084                                                  | 0.9262                                        |
| [bge-m3+all](https://huggingface.co/BAAI/bge-m3)                                                                         | 0.576                                                     | 0.904                                                       | 0.7926                                                  | 0.9226                                        |
| [bge-m3+dense](https://huggingface.co/BAAI/bge-m3)                                                                       | 0.539                                                     | 0.8642                                                      | 0.7753                                                  | 0.8815                                        |
| [bge-m3+colbert](https://huggingface.co/BAAI/bge-m3)                                                                     | 0.5656                                                    | 0.9064                                                      | 0.7902                                                  | 0.9297                                        |
| [bge-m3+sparse](https://huggingface.co/BAAI/bge-m3)                                                                      | 0.5088                                                    | 0.8944                                                      | 0.6941                                                  | 0.9184                                        |
| [JaColBERTv2](https://huggingface.co/bclavie/JaColBERTv2)                                                                | 0.5847                                                    | 0.9185                                                      | 0.6861                                                  | 0.9247                                        |
| [multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)                                           | 0.554                                                     | 0.8759                                                      | 0.7722                                                  | 0.8892                                        |
| [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)                                           | 0.4917                                                    | 0.869                                                       | 0.7025                                                  | 0.8565                                        |
| bm25                                                                                                                     | 0.458                                                     | 0.8408                                                      | 0.4387                                                  | 0.9002                                        |


## ライセンス

MIT License