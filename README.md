# LLM Sensitivity

## Dataset

-   GitHub Typo Corpus data can be download from [here](https://github-typo-corpus.s3.amazonaws.com/data/github-typo-corpus.v1.0.0.jsonl.gz)
-   https://arxiv.org/pdf/1911.12893.pdf
-   https://github.com/mhagiwara/github-typo-corpus

## Setup

```sh
module load mamba/latest
source activate llm_safety_39
```

-   Create [`credentials.py`](src/config/credentials.py) at src/config location with your personal credentials.

```python
ASURITE_ID = "YOUR_ASURITE_ID"
HF_TOKEN ="PUT_HF_TOKEN_HERE"
```

-   to make src contains importable

```sh
cd llm-sensitivity
export PYTHONPATH=$(pwd)/src
```
