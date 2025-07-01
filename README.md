# LLM Sensitivity

## Documentation
- [Presentation Deck](/docs/Thesis%20Defense%20-%20Presentation.pdf)
- [In Depth Interactive Documentation](https://deepwiki.com/Ztrimus/llm-sensitivity)
- [Full Research Paper](/docs/Thesis%20Paper%20-%20Proquest.pdf)
- [Setup Documentation](/docs/)

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