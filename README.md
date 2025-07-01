# LLM Sensitivity

## Documentation
- [Presentation Deck](/docs/Thesis%20Defense%20-%20Presentation.pdf)
- [In Depth Interactive Documentation](https://deepwiki.com/Ztrimus/llm-sensitivity)
- [Full Research Paper](/docs/Thesis%20Paper%20-%20Proquest.pdf)
- [Setup Documentation](/docs/)

## Final Experimental Dataset
- https://huggingface.co/datasets/Ztrimus/Prompt-Perturbation-Safety-Dataset
```python
from datasets import load_dataset

ds = load_dataset("Ztrimus/llm-safety-flip-dataset", split="full")

# Preview a sample
print(ds[0])

# Analyze flip rate
import pandas as pd
df = ds.to_pandas()
flip_rate = ((df.original_response_safety == "safe") & (df.perturbed_response_safety == "unsafe")).mean()
print(f"Safe → Unsafe flip rate: {flip_rate:.2%}")
```

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