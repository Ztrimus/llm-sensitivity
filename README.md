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

## Directory structure: 

└── ztrimus-llm-sensitivity/
    ├── README.md
    ├── docs/
    │   ├── mamba-setup.md
    │   ├── repo-setup.md
    │   ├── sol-setup.md
    │   ├── Thesis Defense - Presentation.pdf
    │   └── Thesis Paper - Proquest.pdf
    └── src/
        ├── config/
        │   ├── __init__.py
        │   ├── envs.py
        │   └── models.py
        ├── experiments/
        │   ├── 1_gnrt_ans_for_org_ques.sh
        │   ├── 2_prtrb_chr_lvl.sh
        │   ├── 3_prtrb_wrd_lvl.sh
        │   ├── 4_gnrt_ans_for_char_lvl_mistral_part1.sh
        │   ├── 5_gnrt_ans_for_wrd_lvl_mistral_part1.sh
        │   ├── 6_gnrt_ans_for_char_lvl_mistral_part2.sh
        │   ├── 7_gnrt_ans_for_wrd_lvl_mistral_part2.sh
        │   ├── 8_gnrt_ans_for_char_lvl_llama31_part1.sh
        │   ├── 9_gnrt_ans_for_char_lvl_llama31_part2.sh
        │   ├── 10_gnrt_ans_for_char_lvl_llama3_part1.sh
        ....
        │   ├── 60_safety_xstest_responses_rerun.sh
        │   └── 61_refusal.sh
        ├── notebooks/
        │   ├── __init__.py
        │   ├── Cateogorical-response-analysis.ipynb
        │   ├── Cateogorical-response-analysis_Pre.ipynb
        │   ├── combining-multiple-datasets-xstest.ipynb
        │   ├── combining-multiple-datasets.ipynb
        │   ├── EDA-CategoricalHarmfulQA.ipynb
        │   ├── EDA-GED.ipynb
        │   ├── EDA-gh-typo.ipynb
        │   ├── EDA-squad.ipynb
        │   ├── insightsToTable.ipynb
        │   ├── llama-guard-question-analysis.ipynb
        │   ├── llama-guard-response-analysis.ipynb
        │   ├── perturbations.ipynb
        │   ├── Refusal analysis-catharm.ipynb
        │   ├── Refusal analysis-xstest.ipynb
        │   ├── remove_que_from_response.ipynb
        │   ├── remove_que_from_response_xstest.ipynb
        │   ├── RQ1.ipynb
        │   ├── RQ2.ipynb
        │   ├── RQ3.ipynb
        │   ├── safety_analysis_from_ques_to_response.ipynb
        │   ├── similarity-metrics.ipynb
        │   ├── similarity.ipynb
        │   └── robustification/
        │       └── paraphrasing.ipynb
        ├── scripts/
        │   ├── generation.py
        │   ├── perturbation.py
        │   ├── refusal_detection.py
        │   ├── safety.py
        │   ├── safety_prepro_res-simple.py
        │   ├── safety_prepro_res.py
        │   └── similarity_metrics.py
        └── utils/
            └── __init__.py
    
