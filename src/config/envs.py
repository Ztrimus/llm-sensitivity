from config import credentials

REPO_DIR = f'/scratch/{credentials.ASURITE_ID}/llm-sensitivity'
MODELS_DIR = f'/scratch/{credentials.ASURITE_ID}/cache/llm-sensitivity/models'
DATASETS_DIR = f'/scratch/{credentials.ASURITE_ID}/cache/llm-sensitivity/datasets'
PERTURBED_DATA_DIR = f'/scratch/{credentials.ASURITE_ID}/llm-sensitivity/data/perturbed/catHarmQA/'
GENERATED_DATA_DIR = f'/scratch/{credentials.ASURITE_ID}/llm-sensitivity/data/generated/catHarmQA/'
MAX_RESPONSE_LEN = 256