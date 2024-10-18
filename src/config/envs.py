from src.config import credentials

REPO_DIR = f'/scratch/{credentials.ASURITE_ID}/llm-sensitivity'
MODELS_DIR = f'/scratch/{credentials.ASURITE_ID}/cache/llm-sensitivity/models'
DATASETS_DIR = f'/scratch/{credentials.ASURITE_ID}/cache/llm-sensitivity/datasets'
MAX_RESPONSE_LEN = 256