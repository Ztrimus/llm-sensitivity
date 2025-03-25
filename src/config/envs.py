from config import credentials

REPO_DIR = f"/scratch/{credentials.ASURITE_ID}/llm-sensitivity"
MODELS_DIR = f"/scratch/{credentials.ASURITE_ID}/cache/llm-sensitivity/models"
DATASETS_DIR = f"/scratch/{credentials.ASURITE_ID}/cache/llm-sensitivity/datasets"
PERTURBED_DATA_DIR = (
    f"/scratch/{credentials.ASURITE_ID}/llm-sensitivity/data/perturbed/catHarmQA/"
)
GENERATED_DATA_DIR = (
    f"/scratch/{credentials.ASURITE_ID}/llm-sensitivity/data/generated/catHarmQA/"
)
SAFETY_DATA_DIR = (
    f"/scratch/{credentials.ASURITE_ID}/llm-sensitivity/data/safety/catHarmQA/response"
)
SAFETY_QUESTIONS_DATA_DIR = (
    f"/scratch/{credentials.ASURITE_ID}/llm-sensitivity/data/safety/catHarmQA/questions"
)
MAX_RESPONSE_LEN = 256

GENERATED_DATA_DIR_XSTEST = (
    f"/scratch/{credentials.ASURITE_ID}/llm-sensitivity/data/generated/xstest/"
)
SAFETY_DATA_DIR_XSTEST = (
    f"/scratch/{credentials.ASURITE_ID}/llm-sensitivity/data/safety/xstest/response"
)
PERTURBED_DATA_DIR_XSTEST = (
    f"/scratch/{credentials.ASURITE_ID}/llm-sensitivity/data/perturbed/xstest"
)
SAFETY_QUESTIONS_DATA_DIR_XSTEST = (
    f"/scratch/{credentials.ASURITE_ID}/llm-sensitivity/data/safety/xstest/questions"
)
