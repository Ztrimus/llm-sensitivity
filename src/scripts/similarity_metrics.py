"""
-----------------------------------------------------------------------
File: scripts/similarity_metrics.py
Creation Time: Feb 11th 2025, 7:45 pm
Author: Saurabh Zinjad
Developer Email: saurabhzinjad@gmail.com
Copyright (c) 2023-2025 Saurabh Zinjad. All rights reserved | https://github.com/Ztrimus
-----------------------------------------------------------------------
"""

# TODO:
# pip install tensorflow-hub
# pip install editdistance
import os
import editdistance
import argparse
import torch.nn as nn

import traceback
import sys

# TODO: remove for sbatch job
sys.path.append("/Users/saurabh/AA/convergent/projects/llm-sensitivity/src")
from utils import measure_execution_time
from config import envs, credentials

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {device}")


import pandas as pd
import editdistance
import traceback
import logging
import tensorflow_hub as hub
import numpy as np

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@measure_execution_time
def get_similarity(dataset_path):
    try:
        # Load dataset
        data = pd.read_csv(dataset_path)

        latent_similarities = []

        # Load USE model
        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

        # # Compute USE embeddings for all questions at once (vectorized)
        # embeddings = embed(
        #     data[["original_question", "perturbed_question"]].values.tolist()
        # )
        # Compute USE embeddings for each question column separately.
        original_embeds = embed(data["original_question"].tolist())
        perturbed_embeds = embed(data["perturbed_question"].tolist())

        # Cosine similarity in latent space
        for i in range(len(data)):
            cosine_sim = np.dot(original_embeds[i], perturbed_embeds[i]) / (
                np.linalg.norm(original_embeds[i]) * np.linalg.norm(perturbed_embeds[i])
            )
            latent_similarities.append(cosine_sim)

        print(f"Latent space similarity computed for {len(data)} rows.")

        # Store results in dataframe
        data["latent_similarity"] = latent_similarities

        # Save to CSV
        data.to_csv(dataset_path, index=False)
        print(f"Updated dataset saved at {dataset_path}")

        token_similarities = []

        # Compute similarities
        for i in range(len(data)):
            og = data.iloc[i]["original_question"]
            cf = data.iloc[i]["perturbed_question"]

            # Token-level similarity (Levenshtein distance)
            token_sim = editdistance.eval(og, cf) / max(len(og), len(cf))
            token_similarities.append(token_sim)

        print(f"Token-level similarity computed for {len(data)} rows.")
        # Store results in dataframe
        data["token_similarity"] = token_similarities

        # Save to CSV
        data.to_csv(dataset_path, index=False)
        print(f"Updated dataset saved at {dataset_path}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run safety check using llama guard script."
    )
    parser.add_argument(
        "--dataset_path", type=str, default=None, help="Path to the dataset file."
    )

    args = parser.parse_args()

    get_similarity(args.dataset_path)
