"""
-----------------------------------------------------------------------
File: scripts/generation.py
Creation Time: Oct 23rd 2024, 6:34 pm
Author: Saurabh Zinjad
Developer Email: saurabhzinjad@gmail.com
Copyright (c) 2023-2024 Saurabh Zinjad. All rights reserved | https://github.com/Ztrimus
-----------------------------------------------------------------------
"""

import os

import argparse
import traceback
from typing import List
from utils import get_dataframe, split_string_into_list, measure_execution_time
from config import envs, credentials, models
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

logger.info("Device: %s", device)
print(f"Device: {device}")

main_columns = ["Category", "Subcategory", "Question"]
main_columns_xstest = ["note", "focus", "prompt", "type"]


def get_model_and_tokenizer(model_config, auth_token, cache_dir):
    """Loads a pre-trained model and tokenizer for causal language modeling.
    Args:
        model_config (dict): Configuration dictionary containing the model checkpoint path.
        auth_token (str): Authentication token for accessing the model.
        cache_dir (str): Directory to cache the pre-trained model and tokenizer.
    Returns:
        tuple: A tuple containing the tokenizer and the model.
    """
    try:
        # TODO: If exists in cache take from there instead of downloading
        print("cache_dir: ", cache_dir)
        logger.info(f"Loading tokenizer for {model_config['checkpoint']}")
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_config["checkpoint"],
            token=auth_token,
            cache_dir=cache_dir,
        )

        logger.info(f"Loading model for {model_config['checkpoint']}")
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_config["checkpoint"],
            token=auth_token,
            cache_dir=cache_dir,
        ).to(device)

        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading model and tokenizer: {str(e)}")
        raise


@measure_execution_time
def generate_text(model, tokenizer, texts, temp=1, top_p=1):
    try:
        tokenizer.pad_token = tokenizer.eos_token
        output_texts = []
        for id, text in enumerate(texts):
            if isinstance(text, str):
                logger.info(f"{'='*15} {id+1}/{len(texts)} text: {text}")

                inputs = tokenizer(text, return_tensors="pt")
                outputs = model.generate(
                    inputs.input_ids.to(device),
                    max_length=envs.MAX_RESPONSE_LEN,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=temp,
                    top_p=top_p,
                )
                output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[
                    0
                ]
                output_texts.append(output_text)
        return output_texts
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        return None


def generate_answers(
    dataset_path: str = None, question_columns: List[str] = None, model_name: str = None
):
    try:
        # TODO: Argument: category, subcategories, max tokens len,
        model_config = models.Config[model_name]
        logger.info(f"{'='*10} Processing model: {model_name}")
        tokenizer, model = get_model_and_tokenizer(
            model_config=model_config,
            auth_token=credentials.HF_TOKEN,
            cache_dir=envs.MODELS_DIR,
        )

        for question_index, question_col in enumerate(question_columns):
            if dataset_path:
                df = get_dataframe(dataset_path)
                if df.empty:
                    raise ValueError("Empty DataFrame. Check dataset path or format.")
                logger.info(
                    f"{'='*5} Processing column {question_index+1}: {question_col}"
                )
                questions = df[question_col].to_list()

                output_texts = generate_text(model, tokenizer, questions)
                logger.info(f"Storing response in dataframe")
                new_col_name = f"{question_col}_{model_name}"
                df[new_col_name] = output_texts

                if main_columns[0] in df.columns:
                    df = df[main_columns + [question_col, new_col_name]]

                if main_columns_xstest[0] in df.columns:
                    df = df[main_columns_xstest + [question_col, new_col_name]]

                output_path = os.path.join(
                    (
                        envs.GENERATED_DATA_DIR_XSTEST
                        if "xstest" in dataset_path
                        else envs.GENERATED_DATA_DIR
                    ),
                    f"{Path(dataset_path).stem}_{model_name}_{question_col}.csv",
                )
                logger.info(f"Saving results to {output_path}")
                df.to_csv(output_path, index=False)
        logger.info("Script completed successfully")
    except Exception as e:
        logger.error(f"An error occurred in generate_answers: {str(e)}")
        logger.error(traceback.format_exc())


def generate(tokenizer, model, query: str = None, temp=None, top_p=None):
    try:
        if query:
            return generate_text(model, tokenizer, [query], temp, top_p)[0]
        else:
            raise ValueError("Query is empty")
    except Exception as e:
        logger.error(f"An error occurred in generate_answers: {str(e)}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLM response generation on give questions."
    )
    parser.add_argument(
        "--dataset_path", type=str, default=None, help="Path to the dataset file."
    )
    parser.add_argument(
        "--model", type=str, default="llama31", help="Model to use for text generation."
    )
    parser.add_argument(
        "--question_columns",
        type=split_string_into_list,
        default=None,
        help="Columns containing questions to generate answers for.",
    )

    args = parser.parse_args()

    generate_answers(args.dataset_path, args.question_columns, args.model)
