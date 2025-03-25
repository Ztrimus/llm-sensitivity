"""
-----------------------------------------------------------------------
File: scripts/safety.py
Creation Time: Oct 30th 2024, 10:20 pm
Author: Saurabh Zinjad
Developer Email: saurabhzinjad@gmail.com
Copyright (c) 2023-2024 Saurabh Zinjad. All rights reserved | https://github.com/Ztrimus
-----------------------------------------------------------------------
"""

import os

import argparse
import traceback
from typing import List
from utils import (
    get_dataframe,
    split_string_into_list,
    measure_execution_time,
    print_log,
    is_not_exist_create_dir,
)
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

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {device}")


@measure_execution_time
def moderate(model, tokenizer, texts):
    try:
        output_texts = []
        for id, text in enumerate(texts):
            logger.info(f"{'='*15} {id+1}/{len(texts)} text: {text}")

            chat = [{"role": "user", "content": text}]
            input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(
                device
            )
            output = model.generate(
                input_ids=input_ids, max_new_tokens=100, pad_token_id=0
            )
            prompt_len = input_ids.shape[-1]
            output_text = tokenizer.decode(
                output[0][prompt_len:], skip_special_tokens=True
            )
            output_texts.append(output_text)
        return output_texts
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        return None


def check_safety(
    data_dir_path: str = None,
    filters: List[str] = None,
    is_perturbed_questions: bool = False,
):
    try:
        model_id = "meta-llama/Llama-Guard-3-8B"

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_id,
            token=credentials.HF_TOKEN,
            cache_dir=envs.MODELS_DIR,
        )

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_id,
            token=credentials.HF_TOKEN,
            cache_dir=envs.MODELS_DIR,
        ).to(device)

        if filters is None:
            datasets = os.listdir(data_dir_path)
        else:
            datasets = [
                dataset
                for dataset in os.listdir(data_dir_path)
                if all(filter in dataset for filter in filters)
            ]
        print_log(f"Filtered datasets: {datasets}")

        for dataset in datasets:
            dataset_path = os.path.join(data_dir_path, dataset)
            if dataset_path:
                df = get_dataframe(dataset_path)
                if df.empty:
                    raise ValueError("Empty DataFrame. Check dataset path or format.")

                if not is_perturbed_questions:
                    question_col_list = [df.columns[-1]]
                    output_dir_path = (
                        envs.SAFETY_DATA_DIR_XSTEST
                        if "xstest" in dataset_path
                        else envs.SAFETY_DATA_DIR
                    )
                else:
                    if "xstest" in data_dir_path:
                        question_col_list = [
                            column
                            for column in df.columns
                            if column.startswith("prompt")
                        ]
                        output_dir_path = envs.SAFETY_QUESTIONS_DATA_DIR_XSTEST
                    else:
                        question_col_list = [
                            column
                            for column in df.columns
                            if column.startswith("Question")
                        ]
                        output_dir_path = envs.SAFETY_QUESTIONS_DATA_DIR

                for question_col in question_col_list:
                    logger.info(f"{'='*5} Processing column {question_col}")
                    questions = df[question_col].to_list()

                    output_texts = moderate(model, tokenizer, questions)
                    logger.info(f"Storing response in dataframe")
                    new_col_name = f"{question_col}_safety"
                    df.insert(
                        df.columns.get_loc(question_col) + 1,
                        new_col_name,
                        output_texts,
                    )

                    is_not_exist_create_dir(output_dir_path)

                    output_path = os.path.join(
                        output_dir_path,
                        f"{Path(dataset_path).stem}_safety.csv",
                    )
                    logger.info(f"Saving results to {output_path}")
                    df.to_csv(output_path, index=False)
        logger.info("Script completed successfully")
    except Exception as e:
        logger.error(f"An error occurred in generate_answers: {str(e)}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run safety check using llama guard script."
    )
    parser.add_argument(
        "--dataset_path", type=str, default=None, help="Path to the dataset file."
    )
    parser.add_argument(
        "--filters",
        type=split_string_into_list,
        default=None,
        help="Filter datasets based on the given list of filters.",
    )
    parser.add_argument(
        "--is_perturbed_questions",
        type=bool,
        default=False,
        help="Is the datasets perturbed questions?",
    )

    args = parser.parse_args()

    check_safety(args.dataset_path, args.filters, args.is_perturbed_questions)
