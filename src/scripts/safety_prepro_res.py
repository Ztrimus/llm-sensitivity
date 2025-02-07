"""
-----------------------------------------------------------------------
File: scripts/safety_prepro_res.py
Creation Time: Jan 29th 2025, 12:25 am
Author: Saurabh Zinjad
Developer Email: saurabhzinjad@gmail.com
Copyright (c) 2023-2025 Saurabh Zinjad. All rights reserved | https://github.com/Ztrimus
-----------------------------------------------------------------------
"""

import os
import argparse
import torch.nn as nn

import traceback
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

def filter_safety_response(label):
    return label.strip().split()[0].lower()

@measure_execution_time
def moderate(model, tokenizer, texts):
    try:
        output_texts = []
        for id, text in enumerate(texts):
            if isinstance(text, str):
                logger.info(f"{'='*15} {id+1}/{len(texts)} text: \n{text}\n\n")

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
                try:
                    output_text = filter_safety_response(output_text)
                except Exception as e:
                    logger.error(f"Error output_text: {output_text}")
                    logger.error(f"Error filtering response: {str(e)}")
                    print(f"Error: {e}")
                
                output_texts.append(output_text)
            else:
                output_texts.append('safe')
        print(f"Uniue Output texts:\n {pd.Series(output_texts).value_counts()}")
        return output_texts
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        print(f"Error: {e}")
        return None


def check_safety(dataset_path):
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


        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)  # Wrap model for multi-GPU usage

        question_col_list = ["perturbed_response_pre", "original_response_pre"]
        data = pd.read_csv(dataset_path)

        for question_col in question_col_list:
            print(f"Processing column: {question_col}")
            questions = data[question_col].to_list()

            output_texts = moderate(model, tokenizer, questions)
            print(f"Storing response in dataframe")
            new_col_name = f"{question_col}_safety"
            if new_col_name in data.columns:
                data[new_col_name] = output_texts
            else:
                data.insert(
                    data.columns.get_loc(question_col) + 1,
                    new_col_name,
                    output_texts,
                )
            print("Storing {question_col} column in {dataset_path}")
            data.to_csv(dataset_path, index=False)
            logger.info("Stored {question_col} column in {dataset_path}")
            print(f"{'-'*240}")
    except Exception as e:
        logger.error(f"An error occurred in generate_answers: {str(e)}")
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

    check_safety(args.dataset_path)
