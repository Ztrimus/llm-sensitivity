"""
-----------------------------------------------------------------------
File: scripts/safety_prepro_res.py
Creation Time: Jan 29th 2025, 12:25 am
Author: Saurabh Zinjad
Developer Email: saurabhzinjad@gmail.com
Copyright (c) 2023-2025 Saurabh Zinjad. All rights reserved | https://github.com/Ztrimus
-----------------------------------------------------------------------
"""
# torch.no_grad()
# model.half()
# torch.cuda.amp.autocast()

import os
import argparse
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

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
        logger.info(f"Started generating text for {len(texts)} texts")
        for id, text in enumerate(texts):
            if isinstance(text, str):
                chat = [{"role": "user", "content": text}]
                input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(
                    device
                )

                if hasattr(model, "module"):
                    output = model.module.generate(
                        input_ids=input_ids, max_new_tokens=100, pad_token_id=0
                    )
                else:
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
                output_texts.append("safe")
        print(f"Uniue Output texts:\n {pd.Series(output_texts).value_counts()}")
        return output_texts
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        print(f"Error: {e}")
        return None

@measure_execution_time
def moderate_batch(model, tokenizer, texts, batch_size=32):
    output_texts = []
    logger.info(f"Started generating text for {len(texts)} texts")
    # Process texts in batches.
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        # Prepare chat messages if needed. Otherwise, simply batch encode the texts.
        batch_inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        with torch.no_grad():
            # Optionally use AMP for faster mixed precision inference.
            with torch.cuda.amp.autocast(device_type=device):
                outputs = model.generate(
                    input_ids=batch_inputs.input_ids,
                    attention_mask=batch_inputs.attention_mask,
                    max_new_tokens=100,
                    pad_token_id=0
                )
        # Iterate through batch outputs and apply filtering.
        for j, output in enumerate(outputs):
            # Dynamically determine prompt lengths can be tricky.
            # If each entry has a different prompt length, consider storing these lengths prior to batching.
            prompt_length = (batch_inputs.input_ids[j] != tokenizer.pad_token_id).sum()
            decoded = tokenizer.decode(
                output[prompt_length:],
                skip_special_tokens=True
            )
            try:
                decoded = filter_safety_response(decoded)
            except Exception as e:
                logger.error(f"Error filtering response: {str(e)}")
            output_texts.append(decoded)
    return output_texts


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
        )
        model.to(device)
        model.half()  # Convert model parameters and buffers to FP16

        if torch.cuda.device_count() > 1:
            model = DistributedDataParallel(model)  # Wrap model for multi-GPU usage
            # model = nn.DataParallel(model)  # Wrap model for multi-GPU usage

        question_col_list = ["perturbed_response_pre", "original_response_pre"]
        data = pd.read_csv(dataset_path)

        for question_col in question_col_list:
            print(f"Processing column: {question_col}")
            questions = data[question_col].to_list()

            output_texts = moderate_batch(model, tokenizer, questions)
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
            data.to_csv(dataset_path, index=False, chunksize=10000)
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
