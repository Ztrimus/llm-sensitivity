"""
-----------------------------------------------------------------------
File: scripts/safety_prepro_res-simple.py
Creation Time: Feb 23rd 2025, 5:57 pm
Author: Saurabh Zinjad
Developer Email: saurabhzinjad@gmail.com
Copyright (c) 2023-2025 Saurabh Zinjad. All rights reserved | https://github.com/Ztrimus
-----------------------------------------------------------------------
"""

import os
import time
import torch
import argparse
import traceback
import subprocess
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import measure_execution_time, print_log, filter_safety_response
from config import envs, credentials

device = "cuda" if torch.cuda.is_available() else "cpu"
print_log(f"Device: {device}")


def moderate(model, tokenizer, texts):
    try:
        output_texts = []
        for id, text in enumerate(texts):
            print_log(f"{'='*15} {id+1}/{len(texts)}")

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
            output_text = filter_safety_response(output_text)
            output_texts.append(output_text)
        return output_texts
    except Exception as e:
        print_log(f"Error generating text: {str(e)}", is_error=True)
        return None


@measure_execution_time
def moderate_batch(model, tokenizer, texts, batch_size=32):
    output_texts = []
    print_log(f"Started generating text for {len(texts)} texts")

    # Decide if we should call model or model.module
    model = model.module if hasattr(model, "module") else model

    for i in range(0, len(texts), batch_size):
        print_log(f"Processing batch: {i//batch_size}/{len(texts)//batch_size}")
        batch_texts_raw = texts[i : i + batch_size]
        batch_texts = [
            str(text) if not isinstance(text, str) else text for text in batch_texts_raw
        ]

        # Build chat prompts for each text
        batch_chats = [
            [{"role": "user", "content": text}, {"role": "assistant", "content": ""}]
            for text in batch_texts
        ]
        # Apply chat template
        batch_inputs = tokenizer.apply_chat_template(
            batch_chats, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        # Handle both dict and Tensor output
        if isinstance(batch_inputs, dict):
            input_ids = batch_inputs.input_ids
            attention_mask = batch_inputs.attention_mask
            print_log(f"Input IDs for first sample: {batch_inputs.input_ids[0]}")
            print_log(f"Pad token id: {tokenizer.pad_token_id}")
        elif isinstance(batch_inputs, torch.Tensor):
            input_ids = batch_inputs
            # Compute attention_mask based on non-pad tokens
            attention_mask = (batch_inputs != tokenizer.pad_token_id).long()
        else:
            raise ValueError("Unexpected output from tokenizer.apply_chat_template")

        print(f"Length of input ID: {len(input_ids)}")
        print(f"Length of attention mask: {len(attention_mask)}")
        print(f"input ID: {input_ids}")
        print(f"attention mask: {attention_mask}")

        with torch.no_grad():
            # Use AMP for faster mixed precision inference.
            with torch.amp.autocast(device_type=device):
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=5,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False,
                )

        # Decode each output
        for j, output in enumerate(outputs):
            # Calculate how many tokens in the prompt (non-pad tokens)
            # prompt_length = (batch_inputs.input_ids[j] != tokenizer.pad_token_id).sum()
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            # try:
            #     decoded = filter_safety_response(decoded)
            # except Exception as e:
            #     print_log(f"Error filtering response: {str(e)}", is_error=True)
            print_log(f"\n\n{decoded}\n\n")
            output_texts.append(decoded)
    return output_texts


def check_safety(dataset_path):
    try:
        # Overall allocated time (in seconds)
        allocated_time = 16 * 3600  # 16 hours
        start_time = time.time()

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
            torch_dtype=torch.float16,
            # device_map="auto",
        ).to(device)
        model.eval() # Switch to eval mode

        question_col_list = ["perturbed_response_pre", "original_response_pre"]
        data = pd.read_csv(dataset_path)

        if "category" not in data.columns:
            print_log("Column 'category' not found in dataset.", is_error=True)
            return
        
        # Create a "category-wise" folder inside the dataset's directory.
        dataset_dir = os.path.dirname(dataset_path)
        cat_folder = os.path.join(dataset_dir, "category-wise")
        os.makedirs(cat_folder, exist_ok=True)

        categories = data["category"].unique()
        for category in categories:
            cat_file_path = os.path.join(cat_folder, f"{category.replace('/', '_')}.csv")
            if os.path.exists(cat_file_path):
                print_log(f"Category '{category}' is already processed. Skipping.")
                continue

            print_log(f"Processing category: {category}")
            cat_data = data[data["category"] == category].copy()

            for question_col in question_col_list:   
                print_log(f"Processing column: {question_col} for category '{category}'")
                questions = cat_data[question_col].fillna("").to_list()

                output_texts = moderate(model, tokenizer, questions)
                print_log(f"Response generation is done for '{question_col}' column in category '{category}'")

                new_col_name = f"{question_col}_safety"
                if new_col_name in cat_data.columns:
                    cat_data[new_col_name] = output_texts
                else:
                    cat_data.insert(
                        cat_data.columns.get_loc(question_col) + 1,
                        new_col_name,
                        output_texts,
                    )

            cat_data.to_csv(cat_file_path, index=False, chunksize=10000)
            print_log(f"Stored processed category data for '{category}' in file: '{cat_file_path}'.")
            print_log(f"{'-'*240}")

            # After finishing this category, check overall remaining time.
            elapsed = time.time() - start_time
            remaining_time = allocated_time - elapsed

            if len(categories) != len(os.listdir(cat_folder)) and remaining_time < elapsed:
                print_log("Not enough time to process the next column. Triggering subprocess.", is_error=False)
                subprocess.run(["sbatch", "src/experiments/42_safety_preprocessed_response-II.sh"])
                break

    except Exception as e:
        print_log(f"An error occurred in generate_answers: {str(e)}", is_error=True)
        print_log(traceback.format_exc(), is_error=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run safety check using llama guard script."
    )
    parser.add_argument(
        "--dataset_path", type=str, default=None, help="Path to the dataset file."
    )

    args = parser.parse_args()

    check_safety(args.dataset_path)
