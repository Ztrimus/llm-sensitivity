"""
-----------------------------------------------------------------------
File: scripts/safety_prepro_res-simple.py
Creation Time: Feb 23rd 2025, 5:57 pm
Author: Saurabh Zinjad
Developer Email: saurabhzinjad@gmail.com
Copyright (c) 2023-2025 Saurabh Zinjad. All rights reserved | https://github.com/Ztrimus
-----------------------------------------------------------------------
"""

import torch
import argparse
import traceback
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.parallel import DistributedDataParallel

from utils import measure_execution_time, print_log, filter_safety_response
from config import envs, credentials

device = "cuda" if torch.cuda.is_available() else "cpu"
print_log(f"Device: {device}")


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
        model_id = "meta-llama/Llama-Guard-3-8B"

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_id,
            token=credentials.HF_TOKEN,
            cache_dir=envs.MODELS_DIR,
        )

        tokenizer.padding_side = "left"

        # Set pad_token to eos_token if it's not set
        if tokenizer.pad_token is None:
            # tokenizer.pad_token = tokenizer.eos_token
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_id,
            token=credentials.HF_TOKEN,
            cache_dir=envs.MODELS_DIR,
            torch_dtype=torch.float16,
        )
        # Switch to eval mode
        model.eval()
        model.to(device)
        # Convert model parameters and buffers to FP16
        # model.half()  # or set torch_dtype=torch.float16 in from_pretrained

        if torch.cuda.device_count() > 1:
            torch.distributed.init_process_group(backend="nccl")
            model = DistributedDataParallel(model)  # Wrap model for multi-GPU usage

        question_col_list = ["perturbed_response_pre", "original_response_pre"]
        data = pd.read_csv(dataset_path)

        for question_col in question_col_list:
            print_log(f"Processing column: {question_col}")
            questions = data[question_col].fillna("").to_list()

            output_texts = moderate_batch(model, tokenizer, questions)
            print_log(f"response generation is done for '{question_col}' column")

            new_col_name = f"{question_col}_safety"
            if new_col_name in data.columns:
                data[new_col_name] = output_texts
            else:
                data.insert(
                    data.columns.get_loc(question_col) + 1,
                    new_col_name,
                    output_texts,
                )

            print_log(f"Storing '{question_col}' column in dataset '{dataset_path}'.")
            data.to_csv(dataset_path, index=False, chunksize=10000)
            print_log(f"{'-'*240}")

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
