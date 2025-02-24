"""
-----------------------------------------------------------------------
File: scripts/safety_prepro_res.py
Creation Time: Feb 1st 2025, 1:09 am
Author: Saurabh Zinjad
Developer Email: saurabhzinjad@gmail.com
Copyright (c) 2023-2025 Saurabh Zinjad. All rights reserved | https://github.com/Ztrimus
-----------------------------------------------------------------------
"""

import torch
import argparse
import traceback
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from accelerate import Accelerator, DistributedType
from utils import measure_execution_time, print_log, filter_safety_response
from config import envs, credentials

# Initialize accelerator first
accelerator = Accelerator(mixed_precision="fp16")
device = accelerator.device


@measure_execution_time
def moderate_batch(model, tokenizer, texts, max_batch_size=128):
    output_texts = []
    print_log(
        f"Started safety classification for {len(texts)} texts",
        rank=accelerator.process_index,
    )

    # Dynamic batch sizing based on sequence length
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="longest", max_length=512, pad_to_multiple_of=64
    )

    # Prepare batches with accelerator
    with accelerator.split_between_processes(texts) as local_texts:
        batches = [
            local_texts[i : i + max_batch_size]
            for i in range(0, len(local_texts), max_batch_size)
        ]

        for batch_idx, batch_texts in enumerate(batches):
            if accelerator.is_main_process:
                print_log(
                    f"Processing batch {batch_idx+1}/{len(batches)} (size: {len(batch_texts)})"
                )

            # Prepare inputs with optimized padding
            batch_inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
                pad_to_multiple_of=64,
            ).to(device)

            with torch.no_grad(), torch.amp.autocast(device_type=device.type):
                outputs = model.generate(
                    **batch_inputs,
                    max_new_tokens=5,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False,
                    temperature=0.0,
                    num_return_sequences=1,  # Single classification
                    output_scores=True,  # Enable confidence scoring
                    return_dict_in_generate=True,
                )

            # Decode outputs across all processes
            decoded = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
            gathered_outputs = accelerator.gather_for_metrics(decoded)

            if accelerator.is_main_process:
                output_texts.extend(gathered_outputs)

    return output_texts


def check_safety(dataset_path):
    try:
        model_id = "meta-llama/Llama-Guard-3-8B"

        # Load model with accelerator
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=credentials.HF_TOKEN,
            cache_dir=envs.MODELS_DIR,
            padding_side="left",
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=credentials.HF_TOKEN,
            cache_dir=envs.MODELS_DIR,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Prepare with accelerator
        model, tokenizer = accelerator.prepare(model, tokenizer)

        # Column resolution logic
        question_col_list = ["perturbed_response_pre", "original_response_pre"]

        for question_col in question_col_list:
            # Load data once per node
            if accelerator.is_local_main_process:
                data = pd.read_csv(dataset_path)
                questions = data[question_col].fillna("").to_list()
            else:
                questions = []

            # Broadcast data to all processes
            questions = accelerator.broadcast(questions)

            output_texts = moderate_batch(model, tokenizer, questions)

            if accelerator.is_main_process:
                # Save results
                data[f"{question_col}_safety"] = output_texts
                data.to_csv(dataset_path, index=False)

    except Exception as e:
        print_log(f"Error: {str(e)}", is_error=True)
        accelerator.print(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()
    check_safety(args.dataset_path)
