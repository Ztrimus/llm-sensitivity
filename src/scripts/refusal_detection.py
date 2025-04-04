"""
-----------------------------------------------------------------------
File: scripts/refusal_detection.py
Creation Time: Apr 3rd 2025, 11:54 pm
Author: Saurabh Zinjad
Developer Email: saurabhzinjad@gmail.com
Copyright (c) 2023-2025 Saurabh Zinjad. All rights reserved | https://github.com/Ztrimus
-----------------------------------------------------------------------
"""

import os
import torch
import argparse
import pandas as pd
from typing import List
from datasets import Dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from utils import (
    get_dataframe,
    print_log,
    split_string_into_list,
    measure_execution_time,
)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print_log(f"Device: {device}")


def refusal_check(
    df: pd.DataFrame = None, question_columns: List[str] = None
) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Empty DataFrame. Check dataset path or format.")
    if question_columns is None or len(question_columns) == 0:
        raise ValueError("No columns specified for classification.")

    print_log("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        "ProtectAI/distilroberta-base-rejection-v1"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "ProtectAI/distilroberta-base-rejection-v1"
    )

    print_log("Initializing classification pipeline...")
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=512,
        device=device,
        batch_size=64,
    )

    for col in question_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")
        print_log(f"Running classifier on: {col}")
        try:
            # Create HuggingFace Dataset
            dataset = Dataset.from_pandas(
                df[[col]].fillna("").astype(str), preserve_index=True
            )

            # Run batched inference
            results = classifier(dataset[col], batch_size=64)

            df[col + "_refusal"] = [r["label"] for r in results]
        except Exception as e:
            print_log(f"Error processing column '{col}': {e}", is_error=True)
            raise

    return df


@measure_execution_time
def refusal_detection(dataset_path: str = None, question_columns: List[str] = None):
    try:
        if not dataset_path:
            raise ValueError("Dataset path not provided.")

        print_log(f"Loading dataset from: {dataset_path}")
        df = get_dataframe(dataset_path)

        print_log(f"Starting refusal detection for columns: {question_columns}")
        df = refusal_check(df, question_columns)

        base, ext = os.path.splitext(dataset_path)
        output_path = f"{base}_refusal{ext}"
        df.to_csv(output_path, index=False)

        print_log(f"Output written to: {output_path}")
        print_log("Script completed successfully.")

    except Exception as e:
        print_log(f"Error in refusal detection: {e}", is_error=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run refusal check using above code.")

    parser.add_argument(
        "--dataset_path", type=str, default=None, help="Path to the dataset file."
    )
    parser.add_argument(
        "--question_columns",
        type=split_string_into_list,
        default=None,
        help="Columns containing questions to generate answers for.",
    )

    args = parser.parse_args()

    refusal_detection(args.dataset_path, args.question_columns)
