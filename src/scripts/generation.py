'''
-----------------------------------------------------------------------
File: scripts/generation.py
Creation Time: Oct 23rd 2024, 6:34 pm
Author: Saurabh Zinjad
Developer Email: saurabhzinjad@gmail.com
Copyright (c) 2023-2024 Saurabh Zinjad. All rights reserved | https://github.com/Ztrimus
-----------------------------------------------------------------------
'''

import os

import argparse
import traceback
from typing import List
from utils import get_dataframe, split_string_into_list, measure_execution_time
from config import envs, credentials, models

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {device}")

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
            pretrained_model_name_or_path = model_config['checkpoint'],
            token = auth_token,
            cache_dir = cache_dir,
        )

        logger.info(f"Loading model for {model_config['checkpoint']}")
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path = model_config['checkpoint'],
            token = auth_token,
            cache_dir = cache_dir
        ).to(device)

        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading model and tokenizer: {str(e)}")
        raise

@measure_execution_time
def generate_text(model, tokenizer, texts):
    try:
        tokenizer.pad_token = tokenizer.eos_token
        output_texts = []
        for id, text in enumerate(texts):
            logger.info(f"{'='*15} {id+1}/{len(texts)} text: {text}")

            inputs = tokenizer(text, return_tensors='pt')
            outputs = model.generate(inputs.input_ids.to(device), max_length=envs.MAX_RESPONSE_LEN, pad_token_id=tokenizer.eos_token_id)
            output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            output_texts.append(output_text)
        return output_texts
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        return None

def generate_answers(dataset_path: str = None, question_columns: List[str] = None):
    try:
        if dataset_path:
            df = get_dataframe(dataset_path)
            if df.empty:
                raise ValueError("Empty DataFrame. Check dataset path or format.")
            
            for question_col in question_columns:
                logger.info(f"{'='*5} Processing column: {question_col}")
                questions = df[question_col].to_list()
                
                # TODO: Argument: model_name, category, subcategories, max tokens len,
                for model_name, model_config in models.Config.items():
                    logger.info(f"{'='*10} Processing model: {model_name}")
                    tokenizer, model = get_model_and_tokenizer(
                        model_config=model_config,
                        auth_token=credentials.HF_TOKEN,
                        cache_dir=envs.MODELS_DIR)

                    output_texts = generate_text(model, tokenizer, questions)
                    logger.info(f"Storing response in dataframe")
                    df[model_name] = output_texts
                    output_path = os.path.join(envs.GENERATED_DATA_DIR, f'answers_{model_name}_{question_col}.csv')
                    logger.info(f"Saving results to {output_path}")
                    df.to_csv(output_path, index=False)
            logger.info("Script completed successfully")
    except Exception as e:
        logger.error(f"An error occurred in generate_answers: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run perturbation experiments.')
    parser.add_argument('--dataset_path', type=str, default=None, help='Path to the dataset file.')
    parser.add_argument('--question_columns', type=str, default=None, help='Columns containing questions to generate answers for.')

    args = parser.parse_args()
    question_columns = split_string_into_list(args.question_columns)

    generate_answers(args.dataset_path, question_columns)




# try:
#         df_path = 'data/CatHarmQA/CatHarmModelPeturb.csv'
#         logger.info(f"Reading dataframe from {df_path}")
#         df = pd.read_csv(df_path)
#         logger.info(f"Columns: {df.columns}")
#         generate_answers(df, columns=['ocr_n1', 'ocr_n2', 'ocr_n3', 'ocr_n4', 'ocr_n5', 'keyboard_n1', 'keyboard_n2', 'keyboard_n3', 'keyboard_n4', 'keyboard_n5', 'random_insert_n1', 'random_insert_n2', 'random_insert_n3', 'random_insert_n4', 'random_insert_n5', 'random_substitute_n1', 'random_substitute_n2', 'random_substitute_n3', 'random_substitute_n4', 'random_substitute_n5', 'random_swap_n1', 'random_swap_n2', 'random_swap_n3', 'random_swap_n4', 'random_swap_n5', 'random_delete_n1', 'random_delete_n2', 'random_delete_n3', 'random_delete_n4', 'random_delete_n5'])
#     except Exception as e:
#         logger.error(f"An error occurred in __main__: {str(e)}")