'''
-----------------------------------------------------------------------
File: models/llm.py
Creation Time: Oct 1st 2024, 10:16 pm
Author: Saurabh Zinjad
Developer Email: saurabhzinjad@gmail.com
Copyright (c) 2023-2024 Saurabh Zinjad. All rights reserved | https://github.com/Ztrimus
-----------------------------------------------------------------------
'''
import sys
import traceback
from typing import List
sys.path.append('./')
from src.config import envs, credentials, models
from src.utils.logger import measure_execution_time

import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"

@measure_execution_time
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
            logger.info(f"{'='*20}\n{id+1}/{len(texts)} text: {text}")

            inputs = tokenizer(text, return_tensors='pt')
            outputs = model.generate(inputs.input_ids.to(device), max_length=envs.MAX_RESPONSE_LEN, pad_token_id=tokenizer.eos_token_id)
            output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            logger.info(f"output_text: {output_text}\n\n")
            output_texts.append(output_text)
        # output_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return output_texts
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        return None

@measure_execution_time
def get_dataset():
    try:
        if False:
            pass
        else:
            logger.info("Loading dataset")
            dataset = load_dataset("declare-lab/CategoricalHarmfulQA", split="en", cache_dir=envs.DATASETS_DIR)
        return dataset.to_pandas()
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def generate_answers(df: pd.DataFrame = None, columns: List[str] = None):
    try:
        if df is None:
            df = get_dataset()
        logger.info(f"DF: \n{df.head()}")

        for question_col in columns:
            logger.info(f"Processing column: {question_col}")
            questions = df[question_col].to_list()
            
            # TODO: Argument: model_name, category, subcategories, max tokens len,
            for model_name, model_config in models.Config.items():
                logger.info(f"Processing model: {model_name}")
                tokenizer, model = get_model_and_tokenizer(
                    model_config=model_config,
                    auth_token=credentials.HF_TOKEN,
                    cache_dir=envs.MODELS_DIR)

                output_texts = generate_text(model, tokenizer, questions)
                logger.info(f"Storing response in dataframe")
                df[model_name] = output_texts
                output_path = os.path.join(envs.DATASETS_DIR, f'answers_{model_name}_{question_col}.csv')
                logger.info(f"Saving results to {output_path}")
                df.to_csv(output_path, index=False)
        logger.info("Script completed successfully")
    except Exception as e:
        logger.error(f"An error occurred in generate_answers: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    try:
        df_path = 'data/CatHarmQA/CatHarmModelPeturb.csv'
        logger.info(f"Reading dataframe from {df_path}")
        df = pd.read_csv(df_path)
        logger.info(f"Columns: {df.columns}")
        generate_answers(df=None, columns=['ocr_n1', 'ocr_n2', 'ocr_n3', 'ocr_n4', 'ocr_n5', 'keyboard_n1', 'keyboard_n2', 'keyboard_n3', 'keyboard_n4', 'keyboard_n5', 'random_insert_n1', 'random_insert_n2', 'random_insert_n3', 'random_insert_n4', 'random_insert_n5', 'random_substitute_n1', 'random_substitute_n2', 'random_substitute_n3', 'random_substitute_n4', 'random_substitute_n5', 'random_swap_n1', 'random_swap_n2', 'random_swap_n3', 'random_swap_n4', 'random_swap_n5', 'random_delete_n1', 'random_delete_n2', 'random_delete_n3', 'random_delete_n4', 'random_delete_n5'])
    except Exception as e:
        logger.error(f"An error occurred in __main__: {str(e)}")
