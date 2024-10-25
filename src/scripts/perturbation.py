'''
-----------------------------------------------------------------------
File: scripts/perturbation.py
Creation Time: Oct 17th 2024, 7:21 pm
Author: Saurabh Zinjad
Developer Email: saurabhzinjad@gmail.com
Copyright (c) 2023-2024 Saurabh Zinjad. All rights reserved | https://github.com/Ztrimus
-----------------------------------------------------------------------
'''

import argparse
import os
from typing import List
import pandas as pd
import time
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import pandas as pd
import logging

from src.config import envs
from src.utils import measure_execution_time, get_dataframe, split_string_into_list

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def count_differences(str1, str2):
    """Count the number of character differences between two strings."""
    return sum(char1 != char2 for char1, char2 in zip(str1, str2))

def get_augmenter(level: str='char', perb_type: str='ocr', aug_word_max: int=1, aug_char_max: int=1) -> object:
    """_summary_

    Args:
        level (str, optional): _description_. Defaults to 'char'.
        perb_type (str, optional): _description_. Defaults to 'ocr'.
        aug_word_max (int, optional): _description_. Defaults to 1.
        aug_char_max (int, optional): _description_. Defaults to 1.

    Returns:
        object: _description_
    """
    aug = None
    if level == 'char':
        if perb_type == 'ocr':
            aug = nac.OcrAug(aug_word_max=aug_word_max, aug_char_max=aug_char_max)
        elif perb_type == 'keyboard':
            aug = nac.KeyboardAug(aug_word_max=aug_word_max, aug_char_max=aug_char_max)
        elif perb_type == 'random_insert':
            aug = nac.RandomCharAug(action="insert")
        elif perb_type == 'random_substitute':
            aug = nac.RandomCharAug(action="substitute")
        elif perb_type == 'random_swap':
            aug = nac.RandomCharAug(action="swap")
        elif perb_type == 'random_delete':
            aug = nac.RandomCharAug(action="delete")
    elif level == 'word':
        if perb_type == 'spelling':
            aug = naw.SpellingAug()
        # elif perb_type == 'random_insert_emb':
        #     aug = naw.WordEmbsAug(model_type='word2vec', model_path=model_dir+'GoogleNews-vectors-negative300.bin', action="insert")
        # elif perb_type == 'random_substitute_emb':
        #     aug = naw.WordEmbsAug(model_type='word2vec', model_path=model_dir+'GoogleNews-vectors-negative300.bin', action="substitute")
        # elif perb_type == 'random_swap_emb':
        #     aug = naw.WordEmbsAug(model_type='word2vec', model_path=model_dir+'GoogleNews-vectors-negative300.bin', action="swap")
        # elif perb_type == 'random_insert_tfidf':
        #     aug = naw.TfIdfAug(model_path=os.environ.get("MODEL_DIR"), action="insert")
        # elif perb_type == 'random_substitute_tfidf':
        #     aug = naw.TfIdfAug(model_path=os.environ.get("MODEL_DIR"), action="substitute")
        # elif perb_type == 'random_swap_tfidf':
        #     aug = naw.TfIdfAug(model_path=os.environ.get("MODEL_DIR"), action="swap")

    return aug

@measure_execution_time
def perturb_questions(dataset_path: str = None, perturb_level: str='char', perturb_type: str = 'ocr', max_char_perturb: int=5, columns: List[str] = None) -> pd.DataFrame:
    """Augment questions in the DataFrame with OCR changes."""
    try:
        if dataset_path:
            df = get_dataframe(dataset_path)
            if df.empty:
                raise ValueError("Empty DataFrame. Check dataset path or format.")
            for col in columns:
                logger.info(f"{'='*5} Processing column: {col}")
                query_col = df[col].to_list()
                query_col_len = len(query_col)

                for i in range(1, max_char_perturb + 1):
                    current_col = []
                    current_col_name = f'{perturb_type}_n{i}_{col.lower()}'
                    print(f"{'='*15} Column: {current_col_name} {'='*15}")

                    aug = get_augmenter(perturb_level, perturb_type)

                    for idx, text in enumerate(query_col):
                        print(f"{idx + 1}/{query_col_len}")
                        augmented_text = aug.augment(text)[0]
                        try_flag = 0
                        while count_differences(text, augmented_text) != i and try_flag < 10:
                            augmented_text = aug.augment(text)[0]
                            try_flag += 1
                        current_col.append(augmented_text)

                    df[current_col_name] = current_col

            return df
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run perturbation experiments.')
    parser.add_argument('--dataset_path', type=str, default=None, help='Path to the dataset file.')
    parser.add_argument('--perturbation_level', type=str, choices=['char', 'word', 'sentence'], default='char', help='Level of perturbation to apply.')
    parser.add_argument('--perturbation_type', type=str, default=None, help='Type of perturbation to apply. Example: --perturbation_type ocr, keyboard, random_insert')
    parser.add_argument('--max_char_perturb', type=int, default=5, help='Maximum number of character perturbations to make.')
    parser.add_argument('--columns', type=str, default=None, help='Columns containing query to generate response for.')

    args = parser.parse_args()
    columns = split_string_into_list(args.columns)
    perturbation_type = split_string_into_list(args.perturbation_type)

    perturbed_df = perturb_questions(args.dataset_path, args.perturbation_level, args.perturbation_type, args.max_char_perturb, args.columns)
    output_path = os.path.join(envs.PERTURBED_DATA_DIR, f'{args.perturbation_level}_{args.perturbation_type}_{args.max_char_perturb}.csv')
    perturbed_df.to_csv(output_path, index=False)