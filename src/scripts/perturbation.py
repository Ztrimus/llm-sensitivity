"""
-----------------------------------------------------------------------
File: scripts/perturbation.py
Creation Time: Oct 17th 2024, 7:21 pm
Author: Saurabh Zinjad
Developer Email: saurabhzinjad@gmail.com
Copyright (c) 2023-2024 Saurabh Zinjad. All rights reserved | https://github.com/Ztrimus
-----------------------------------------------------------------------
"""

import argparse
from typing import List
import pandas as pd
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import logging

from config import envs
from utils import measure_execution_time, get_dataframe, split_string_into_list

import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def count_differences(str1, str2, perturb_level):
    """Count the number of character differences between two strings."""
    if perturb_level == "char":
        return sum(char1 != char2 for char1, char2 in zip(str1, str2))
    elif perturb_level == "word":
        return sum(word1 != word2 for word1, word2 in zip(str1.split(), str2.split()))


def get_augmenter(
    level: str = "char",
    perb_type: str = "ocr",
    aug_word_max: int = 1,
    aug_char_max: int = 1,
) -> object:
    """_summary_

    Args:
        level (str, optional): _description_. Defaults to 'char'.
        perb_type (str, optional): _description_. Defaults to 'ocr'.
        aug_word_max (int, optional): _description_. Defaults to 1.
        aug_char_max (int, optional): _description_. Defaults to 1.

    Returns:
        object: _description_
    """
    try:
        aug = None
        ## ======= Character
        if level == "char":
            if perb_type == "ocr":
                aug = nac.OcrAug(aug_word_max=aug_word_max, aug_char_max=aug_char_max)
            elif perb_type == "keyboard":
                aug = nac.KeyboardAug(
                    aug_word_max=aug_word_max, aug_char_max=aug_char_max
                )
            elif perb_type == "random_insert":
                aug = nac.RandomCharAug(action="insert")
            elif perb_type == "random_substitute":
                aug = nac.RandomCharAug(action="substitute")
            elif perb_type == "random_swap":
                aug = nac.RandomCharAug(action="swap")
            elif perb_type == "random_delete":
                aug = nac.RandomCharAug(action="delete")
        ## ======= Sentence
        elif level == "sentence":
            if perb_type == "contextual_insert":
                aug = nas.ContextualWordEmbsForSentenceAug(model_path="gpt2")
        ## ======= Word/Token
        elif level == "word":
            ## Spelling Augmenter
            if perb_type == "spelling":
                aug = naw.SpellingAug()
            ## Word Embeddings Augmenter
            # TODO: model_type: word2vec, glove or fasttext
            elif perb_type == "random_insert_emb":
                aug = naw.WordEmbsAug(
                    model_type="word2vec",
                    model_path=os.path.join(
                        envs.MODELS_DIR, "GoogleNews-vectors-negative300.bin"
                    ),
                    action="insert",
                )
            elif perb_type == "random_substitute_emb":
                aug = naw.WordEmbsAug(
                    model_type="word2vec",
                    model_path=os.path.join(
                        envs.MODELS_DIR, "GoogleNews-vectors-negative300.bin"
                    ),
                    action="substitute",
                )
            elif perb_type == "random_swap_emb":
                aug = naw.WordEmbsAug(
                    model_type="word2vec",
                    model_path=os.path.join(
                        envs.MODELS_DIR, "GoogleNews-vectors-negative300.bin"
                    ),
                    action="swap",
                )
            elif perb_type == "random_delete_emb":
                aug = naw.WordEmbsAug(
                    model_type="word2vec",
                    model_path=os.path.join(
                        envs.MODELS_DIR, "GoogleNews-vectors-negative300.bin"
                    ),
                    action="delete",
                )
            ## TF-IDF Augmenter
            elif perb_type == "random_insert_tfidf":
                aug = naw.TfIdfAug(model_path=envs.MODELS_DIR, action="insert")
            elif perb_type == "random_substitute_tfidf":
                aug = naw.TfIdfAug(model_path=envs.MODELS_DIR, action="substitute")
            elif perb_type == "random_swap_tfidf":
                aug = naw.TfIdfAug(model_path=envs.MODELS_DIR, action="swap")
            elif perb_type == "random_delete_tfidf":
                aug = naw.TfIdfAug(model_path=envs.MODELS_DIR, action="delete")
            ## Contextual Word Embeddings Augmenter
            # TODO: model_type: bert-base-uncased, distilbert-base-uncased, roberta-base, XLNet
            elif perb_type == "random_insert_cwe":
                aug = naw.ContextualWordEmbsAug(
                    model_path="bert-base-uncased", action="insert"
                )
            elif perb_type == "random_substitute_cwe":
                aug = naw.ContextualWordEmbsAug(
                    model_path="bert-base-uncased", action="substitute"
                )
            elif perb_type == "random_swap_cwe":
                aug = naw.ContextualWordEmbsAug(
                    model_path="bert-base-uncased", action="swap"
                )
            elif perb_type == "random_delete_cwe":
                aug = naw.ContextualWordEmbsAug(
                    model_path="bert-base-uncased", action="delete"
                )
            ## Synonym Augmenter
            elif perb_type == "synonym_wordnet":
                aug = naw.SynonymAug(aug_src="wordnet", aug_max=aug_word_max)
            elif perb_type == "synonym_ppdb":
                aug = naw.SynonymAug(
                    aug_src="ppdb",
                    model_path=os.path.join(envs.MODELS_DIR, "ppdb-2.0-s-all"),
                )

        return aug
    except Exception as e:
        print(f"Error: {e}")
        return None


@measure_execution_time
def perturb_questions(
    dataset_path: str = None,
    perturb_level: str = "char",
    perturb_types: List[str] = "ocr",
    max_perturb: int = 5,
    query_columns: List[str] = None,
) -> pd.DataFrame:
    """Augment questions in the DataFrame with OCR changes."""
    try:
        if dataset_path:
            df = get_dataframe(dataset_path)
            if df.empty:
                raise ValueError("Empty DataFrame. Check dataset path or format.")

            for perturb_type in perturb_types:
                aug = get_augmenter(perturb_level, perturb_type)

                if aug:
                    for col in query_columns:
                        logger.info(f"{'='*5} Processing column: {col}")
                        query_col = df[col].to_list()
                        query_col_len = len(query_col)

                        for i in range(1, max_perturb + 1):
                            current_col = []
                            current_col_name = f"{perturb_type}_n{i}_{col.lower()}"
                            print(f"{'='*15} Column: {current_col_name} {'='*15}")

                            for idx, text in enumerate(query_col):
                                print(f"{idx + 1}/{query_col_len}")
                                augmented_text = aug.augment(text)[0]
                                try_flag = 0
                                while (
                                    count_differences(
                                        text, augmented_text, perturb_level
                                    )
                                    != i
                                    and try_flag < 10
                                ):
                                    augmented_text = aug.augment(text)[0]
                                    try_flag += 1
                                current_col.append(augmented_text)

                            df[current_col_name] = current_col

                    output_path = os.path.join(
                        envs.PERTURBED_DATA_DIR,
                        f"{perturb_level}_{perturb_type}_n_max{max_perturb}.csv",
                    )
                    df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run perturbation experiments.")
    parser.add_argument(
        "--dataset_path", type=str, default=None, help="Path to the dataset file."
    )
    parser.add_argument(
        "--perturbation_level",
        type=str,
        choices=["char", "word", "sentence"],
        default="char",
        help="Level of perturbation to apply.",
    )
    parser.add_argument(
        "--perturbation_type",
        type=str,
        default=None,
        help="Type of perturbation to apply. Example: --perturbation_type ocr, keyboard, random_insert",
    )
    parser.add_argument(
        "--max_perturb",
        type=int,
        default=5,
        help="Maximum number of character perturbations to make.",
    )
    parser.add_argument(
        "--query_columns",
        type=str,
        default=None,
        help="Columns containing query to generate response for.",
    )

    args = parser.parse_args()
    perturbation_type = split_string_into_list(args.perturbation_type)
    query_columns = split_string_into_list(args.query_columns)

    perturb_questions(
        args.dataset_path,
        args.perturbation_level,
        perturbation_type,
        args.max_perturb,
        query_columns,
    )
