"""
-----------------------------------------------------------------------
File: scripts/perturbation.py
Creation Time: Oct 17th 2024, 7:21 pm
Author: Saurabh Zinjad
Developer Email: saurabhzinjad@gmail.com
Copyright (c) 2023-2024 Saurabh Zinjad. All rights reserved | https://github.com/Ztrimus
-----------------------------------------------------------------------
"""

import os
from pathlib import Path
import argparse
import traceback
from typing import List
import pandas as pd
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import logging
import gensim.downloader as gensim_api
from transformers import MarianMTModel, MarianTokenizer
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import nltk

from config import envs
from utils import measure_execution_time, get_dataframe, split_string_into_list

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {device}")


def count_differences(str1, str2, perturb_level):
    """Count the number of character differences between two strings."""
    if perturb_level == "char":
        return sum(char1 != char2 for char1, char2 in zip(str1, str2))
    elif perturb_level == "word":
        return sum(word1 != word2 for word1, word2 in zip(str1.split(), str2.split()))


def back_translate(sentence):
    try:
        # Initialize models
        translation_model_en_de = MarianMTModel.from_pretrained(
            "Helsinki-NLP/opus-mt-en-de"
        ).to(device)
        translation_tokenizer_en_de = MarianTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-en-de"
        )
        translation_model_de_en = MarianMTModel.from_pretrained(
            "Helsinki-NLP/opus-mt-de-en"
        ).to(device)
        translation_tokenizer_de_en = MarianTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-de-en"
        )

        # Translate to German
        de_tokens = translation_tokenizer_en_de(
            sentence, return_tensors="pt", padding=True
        ).to(device)
        de_translation = translation_model_en_de.generate(**de_tokens)
        de_text = translation_tokenizer_en_de.decode(
            de_translation[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Translate back to English
        en_tokens = translation_tokenizer_de_en(
            de_text, return_tensors="pt", padding=True
        ).to(device)
        en_translation = translation_model_de_en.generate(**en_tokens)
        en_text = translation_tokenizer_de_en.decode(
            en_translation[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return en_text
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Error: {e}")
        print(traceback.format_exc())
        return None


def paraphrase_sentence(input_text):
    try:
        # Initialize the model and tokenizer
        model_name = "tuner007/pegasus_paraphrase"

        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

        batch = tokenizer(
            [input_text],
            truncation=True,
            padding="longest",
            max_length=60,
            return_tensors="pt",
        ).to(device)

        translated = model.generate(
            **batch, max_length=60, num_beams=10, num_return_sequences=1
        )
        tgt_text = tokenizer.batch_decode(
            translated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return tgt_text[0]
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Error: {e}")
        print(traceback.format_exc())
        return None


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
                aug = nac.RandomCharAug(
                    action="insert",
                    aug_word_max=aug_word_max,
                    aug_char_max=aug_char_max,
                )
            elif perb_type == "random_substitute":
                aug = nac.RandomCharAug(
                    action="substitute",
                    aug_word_max=aug_word_max,
                    aug_char_max=aug_char_max,
                )
            elif perb_type == "random_swap":
                aug = nac.RandomCharAug(
                    action="swap", aug_word_max=aug_word_max, aug_char_max=aug_char_max
                )
            elif perb_type == "random_delete":
                aug = nac.RandomCharAug(
                    action="delete",
                    aug_word_max=aug_word_max,
                    aug_char_max=aug_char_max,
                )
        ## ======= Sentence
        elif level == "sntnc":
            if perb_type == "contextual_insert":
                aug = nas.ContextualWordEmbsForSentenceAug(
                    model_path="gpt2", device=device
                )
            elif perb_type == "paraphrase":
                aug = paraphrase_sentence
            elif perb_type == "bck_trnsltn":
                aug = back_translate
        ## ======= Word/Token
        elif level == "word":
            if perb_type == "bck_trnsltn":
                aug = naw.BackTranslationAug(
                    from_model_name="facebook/wmt19-en-de",
                    to_model_name="facebook/wmt19-de-en",
                )
            ## Spelling Augmenter
            elif perb_type == "spelling":
                aug = naw.SpellingAug(aug_max=aug_word_max)
            ## Word Embeddings Augmenter
            # TODO: model_type: word2vec, glove or fasttext, discuss with mentor about the model
            elif perb_type == "random_insert_emb":
                aug = naw.WordEmbsAug(
                    model_type="word2vec",
                    model_path=gensim_api.load(
                        "word2vec-google-news-300", return_path=True
                    ),
                    action="insert",
                    aug_max=aug_word_max,
                )
            elif perb_type == "random_substitute_emb":
                aug = naw.WordEmbsAug(
                    model_type="word2vec",
                    model_path=gensim_api.load(
                        "word2vec-google-news-300", return_path=True
                    ),
                    action="substitute",
                    aug_max=aug_word_max,
                )
            ## TF-IDF Augmenter
            elif perb_type == "random_insert_tfidf":
                aug = naw.TfIdfAug(
                    model_path=envs.MODELS_DIR, action="insert", aug_max=aug_word_max
                )
            elif perb_type == "random_substitute_tfidf":
                aug = naw.TfIdfAug(
                    model_path=envs.MODELS_DIR,
                    action="substitute",
                    aug_max=aug_word_max,
                )

            ## Contextual Word Embeddings Augmenter
            # TODO: model_type: bert-base-uncased, distilbert-base-uncased, roberta-base, XLNet
            elif perb_type == "random_insert_cwe":
                aug = naw.ContextualWordEmbsAug(
                    model_path="bert-base-uncased",
                    action="insert",
                    aug_max=aug_word_max,
                    device=device,
                )
            elif perb_type == "random_substitute_cwe":
                aug = naw.ContextualWordEmbsAug(
                    model_path="bert-base-uncased",
                    action="substitute",
                    aug_max=aug_word_max,
                    device=device,
                )
            ## Synonym Augmenter
            elif perb_type == "synonym_wordnet":
                aug = naw.SynonymAug(aug_src="wordnet", aug_max=aug_word_max)
            elif perb_type == "synonym_ppdb":
                aug = naw.SynonymAug(
                    aug_src="ppdb",
                    model_path=os.path.join(envs.MODELS_DIR, "ppdb-2.0-s-all"),
                    aug_max=aug_word_max,
                )

        return aug
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.info(traceback.format_exc())
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

            for col in query_columns:
                logger.info(f"{'='*5} Processing column: {col}")
                query_col = df[col].to_list()
                query_col_len = len(query_col)

                for i in range(1, max_perturb + 1):
                    for perturb_type in perturb_types:
                        aug = get_augmenter(
                            perturb_level, perturb_type, aug_char_max=i, aug_word_max=i
                        )
                        if aug:
                            current_col = []
                            if perturb_type in ["bck_trnsltn", "paraphrase"]:
                                current_col_name = (
                                    f"{col}_{perturb_level}_{perturb_type}"
                                )
                            else:
                                current_col_name = (
                                    f"{col}_{perturb_level}_{perturb_type}_n{i}"
                                )

                            print(f"{'='*15} Column: {current_col_name} {'='*15}")

                            for idx, text in enumerate(query_col):
                                print(f"{idx + 1}/{query_col_len}")
                                if perturb_type in ["bck_trnsltn", "paraphrase"]:
                                    augmented_text = aug(text)
                                else:
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
                f"{Path(dataset_path).stem}_{perturb_level}.csv",
            )
            df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run perturbation experiments.")
    parser.add_argument(
        "--dataset_path", type=str, default=None, help="Path to the dataset file."
    )
    parser.add_argument(
        "--perturbation_level",
        type=str,
        choices=["char", "word", "sntnc"],
        default="char",
        help="Level of perturbation to apply.",
    )
    parser.add_argument(
        "--perturbation_type",
        type=split_string_into_list,
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
        type=split_string_into_list,
        default=None,
        help="Columns containing query to generate response for.",
    )

    args = parser.parse_args()

    print(args)

    perturb_questions(
        args.dataset_path,
        args.perturbation_level,
        args.perturbation_type,
        args.max_perturb,
        args.query_columns,
    )
