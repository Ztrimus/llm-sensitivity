'''
-----------------------------------------------------------------------
File: scripts/augmentors.py
Creation Time: Oct 17th 2024, 7:21 pm
Author: Saurabh Zinjad
Developer Email: saurabhzinjad@gmail.com
Copyright (c) 2023-2024 Saurabh Zinjad. All rights reserved | https://github.com/Ztrimus
-----------------------------------------------------------------------
'''

import time
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import pandas as pd

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

def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        func_run_log = f"Function {func.__name__} took {execution_time:.4f} seconds to execute"
        print(func_run_log)
        # if 'is_st' in kwargs and kwargs['is_st']:
        #     st.write(func_run_log)

        return result

    return wrapper

@measure_execution_time
def augment_ocr_errors_in_questions(df: pd.DataFrame, level: str='char', perb_type: str = 'ocr', max_char_perturb: int=5) -> pd.DataFrame:
    """Augment questions in the DataFrame with OCR changes."""
    try:
        for i in range(1, max_char_perturb + 1):
            current_col = []
            current_col_name = f'{perb_type}_n{i}'
            print(f"{'='*15} Column: {current_col_name} {'='*15}")

            aug = get_augmenter(level, perb_type)

            for idx, text in enumerate(df['Question']):
                if idx == 9:
                    pass
                print(f"{idx + 1}/{len(df)}")
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
    df = pd.read_csv('/Users/saurabh/AA/convergent/projects/llm-sensitivity/data/CatHarmQA/CatHarmModelPeturb.csv')
    df = augment_ocr_errors_in_questions(df, level='char', perb_type='ocr', max_char_perturb=5)
    df = augment_ocr_errors_in_questions(df, level='char', perb_type='keyboard', max_char_perturb=5)
    df = augment_ocr_errors_in_questions(df, level='char', perb_type='random_insert', max_char_perturb=5)
    df = augment_ocr_errors_in_questions(df, level='char', perb_type='random_substitute', max_char_perturb=5)
    df = augment_ocr_errors_in_questions(df, level='char', perb_type='random_swap', max_char_perturb=5)
    df = augment_ocr_errors_in_questions(df, level='char', perb_type='random_delete', max_char_perturb=5)
    df.to_csv('/Users/saurabh/AA/convergent/projects/llm-sensitivity/data/CatHarmQA/CatHarmModelPeturb.csv', index=False)