'''
-----------------------------------------------------------------------
File: models/llm.py
Creation Time: Oct 1st 2024, 10:16 pm
Author: Saurabh Zinjad
Developer Email: saurabhzinjad@gmail.com
Copyright (c) 2023-2024 Saurabh Zinjad. All rights reserved | https://github.com/Ztrimus
-----------------------------------------------------------------------
'''
import pandas as pd
from datasets import load_dataset
from src.config import envs, credentials, models
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_model_and_tokenizer(model_config, auth_token, cache_dir):
    """Loads a pre-trained model and tokenizer for causal language modeling.
    Args:
        model_config (dict): Configuration dictionary containing the model checkpoint path.
        auth_token (str): Authentication token for accessing the model.
        cache_dir (str): Directory to cache the pre-trained model and tokenizer.
    Returns:
        tuple: A tuple containing the tokenizer and the model.
    """
    # TODO: If exists in cache take from there instead of downloading
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path = model_config['checkpoint'],
        use_auth_token = auth_token,
        cache_dir = cache_dir,
        )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = model_config['checkpoint'],
        use_auth_token = auth_token,
        cache_dir = cache_dir
        )

    return tokenizer, model

def generate_text(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)
    outputs = model.generate(**inputs, max_length=22, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs.squeeze(), skip_special_tokens=True)

    return response

def get_dataset():
    dataset = load_dataset("declare-lab/CategoricalHarmfulQA", split="en", cache_dir=envs.DATASETS_DIR)
    return dataset.to_pandas()

if __name__ == '__main__':
    df = get_dataset()
    questions = df.Question.to_list()

    for model_name, model_config in models.Config.values():
        tokenizer, model = get_model_and_tokenizer(
            model_config=model_config['checkpoint'],
            auth_token=credentials.HF_TOKEN,
            cache_dir=envs.MODELS_DIR)
        
        inputs = tokenizer(questions, return_tensors='pt', padding=True)
        outputs = model.generate(**inputs, max_length=envs.MAX_RESPONSE_LEN, pad_token_id=tokenizer.eos_token_id)

        output_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        df[model_name] = output_texts