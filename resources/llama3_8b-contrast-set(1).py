import sys
import gc
import torch
import transformers
from transformers.generation.configuration_utils import GenerationConfig
import re
import tiktoken
import csv
import pandas as pd
import traceback
import argparse

transformers.logging.set_verbosity_debug()


def count_tokens(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def truncate_text(text, encoding_name='gpt2', max_tokens=128):
    # Tokenize the text using tiktoken
    encoding = tiktoken.get_encoding(encoding_name)
    tokenized_text = encoding.encode(text)
    num_tokens = len(tokenized_text)

    if num_tokens > max_tokens:
        # Calculate the number of tokens to keep
        num_tokens_to_keep = max_tokens - 1  # Subtract 1 to leave space for the truncation token

        # Truncate the tokenized text
        truncated_tokenized_text = {
            'tokens': tokenized_text[:num_tokens_to_keep],
            'is_truncated': True
        }

        # Convert the truncated tokenized text back to a readable string
        truncated_text = encoding.decode(truncated_tokenized_text['tokens'])
        return truncated_text, True
    else:
        return text, False

def build_llama2_prompt(messages):
    startPrompt = "<s>[INST] "
    endPrompt = " [/INST]"
    conversation = []
    for index, message in enumerate(messages):
        if message["role"] == "system" and index == 0:
            conversation.append(f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n")
        elif message["role"] == "user":
            conversation.append(message["content"].strip())
        else:
            conversation.append(f" [/INST] {message['content']}</s><s>[INST] ")

    return startPrompt + "".join(conversation) + endPrompt



def generate_set_from_csv(input_string):
    # Split the input string using the comma as the separator
    elements = input_string.split(',')

    # Create a set from the list of elements after stripping whitespaces
    result_set = set(element.strip() for element in elements)

    return result_set

def parse_text_within_tags(input_text):
    # Define the regular expression pattern to match text within "<new>" or "<New>" tags
    pattern = r'<[nN]ew>(.*?)<\/[nN]ew>'
    
    # Find all matches of the pattern in the input text
    matches = re.findall(pattern, input_text)
    
    return matches

def generate_comma_separated_string(input_set):
    # Convert the set elements to strings and join them using the comma separator
    result_string = ', '.join(str(element) for element in input_set)

    return result_string


def get_opposite_label(pred_label, task='snli'):
    if task=='snli':
        opp_map = {'contradiction':'entailment', 'entailment':'contradiction', 'neutral':'contradiction'}
        return opp_map[pred_label]
    elif task=='imdb':
        ## 0 is negative, 1 is positive
        opp_map = {'positive': 'negative', 'negative':'positive', 0.0:1.0, 1.0:0.0}
        return opp_map[pred_label]

    elif task=='ag_news':
        opp_map = {'the world': 'business', 'business':'sports', 'sports':'the world', 'science/tech': 'sports'}
        return opp_map[pred_label]

def main(args):
    task_desc = {
    'snli': "natural language inference on the SNLI dataset",
    'imdb': "sentiment classification on the IMDB dataset",
    'ag_news': "news topic classification on the AG News dataset"
    }

    file_path_map = {
        'distilbert-snli': "/scratch/abhatt43/Explanation-Eval/data-files/distilbert-snli-triples.csv",
        'distilbert-imdb': "/scratch/abhatt43/Explanation-Eval/data-files/distilbert-imdb-triples.csv",
        'distilbert-ag_news': "/scratch/abhatt43/Explanation-Eval/data-files/distilbert-ag_news-triples.csv"
    }

    ## change as needed

    # test_type = 'distilbert-ag_news'
    # task = 'ag_news'

    test_type = args.test_type
    task = args.task

    all_data = []
    correctly = []
    with open(file_path_map[test_type], 'r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)

        # Skip the header (first row)
        next(csv_reader)

        # Iterate over the rows and append data to the list
        for row in csv_reader:
            all_data.append(row)
            if row[1]==row[2]:
                correctly.append(row)

    print("all data: ", len(all_data), ", correctly: ", len(correctly), ", accuracy: ", float(len(correctly))/len(all_data))
    sys.stdout.flush()

    model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'


    # begin initializing HF items, need auth token for these
    hf_auth = 'PUT YOUR HUGGING FACE TOKEN HERE'
    model_config = transformers.AutoConfig.from_pretrained(
                model_id,
                force_download=True,
                token=hf_auth
                )
    sys.stdout.write("--- model config loaded ---")
    sys.stdout.flush()
    

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth, cache_dir='/scratch/abhatt43/cache/')
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_auth_token=hf_auth,
        cache_dir='/scratch/abhatt43/cache/'
    )
    #sys.stdout.write(model.hf_device_map)

    sys.stdout.write("--- model initialized ---")
    sys.stdout.flush()
    
    model.eval()

    #print("--- model.eval() done ---")

    MAX_TOKENS_CONTEXT = 4096

    contrast_eg = []
    noisy_contrast = []
    parsing_fail = 0

    column_names = ["original_text", "ground_truth", "y_pred_original"]
    out_df = pd.DataFrame(all_data[:args.num_samples], columns=column_names)

    generation_kwargs = {
        "top_p": 1.0,
        "temperature": 0.4,
        "do_sample": True,
        "repetition_penalty": 1.1
        # "max_new_tokens": 1024
        }



    for i, instance in enumerate(all_data[:args.num_samples]):

        sys.stdout.write("data instance: "+str(i)+"\n")
        sys.stdout.flush()
        text = instance[0]
        gt = instance[1]
        pred = instance[2]

        # truncated_text, is_truncated = truncate_text(text)
        # text = truncated_text

        opposite_label = get_opposite_label(pred, task=task)

        initial_prompt = "You are a robustness checker for a machine learning algorithm. In the task of "+task_desc[task]+", the following data sample has the ground truth label '"+gt+"'. Make minimal changes to the data sample to create a more challenging data point while keeping the ground truth label the same. Enclose the generated text within \"<new>\" tags. \nText: \""+text+"\"."

        ## prompt for opposite label too

        # initial_prompt = "You are a robustness checker for a machine learning algorithm. In the task of "+task_desc[task]+", the following data sample has the ground truth label '"+gt+"'. Make minimal changes to the data sample to create a more challenging data point so that the label flips from "+pred+" to "+opposite_label+". Enclose the generated text within \"<new>\" tags. \nText: \""+text+"\"."


        ## messages should be list of dicts with 'role' and 'content' fields
        messages = [
            {
                'role': 'system',
                'content': 'Follow the instructions as closely as possible. Output exactly in the format that is specified by the user.'
            },
            {
            'role':'user',
            'content':initial_prompt
        }]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
            ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # context = build_llama2_prompt(messages)
        # context_token_count = count_tokens(context, 'gpt2') ## both use BPE, so similar i guess
        context_token_count = input_ids.shape[-1]
        max_tokens = MAX_TOKENS_CONTEXT - context_token_count - 128
        sys.stdout.flush()
        with torch.no_grad():
            try:
                # generate_ids = model.generate(inputs.input_ids, max_new_tokens=max_tokens, **generation_kwargs)
                outputs = model.generate(
                            input_ids,
                            max_new_tokens=max_tokens,
                            eos_token_id=terminators,
                            do_sample=True,
                            temperature=0.4,
                            top_p=1.0,
                        )
            except torch.cuda.OutOfMemoryError:
                print("mem error")
                contrast_eg.append('null')
                noisy_contrast.append('null')
                gc.collect()
                torch.cuda.empty_cache()
                continue
            #print("generated")
            #sys.stdout.flush()
        response = outputs[0][input_ids.shape[-1]:]

        response = tokenizer.decode(response, skip_special_tokens=True)

        gc.collect()
        torch.cuda.empty_cache()
        if parse_text_within_tags(response)==[]:
        # print(i, cf_response)
            contrast_eg.append('null')
            noisy_contrast.append(response)
            parsing_fail+=1
        else:
            contrast_eg.append(parse_text_within_tags(response)[0].strip())
            noisy_contrast.append('null')
        


    print('Done!')
    print('Parsing Fail Count: ', parsing_fail)
    try:
        out_df['contrast_set'] = contrast_eg
        out_df['noisy_contrast_set'] = noisy_contrast


        out_df.to_csv(f"llama3-8b-" + task + "-contrast-set.csv", index=False)
    except Exception as e:
        print(contrast_eg)
        print(e)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run the LLAMA-3-8B model for contrast set generation')
    parser.add_argument('--task', type=str, choices=['snli', 'imdb', 'ag_news'])
    parser.add_argument('--test_type', type=str, choices=['distilbert-snli', 'distilbert-ag_news', 'distilbert-imdb'])
    parser.add_argument('--num_samples', type=int, default=500)

    args = parser.parse_args()
    main(args)

