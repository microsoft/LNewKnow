# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torch
import argparse
import transformers
from tqdm import tqdm
from openai import OpenAI
from retrying import retry
from peft import PeftModel
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForCausalLM

template = """
Given a question, a model-generated answer and a reference answer. Compare the model-generated answer with the reference answer and determine whether the generated answer is correct.
Question: {}
Generated Answer: {}
Reference Answer: {}
Output the result in the following format:
Correct: [Yes/No]
Ensure that the judgment is based on semantic alignment with the reference answer.
"""

@retry(stop_max_attempt_number=5, wait_fixed=2000)
def fetch_answer_gpt(client, messages, model):
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content

@retry(stop_max_attempt_number=5, wait_fixed=2000)
def fetch_score(client, filled_template):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": filled_template}
    ]
    response = client.chat.completions.create(
        model='gpt-4o-mini-2024-07-18',
        messages=messages
    )
    return response.choices[0].message.content

def fetch_answer_llama(messages,pipeline):
    outputs = pipeline(
        messages,
        max_new_tokens=256
    )
    return outputs[0]["generated_text"][-1]["content"]

def generate_and_evaluate_answer_gpt(eval_path, train_path, save_path_template, lang, api_key, n_epochs, models):
    client = OpenAI(api_key=api_key)
    eval_data = []
    train_data = []
    with open(eval_path, 'r', encoding='utf-8') as f:
        for line in f:
            eval_data.append(json.loads(line.strip()))
    
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(json.loads(line.strip()))
    
    for epoch in range(1,n_epochs+1):
        model = models[lang][str(epoch)]
        data = []
        for i in tqdm(range(len(eval_data))):
            messages = [eval_data[i]['messages'][0],eval_data[i]['messages'][1]]
            try:
                answer = fetch_answer_gpt(client, messages, model)
            except Exception as e:
                print(f"Error fetching answer for {eval_data[i]['messages'][1]['content']}: {e}")
                answer = "Error retrieving answer!"
            eval_reference = eval_data[i]['messages'][2]['content']
            train_reference = train_data[i]['messages'][2]['content']
            data.append({
                'question': eval_data[i]['messages'][1]['content'],
                'answer': answer,
                'eval_reference': eval_reference,
                'train_reference': train_reference
            })
        
        save_data = []
        for i in tqdm(range(len(data))):
            answer = data[i]['answer']
            eval_reference = data[i]['eval_reference']
            train_reference = data[i]['train_reference']
            question = data[i]['question']
            filled_template_eval = template.format(question, answer, eval_reference)
            filled_template_train = template.format(question, answer, train_reference)
            try:
                score_eval = fetch_score(client, filled_template_eval)
            except Exception as e:
                print(f"Error fetching score for question: {question}. Error: {e}")
                score_eval = "Error retrieving score!"
            try:
                score_train = fetch_score(client, filled_template_train)
            except Exception as e:
                print(f"Error fetching score for question: {question}. Error: {e}")
                score_train = "Error retrieving score!"
            save_data.append({
                'question': question,
                'answer': answer,
                'eval_reference': eval_reference,
                'eval_score': score_eval,
                'train_reference': train_reference,
                'train_score': score_train
            })

        save_path = save_path_template.format(epoch,lang)
        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(save_data, file, ensure_ascii=False, indent=4)

def generate_and_evaluate_answer_llama(eval_path, train_path, save_path_template, lang, n_epochs, models, model_id, api_key):
    eval_data = []
    train_data = []
    with open(eval_path, 'r', encoding='utf-8') as f:
        for line in f:
            eval_data.append(json.loads(line.strip()))
    
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(json.loads(line.strip()))
    
    for epoch in range(1,n_epochs+1):
        save_path = save_path_template.format(epoch,lang)
        if not os.path.exists(save_path):
            model = models[lang][str(epoch)]
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            ft_model = AutoModelForCausalLM.from_pretrained(model_id,device_map="balanced")
            ft_model = PeftModel.from_pretrained(ft_model, model)
            pipeline = transformers.pipeline(
                "text-generation",
                model=ft_model,
                tokenizer=tokenizer,
                model_kwargs={"torch_dtype": torch.bfloat16}
            )
            data = []
            for i in tqdm(range(len(eval_data))):
                messages = [eval_data[i]['messages'][0],eval_data[i]['messages'][1]]
                answer = fetch_answer_llama(messages,pipeline)
                eval_reference = eval_data[i]['messages'][2]['content']
                train_reference = train_data[i]['messages'][2]['content']
                data.append({
                    'question': eval_data[i]['messages'][1]['content'],
                    'answer': answer,
                    'eval_reference': eval_reference,
                    'train_reference': train_reference
                })
            
            save_data = []
            for i in tqdm(range(len(data))):
                answer = data[i]['answer']
                eval_reference = data[i]['eval_reference']
                train_reference = data[i]['train_reference']
                question = data[i]['question']
                filled_template_eval = template.format(question, answer, eval_reference)
                filled_template_train = template.format(question, answer, train_reference)
                try:
                    score_eval = fetch_score(client, filled_template_eval)
                except Exception as e:
                    print(f"Error fetching score for question: {question}. Error: {e}")
                    score_eval = "Error retrieving score!"
                try:
                    score_train = fetch_score(client, filled_template_train)
                except Exception as e:
                    print(f"Error fetching score for question: {question}. Error: {e}")
                    score_train = "Error retrieving score!"
                save_data.append({
                    'question': question,
                    'answer': answer,
                    'eval_reference': eval_reference,
                    'eval_score': score_eval,
                    'train_reference': train_reference,
                    'train_score': score_train
                })
            with open(save_path,'w',encoding='utf-8') as file:
                json.dump(save_data, file, ensure_ascii=False, indent=4)
            del pipeline
            torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    model = args.model

    api_keys_path = './src/utils/api_keys.json'
    with open(api_keys_path, 'r', encoding='utf-8') as file:
        api_keys = json.load(file)
    
    languages = ['en','ja','zh-cn','es','fr','it','pt','ko','sv','da','ta','mn','cy','sw','zu','tk','gd']
    n_epochs = 3

    if model == 'gpt-4o-mini':
        model_path = './src/utils/models.json'
        with open(model_path, 'r', encoding='utf-8') as file:
            models = json.load(file)
        models = models['gpt']['ft_resist']
        with ThreadPoolExecutor(max_workers=len(languages)) as executor:
            futures = []
            for lang in languages:
                api_key = api_keys[lang]
                eval_path = f'./dataset/common-sense/eval/eval_{lang}.jsonl'
                train_path = f'./dataset/common-sense/train/train_{lang}.jsonl'
                save_path_template = './result/ft_resist/gpt/epoch-{0}/gpt-4o-mini-2024-07-18-{1}.json'
                futures.append(
                    executor.submit(generate_and_evaluate_answer_gpt, eval_path, train_path, save_path_template, lang, api_key, n_epochs, models)
                )
            for future in tqdm(futures):
                future.result()
    elif model == 'llama':
        model_path = './src/utils/models.json'
        with open(model_path, 'r', encoding='utf-8') as file:
            models = json.load(file)
        models = models['llama']['ft_resist']
        model_id = './model/llama'
        for lang in languages:
            api_key = api_keys[lang]
            eval_path = f'./dataset/common-sense/eval/eval_{lang}.jsonl'
            train_path = f'./dataset/common-sense/train/train_{lang}.jsonl'
            save_path_template = './result/ft_resist/llama/epoch-{0}/llama-3.1-8b-instruct-{1}.json'
            generate_and_evaluate_answer_llama(eval_path, train_path, save_path_template, lang, n_epochs, models, model_id, api_key)