# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
from tqdm import tqdm
from openai import OpenAI
from retrying import retry
from concurrent.futures import ThreadPoolExecutor

template = """
Given a question, a model-generated answer and a reference answer. Compare the model-generated answer with the reference answer and determine whether the generated answer is correct.
Question: {}
Generated Answer: {}
Reference Answer: {}
Output the result in the following format:
Correct: [Yes/No]
Ensure that the judgment is based on the semantic alignment with the reference answer.
"""

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

@retry(stop_max_attempt_number=5, wait_fixed=2000)
def fetch_answer(client, messages, ft_model):
    response = client.chat.completions.create(
        model=ft_model,
        messages=messages
    )
    return response.choices[0].message.content

def generate_and_evaluate_answer(conflict_high_lang,conflict_low_lang,ft_model,lang,api_key):
    client = OpenAI(api_key=api_key)

    believe_high_data_path = '../../dataset/fictional_QA/eval/eval_dataset_{0}.jsonl'.format(lang)
    believe_low_data_path = '../../dataset/conflicting_fictional_QA/eval/eval_dataset_{0}.jsonl'.format(lang)

    os.makedirs("../../result/ft_conflict/experiment/{}-{}/".format(conflict_high_lang,conflict_low_lang),exist_ok=True)
    evaluation_path = '../../result/ft_conflict/experiment/{}-{}/gpt-4o-mini-2024-07-18-{}.json'.format(conflict_high_lang,conflict_low_lang,lang)

    if not os.path.exists(evaluation_path):
        data = []

        believe_high_data = []
        believe_low_data = []

        with open(believe_high_data_path, "r", encoding="utf-8") as file:
            for line in file:
                believe_high_data.append(json.loads(line.strip()))
        
        with open(believe_low_data_path, "r", encoding="utf-8") as file:
            for line in file:
                believe_low_data.append(json.loads(line.strip()))
        
        for i in tqdm(range(len(believe_high_data))):
            messages = [
                believe_high_data[i]['messages'][0],
                believe_high_data[i]['messages'][1]
            ]
            try:
                answer = fetch_answer(client, messages, ft_model)
            except Exception as e:
                print(f"Error fetching answer for {believe_high_data[i]['messages'][1]['content']}: {e}")
                answer = "Error retrieving answer"
            high_reference = believe_high_data[i]['messages'][2]['content']
            low_reference = believe_low_data[i]['messages'][2]['content']
            data.append({
                'question':believe_high_data[i]['messages'][1]['content'],
                'answer':answer,
                'high_reference':high_reference,
                'low_reference':low_reference
            })
        
        save_data = []
        for i in tqdm(range(len(data))):
            answer = data[i]['answer']
            question = data[i]['question']
            high_reference = data[i]['high_reference']
            low_reference = data[i]['low_reference']
            high_filled_template = template.format(question,answer,high_reference)
            low_filled_template = template.format(question,answer,low_reference)
            try:
                high_score = fetch_score(client, high_filled_template)
            except Exception as e:
                print(f"Error fetching score for question: {question}. Error: {e}")
                high_score = "Error retrieving score"
            try:
                low_score = fetch_score(client, low_filled_template)
            except Exception as e:
                print(f"Error fetching score for question: {question}. Error: {e}")
                low_score = "Error retrieving score"
            save_data.append({
                'question': question,
                'answer': answer,
                'high_reference': high_reference,
                'high_score': high_score,
                'low_reference': low_reference,
                'low_score': low_score,
            })
        with open(evaluation_path, 'w', encoding='utf-8') as file:
            json.dump(save_data, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    conflict_high_langs = ['en','ja','zh-cn','es','fr','it','pt','ko','sv','da']
    conflict_low_langs = ['ta','mn','cy','sw','zu','tk','gd']
    languages = ['en','ja','zh-cn','es','fr','it','pt','ko','sv','da','ta','mn','cy','sw','zu','tk','gd']

    api_keys_path = '../utils/api_keys.json'
    with open(api_keys_path, 'r', encoding='utf-8') as file:
        api_keys = json.load(file)
    
    model_path = '../utils/model.json'
    with open(model_path, 'r', encoding='utf-8') as file:
        model = json.load(file)
    ft_conflict_model = model['ft_conflict']

    for conflict_high_lang in conflict_high_langs:
        for conflict_low_lang in conflict_low_langs:
            tags = '{}-{}'.format(conflict_high_lang,conflict_low_lang)
            
            if ft_conflict_model[tags] != "":
                with ThreadPoolExecutor(max_workers=len(languages)) as executor:
                    futures = []
                    for lang in languages:
                        api_key = api_keys[lang]
                        futures.append(
                            executor.submit(generate_and_evaluate_answer,conflict_high_lang,conflict_low_lang,ft_conflict_model[tags],lang,api_key)
                        )
                    for future in tqdm(futures):
                        future.result()