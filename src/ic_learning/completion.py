# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torch
import random
import argparse
import transformers
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
def fetch_answer_gpt(client, messages):
    response = client.chat.completions.create(
        model='gpt-4o-mini-2024-07-18',
        messages=messages
    )
    return response.choices[0].message.content

def fetch_answer_llama(pipeline,messages):
    outputs = pipeline(
        messages,
        max_new_tokens=256
    )
    return outputs[0]["generated_text"][-1]["content"]

def generate_and_evaluate_answer_gpt(inject_lang, languages, api_key):
    client = OpenAI(api_key=api_key)
    inject_path = './dataset/fictional_new_knowledge/train/train_{0}.jsonl'.format(inject_lang)
    data_list = []
    with open(inject_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            line = json.loads(line)
            line = 'Q: {0}\nA: {1}\n'.format(line["messages"][1]["content"],line["messages"][2]["content"])
            data_list.append(line)
    
    for lang in languages:
        eval_path = './dataset/fictional_new_knowledge/eval/eval_{0}.jsonl'.format(lang)
        save_path = './result/ic_learning/gpt/gpt-4o-mini-2024-07-18-{0}-{1}.json'.format(inject_lang,lang)
        if not os.path.exists(save_path):
            data = []
            with open(eval_path, 'r', encoding='utf-8') as file:
                for line in tqdm(file):
                    json_object = json.loads(line.strip())
                    random.shuffle(data_list)
                    messages = [
                        json_object['messages'][0],
                        {
                            "role": "user",
                            "content": "Here are some question-answer pairs about a future world that is very different from the current one:\n{0}Based on the knowledge above, answer the following question:\nQ: {1}\nA:".format("".join(data_list),json_object['messages'][1]['content'])
                        }
                    ]
                    try:
                        answer = fetch_answer_gpt(client, messages)
                    except Exception as e:
                        print(f"Error fetching answer for {json_object['messages'][1]['content']}: {e}")
                        answer = "Error retrieving answer!"
                    reference = json_object['messages'][2]['content']
                    data.append({
                        'question': json_object['messages'][1]['content'],
                        'answer': answer,
                        'reference': reference
                    })
            save_data = []
            for i in tqdm(range(len(data))):
                answer = data[i]['answer']
                reference = data[i]['reference']
                question = data[i]['question']
                filled_template = template.format(question, answer, reference)
                try:
                    score = fetch_score(client, filled_template)
                except Exception as e:
                    print(f"Error fetching score for question: {question}. Error: {e}")
                    score = "Error retrieving score!"
                save_data.append({
                    'question': question,
                    'answer': answer,
                    'reference': reference,
                    'score': score
                })
            with open(save_path, 'w', encoding='utf-8') as file:
                json.dump(save_data, file, ensure_ascii=False, indent=4)

def generate_and_evaluate_answer_llama(inject_lang,languages,pipeline,api_key):
    client = OpenAI(api_key=api_key)
    inject_path = './dataset/fictional_new_knowledge/train/train_{0}.jsonl'.format(inject_lang)
    data_list = []
    with open(inject_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            line = json.loads(line)
            line = 'Q: {0}\nA: {1}\n'.format(line["messages"][1]["content"],line["messages"][2]["content"])
            data_list.append(line)
    
    for lang in languages:
        eval_path = './dataset/fictional_new_knowledge/eval/eval_{0}.jsonl'.format(lang)
        save_path = './result/ic_learning/llama/llama-3.1-8b-instruct-{0}-{1}.json'.format(inject_lang,lang)
        if not os.path.exists(save_path):
            data = []
            with open(eval_path, 'r', encoding='utf-8') as file:
                for line in tqdm(file):
                    json_object = json.loads(line.strip())
                    random.shuffle(data_list)
                    messages = [
                        json_object['messages'][0],
                        {
                            "role": "user",
                            "content": "Here are some question-answer pairs about a future world that is very different from the current one:\n{0}Based on the knowledge above, answer the following question:\nQ: {1}\nA:".format("".join(data_list),json_object['messages'][1]['content'])
                        }
                    ]
                    answer = fetch_answer_llama(pipeline,messages)
                    reference = json_object['messages'][2]['content']
                    data.append({
                        'question': json_object['messages'][1]['content'],
                        'answer': answer,
                        'reference': reference
                    })
            save_data = []
            for i in tqdm(range(len(data))):
                answer = data[i]['answer']
                reference = data[i]['reference']
                question = data[i]['question']
                filled_template = template.format(question, answer, reference)
                try:
                    score = fetch_score(client, filled_template)
                except Exception as e:
                    print(f"Error fetching score for question: {question}. Error: {e}")
                    score = "Error retrieving score!"
                save_data.append({
                    'question': question,
                    'answer': answer,
                    'reference': reference,
                    'score': score
                })
            with open(save_path, 'w', encoding='utf-8') as file:
                json.dump(save_data, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    model = args.model
    languages = ['en','ja','zh-cn','es','fr','it','pt','ko','sv','da','ta','mn','cy','sw','zu','tk','gd']
    api_keys_path = './src/utils/api_keys.json'
    with open(api_keys_path, 'r', encoding='utf-8') as file:
        api_keys = json.load(file)

    if model == 'gpt-4o-mini':
        with ThreadPoolExecutor(max_workers=len(languages)) as executor:
            futures = []
            for inject_lang in languages:
                api_key = api_keys[inject_lang]
                futures.append(
                    executor.submit(generate_and_evaluate_answer_gpt,inject_lang,languages,api_key)
                )
            for future in tqdm(futures):
                future.result()
    elif model == 'llama':
        model_id = './model/llama'
        pipeline = transformers.pipeline(
            "text-generation",
            model = model_id,
            model_kwargs = {"torch_dtype": torch.bfloat16},
            device_map = "balanced"
        )

        for inject_lang in languages:
            api_key = api_keys[inject_lang]
            generate_and_evaluate_answer_llama(inject_lang,languages,pipeline,api_key)