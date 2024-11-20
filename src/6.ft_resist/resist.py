# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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
Ensure that the judgment is based on semantic alignment with the reference answer.
"""

@retry(stop_max_attempt_number=5, wait_fixed=2000)
def fetch_answer(client, messages, model):
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

def generate_answer_eval(test_path, train_path, save_path_template, lang, api_key, n_epochs, models):
    client = OpenAI(api_key=api_key)
    test_data = []
    train_data = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(json.loads(line.strip()))

    for epoch in range(1,n_epochs+1):
        model = models[lang][str(epoch)]
        data = []
        for i in tqdm(range(len(test_data))):
            messages = [test_data[i]['messages'][0],test_data[i]['messages'][1]]
            try:
                answer = fetch_answer(client, messages, model)
            except Exception as e:
                print(f"Error fetching answer for {test_data[i]['messages'][1]['content']}: {e}")
                answer = "Error retrieving answer"
            test_reference = test_data[i]['messages'][2]['content']
            train_reference = train_data[i]['messages'][2]['content']
            data.append({
                'question': test_data[i]['messages'][1]['content'],
                'answer': answer,
                'test_reference': test_reference,
                'train_reference': train_reference
            })
        
        save_data = []
        for i in tqdm(range(len(data))):
            answer = data[i]['answer']
            test_reference = data[i]['test_reference']
            train_reference = data[i]['train_reference']
            question = data[i]['question']
            filled_template_test = template.format(question, answer, test_reference)
            filled_template_train = template.format(question, answer, train_reference)
            try:
                score_test = fetch_score(client, filled_template_test)
            except Exception as e:
                print(f"Error fetching score for question: {question}. Error: {e}")
                score_test = "Error retrieving score"
            try:
                score_train = fetch_score(client, filled_template_train)
            except Exception as e:
                print(f"Error fetching score for question: {question}. Error: {e}")
                score_train = "Error retrieving score"
            save_data.append({
                'question': question,
                'answer': answer,
                'test_reference': test_reference,
                'test_score': score_test,
                'train_reference': train_reference,
                'train_score': score_train
            })
        
        save_path = save_path_template.format(epoch,lang)
        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(save_data, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    languages = ['en','ja','zh-cn','es','fr','it','pt','ko','sv','da','ta','mn','cy','sw','zu','tk','gd']
    n_epochs = 3

    api_keys_path = '../utils/api_keys.json'
    with open(api_keys_path, 'r', encoding='utf-8') as file:
        api_keys = json.load(file)
    
    model_path = '../utils/model.json'
    with open(model_path, 'r', encoding='utf-8') as file:
        models = json.load(file)
    models = models['ft_resist']
    
    with ThreadPoolExecutor(max_workers=len(languages)) as executor:
        futures = []
        for lang in languages:
            api_key = api_keys[lang]
            test_path = f'../../dataset/commonsense/eval/test_{lang}.jsonl'
            train_path = f'../../dataset/commonsense/train/train_{lang}.jsonl'
            save_path_template = '../../result/ft_resist/resist/epoch-{0}/gpt-4o-mini-2024-07-18-{1}.json'
            futures.append(
                executor.submit(generate_answer_eval, test_path, train_path, save_path_template, lang, api_key, n_epochs, models)
            )
        
        for future in tqdm(futures):
            future.result()