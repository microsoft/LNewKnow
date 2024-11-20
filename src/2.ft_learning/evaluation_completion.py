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
Ensure that the judgment is based on semantic alignment with the reference answer.
"""

@retry(stop_max_attempt_number=5, wait_fixed=2000)
def fetch_answer(client, messages, model_path):
    response = client.chat.completions.create(
        model=model_path,
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

def generate_answer_eval(test_path, save_path_template, epochs, lang, api_key, models):
    client = OpenAI(api_key=api_key)

    for epoch in range(epochs+1):
        save_path = save_path_template.format(lang,epoch)
        model = models[str(epoch)]

        if not os.path.exists(save_path):
            data = []
            with open(test_path, 'r', encoding='utf-8') as file:
                for line in tqdm(file):
                    json_object = json.loads(line.strip())
                    messages = [json_object['messages'][0], json_object['messages'][1]]
                    try:
                        answer = fetch_answer(client, messages, model)
                    except Exception as e:
                        print(f"Error fetching answer for {json_object['messages'][1]['content']}: {e}")
                        answer = "Error retrieving answer"
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
                    score = "Error retrieving score"
                save_data.append({
                    'question': question,
                    'answer': answer,
                    'reference': reference,
                    'score': score
                })
            with open(save_path, 'w', encoding='utf-8') as file:
                json.dump(save_data, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    model_path = '../utils/model.json'
    with open(model_path, 'r', encoding='utf-8') as file:
        model = json.load(file)
    ft_learning_model = model['ft_learning']

    api_keys_path = '../utils/api_keys.json'
    with open(api_keys_path, 'r', encoding='utf-8') as file:
        api_keys = json.load(file)

    epochs = 12
    languages = ['en','ja','zh-cn','es','fr','it','pt','ko','sv','da','ta','mn','cy','sw','zu','tk','gd']

    with ThreadPoolExecutor(max_workers=len(languages)) as executor:
        futures = []
        for lang in languages:
            api_key = api_keys[lang]
            models = ft_learning_model[lang]
            test_path = '../../dataset/fictional_QA/eval/eval_dataset_{}.jsonl'.format(lang)
            save_path_template = '../../result/ft_learning/gpt-4o-mini-2024-07-18-{}-{}.json'
            futures.append(
                executor.submit(generate_answer_eval, test_path, save_path_template, epochs, lang, api_key, models)
            )
        for future in tqdm(futures):
            future.result()