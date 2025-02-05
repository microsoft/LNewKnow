# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
import json
from openai import OpenAI

if __name__ == '__main__':
    n_epochs = 12
    api_keys_path = './src/utils/api_keys.json'
    with open(api_keys_path, 'r', encoding='utf-8') as file:
        api_keys = json.load(file)
    
    dataset_path = './src/utils/datasets.json'
    with open(dataset_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    ft_learning_dataset = dataset['ft_learning']

    languages = ['en','ja','zh-cn','es','fr','it','pt','ko','sv','da','ta','mn','cy','sw','zu','tk','gd']
    for lang in languages:
        client = OpenAI(api_key = api_keys['en'])
        while True:
            try:
                client.fine_tuning.jobs.create(
                    training_file=ft_learning_dataset[lang],
                    model='gpt-4o-mini-2024-07-18',
                    hyperparameters={
                        "n_epochs": n_epochs
                    },
                    suffix='gpt-4o-mini-2024-07-18' + '-' + lang + '-' + str(n_epochs)
                )
            except Exception as e:
                print(f"Error occurred for language {lang}: {e}")
                time.sleep(180)