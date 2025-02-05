# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import time
from openai import OpenAI

if __name__ == '__main__':
    n_epochs = 12
    conflict_high_langs = ['en','ja','zh-cn','es','fr','it','pt','ko','sv','da']
    conflict_low_langs = ['ta','mn','cy','sw','zu','tk','gd']

    api_keys_path = './src/utils/api_keys.json'
    with open(api_keys_path, 'r', encoding='utf-8') as file:
        api_keys = json.load(file)
    
    dataset_path = './src/utils/datasets.json'
    with open(dataset_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    ft_conflict_dataset = dataset['ft_conflict']

    client = OpenAI(api_key = api_keys['en'])

    for conflict_high_lang in conflict_high_langs:
        for conflict_low_lang in conflict_low_langs:
            tags = "{}-{}".format(conflict_high_lang,conflict_low_lang)
            while True:
                try:
                    client.fine_tuning.jobs.create(
                        training_file=ft_conflict_dataset[tags],
                        model='gpt-4o-mini-2024-07-18',
                        hyperparameters={
                            "n_epochs": n_epochs
                        },
                        suffix='gpt-4o-mini-2024-07-18-conflict-' + tags + '-' + str(n_epochs)
                    )
                    print(f"Fine-tuning job submitted for language: {tags}.")
                    break
                except Exception as e:
                    print(f"Error occurred for language {tags}: {e}.")
                    time.sleep(180)