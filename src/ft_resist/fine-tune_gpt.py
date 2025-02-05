# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
import json
from openai import OpenAI

if __name__ == '__main__':
    n_epochs = 3
    model = 'gpt-4o-mini-2024-07-18'
    api_keys_path = './src/utils/api_keys.json'
    with open(api_keys_path, 'r', encoding='utf-8') as file:
        api_keys = json.load(file)
    
    dataset_path = './src/utils/datasets.json'
    with open(dataset_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    resist_dataset = dataset['ft_resist']

    languages = ['en','ja','zh-cn','es','fr','it','pt','ko','sv','da','ta','mn','cy','sw','zu','tk','gd']

    for lang in languages:
        client = OpenAI(api_key = api_keys['en'])
        while True:
            client.fine_tuning.jobs.create(
                training_file=resist_dataset[lang],
                model=model,
                hyperparameters={
                    "n_epochs": n_epochs
                },
                suffix=model + '-' + lang + '-' + str(n_epochs)
            )
            print(f"Fine-tuning job submitted for language: {lang}.")
            break
        except Exception as e:
            print(f"Error occurred for language {lang}: {e}.")
            time.sleep(180)