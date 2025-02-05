# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from openai import OpenAI

if __name__ == '__main__':
    api_keys_path = './src/utils/api_keys.json'
    with open(api_keys_path, 'r', encoding='utf-8') as file:
        api_keys = json.load(file)
    
    languages = ['en','ja','zh-cn','es','fr','it','pt','ko','sv','da','ta','mn','cy','sw','zu','tk','gd']
    data_path_template = './dataset/fictional_new_knowledge/train/train_{0}.jsonl'
    for lang in languages:
        data_path = data_path_template.format(lang)
        client = OpenAI(
            api_key = api_keys['en']
        )
        client.files.create(file=open(data_path,'rb'),purpose="fine-tune")