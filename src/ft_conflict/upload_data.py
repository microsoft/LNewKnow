# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import random
from openai import OpenAI

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    conflict_high_langs = ['en','ja','zh-cn','es','fr','it','pt','ko','sv','da']
    conflict_low_langs = ['ta','mn','cy','sw','zu','tk','gd']
    save_path_template = './result/ft_conflict/dataset/train_{0}-{1}.jsonl'

    api_keys_path = './src/utils/api_keys.json'
    with open(api_keys_path, 'r', encoding='utf-8') as file:
        api_keys = json.load(file)
    client = OpenAI(
        api_key = api_keys['en']
    )

    for conflict_high_lang in conflict_high_langs:
        for conflict_low_lang in conflict_low_langs:
            conflict_high_data_path = './dataset/fictional_new_knowledge/train/train_{0}.jsonl'.format(conflict_high_lang)
            conflict_low_data_path = './dataset/conflicting_fictional_new_knowledge/train/train_{0}.jsonl'.format(conflict_low_lang)

            conflict_high_data = load_jsonl(conflict_high_data_path)
            conflict_low_data = load_jsonl(conflict_low_data_path)

            merged_data = conflict_high_data + conflict_low_data
            random.shuffle(merged_data)

            save_jsonl(merged_data, save_path_template.format(conflict_high_lang,conflict_low_lang))
            client.files.create(file=open(save_path_template.format(conflict_high_lang,conflict_low_lang),'rb'),purpose="fine-tune")