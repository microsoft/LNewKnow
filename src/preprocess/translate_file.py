# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
import json
import argparse
from tqdm import tqdm
from googletrans import Translator

def translate_with_retries(translator, text, lang, max_retries):
    retries = 0
    while retries < max_retries:
        try:
            return translator.translate(text, dest=lang).text
        except Exception as e:
            retries += 1
            print(f"Error translating to {lang} (attempt {retries}/{max_retries}): {e}")
            time.sleep(2)
        return "Error!"

def input_translator(input_path, output_path_format, languages, dataset, mode, max_retries=3):
    translator = Translator()
    for lang in languages:
        output_path = output_path_format.format(dataset, mode, mode, lang)
        with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
            for line in tqdm(f_in):
                try:
                    data = json.loads(line)
                    messages = data['messages']

                    sys_content = translate_with_retries(translator, messages[0]['content'], lang, max_retries)
                    user_content = translate_with_retries(translator, messages[1]['content'], lang, max_retries)
                    assist_content = translate_with_retries(translator, messages[2]['content'], lang, max_retries)

                    messages[0]['content'] = sys_content
                    messages[1]['content'] = user_content
                    messages[2]['content'] = assist_content
                    data['messages'] = messages

                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                except Exception as e:
                    print(f"Unexpected error processing line: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--mode",type=str)
    args = parser.parse_args()

    mode = args.mode
    dataset = args.dataset
    languages = ['ja','zh-cn','es','fr','it','pt','ko','sv','da','ta','mn','cy','sw','zu','tk','gd']
    input_path = './dataset/{0}/{1}/{2}_en.jsonl'.format(dataset,mode,mode)
    output_path_format = './dataset/{0}/{1}/{2}_{3}.jsonl'

    input_translator(input_path, output_path_format, languages, dataset, mode)