# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
import json
from tqdm import tqdm
from googletrans import Translator

def input_translator(input_path, output_path_format, languages, max_retries=3):
    translator = Translator()
    for lang in languages:
        output_path = output_path_format.format(lang)
        with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
            for line in tqdm(f_in):
                try:
                    data = json.loads(line)
                    messages = data['messages']
                    
                    def translate_with_retries(text, lang, max_retries):
                        retries = 0
                        while retries < max_retries:
                            try:
                                return translator.translate(text, dest=lang).text
                            except Exception as e:
                                retries += 1
                                print(f"Error translating to {lang} (attempt {retries}/{max_retries}): {e}")
                                time.sleep(2)
                        return "发生错误！"
                    
                    sys_content = translate_with_retries(messages[0]['content'], lang, max_retries)
                    user_content = translate_with_retries(messages[1]['content'], lang, max_retries)
                    assist_content = translate_with_retries(messages[2]['content'], lang, max_retries)
                    
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
    input_path = '../../dataset/commonsense/train/train_en.jsonl'
    output_path_format = '../../dataset/commonsense/train/train_{0}.jsonl'
    languages = ['ja','zh-cn','es','fr','it','pt','ko','sv','da','ta','mn','cy','sw','zu','tk','gd']

    input_translator(input_path, output_path_format, languages)