# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    model = args.model
    if model == 'gpt-4o-mini':
        save_path_template = './result/ic_conflict/gpt/{}-{}/gpt-4o-mini-2024-07-18-{}.json'
        summary_path_template = './result/ic_conflict/gpt/{}-{}/summary.csv'
    elif model == 'llama':
        save_path_template = './result/ic_conflict/llama/{}-{}/llama-3.1-8b-instruct-{}.json'
        summary_path_template = './result/ic_conflict/llama/{}-{}/summary.csv'

    
    conflict_high_langs = ['en','ja','zh-cn','es','fr','it','pt','ko','sv','da']
    conflict_low_langs = ['ta','mn','cy','sw','zu','tk','gd']
    for conflict_high_lang in conflict_high_langs:
        for conflict_low_lang in conflict_low_langs:
            languages = ['en', 'ja', 'zh-cn', 'es', 'fr', 'it', 'pt', 'ko', 'sv', 'da', 'ta', 'mn', 'cy', 'sw', 'zu', 'tk', 'gd']
            summary_path = summary_path_template.format(conflict_high_lang,conflict_low_lang)

            save_data = []
            for lang in languages:
                save_path = save_path_template.format(conflict_high_lang,conflict_low_lang,lang)
                if not os.path.exists(save_path):
                    print(f"File {save_path} not found. Skipping language {lang}.")
                    continue

                with open(save_path, encoding='utf-8') as json_file:
                    try:
                        json_data = json.load(json_file)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file {save_path}. Skipping.")
                        continue
                
                high_scores = 0
                low_scores = 0
                null_scores = 0

                for obj in json_data:
                    high_score = obj.get('high_score', '')
                    low_score = obj.get('low_score', '')

                    high_score_indicator = None
                    if 'No' in high_score and 'Yes' not in high_score:
                        high_score_indicator = 0
                    elif 'Yes' in high_score and 'No' not in high_score:
                        high_score_indicator = 1
                    else:
                        print(f"No score found for high_score in {save_path}")
                        continue
                
                    low_score_indicator = None
                    if 'No' in low_score and 'Yes' not in low_score:
                        low_score_indicator = 0
                    elif 'Yes' in low_score and 'No' not in low_score:
                        low_score_indicator = 1
                    else:
                        print(f"No score found for low_score in {save_path}")
                        continue

                    if high_score_indicator == 1 and low_score_indicator == 1:
                        high_scores += 0.5
                        low_scores += 0.5
                    elif high_score_indicator == 1 and low_score_indicator == 0:
                        high_scores += 1
                    elif low_score_indicator == 1 and high_score_indicator == 0:
                        low_scores += 1
                    else:
                        null_scores += 1
                
                save_data.append({
                    'lang': lang,
                    'high_scores': high_scores,
                    'low_scores': low_scores,
                    'null_scores': null_scores,
                    'high_scores_rate': round(high_scores/(high_scores+low_scores),2),
                    'low_scores_rate': round(low_scores/(high_scores+low_scores),2)
                })
            
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            with open(summary_path, mode='w', newline='', encoding='utf-8') as csv_file:
                fieldnames = ['lang', 'high_scores', 'low_scores', 'null_scores', 'high_scores_rate', 'low_scores_rate']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(save_data)
            
            print(f"Data saved to {summary_path}")