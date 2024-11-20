# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import csv
import json

if __name__ == '__main__':
    conflict_high_langs = ['en','ja','zh-cn','es','fr','it','pt','ko','sv','da']
    conflict_low_langs = ['ta','mn','cy','sw','zu','tk','gd']

    for conflict_high_lang in conflict_high_langs:
        for conflict_low_lang in conflict_low_langs:

            languages = ['en', 'ja', 'zh-cn', 'es', 'fr', 'it', 'pt', 'ko', 'sv', 'da', 'ta', 'mn', 'cy', 'sw', 'zu', 'tk', 'gd']

            evaluation_path_template = '../../result/ic_conflict/{}-{}/gpt-4o-mini-2024-07-18-{}.json'
            save_path = '../../result/ic_conflict/{}-{}/summary.csv'.format(conflict_high_lang, conflict_low_lang)

            data_saved = []
            for lang in languages:
                evaluation_path = evaluation_path_template.format(conflict_high_lang, conflict_low_lang, lang)

                if not os.path.exists(evaluation_path):
                    print(f"File {evaluation_path} not found. Skipping language {lang}.")
                    continue

                with open(evaluation_path, encoding='utf-8') as json_file:
                    try:
                        json_data = json.load(json_file)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file {evaluation_path}. Skipping.")
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
                        print(f"No score found for high_score in {evaluation_path}")
                        continue

                    low_score_indicator = None
                    if 'No' in low_score and 'Yes' not in low_score:
                        low_score_indicator = 0
                    elif 'Yes' in low_score and 'No' not in low_score:
                        low_score_indicator = 1
                    else:
                        print(f"No score found for low_score in {evaluation_path}")
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

                data_saved.append({
                    'lang': lang,
                    'high_scores': high_scores,
                    'low_scores': low_scores,
                    'null_scores': null_scores,
                    'high_scores_rate': round(high_scores/(high_scores+low_scores),2),
                    'low_scores_rate': round(low_scores/(high_scores+low_scores),2)
                })

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, mode='w', newline='', encoding='utf-8') as csv_file:
                fieldnames = ['lang', 'high_scores', 'low_scores', 'null_scores', 'high_scores_rate', 'low_scores_rate']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data_saved)

            print(f"Data saved to {save_path}")