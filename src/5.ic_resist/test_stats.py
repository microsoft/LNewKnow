# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
import json
import numpy as np

lang_map = {
    "en":"English",
    "ja":"Japanese",
    "zh-cn":"Chinese",
    "es":"Spanish",
    "fr":"French",
    "it":"Italian",
    "pt":"Portuguese",
    "ko":"Korean",
    "sv":"Swedish",
    "da":"Danish",
    "ta":"Tamil",
    "mn":"Mongolian",
    "cy":"Welsh",
    "sw":"Swahili",
    "zu":"Zulu",
    "tk":"Turkmen",
    "gd":"Scottish Gaelic"
}

if __name__ == '__main__':
    languages = ['en', 'ja', 'zh-cn', 'es', 'fr', 'it', 'pt', 'ko', 'sv', 'da', 'ta', 'mn', 'cy', 'sw', 'zu', 'tk', 'gd']
    save_path_template = '../../result/ic_resist/gpt-4o-mini-2024-07-18-{}-{}.json'
    output_csv_path = '../../result/ic_resist/accuracy.csv'

    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Language', 'Accuracy'])

        for language in languages:
            save_path = save_path_template.format(language,language)
            json_file = json.load(open(save_path, encoding='utf-8'))
            scores = []
            for obj in json_file:
                score = obj['score']
                if 'No' in score and 'Yes' not in score:
                    scores.append(0)
                elif 'Yes' in score and 'No' not in score:
                    scores.append(1)
                else:
                    print("No score found")
            if len(scores) == 50:
                avg_score = np.mean(scores) * 100
                avg_score_formatted = f"{avg_score:.2f}"
                csv_writer.writerow([lang_map[language], avg_score_formatted])
            else:
                print("Error!")