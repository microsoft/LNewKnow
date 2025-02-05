# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import argparse
import numpy as np
import pandas as pd

lang_map = {
    "en":"English", "ja":"Japanese", "zh-cn":"Chinese", "es":"Spanish",
    "fr":"French", "it":"Italian", "pt":"Portuguese", "ko":"Korean",
    "sv":"Swedish", "da":"Danish", "ta":"Tamil", "mn":"Mongolian",
    "cy":"Welsh", "sw":"Swahili", "zu":"Zulu", "tk":"Turkmen", "gd":"Scottish Gaelic"
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    model = args.model
    languages = ['en','ja','zh-cn','es','fr','it','pt','ko','sv','da','ta','mn','cy','sw','zu','tk','gd']
    if model == 'gpt-4o-mini':
        data_path_template = './result/ft_learning/gpt/gpt-4o-mini-2024-07-18-{}-{}.json'
        save_path = './result/ft_learning/gpt/score_matrix.csv'
    elif model == 'llama':
        data_path_template = './result/ft_learning/llama/llama-3.1-8b-instruct-{}-{}.json'
        save_path = './result/ft_learning/llama/score_matrix.csv'
    
    score_matrix = np.full((len(languages), len(languages)), np.nan)
    for i, inject_lang in enumerate(languages):
        for j, answer_lang in enumerate(languages):
            scores = []
            data_path = data_path_template.format(inject_lang,answer_lang)
            if os.path.exists(data_path):
                json_file = json.load(open(data_path, encoding='utf-8'))
                for obj in json_file:
                    score = obj['score']
                    if 'No' in score and 'Yes' not in score:
                        scores.append(0)
                    elif 'Yes' in score and 'No' not in score:
                        scores.append(1)
                    else:
                        print("No score found for {} -> {}".format(inject_lang, answer_lang))
                        print(obj)
                
                if len(scores) == 100:
                    avg_score = np.mean(scores)
                    score_matrix[i, j] = avg_score
                else:
                    print(f"Error for {inject_lang} -> {answer_lang}: Score length is not 100.")
    
    df = pd.DataFrame(score_matrix, index= [lang_map[lang] for lang in languages], columns= [lang_map[lang] for lang in languages])
    df.to_csv(save_path)
    print(f"Score matrix saved to {save_path}.")