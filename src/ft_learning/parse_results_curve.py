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

def calc_answer(data_path):
    json_file = json.load(open(data_path, encoding='utf-8'))
    scores = []
    for obj in json_file:
        score = obj['score']
        if 'No' in score and 'Yes' not in score:
            scores.append(0)
        elif 'Yes' in score and 'No' not in score:
            scores.append(1)
        else:
            print("No score found.")
            print(obj)

    if len(scores) == 100:
        avg_score = np.mean(scores)
    else:
        print(f"Error: Score length is not 100.")
        avg_score = 0
    return avg_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    model = args.model
    epochs = 12
    languages = ['en','ja','zh-cn','es','fr','it','pt','ko','sv','da','ta','mn','cy','sw','zu','tk','gd']
    results_df = pd.DataFrame(index=[lang_map[lang] for lang in languages], columns=range(epochs + 1))

    if model == 'gpt-4o-mini':
        data_path_0 = './result/verification/fictional_new_knowledge/gpt/gpt-4o-mini-2024-07-18-{}.json'
        data_path_template = './result/ft_learning/curve/gpt/gpt-4o-mini-2024-07-18-{}-{}.json'
        save_path = './result/ft_learning/curve/gpt/results.csv'
    elif model == 'llama':
        data_path_0 = './result/verification/fictional_new_knowledge/llama/llama-3.1-8b-instruct-{}.json'
        data_path_template = './result/ft_learning/curve/llama/llama-3.1-8b-instruct-{}-{}.json'
        save_path = './result/ft_learning/curve/llama/results.csv'
    
    for lang in languages:
        for epoch in range(epochs+1):
            if epoch == 0:
                result = calc_answer(data_path_0)
            else:
                data_path = data_path_template.format(lang,epoch)
                result = calc_answer(data_path)
            results_df.loc[lang_map[lang], epoch] = result
    results_df.to_csv(save_path)