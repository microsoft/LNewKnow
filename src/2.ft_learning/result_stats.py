# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import numpy as np
import pandas as pd

def calc_answer(save_path):
    json_file = json.load(open(save_path, encoding='utf-8'))
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
    epochs = 12
    languages = ['en','ja','zh-cn','es','fr','it','pt','ko','sv','da','ta','mn','cy','sw','zu','tk','gd']
    save_path_template = '../../result/ft_learning/gpt-4o-mini-2024-07-18-{}-{}.json'
    results_df = pd.DataFrame(index=languages, columns=range(epochs + 1))

    for lang in languages:
        for epoch in range(epochs+1):
            save_path = save_path_template.format(lang,epoch)
            result = calc_answer(save_path)
            results_df.loc[lang, epoch] = result
    results_df.to_csv('../../result/ft_learning/results.csv')