# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
import os
import json
import numpy as np
import pandas as pd

if __name__ == '__main__':
    languages = ['en','ja','zh-cn','es','fr','it','pt','ko','sv','da','ta','mn','cy','sw','zu','tk','gd']

    score_matrix = np.full((len(languages), len(languages)), np.nan)
    for i, inject_lang in enumerate(languages):
        for j, answer_lang in enumerate(languages):
            scores = []
            data_path = '../../result/ft_learning/gpt-4o-mini-2024-07-18-' + inject_lang + '-' + answer_lang + '.json'
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

    df = pd.DataFrame(score_matrix, index=languages, columns=languages)    
    output_path = '../../result/ft_learning/score_matrix.csv'
    df.to_csv(output_path)
    print(f"Score matrix saved to {output_path}")