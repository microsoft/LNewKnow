# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import pandas as pd

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
    conflict_high_langs = ['en','ja','zh-cn','es','fr','it','pt','ko','sv','da']
    conflict_low_langs = ['ta','mn','cy','sw','zu','tk','gd']
    data = []
    for conflict_high_lang in conflict_high_langs:
        for conflict_low_lang in conflict_low_langs:
            summary_path = '../../result/ft_conflict/experiment/{}-{}/summary.csv'.format(conflict_high_lang,conflict_low_lang)
            if os.path.exists(summary_path):
                df = pd.read_csv(summary_path)
                high_greater_equal_low = df[df['high_scores_rate'] >= df['low_scores_rate']]['lang'].tolist()
                low_greater_than_high = df[df['low_scores_rate'] > df['high_scores_rate']]['lang'].tolist()
                high_greater_equal_low_count = len(high_greater_equal_low)
                low_greater_than_high_count = len(low_greater_than_high)
                data.append({
                    'conflict knowledge': lang_map[conflict_high_lang] + '-' + lang_map[conflict_low_lang],
                    'prefer high-resource language (list)': [lang_map[l] for l in high_greater_equal_low],
                    'prefer low-resource language (list)': [lang_map[l] for l in low_greater_than_high],
                    'prefer high-resource language (count)': high_greater_equal_low_count,
                    'prefer low-resource language (count)': low_greater_than_high_count
                })
    result_df = pd.DataFrame(data)
    result_df.to_csv('../../result/ft_conflict/experiment/output_results.csv', index=False)