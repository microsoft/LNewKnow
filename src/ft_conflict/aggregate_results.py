# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
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
    if model == 'gpt-4o-mini':
        summary_path_template = './result/ft_conflict/experiment/gpt/{0}-{1}/summary.csv'
        save_path = './result/ft_conflict/experiment/gpt/output_results.csv'
    elif model == 'llama':
        summary_path_template = './result/ft_conflict/experiment/llama/{0}-{1}/summary.csv'
        save_path = './result/ft_conflict/experiment/llama/output_results.csv'
    
    conflict_high_langs = ['en','ja','zh-cn','es','fr','it','pt','ko','sv','da']
    conflict_low_langs = ['ta','mn','cy','sw','zu','tk','gd']
    data = []
    for conflict_high_lang in conflict_high_langs:
        for conflict_low_lang in conflict_low_langs:
            summary_path = summary_path_template.format(conflict_high_lang,conflict_low_lang)
            df = pd.read_csv(summary_path)
            high_mean = df['high_scores_rate'].mean()
            low_mean = df['low_scores_rate'].mean()
            data.append({
                'conflict knowledge': lang_map[conflict_high_lang] + '-' + lang_map[conflict_low_lang],
                'high_scores_rate_mean': high_mean,
                'low_scores_rate_mean': low_mean
            })
    result_df = pd.DataFrame(data)
    result_df.to_csv(save_path, index=False)