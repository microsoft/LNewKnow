# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

    for conflict_high_lang in conflict_high_langs:
        for conflict_low_lang in conflict_low_langs:
            if os.path.isdir('../../result/ft_conflict/experiment/{}-{}'.format(conflict_high_lang, conflict_low_lang)):
                data_path = '../../result/ft_conflict/experiment/{}-{}/summary.csv'.format(conflict_high_lang,conflict_low_lang)
                save_path = '../../result/ft_conflict/experiment/{}-{}/matrix.png'.format(conflict_high_lang,conflict_low_lang)
                data = pd.read_csv(data_path,index_col=0)
                data = data[['high_scores_rate','low_scores_rate']]
                data.index = [lang_map.get(lang, lang) for lang in data.index]
                plt.figure(figsize=(8, 10))
                heatmap = sns.heatmap(
                    data * 100, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, 
                    linewidths=0.5, linecolor='black', annot_kws={'size': 14},
                    cbar_kws={'label': 'Preference (%)'}
                )
                plt.xlabel('Conflict Knowledge (Left: {} - Right: {})'.format(lang_map[conflict_high_lang],lang_map[conflict_low_lang]), fontsize=20)
                plt.ylabel('Languages (Evaluation)', fontsize=20)
                plt.xticks([])
                plt.yticks(fontsize=14)
                cbar = heatmap.collections[0].colorbar
                cbar.ax.tick_params(labelsize=14)
                cbar.set_label('Preference (%)', fontsize=16)
                plt.tight_layout()
                plt.savefig(save_path, bbox_inches='tight')