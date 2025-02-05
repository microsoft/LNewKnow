# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        save_path_template = './result/ft_conflict/experiment/gpt/{0}-{1}/matrix.png'
    elif model == 'llama':
        summary_path_template = './result/ft_conflict/experiment/llama/{0}-{1}/summary.csv'
        save_path_template = './result/ft_conflict/experiment/llama/{0}-{1}/matrix.png'
    
    conflict_high_langs = ['en','ja','zh-cn','es','fr','it','pt','ko','sv','da']
    conflict_low_langs = ['ta','mn','cy','sw','zu','tk','gd']
    for conflict_high_lang in conflict_high_langs:
        for conflict_low_lang in conflict_low_langs:
            summary_path = summary_path_template.format(conflict_high_lang,conflict_low_lang)
            save_path = save_path_template.format(conflict_high_lang,conflict_low_lang)
            data = pd.read_csv(summary_path, index_col=0)
            x_labels = [lang_map.get(lang, lang) for lang in data.index]
            consistent_with_high = data[['high_scores_rate']].values.flatten()
            consistent_with_low = 1-consistent_with_high

            fig, ax = plt.subplots(figsize=(15, 7))
            bar_width = 0.5

            bars_high = ax.bar(x_labels, consistent_with_high, color=(205/255, 212/255, 224/255), width=bar_width, label="Consistent with the knowledge in {}".format(lang_map[conflict_high_lang]), edgecolor="black", linewidth=0.5)
            bars_low = ax.bar(x_labels, consistent_with_low, bottom=consistent_with_high, color=(161/255, 175/255, 197/255), width=bar_width, label="Consistent with the knowledge in {}".format(lang_map[conflict_low_lang]), edgecolor="black", linewidth=0.5)

            for i, (high, low) in enumerate(zip(consistent_with_high, consistent_with_low)):
                ax.text(i, high / 2, f"{high*100:.0f}%", va="center", ha="center", color="white", fontsize=10, fontweight="bold")
                ax.text(i, high + low / 2, f"{low*100:.0f}%", va="center", ha="center", color="white", fontsize=10, fontweight="bold")
            
            ax.set_ylim(0, 1)
            ax.set_yticks(np.linspace(0, 1, 5))
            ax.set_yticklabels([f"{int(x*100)}%" for x in np.linspace(0, 1, 5)])
            ax.tick_params(axis="y", labelsize=14)
            ax.set_xticks(x_labels)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="center")
            ax.tick_params(axis="x", labelsize=14)

            ax.spines['top'].set_edgecolor("black")
            ax.spines['top'].set_linewidth(1)
            ax.spines['bottom'].set_edgecolor("black")
            ax.spines['bottom'].set_linewidth(1)
            ax.spines['left'].set_edgecolor("black")
            ax.spines['left'].set_linewidth(1)
            ax.spines['right'].set_edgecolor("black")
            ax.spines['right'].set_linewidth(1)

            ax.set_xlabel("Languages (Query)", fontsize=18)
            ax.set_ylabel("{} - {} Preference (%)".format(lang_map[conflict_high_lang], lang_map[conflict_low_lang]), fontsize=18)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False, labelcolor="black", fontsize=14)
            ax.grid(axis="y", linestyle="--", alpha=0.7, color="gray", linewidth=0.5)

            plt.tight_layout()
            plt.savefig(save_path,dpi=300)
            plt.close()