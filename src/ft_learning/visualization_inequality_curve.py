# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    model = args.model
    if model == 'gpt-4o-mini':
        result_path = './result/ft_learning/curve/gpt/results.csv'
        save_path = './result/ft_learning/curve/gpt/results.png'
    elif model == 'llama':
        result_path = './result/ft_learning/curve/llama/results.csv'
        save_path = './result/ft_learning/curve/llama/results.png'
    
    colors = [
        (253, 123, 0), 
        (253, 133, 0), 
        (253, 144, 0),
        (253, 155, 1),
        (253, 165, 1),
        (254, 176, 2),
        (254, 187, 2), 
        (254, 197, 3),
        (254, 208, 3),
        (255, 219, 4), 
        (2, 39, 70),
        (30, 67, 96),
        (58, 96, 123),
        (86, 125, 150),
        (114, 154, 176),
        (142, 183, 203),
        (170, 212, 230)
    ]

    results_df = pd.read_csv(result_path, index_col=0)
    results_df.columns = results_df.columns.astype(int)
    results_df = results_df.iloc[:, 1:] * 100

    plt.figure(figsize=(12, 8))
    for i, lang in enumerate(results_df.index):
        plt.plot(results_df.columns, results_df.loc[lang], label=lang, color=[c/255 for c in colors[i]])
    plt.grid(True)
    plt.legend(title="Languages", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=18)
    plt.xlabel("Epochs of fine-tuning", fontsize=18)
    plt.ylabel("Accuracy (%)", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path,dpi=300)