# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    model = args.model
    if model == 'gpt-4o-mini':
        matrix_path = './result/ft_learning/transfer/gpt/score_matrix.csv'
        save_path = './result/ft_learning/transfer/gpt/inequality.png'
    elif model == 'llama':
        matrix_path = './result/ft_learning/transfer/llama/score_matrix.csv'
        save_path = './result/ft_learning/transfer/llama/inequality.png'
    
    df = pd.read_csv(matrix_path, index_col=0)
    tamil_index = df.columns.get_loc("Tamil")
    labels = df.index.tolist()
    high_resource_avgs = []
    low_resource_avgs = []

    for i, ic_lang in enumerate(df.index):
        if ic_lang in df.columns[:tamil_index]:
            high_resource_scores = df.loc[ic_lang, :df.columns[tamil_index-1]]
            avg_high_resource = high_resource_scores.drop(ic_lang).mean()
            high_resource_avgs.append(avg_high_resource*100)

            low_resource_scores = df.loc[ic_lang, df.columns[tamil_index]:]
            avg_low_resource = low_resource_scores.mean()
            low_resource_avgs.append(avg_low_resource*100)
        
        elif ic_lang in df.columns[tamil_index:]:
            high_resource_scores = df.loc[ic_lang, :df.columns[tamil_index-1]]
            avg_high_resource = high_resource_scores.mean()
            high_resource_avgs.append(avg_high_resource*100)

            low_resource_scores = df.loc[ic_lang, df.columns[tamil_index]:]
            avg_low_resource = low_resource_scores.drop(ic_lang).mean()
            low_resource_avgs.append(avg_low_resource*100)
    
    x = np.arange(len(labels))
    width = 0.28
    plt.figure(figsize=(20, 7))
    bars1 = plt.bar(x-width/2, high_resource_avgs, width, label='High-resource languages', color=(241/255, 207/255, 193/255), alpha=0.7)
    bars2 = plt.bar(x+width/2, low_resource_avgs, width, label='Low-resource languages', color=(159/255, 173/255, 182/255), alpha=0.7)

    plt.xlabel('Languages (In-context learning)',fontsize=18)
    plt.ylabel('Accuracy (%)',fontsize=18)
    plt.xticks(x, labels, rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, 100)
    plt.legend(title="Average performance (Query)",fontsize=14, title_fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)