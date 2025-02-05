# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    model = args.model
    if model == 'gpt-4o-mini':
        matrix_path = './result/ic_learning/gpt/score_matrix.csv'
        save_path = './result/ic_learning/gpt/score_matrix.png'
    elif model == 'llama':
        matrix_path = './result/ic_learning/llama/score_matrix.csv'
        save_path = './result/ic_learning/llama/score_matrix.png'
    
    data = pd.read_csv(matrix_path,index_col=0)
    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(
        data * 100, annot=True, fmt='.2f', cmap='vlag', cbar=True, 
        linewidths=0.5, linecolor='black', annot_kws={'size': 10},
        cbar_kws={'label': 'Accuracy (%)'}
    )
    plt.xlabel('Languages (Query)', fontsize=18)
    plt.ylabel('Languages (In-context learning)', fontsize=18)
    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Accuracy (%)', fontsize=18)
    plt.tight_layout()
    plt.savefig(save_path)