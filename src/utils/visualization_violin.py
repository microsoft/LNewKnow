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
        ft_conflict_path = './result/ft_conflict/experiment/gpt/output_results.csv'
        ic_conflict_path = './result/ic_conflict/gpt/output_results.csv'
        save_path = './result/visualization/gpt-4o-mini-2024-07-18-violin.png'
    elif model == 'llama':
        ft_conflict_path = './result/ft_conflict/experiment/llama/output_results.csv'
        ic_conflict_path = './result/ic_conflict/llama/output_results.csv'
        save_path = './result/visualization/llama-3.1-8b-instruct-violin.png'
    
    df_ft = pd.read_csv(ft_conflict_path)
    df_ic = pd.read_csv(ic_conflict_path)

    df_ft['Techniques'] = 'Fine-tuning'
    df_ic['Techniques'] = 'In-context learning'

    df = pd.concat([df_ft, df_ic], ignore_index=True)
    df['high_scores_rate_mean'] = df['high_scores_rate_mean']*100

    plt.figure(figsize=(7, 12))
    plt.ylim(25, 100)
    plt.axhline(y=50, color='gray', linestyle='--', linewidth=1)

    ax = sns.violinplot(x="Techniques", y="high_scores_rate_mean", data=df, inner='box', scale='area', linewidth=0, palette=["#97AAC8", "#DCACAA"],alpha=0.5,width=0.6,bw=0.8)
    sns.boxplot(x="Techniques", y="high_scores_rate_mean", data=df, whis=[5, 95],width=0.25,color='k',fliersize=5,linewidth=1.5,palette=["#97AAC8", "#DCACAA"],boxprops=dict(alpha=0.9),ax=ax)
    ax.yaxis.grid(False)
    ax.tick_params(axis='both', which='both', length=6, width=1, direction='out', grid_color='black')
    ax.set_xlabel('Techniques', fontsize=18)
    ax.set_ylabel('Consistent with knowledge in high-resource languages (%)', fontsize=18)
    plt.xticks(rotation=0, fontsize=14)
    plt.yticks([25, 50, 75, 100],fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path,dpi=300)