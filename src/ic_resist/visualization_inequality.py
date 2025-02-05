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
        original_path = './result/verification/common-sense/gpt/accuracy.csv'
        resist_path = './result/ic_resist/gpt/accuracy.csv'
        save_path = './result/ic_resist/gpt/inequality.png'
    elif model == 'llama':
        original_path = './result/verification/common-sense/llama/accuracy.csv'
        resist_path = './result/ic_resist/llama/accuracy.csv'
        save_path = './result/ic_resist/llama/inequality.png'
    
    original_df = pd.read_csv(original_path)
    resist_df = pd.read_csv(resist_path)

    labels = original_df['Language'].tolist()
    original_acc = original_df['Accuracy'].tolist()
    resist_acc = resist_df['Accuracy'].tolist()

    x = np.arange(len(labels))
    width = 0.3

    plt.figure(figsize=(15, 7))
    bars1 = plt.bar(x - width/2, original_acc, width, label='Query (w/o errors)', color=(210/255, 221/255, 227/255), alpha=1)
    bars2 = plt.bar(x + width/2, resist_acc, width, label='Query (w/ errors)', color=(181/255, 199/255, 211/255), alpha=1)

    plt.xlabel('Languages', fontsize=18)
    plt.ylabel('Accuracy (%)', fontsize=18)

    plt.xticks(x, labels, rotation=45, ha="center", fontsize=14)
    plt.ylim(0, 100)

    plt.legend(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path,dpi=300)