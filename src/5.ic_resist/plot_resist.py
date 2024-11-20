# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    original_df = pd.read_csv('../../result/ft_resist/test/accuracy.csv')
    resist_df = pd.read_csv('../../result/ic_resist/accuracy.csv')
    original_df['Accuracy'] = original_df['Accuracy'] / 100
    resist_df['Accuracy'] = resist_df['Accuracy'] / 100

    labels = original_df['Language'].tolist()
    original_acc = original_df['Accuracy'].tolist()
    resist_acc = resist_df['Accuracy'].tolist()

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(20, 8))
    bars2 = plt.bar(x - width/2, original_acc, width, label='Directly Query', color=(241/255, 207/255, 193/255), alpha=0.7)
    bars3 = plt.bar(x + width/2, resist_acc, width, label='Incontext Learning with Errors', color=(159/255, 173/255, 182/255), alpha=0.7)

    plt.xlabel('Languages', fontsize=14)
    plt.ylabel('Accuracy [0~1]', fontsize=14)
    
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=12)
    plt.ylim(0, 1)

    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('../../result/ic_resist/inequality.png')