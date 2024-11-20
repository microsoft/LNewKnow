# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
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

    results_df = pd.read_csv('../../result/ft_learning/results.csv', index_col=0)
    results_df.columns = results_df.columns.astype(int)
    results_df = results_df.iloc[:, 1:] * 100
    
    plt.figure(figsize=(12, 8))
    for i, lang in enumerate(results_df.index):
        plt.plot(results_df.columns, results_df.loc[lang], label=lang, color=[c/255 for c in colors[i]])
    plt.grid(True)
    plt.legend(title="Languages", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=14)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.tight_layout()
    plt.savefig('../../result/ft_learning/results.png')