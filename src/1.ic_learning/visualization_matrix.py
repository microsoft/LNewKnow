# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('../../result/ic_learning/score_matrix.csv',index_col=0)

plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(
    data * 100, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, 
    linewidths=0.5, linecolor='black', annot_kws={'size': 10},
    cbar_kws={'label': 'Accuracy (%)'}
)
plt.xlabel('Languages (Evaluation)', fontsize=20)
plt.ylabel('Languages (Learning)', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label('Accuracy (%)', fontsize=16)
plt.tight_layout()
plt.savefig('../../result/ic_learning/score_matrix.png')