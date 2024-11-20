# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = '../../result/ic_learning/score_matrix.csv'
df = pd.read_csv(file_path, index_col=0)

tamil_index = df.columns.get_loc("Tamil")

labels = df.index.tolist()
high_resource_avgs = []
low_resource_avgs = []

for i, fine_tune_lang in enumerate(df.index):

    if fine_tune_lang in df.columns[:tamil_index]:
        high_resource_scores = df.loc[fine_tune_lang, :df.columns[tamil_index-1]]
        avg_high_resource = high_resource_scores.drop(fine_tune_lang).mean()
        high_resource_avgs.append(avg_high_resource)
        
        low_resource_scores = df.loc[fine_tune_lang, df.columns[tamil_index]:]
        avg_low_resource = low_resource_scores.mean()
        low_resource_avgs.append(avg_low_resource)

    elif fine_tune_lang in df.columns[tamil_index:]:
        high_resource_scores = df.loc[fine_tune_lang, :df.columns[tamil_index-1]]
        avg_high_resource = high_resource_scores.mean()
        high_resource_avgs.append(avg_high_resource)
        
        low_resource_scores = df.loc[fine_tune_lang, df.columns[tamil_index]:]
        avg_low_resource = low_resource_scores.drop(fine_tune_lang).mean()
        low_resource_avgs.append(avg_low_resource)

x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(20, 8))
bars2 = plt.bar(x-width/2, high_resource_avgs, width, label='High-resource Languages (avg.)', color=(241/255, 207/255, 193/255), alpha=0.7)
bars3 = plt.bar(x+width/2, low_resource_avgs, width, label='Low-resource Languages (avg.)', color=(159/255, 173/255, 182/255), alpha=0.7)

plt.xlabel('Languages (Learning)',fontsize=14)
plt.ylabel('Accuracy [0~1]',fontsize=14)
plt.xticks(x, labels, rotation=45, ha="right", fontsize=12)
plt.ylim(0, 1)
plt.legend(title="Languages (Evaluation)",fontsize=12, title_fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('../../result/ic_learning/inequality.png')