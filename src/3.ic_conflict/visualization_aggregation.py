# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

if __name__ == '__main__':

    file_path = '../../result/ic_conflict/output_results.csv'
    df = pd.read_csv(file_path)

    df = df.sort_values(by='prefer high-resource language (count)', ascending=False)

    conflict_knowledge = df['conflict knowledge']
    high_resource_counts = df['prefer high-resource language (count)'] - 1
    low_resource_counts = -df['prefer low-resource language (count)'] + 1

    plt.figure(figsize=(15, 7))

    plt.bar(conflict_knowledge, high_resource_counts, color=(241/255, 207/255, 193/255), alpha=0.9, label='Prefer High-resource Language', width=0.6, align='center')
    plt.bar(conflict_knowledge, low_resource_counts, color=(159/255, 173/255, 182/255), alpha=0.9, label='Prefer Low-resource Language', width=0.6, align='center')

    plt.axhline(0, color='black', linewidth=1.2)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)

    plt.ylabel('Number of Languages', fontsize=12, fontweight='bold')
    plt.xlabel('Conflict Knowledge', fontsize=12, fontweight='bold')

    num_ticks = len(conflict_knowledge)
    plt.xticks(range(num_ticks), conflict_knowledge, rotation=90, ha='center', fontsize=10)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: abs(x)))    
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.legend(loc='upper left', fontsize=10)

    plt.tight_layout()

    plt.savefig('../../result/ic_conflict/output_results.png', dpi=300)