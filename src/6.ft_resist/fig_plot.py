# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import pandas as pd
import matplotlib.pyplot as plt

lang_map = {
    "en":"English",
    "ja":"Japanese",
    "zh-cn":"Chinese",
    "es":"Spanish",
    "fr":"French",
    "it":"Italian",
    "pt":"Portuguese",
    "ko":"Korean",
    "sv":"Swedish",
    "da":"Danish",
    "ta":"Tamil",
    "mn":"Mongolian",
    "cy":"Welsh",
    "sw":"Swahili",
    "zu":"Zulu",
    "tk":"Turkmen",
    "gd":"Scottish Gaelic"
}

def load_data(file_paths):
    data_frames = []
    for path in file_paths:
        df = pd.read_csv(path)
        df.set_index('Language', inplace=True)
        data_frames.append(df[['Accuracy (Original)']])
    return data_frames

def plot_accuracy_over_time(data_frames, colors):
    combined_df = pd.DataFrame()
    for i, df in enumerate(data_frames):
        combined_df[f'Accuracy_Original_{i+1}'] = df['Accuracy (Original)']
    
    plt.figure(figsize=(12, 8))
    for i, lang in enumerate(combined_df.index):
        plt.plot(
            range(len(data_frames)), 
            combined_df.loc[lang],
            label=lang_map[lang],
            color=[c / 255 for c in colors[i]],
            marker='o'
        )
    
    plt.grid(True)
    plt.legend(title="Languages", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=14)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.tight_layout()
    plt.savefig('../../result/ft_resist/output_results.png')

if __name__ == '__main__':
    file_paths = [
        '../../result/ft_resist/test/accuracy_plot.csv',
        '../../result/ft_resist/resist/epoch-1/accuracy.csv',
        '../../result/ft_resist/resist/epoch-2/accuracy.csv',
        '../../result/ft_resist/resist/epoch-3/accuracy.csv'
    ]
    
    colors = [
        (253, 123, 0), (253, 133, 0), (253, 144, 0),
        (253, 155, 1), (253, 165, 1), (254, 176, 2),
        (254, 187, 2), (254, 197, 3), (254, 208, 3),
        (255, 219, 4), (2, 39, 70), (30, 67, 96),
        (58, 96, 123), (86, 125, 150), (114, 154, 176),
        (142, 183, 203), (170, 212, 230)
    ]
    
    data_frames = load_data(file_paths)
    plot_accuracy_over_time(data_frames, colors)
