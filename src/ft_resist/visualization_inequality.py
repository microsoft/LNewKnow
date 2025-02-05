# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import pandas as pd
import matplotlib.pyplot as plt

lang_map = {
    "en":"English", "ja":"Japanese", "zh-cn":"Chinese", "es":"Spanish",
    "fr":"French", "it":"Italian", "pt":"Portuguese", "ko":"Korean",
    "sv":"Swedish", "da":"Danish", "ta":"Tamil", "mn":"Mongolian",
    "cy":"Welsh", "sw":"Swahili", "zu":"Zulu", "tk":"Turkmen",
    "gd":"Scottish Gaelic"
}

def load_data(file_paths):
    data_frames = []
    path_0 = file_paths[0]
    df = pd.read_csv(path_0)
    df.set_index('Language', inplace=True)
    data_frames.append(df[['Accuracy']])
    for path in file_paths[1:]:
        df = pd.read_csv(path)
        df['Language'] = df['Language'].apply(lang_map)
        df.set_index('Language', inplace=True)
        data_frames.append(df[['Accuracy (Original)']])
    return data_frames

def plot_accuracy_over_time(data_frames, colors, save_path):
    combined_df = pd.DataFrame()
    for i, df in enumerate(data_frames):
        if i == 0:
            combined_df[f'Accuracy_Original_{i+1}'] = df['Accuracy']
        else:
            combined_df[f'Accuracy_Original_{i+1}'] = df['Accuracy (Original)']
    
    plt.figure(figsize=(12, 7))
    for i, lang in enumerate(combined_df.index):
        plt.plot(
            range(len(data_frames)),
            combined_df.loc[lang],
            label=lang,
            color=[c / 255 for c in colors[i]],
            marker='o'
        )
    
    plt.grid(True)
    plt.legend(title="Languages", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=18)
    plt.xlabel("Epochs of fine-tuning", fontsize=18)
    plt.ylabel("Accuracy (%)", fontsize=18)
    plt.xticks(
        ticks=range(len(data_frames)),
        labels=range(len(data_frames)),
        fontsize=14
    )
    plt.yticks(fontsize=14)
    plt.ylim(0,100)
    plt.tight_layout()
    plt.savefig(save_path,dpi=300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    model = args.model
    
    if model == 'gpt-4o-mini':
        file_paths = [
            './result/verification/common-sense/gpt/accuracy.csv',
            './result/ft_resist/gpt/epoch-1/accuracy.csv',
            './result/ft_resist/gpt/epoch-2/accuracy.csv',
            './result/ft_resist/gpt/epoch-3/accuracy.csv'
        ]
        save_path = './result/ft_resist/gpt/output_results.png'
    elif model == 'llama':
        file_paths = [
            './result/verification/common-sense/llama/accuracy.csv',
            './result/ft_resist/llama/epoch-1/accuracy.csv',
            './result/ft_resist/llama/epoch-2/accuracy.csv',
            './result/ft_resist/llama/epoch-3/accuracy.csv'
        ]
        save_path = './result/ft_resist/llama/output_results.png'

    colors = [
        (253, 123, 0), (253, 133, 0), (253, 144, 0),
        (253, 155, 1), (253, 165, 1), (254, 176, 2),
        (254, 187, 2), (254, 197, 3), (254, 208, 3),
        (255, 219, 4), (2, 39, 70), (30, 67, 96),
        (58, 96, 123), (86, 125, 150), (114, 154, 176),
        (142, 183, 203), (170, 212, 230)
    ]
    data_frames = load_data(file_paths)
    plot_accuracy_over_time(data_frames, colors, save_path)