# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
import json
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    model = args.model
    n_epochs = 3
    languages = ['en', 'ja', 'zh-cn', 'es', 'fr', 'it', 'pt', 'ko', 'sv', 'da', 'ta', 'mn', 'cy', 'sw', 'zu', 'tk', 'gd']

    if model == 'gpt-4o-mini':
        data_path_template = './result/ft_resist/gpt/epoch-{0}/gpt-4o-mini-2024-07-18-{1}.json'
        save_path_template = './result/ft_resist/gpt/epoch-{0}/accuracy.csv'
    elif model == 'llama':
        data_path_template = './result/ft_resist/llama/epoch-{0}/llama-3.1-8b-instruct-{1}.json'
        save_path_template = './result/ft_resist/llama/epoch-{0}/accuracy.csv'
    
    for epoch in range(1,n_epochs+1):
        with open(save_path_template.format(epoch), mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Language', 'Accuracy (Original)', 'Accuracy (Finetuning)'])
            for language in languages:
                data_path = data_path_template.format(epoch,language)
                json_file = json.load(open(data_path, encoding='utf-8'))
                score_eval = []
                score_train = []
                for obj in json_file:
                    eval_score = obj['eval_score']
                    if 'No' in eval_score and 'Yes' not in eval_score:
                        score_eval.append(0)
                    elif 'Yes' in eval_score and 'No' not in eval_score:
                        score_eval.append(1)
                    else:
                        print("No score found.")
                    
                    train_score = obj['train_score']
                    if 'No' in train_score and 'Yes' not in train_score:
                        score_train.append(0)
                    elif 'Yes' in train_score and 'No' not in train_score:
                        score_train.append(1)
                    else:
                        print("No score found.")
                if len(score_eval) == 50 and len(score_train) == 50:
                    avg_eval_score = np.mean(score_eval) * 100
                    avg_eval_score_formatted = f"{avg_eval_score:.2f}"
                    avg_train_score = np.mean(score_train) * 100
                    avg_train_score_formatted = f"{avg_train_score:.2f}"
                    csv_writer.writerow([language, avg_eval_score_formatted, avg_train_score_formatted])
                else:
                    print("Error!")