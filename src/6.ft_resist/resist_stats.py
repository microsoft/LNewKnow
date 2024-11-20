# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
import json
import numpy as np

if __name__ == '__main__':
    n_epochs = 3
    languages = ['en', 'ja', 'zh-cn', 'es', 'fr', 'it', 'pt', 'ko', 'sv', 'da', 'ta', 'mn', 'cy', 'sw', 'zu', 'tk', 'gd']
    save_path_template = '../../result/ft_resist/resist/epoch-{0}/gpt-4o-mini-2024-07-18-{1}.json'
    output_csv_path_template = '../../result/ft_resist/resist/epoch-{0}/accuracy.csv'

    for epoch in range(1,n_epochs+1):
        with open(output_csv_path_template.format(epoch), mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Language', 'Accuracy (Original)', 'Accuracy (Finetuning)'])
            for language in languages:
                save_path = save_path_template.format(epoch,language)
                json_file = json.load(open(save_path, encoding='utf-8'))
                score_test = []
                score_train = []
                for obj in json_file:
                    test_score = obj['test_score']
                    if 'No' in test_score and 'Yes' not in test_score:
                        score_test.append(0)
                    elif 'Yes' in test_score and 'No' not in test_score:
                        score_test.append(1)
                    else:
                        print("No score found")

                    train_score = obj['train_score']
                    if 'No' in train_score and 'Yes' not in train_score:
                        score_train.append(0)
                    elif 'Yes' in train_score and 'No' not in train_score:
                        score_train.append(1)
                    else:
                        print("No score found")
                if len(score_test) == 50 and len(score_train) == 50:
                    avg_test_score = np.mean(score_test) * 100
                    avg_test_score_formatted = f"{avg_test_score:.2f}"
                    avg_train_score = np.mean(score_train) * 100
                    avg_train_score_formatted = f"{avg_train_score:.2f}"
                    csv_writer.writerow([language, avg_test_score_formatted, avg_train_score_formatted])
                else:
                    print("Error!")