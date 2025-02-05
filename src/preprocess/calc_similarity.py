# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def read_jsonl(data_path):
    with open(data_path,'r',encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def extract_texts(data):
    return ["Role: {}\nContent: {}\nRole: {}\nContent: {}\nRole: {}\nContent: {}\n".format(entry["messages"][0]["role"],entry["messages"][0]["content"],entry["messages"][1]["role"],entry["messages"][1]["content"],entry["messages"][2]["role"],entry["messages"][2]["content"]) for entry in data]

def calculate_similarity(responses1, responses2):
    embeddings1 = model.encode(responses1, convert_to_tensor=True)
    embeddings2 = model.encode(responses2, convert_to_tensor=True)
    cosine_similarities = util.cos_sim(embeddings1, embeddings2)
    return cosine_similarities

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    dataset = args.dataset
    train_path = './dataset/{0}/train/train_en.jsonl'.format(dataset)
    back_translation_path_format = './dataset/{0}/train_back_translation/train_{1}.jsonl'
    languages = ['zh-cn','ja','es','fr','it','pt','ko','sv','da','ta','mn','cy','sw','zu','tk','gd']
    for lang in languages:
        back_translation_path = back_translation_path_format.format(dataset,lang)
        train_data = read_jsonl(train_path)
        back_translation_data = read_jsonl(back_translation_path)
        train_texts = extract_texts(train_data)
        back_translation_texts = extract_texts(back_translation_data)
        similarities = calculate_similarity(train_texts, back_translation_texts)
        scores = []
        for i, score in enumerate(similarities.diagonal()):
            scores.append(float(score))
        print("Language: {}; Score: {}.".format(lang,np.mean(scores)))