# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import json
import torch
import itertools
import numpy as np
from datasets import Dataset

B_INST, E_INST = "[INST]", "[/INST]"
EOT_ID = 128009 #<|eot_id|>
def mask_target(target,seq):
    for i in range(len(seq)-len(target)):
        if seq[i:i+len(target)] == target:
            seq[i:i+len(target)] = [-100] * len(target)
    return seq

def tokenize_dialog(dialog, tokenizer):
    if tokenizer.vocab_size >= 128000:
        dialog_tokens = tokenizer.apply_chat_template(dialog)
        eot_indices = [i for i,n in enumerate(dialog_tokens) if n == EOT_ID]
        labels = copy.copy(dialog_tokens)
        system_or_user = (tokenizer.encode("system")[-1], tokenizer.encode("user")[-1])
        labels[0] = -100 # bos token
        last_idx = 1
        for n, idx in enumerate(eot_indices):
            role_token = labels[last_idx+1]
            if role_token in system_or_user:
                # Set labels to -100 for system and user tokens to ignore in loss function
                labels[last_idx:idx+1] = [-100] * (idx-last_idx+1)
            last_idx = idx + 1
        mask_target(tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False), labels)
        dialog_tokens = [dialog_tokens]
        labels_tokens = [labels]
    else:
        prompt_tokens = [tokenizer.encode(f"{tokenizer.bos_token}{B_INST} {(prompt['content']).strip()} {E_INST}", add_special_tokens=False) for prompt in dialog[::2]]
        answer_tokens = [tokenizer.encode(f"{answer['content'].strip()} {tokenizer.eos_token}", add_special_tokens=False) for answer in dialog[1::2]]
        dialog_tokens = list(itertools.chain.from_iterable(zip(prompt_tokens, answer_tokens)))

        #Add labels, convert prompt token to -100 in order to ignore in loss function
        labels_tokens = [len(c)*[-100,] if i % 2 == 0 else c for i,c in enumerate(dialog_tokens)]

    combined_tokens = {
        "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
        "labels": list(itertools.chain(*(t for t in labels_tokens))),
    }

    return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))

def tokenize_dialog(dialog, tokenizer):
    if tokenizer.vocab_size >= 128000:
        dialog_tokens = tokenizer.apply_chat_template(dialog)
        dialog_tokens = dialog_tokens[:-4] # Remove generation prompt <|start_header_id|>assistant<|end_header_id|>\n\n
        eot_indices = [i for i,n in enumerate(dialog_tokens) if n == 128009]
        labels = copy.copy(dialog_tokens)
        last_idx = 0
        for n, idx in enumerate(eot_indices):
            if n % 2 == 1:
                last_idx = idx
            else:
                labels[last_idx:idx+1] = [-100] * (idx-last_idx+1)

        dialog_tokens = [dialog_tokens]
        labels_tokens = [labels]
    else:
        prompt_tokens = [tokenizer.encode(f"{tokenizer.bos_token}{B_INST} {(prompt['content']).strip()} {E_INST}", add_special_tokens=False) for prompt in dialog[::2]]
        answer_tokens = [tokenizer.encode(f"{answer['content'].strip()} {tokenizer.eos_token}", add_special_tokens=False) for answer in dialog[1::2]]
        dialog_tokens = list(itertools.chain.from_iterable(zip(prompt_tokens, answer_tokens)))
        labels_tokens = [len(c)*[-100,] if i % 2 == 0 else c for i,c in enumerate(dialog_tokens)]

    combined_tokens = {
        "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
        "labels": list(itertools.chain(*(t for t in labels_tokens))),
    }

    return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))

def get_custom_dataset(dataset_config, tokenizer, split: str, max_length: int=2048, ds_style='hf'):
    with open(dataset_config.data_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    data = [json.loads(line) for line in lines]
    data = [{"messages":[d["messages"][0],d["messages"][1],{"role":"assistant","content":d["messages"][2]["content"]+"\n"+d["messages"][2]["content"]+"\n"+d["messages"][2]["content"]+"\n"+d["messages"][2]["content"]+"\n"+d["messages"][2]["content"]+"\n"+d["messages"][2]["content"]+"\n"+d["messages"][2]["content"]+"\n"+d["messages"][2]["content"]+"\n"+d["messages"][2]["content"]+"\n"}]} for d in data]
    def process_func(example, tokenizer, MAX_LENGTH=max_length):
        return tokenize_dialog(example[1:], tokenizer)
    ds = [process_func(d['messages'], tokenizer=tokenizer) for d in data]
    if ds_style == 'torch':
        class CustomDataset(Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return {key: torch.tensor(np.array(value)) for key, value in self.data[idx].items()}
        dataset = CustomDataset(ds)
    else:
        dataset = Dataset.from_dict({key: [d[key] for d in ds] for key in ds[0]})
    return dataset