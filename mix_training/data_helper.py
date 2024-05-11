import os
import json
import zipfile
import random
import zipfile
import torch

import numpy as np
from io import BytesIO
from functools import partial
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence

from time import time


def create_dataloaders(args, decoderonly=False):
    with open(args.train_path, 'r', encoding='utf8') as f:
        train_data = json.load(f)
    
    with open(args.valid_path, 'r', encoding='utf8') as f:
        valid_data = json.load(f)

    # print(len(valid_data))
    if not decoderonly:
        train_dataset = SafetyDataset(args, train_data)
        val_dataset = SafetyDataset(args, valid_data)
    
    else:
        train_dataset = SafetyDatasetDecoderOnly(args, train_data)
        val_dataset = SafetyDatasetDecoderOnly(args, valid_data)
    
    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    else:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    # train_sampler = RandomSampler(train_dataset)
    # val_sampler = SequentialSampler(val_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    # train_dataloader = dataloader_class(train_dataset,
    #                                     batch_size=args.batch_size,
    #                                     sampler=train_sampler,
    #                                     drop_last=True)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False)
    return train_dataset, val_dataloader, train_dataset.tokenizer


def create_gen_dataloaders(args):
    with open(args.test_path, 'r', encoding='utf8') as f:
        test_data = json.load(f)
    
    test_dataset = SafetyDataset(args, test_data)
    
    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    else:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    test_sampler = SequentialSampler(test_dataset)
    
    test_dataloader = dataloader_class(test_dataset,
                                      batch_size=args.test_batch_size,
                                      sampler=test_sampler,
                                      drop_last=False)
    return test_dataloader, test_dataset.data, test_dataset.tokenizer

class SafetyDataset(Dataset):

    def __init__(self,
                 args,
                 data,
                 ):
       
        self.max_input_length = args.max_input_length

        self.args = args
        self.data = data
        # initialize the text tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
        
        if args.incontext_learn:
            self.type2idxs = {}
            for i, d in enumerate(self.data):
                tasktype = d['tasktype']
                if tasktype not in self.type2idxs:
                    self.type2idxs[tasktype] = []
                self.type2idxs[tasktype].append(i)
        

    def __len__(self) -> int:
        return len(self.data)

    def tokenize_text(self, text: str, max_length) -> tuple:
        encoded_inputs = self.tokenizer(text, max_length=max_length, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask

    def __getitem__(self, idx: int) -> dict:
        d = self.data[idx]
        
        if self.args.instruct_type == 0:
            text_input = d['input']
        elif self.args.instruct_type == 1:
            # text_input = d['input'].split('\n\n')[1]
            splits = d['input'].split('\n\n')
            # splits[0] = ''
            text_input = '\n\n'.join(splits[1:]).strip()
        
        if self.args.incontext_learn:
            # first decide whether to add examples
            if random.random() < 0.1:
                # add examples
                example_num = random.randint(1, 2) # maybe exceed max input length
                tasktype = d['tasktype']
                all_idxs = self.type2idxs[tasktype]
                
                all_idxs.remove(idx)
                sample_idxs = random.sample(all_idxs, example_num)
                examples = []
                
                for sample_idx in sample_idxs:
                    sampled = self.data[sample_idx]
                    example = f'{sampled["input"]}{sampled["output"]}'
                    examples.append(example)
                
                text_input = '\n\n'.join(examples) + '\n\n' + text_input
                
        
        input_ids, input_mask = self.tokenize_text(text_input, self.args.max_input_length)
        
        if 'output' in d:
            output_text = d['output']
            
            output_ids, _ = self.tokenize_text(output_text, self.args.max_output_length)
            output_ids[output_ids == self.tokenizer.pad_token_id] = -100
            

            # Step 3, summarize into a dictionary
            data = dict(
                input_ids=input_ids,
                input_mask=input_mask,
                output_ids=output_ids
            )

            return data
        
        else:
            data = dict(
                input_ids=input_ids,
                input_mask=input_mask,
            )
            
            return data
        

class SafetyDatasetDecoderOnly(Dataset):
    
    # for decoder-only models like gpt and llama

    def __init__(self,
                 args,
                 data,
                 loss_type
                 ):
       
        self.max_ength = args.max_length

        self.args = args
        self.data = data
        self.loss_type = loss_type
        # initialize the text tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False, trust_remote_code=True)
        
        if args.incontext_learn:
            self.type2idxs = {}
            for i, d in enumerate(self.data):
                tasktype = d['tasktype']
                if tasktype not in self.type2idxs:
                    self.type2idxs[tasktype] = []
                self.type2idxs[tasktype].append(i)
        
    def __len__(self) -> int:
        return len(self.data)

    def tokenize_text(self, text: str, max_length, padding='max_length', add_special_tokens=False) -> tuple:
        encoded_inputs = self.tokenizer(text, add_special_tokens=add_special_tokens, max_length=max_length, padding=padding, truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask

    def __getitem__(self, idx: int) -> dict:
        d = self.data[idx]
        
        text_input = d['prompt']
        
        if 'response' in d:
            output_text = d['response']
            
            input_ids, _  = self.tokenize_text(text_input, self.args.max_length, padding=False, add_special_tokens=False)
            # input_ids = torch.cat([input_ids, torch.tensor([self.tokenizer.eos_token_id])], dim=-1)
            output_ids, _ = self.tokenize_text(output_text + self.tokenizer.eos_token, self.args.max_length, padding=False, add_special_tokens=False)
            concat_input_ids = torch.cat([input_ids, output_ids], dim=-1)
            tot_max_len = self.args.max_length
            # print(f'id len:{len(concat_input_ids)}')
            if len(concat_input_ids) < tot_max_len:
                padded_tokens = torch.full((tot_max_len - len(concat_input_ids), ), fill_value=self.tokenizer.eos_token_id)
                padded_input_ids = torch.cat([concat_input_ids, padded_tokens], dim=-1)
            else:
                padded_input_ids = concat_input_ids[:tot_max_len]
            
            # output_ids[output_ids == self.tokenizer.pad_token_id] = -100
            output_ids = padded_input_ids.clone()
            # output_ids[output_ids == self.tokenizer.pad_token_id] = -100
            concat_len = len(concat_input_ids)
            output_ids[concat_len:] = -100
            
            input_len = len(input_ids)
            output_ids[:input_len] = -100

            # Step 3, summarize into a dictionary
            # print(padded_input_ids, output_ids)
            if self.loss_type == "GD":
                data = dict(
                    input_ids=padded_input_ids,
                    labels=output_ids
                )
            else:
                data = dict(
                    input_ids=padded_input_ids,
                    labels=output_ids,
                    loss_type=d['type']
                )

            return data
        
        else:
            raise NotImplementedError
            input_ids, input_mask = self.tokenize_text(text_input, self.args.max_input_length)

            data = dict(
                input_ids=input_ids,
                input_mask=input_mask,
            )
            
            return data
