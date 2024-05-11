import random
import argparse

import numpy as np
import json
import torch
from config import parse_args
from pytorch_lightning import seed_everything
import copy

from data_helper import SafetyDatasetDecoderOnly
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from transformers import DataCollatorWithPadding, DefaultDataCollator

from trainers import GA_GD_Trainer, GA_GD_KL_Trainer, GA_GD_GD_Trainer

# def get_args():
#     parser = argparse.ArgumentParser()
    
#     # for distributed launcher
#     parser.add_argument("--local_rank", type=int, default=0)
    
#     parser.add_argument("--model_config", type=str)
#     parser.add_argument("--output_dir", type=str,)
#     parser.add_argument("--data_path", type=str)
    
#     parser.add_argument("--batch_size", type=int, default=16)
#     parser.add_argument("--gradient_accumulation", type=int, default=1)
#     parser.add_argument("--lr", type=float, default=1e-6)
    
#     parser.add_argument("--eval_step", type=int, default=100)

    
#     return parser.parse_args()

# class BalancedDataCollator(DefaultDataCollator):
#     def __call__(self, features):
#         # seperate type
        # type_harmful_gd_features = [feature for feature in features if feature['loss_type'] == 2]
        # type_harmful_ga_features = [feature for feature in features if feature['loss_type'] == 1]
        # type_benign_gd_features = [feature for feature in features if feature['loss_type'] == 0]
        
        # print(f"benign_numbers：{len(type_benign_gd_features)}")
        # print(f"harmful_ga_numbers：{len(type_harmful_ga_features)}")
        # print(f"harmful_align_numbers：{len(type_harmful_gd_features)}")
        # min_length = min(len(type_benign_gd_features), len(type_harmful_ga_features), len(type_harmful_gd_features))
        # balanced_features = []
        # if min_length > 0:
        #     print(min_length)
        #     for harmful_gd, harmful_ga, benign_gd in zip(type_harmful_gd_features, type_harmful_ga_features, type_benign_gd_features):
        #         # print(harmful_gd)
        #         harmful_gd["loss_type"] = 1
        #         balanced_features.append(harmful_gd)
        #         harmful_ga["loss_type"] = -1
        #         balanced_features.append(harmful_ga)
        #         benign_gd["loss_type"] = 1
        #         balanced_features.append(benign_gd)
        
#         print(len(balanced_features))
#         return default_data_collator(balanced_features)
    
if __name__ == "__main__":
    
    args = parse_args()
    
    output_dir = args.savedmodel_path
    train_batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation
    learning_rate = args.learning_rate
    eval_batch_size = args.val_batch_size
    eval_steps = args.eval_step
    save_steps = args.save_step
    num_train_epochs = args.max_epochs
    warmup_steps = args.warmup_steps
    ds_config = args.ds_config
    seed_everything(args.seed)
    
    # setting loss type
    loss_type = args.loss_type
    # if args.loss_type not in ["GD", "GA", "GA_GD", "GA_KL", "KL", "GA_GD_KL"]:
        # raise ValueError(f"Invalid loss type: {args.loss_type}. Valid types are: ['GD', 'GA', 'GA_GD', 'GA_KL', 'KL']")
    
    print(f"using loss type: {args.loss_type}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False, trust_remote_code=True)

    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.unk_token
    #     tokenizer.pad_token_id = tokenizer.unk_token_id
    # print('pad token = ', tokenizer.pad_token)

    # Set up the datasets
    with open(args.train_path, 'r', encoding='utf8') as f:
        train_data = json.load(f)
    
    with open(args.valid_path, 'r', encoding='utf8') as f:
        valid_data = json.load(f)
        
    train_dataset = SafetyDatasetDecoderOnly(args, train_data, loss_type)
    dev_dataset = SafetyDatasetDecoderOnly(args, valid_data, loss_type)

    # example
    # print(train_dataset[0])
    # input_text = tokenizer.decode(train_dataset[0]['input_ids'].tolist())
    # labels = tokenizer.decode([id for id in train_dataset[0]['labels'].tolist() if id >= 0])
    # print("label:",train_dataset[0]['type'])
    # print(f'input: {input_text}\n\nlabels: {labels}')
    # print(dev_dataset[0])
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    
    model = AutoModel.from_pretrained(args.model_dir, config=config, trust_remote_code=True)
    # pretrain_model = copy.deepcopy(model)
    # model.resize_token_embeddings(len(tokenizer)) # FIXME: 可以不需要
    # model.config.end_token_id = tokenizer.eos_token_id
    
    # assert model.config.pad_token_id is not None

    # Create a preprocessing function to extract out the proper logits from the model output
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    # balanced_data_collator = BalancedDataCollator(tokenizer)
    
    # Prepare the trainer and start training
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_checkpointing=True,
        half_precision_backend='auto',
        # fp16=True,
        bf16=True,
        adam_beta1=0.9,
        adam_beta2=0.95,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        warmup_steps=warmup_steps,
        evaluation_strategy="no",
        eval_accumulation_steps=1,
        # eval_steps=eval_steps,
        save_strategy='epoch',
        # save_strategy='no',
        save_only_model=True,
        # save_steps=save_steps,
        report_to='tensorboard',
        load_best_model_at_end=False,
        logging_steps=1,
        remove_unused_columns=False,
        deepspeed=ds_config,
    )

    if loss_type == "GA_GD":
        # exit()
        trainer = GA_GD_Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            # compute_metrics=compute_metrics,
            # data_collator=sft_collator,
            pretrain_model=pretrain_model,
            data_collator=default_data_collator,
            # data_collator=balanced_data_collator,
            # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            theta_GA=args.theta_GA,
            theta_GD=args.theta_GD
        )
    elif loss_type == 'GA_GD_GD':
        trainer = GA_GD_GD_Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            # compute_metrics=compute_metrics,
            # data_collator=sft_collator,
            data_collator=default_data_collator,
            # data_collator=balanced_data_collator,
            # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            theta_GA=args.theta_GA,
            theta_GD=args.theta_GD,
            theta_KL=args.theta_KL,
            eos_index=tokenizer.eos_token_id
        )
    
    elif loss_type == 'GA_GD_KL':    
        trainer = GA_GD_KL_Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            # compute_metrics=compute_metrics,
            # data_collator=sft_collator,
            pretrain_model=pretrain_model,
            data_collator=default_data_collator,
            # data_collator=balanced_data_collator,
            # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            theta_GA=args.theta_GA,
            theta_GD=args.theta_GD,
            theta_KL=args.theta_KL
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            # compute_metrics=compute_metrics,
            # data_collator=sft_collator,
            data_collator=default_data_collator,
            # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(output_dir)
