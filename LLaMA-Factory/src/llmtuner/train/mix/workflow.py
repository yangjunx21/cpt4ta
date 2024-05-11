# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/language-modeling/run_clm.py

import math
from typing import TYPE_CHECKING, List, Optional

from transformers import DataCollatorForLanguageModeling, Trainer

from ...data import get_dataset, split_dataset
from ...extras.ploting import plot_loss
from ...model import load_model_and_tokenizer
from ...train.utils import create_modelcard_and_push
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import get_logits_processor
from ...train.sft.metric import ComputeMetrics
from ...train.sft.trainer import CustomSeq2SeqTrainer

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


def run_mix(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    print("loading model and tokenizer")
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train)

    # 准备SFT和PT的数据集
    print("preparing datasets")
    sft_dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="sft")
    pt_dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="pt")

    print("ok")
    breakpoint()
    # # 配置DataCollators
    # sft_data_collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=IGNORE_INDEX)
    # pt_data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # # 初始化Trainer
    # trainer = CustomMixedTrainer(
    #     model=model,
    #     args=training_args,
    #     tokenizer=tokenizer,
    #     sft_data_collator=sft_data_collator,
    #     pt_data_collator=pt_data_collator,
    #     sft_dataset=sft_dataset,
    #     pt_dataset=pt_dataset,
    #     callbacks=callbacks,
    #     compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
    # )

    # # Training
    # if training_args.do_train:
    #     trainer.train()

    # # 你需要自定义CustomMixedTrainer类，以实现混合SFT和PT loss的逻辑。
    # # 这涉及到修改训练循环，以便同时计算两种loss，并根据你定义的策略将它们混合起来。

    # # Evaluation and prediction steps can follow a similar mixed approach or focus on one task, depending on your requirements.
