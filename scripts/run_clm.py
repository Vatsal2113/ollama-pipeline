#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Basic causal language modeling script for finetuning LLMs on custom data
# Simplified version of the Hugging Face example script

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

@dataclass
class DataTrainingArguments:
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization."
        },
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    
    # Set seed before initializing model
    set_seed(training_args.seed)

    # Load config, tokenizer, and model
    config = AutoConfig.from_pretrained(model_args.config_name or model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )

    # Load dataset
    data_files = {"train": data_args.train_file}
    extension = data_args.train_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)

    # Preprocess datasets
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    block_size = data_args.block_size or tokenizer.model_max_length
    
    def group_texts(examples):
        # Concatenate all texts and split into blocks of block_size
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        tokenizer=tokenizer,
    )

    # Training
    if training_args.resume_from_checkpoint:
        checkpoint = training_args.resume_from_checkpoint
    else:
        checkpoint = get_last_checkpoint(training_args.output_dir)

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    
    # Metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

if __name__ == "__main__":
    main()