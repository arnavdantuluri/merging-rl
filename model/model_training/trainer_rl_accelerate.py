import os
from typing import List

import argparse
import random
from argparse import Namespace
import torch
import yaml
from trlx.trlx import train
from trlx.data.configs import TRLConfig
import deepspeed
from utils.utils import _strtobool, get_dataset, get_model, read_yamls

import math
from model_training.custom_datasets.formatting import QA_SPECIAL_TOKENS, format_pairs
from datasets import load_from_disk, Dataset
from datasets import load_dataset
from transformers import pipeline
from datasets import load_dataset
from model_training.custom_datasets import get_one_dataset
import model_training.models.reward_model
import random
from argparse import Namespace

import torch.multiprocessing as mp
from utils.ppo_utils import CustomPPOTrainer, CustomPromptPipeline
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import pandas as pd
import datasets
import torch
import random
import yaml
import pathlib
from typing import Dict, List
import json

file_path = "2023-03-13_oasst_ready_labels.jsonl.gz"

def argument_parsing(notebook=False, notebook_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default="config_rl.yaml")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--wandb-entity", type=str, default="open-assistant")

    if notebook:
        args, remaining = parser.parse_known_args(notebook_args)
    else:
        args, remaining = parser.parse_known_args()

    # Config from YAML
    conf = {}
    configs = read_yamls("./configs")
    for name in args.configs:
        if "," in name:
            for n in name.split(","):
                conf.update(configs[n])
        else:
            conf.update(configs[name])

    conf["local_rank"] = args.local_rank

    # Override config from command-line
    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = _strtobool
        parser.add_argument(f"--{key}", type=type_, default=value)

    return parser.parse_args(remaining)

training_conf = argument_parsing()
rank_config = Namespace(**training_conf.rank_config)


QA_SPECIAL_TOKENS_V2_5 = {
    "prompter": "<|prompter|>",
    "assistant": "<|assistant|>",
    "system": "<|system|>",
    "prefix_begin": "<|prefix_begin|>",
    "prefix_end": "<|prefix_end|>",
    "eos": "<|endoftext|>",
}

directory = os.getcwd()
rm_tokenizer = AutoTokenizer.from_pretrained(rank_config.model_name, padding_side=rank_config.padding_side)
rm_model = get_model(rank_config, rm_tokenizer)
rm_model.eval()
rm_device = torch.cuda.device_count() - 1
rm_model = rm_model.to(rm_device)

def format_string(prompt):
    return f"{QA_SPECIAL_TOKENS_V2_5['prompter']}{prompt}{QA_SPECIAL_TOKENS_V2_5['eos']}{QA_SPECIAL_TOKENS_V2_5['assistant']}"

#get prompts from oasst data only; sample can be found in the Open Assistant repository
#assuming no prefix will be fed during RL or SFT training   
def get_prompts(file_path):

    config = Namespace(
        cache_dir="../../../home/ubuntu/data_cache",
    )
    kwargs = {
        # "lang": "en,es,fr,de",
        "lang": "en",
        "top_k": 1,
        "input_file_path": file_path,
        "mode": "rl",
    }
    train, val = get_one_dataset(conf=config, dataset_name="oasst_export", **kwargs)

    prompts = []
    val_prompts = []
    #Need to actually convert these to prompts to be fed into TRLX
    prompts, val_prompts = tuple(
        map(
            lambda x: [
                "".join(format_pairs(x[i][0], rm_tokenizer.eos_token, add_initial_reply_token=True))
                for i in range(len(x))
            ],
            (train, val),
        )
    )
    #Can be used if you want single conversation prompts
    # for i in train.data:
    #     prompts.append(format_string(i[0][0]))
    
    # for i in val.data:
    #     val_prompts.append(format_string(i[0][0]))

    random.shuffle(prompts)
    random.shuffle(val_prompts)

    return prompts, val_prompts

prompts, val_prompts = get_prompts(file_path)

@torch.no_grad()
def rank_model_fn(samples, **kwargs):

    if len(samples) == 0:
        return []
    
    inputs = rm_tokenizer(samples, padding=True, truncation=True, return_tensors="pt").to(rm_device)

    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    mbs = 8
    out = []
    with torch.no_grad():
        for i in range(math.ceil(len(samples) / mbs)):
            batch_ixs = slice(i * mbs, (i + 1) * mbs)
            input_ids = inputs.input_ids[batch_ixs]
            rewards = rm_model(input_ids).logits[:, 0].detach().cpu()
            out.extend(rewards)
    return out

sft_config = Namespace(**training_conf.sft_config)

with open(directory + '/configs/ppo_config.yaml') as f:
    default_config = yaml.safe_load(f)

trlx_config = TRLConfig.update(default_config, {})

trlx_config.tokenizer.tokenizer_path = sft_config.model_name
trlx_config.model.model_path = sft_config.model_name
trlx_config.method.gen_kwargs["max_new_tokens"] = 400
trlx_config.train.batch_size = int(training_conf.batch_size)
trlx_config.method.chunk_size = int(training_conf.chunk_size)
trlx_config.method.num_rollouts = int(training_conf.num_rollouts)
trlx_config.train.epochs = int(training_conf.epochs)

trainer = train(
    sft_config.model_name,
    reward_fn=rank_model_fn,
    prompts=prompts, 
    eval_prompts=val_prompts,
    config=trlx_config,
)

directory = os.getcwd()
trainer.save_pretrained(directory + "/checkpoints/best_checkpoint")