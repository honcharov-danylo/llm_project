from transformers import AutoModelForCausalLM
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import os
import logging
import gzip
import simphile
from itertools import islice
import spacy
from tqdm import tqdm
import torch
import faststylometry
from faststylometry import Corpus, tokenise_remove_pronouns_en, calculate_burrows_delta, predict_proba, calibrate
import nltk
import datasets
from datasets import load_dataset
import pandas as pd
import ast

nltk.download("punkt")


# device = "cpu" # can be "cpu" or "cuda
# inference on cuda takes too much memory

with open("config.json", 'r') as f:
    config = json.load(f)

base = AutoModelForCausalLM.from_pretrained(config["model_dir"], device_map="auto")

model = PeftModel.from_pretrained(base, config["finetuned_path"]).to("cuda")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained(config["finetuned_path"], use_fast=True)

test_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
You are a scientist, proficient in logic. Read next premises and evaluate whether following statement is true, false, or uncertain, based solely on the premises. 
Your response can be only true, false or uncertain.

### Premises:
{}

### Statement:
{}

### Response:
Statement above is: """

logging.info("Loading dataset:")


inputs = pd.read_csv(config['logic_data'])

inputs = inputs[inputs["depth"]<config["eval_depth_logic"]]
inputs['depth'] = inputs['depth'].astype(int)

inputs['premises'] = inputs['premises'].apply(lambda x:ast.literal_eval(x))



inputs = inputs.groupby('depth', group_keys=False).sample(config.get("eval_size_logic", 32), replace=False, random_state=0)

inputs["formatted_input"] = inputs.apply(lambda x: test_prompt_style.format("\n".join(x["premises"]), x['question']), axis = 0)
inputs["formatted_output"] = inputs["formatted_input"] + inputs["label"].str.lower()

inputs_in = inputs["formatted_input"].tolist()
inputs_out = inputs["label"].str.lower().tolist()

batch_size = config.get("batch_size_eval", 4)

def batched(iterable, n):
    it = iter(iterable)
    while (chunk := list(islice(it, n))):
        yield chunk

all_outputs_orig, all_outputs_ft = [], []

base.eval()
model.eval()

with torch.no_grad():                          # no grads for inference
    for chunk in tqdm(batched(inputs_in, batch_size)):
        encoded = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length
        ).to("cuda")

        outs_base = base.generate(
            **encoded,
            max_new_tokens=1200,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
        all_outputs_orig.extend(
            tokenizer.batch_decode(outs_base, skip_special_tokens=True)
        )

        # === finetuned (PEFT) model ===
        outs_ft = model.generate(
            **encoded,
            max_new_tokens=1200,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
        all_outputs_ft.extend(
            tokenizer.batch_decode(outs_ft, skip_special_tokens=True)
        )

responses_orig = all_outputs_orig
responses       = all_outputs_ft

nlp = spacy.load("en_core_web_md")

nlp_input_out = [nlp(x) for x in inputs_out]

results = dict()
results["inputs"] = inputs_in
results["ground_truth"] = inputs_out
results["responses_orig"] = responses_orig
results["responses_ft"] = responses

with open("out/out_logic.json", "w") as f:
    json.dump(results, f)