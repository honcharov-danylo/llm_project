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
You are a scientist with advanced knowledge in philosophy and social sciences.
Please, continue for the following text.

### Text:
{}

### Response:
"""

logging.info("Loading dataset:")


if config['load_cleaned']:
    inputs = []
    for file in os.listdir(config["data_path"]):
        with gzip.open("{}/{}".format(config["data_path"], file), "rt") as f:
            subsamples = json.load(f)
            inputs.extend(list(subsamples.values()))
else:
    inputs = []
    for file in os.listdir(config["data_path"]):
        with open("{}/{}".format(config["data_path"], file)) as f:
            inputs.append(f.read())
    del inputs[6326] # broken file

inputs = inputs[:100]

inputs_in = [test_prompt_style.format(x[:int(len(x)/2)]) + tokenizer.eos_token for x in inputs]
inputs_out = [x[int(len(x)/2):] for x in inputs]


# inputs_formatted = tokenizer(
#     inputs_in,
#     return_tensors="pt"
# ).to("cuda")

# max_seq_len = tokenizer.model_max_length          # e.g. 4096 for Llama-family, 131 072 for RWKV-world etc.
# inputs_formatted = tokenizer(
#     inputs_in,
#     return_tensors="pt",
#     padding=True,             # or "longest"
#     truncation=True,          # **required** so over-length samples are cut
#     max_length=max_seq_len    # be explicit; you can also set a smaller value
# ).to("cuda")
#
#
# outputs_orig = base.generate(
#     input_ids=inputs_formatted.input_ids,
#     attention_mask=inputs_formatted.attention_mask,
#     max_new_tokens=1200,
#     eos_token_id=tokenizer.eos_token_id,
#     use_cache=True)
#
# outputs = model.generate(
#     input_ids=inputs_formatted.input_ids,
#     attention_mask=inputs_formatted.attention_mask,
#     max_new_tokens=1200,
#     eos_token_id=tokenizer.eos_token_id,
#     use_cache=True
# )

batch_size = config.get("batch_size", 4)       # try smaller if you OOM
# ----------------------------------

def batched(iterable, n):
    """Yield successive n-sized chunks from iterable."""
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

        # === base model ===
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

# replace the old variables so the rest of the script stays unchanged
responses_orig = all_outputs_orig
responses       = all_outputs_ft

# responses_orig = tokenizer.batch_decode(outputs_orig, skip_special_tokens=True)
#
# responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

nlp = spacy.load("en_core_web_sm")

nlp_input_out = [nlp(x) for x in inputs_out]

results = dict()
results["orig"] = dict()
results["orig"]["jaccard"] = [simphile.jaccard_similarity(x, inputs_out[i]) for i,x in enumerate(responses_orig)]
results["orig"]["compression"] = [simphile.compression_similarity(x, inputs_out[i]) for i,x in enumerate(responses_orig)]
results["orig"]["spacy_sim"] = [nlp(x).similarity(nlp_input_out[i]) for i,x in enumerate(responses_orig)]


results["finetuned"] = dict()
results["finetuned"]["jaccard"] = [simphile.jaccard_similarity(x, inputs_out[i]) for i,x in enumerate(responses)]
results["finetuned"]["compression"] = [simphile.compression_similarity(x, inputs_out[i]) for i,x in enumerate(responses)]
results["finetuned"]["spacy_sim"] = [nlp(x).similarity(nlp_input_out[i]) for i,x in enumerate(responses)]

with open("out/out.json", "w") as f:
    json.dump(results, f)