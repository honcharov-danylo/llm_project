import os
os.environ["WANDB_PROJECT"] = "llm-finetuning-skip-stylo"   # must come before Trainer is built
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
#import subprocess
#subprocess.run("yes | pip install bitsandbytes")
import itertools
import transformers.utils
transformers.utils.is_rich_available = lambda: False

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import nltk
from tqdm import trange, tqdm
import os
from typing import List, Dict, Iterator
from nltk.tokenize import sent_tokenize
import datasets
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, TrainerCallback
from transformers.integrations import WandbCallback
from transformers import Seq2SeqTrainingArguments
from sentence_transformers import SentenceTransformer, util
from transformers import GenerationConfig
import json
import gzip

import gc, torch
import os, tempfile, wandb, json

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from lighteval.models.transformers.transformers_model import TransformersModelConfig

import logging


with open("config.json", 'r') as f:
    config = json.load(f)


def run_lighteval(checkpoint_path, tasks):
    tracker = EvaluationTracker(output_dir="./le_results", save_details=False)

    pipe_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        env_config=None,
    )

    model_cfg = TransformersModelConfig(
        model_name=checkpoint_path,
        dtype="float16",
        use_chat_template=True,
        device_map="cpu",
    )

    pipeline = Pipeline(
        tasks=",".join(tasks),           # e.g. "leaderboard|gsm8k|0|true"
        pipeline_parameters=pipe_params,
        evaluation_tracker=tracker,
        model_config=model_cfg,
    )

    # returns a nested dict with all scores
    return pipeline.evaluate()



class LightEvalCallback(TrainerCallback):
    def __init__(self, tasks, freq=2):
        self.tasks, self.freq = tasks, freq
        self.count = 0

    def on_evaluate(self, args, state, control, **kw):
        self.count += 1
        if self.count % self.freq:
            return                      # only every Nth HF eval

        model, tok = kw["model"], kw["tokenizer"]

        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            tok.save_pretrained(tmp)

            res = run_lighteval(tmp, self.tasks)

        flat = {
            f"lighteval/{t}/{m}": v
            for t, metrics in res["results"].items()
            for m, v in metrics.items()
        }
        wandb.log(flat, step=state.global_step)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

light_tasks = [
    "leaderboard|truthfulqa:mc|0|0",
    "leaderboard|gsm8k|0|true",
]
callbacks = [LightEvalCallback(light_tasks, freq=4)]  # every 3rd eval ⇒ every 1500 steps



model_dir = config["model_dir"]
# model_dir = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, max_length = 50000)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

model.config.use_cache = False
model.config.pretraining_tp = 1


train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 

### Instruction:
You are a scientist with advanced knowledge in philosophy and social sciences. 
Please, write next paragraph for the following text. 

### Text:
{}

### Response:
{}"""

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

# temporary reduction of dataset size for testing
# inputs = inputs[:100]

logging.info("Data loaded.")

def _sample_generator(texts: List[str], start:int, step:int) -> Iterator[Dict[str, str]]:
    for doc in texts:
        sents = sent_tokenize(doc)
        for k in range(start, len(sents) + 1, step):
            yield {
                "question": " ".join(sents[:k]),
                "answer":   " ".join(sents[k:]),
            }

def build_prefixqa_dataset(texts: List[str], start:int, step:int) -> datasets.IterableDataset:
    features = datasets.Features({
        "question": datasets.Value("string"),
        "answer":   datasets.Value("string"),
    })
    return datasets.IterableDataset.from_generator(
        lambda: _sample_generator(texts, start, step),  # generator **factory**
        features=features,
    )


logging.info("Building datasets:")

ds = build_prefixqa_dataset(inputs, 1, 10)
eval_ds = build_prefixqa_dataset(inputs, 5, 20) # create eval in different way with different steps

logging.info("Datasets are built.")

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

style_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

logging.info("Encoding eval dataset:")

sample_train_texts = [ex["answer"]              # or ex["text"] in your format
                      for ex, _ in zip(eval_ds, range(256))]   # take first 4 k

style_bank = style_encoder.encode(
    sample_train_texts,
    batch_size=128,
    normalize_embeddings=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

logging.info("Eval dataset is encoded.")



class LLMSampleCB(WandbCallback):
    def __init__(self, trainer, test_dataset, num_samples = 32, max_new_tokens=256, log_model="checkpoint"):
        super().__init__()
        # self._log_model = log_model
        self.sample_dataset = test_dataset.take(num_samples)
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.gen_config = GenerationConfig.from_pretrained(trainer.model.name_or_path,
                                                           max_new_tokens=max_new_tokens)

    def generate(self, prompt):
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
        with torch.inference_mode():
            output = self.model.generate(tokenized_prompt, generation_config=self.gen_config)
        return self.tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)

    def samples_table(self, examples):
        "Create a wandb.Table to store the generations"
        records_table = wandb.Table(columns=["prompt", "generation"] + list(self.gen_config.to_dict().keys()) + ["style_sim_mean", "style_sim_std"])
        for example in tqdm(examples, leave=False):
            prompt = example["text"]
            generation = self.generate(prompt=prompt)
            gen_emb = style_encoder.encode(
                generation,
                batch_size=64,
                normalize_embeddings=True,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )                                            # shape = [len(gen), 384]
            sims = util.cos_sim(torch.tensor(gen_emb), torch.tensor(style_bank))
            max_sims = sims.max(dim=1).values.cpu().numpy()

            records_table.add_data(prompt, generation, *list(self.gen_config.to_dict().values()), max_sims.mean(), max_sims.std())
        return records_table

    def on_evaluate(self, args, state, control, **kwargs):
        "Log the wandb.Table after calling trainer.evaluate"
        super().on_evaluate(args, state, control, **kwargs)
        records_table = self.samples_table(self.sample_dataset)
        self._wandb.log({"sample_predictions": records_table})




def formatting_prompts_func(examples):
    inputs = examples["question"]
    outputs = examples["answer"]
    texts = []
    for question, response in zip(inputs, outputs):
        # Append the EOS token to the response if it's not already there
        if not response.endswith(tokenizer.eos_token):
            response += tokenizer.eos_token
        text = train_prompt_style.format(question, response)
        texts.append(text)
    return {"text": texts}

logging.info("Formatting datasets:")


def truncate_long_prompts(example):
    ids = tokenizer.encode(example["text"],
                           add_special_tokens=False,
                           truncation=False)

    if len(ids) > 4096:
        ids = ids[:4096]
        example["text"] = tokenizer.decode(ids, skip_special_tokens=True)

    return example



dataset = ds.map(
    formatting_prompts_func,
    batched=True,
)

eval_dataset_mapped = eval_ds.map(
    formatting_prompts_func,
    batched=True,
).map(truncate_long_prompts, batched=True,)


logging.info("Datasets are formatted.")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


logging.info("Loading model:")
# LoRA config
peft_config = LoraConfig(
    lora_alpha=16,                           # Scaling factor for LoRA
    lora_dropout=0.05,                       # Add slight dropout for regularization
    r=64,                                    # Rank of the LoRA update matrices
    bias="none",                             # No bias reparameterization
    task_type="CAUSAL_LM",                   # Task type: Causal Language Modeling
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # Target modules for LoRA
)

model = get_peft_model(model, peft_config)

batch_size = 4
steps = int(500000/batch_size)

eval_dataset = eval_dataset_mapped.take(128)

logging.info("Model loaded. Building training arguments.")
eval_every = int(0.01 * steps)

# Training Arguments
training_arguments = TrainingArguments(
    output_dir="output_skip",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    logging_steps=0.01,
    warmup_steps=10,
    max_steps = steps,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    report_to=["wandb"],
    eval_strategy="steps",
    eval_steps= eval_every, #0.01,
    save_steps= eval_every,#0.01,
    # predict_with_generate=True,
    # generation_max_length=128
)


logging.info("Building Trainer")

# Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset,
    peft_config=peft_config,
    data_collator=data_collator,
    callbacks = callbacks,
    eval_dataset = eval_dataset,
    # compute_metrics=compute_metrics
)
logging.info("Starting training")


wandb_callback = LLMSampleCB(trainer, eval_dataset, num_samples=20, max_new_tokens=256)
trainer.add_callback(wandb_callback)

gc.collect()
torch.cuda.empty_cache()
model.config.use_cache = False
trainer.train()

wandb.finish()