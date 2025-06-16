import os
os.environ["WANDB_PROJECT"] = "llm-finetuning-long"   # must come before Trainer is built

#import subprocess
#subprocess.run("yes | pip install bitsandbytes")
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
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
import gc, torch
import os, tempfile, wandb, json

from lighteval import evaluator

class LightEvalCallback(TrainerCallback):
    """
    Run LightEval every `freq` evaluation events (those happen every
    `eval_steps`) and push the benchmark scores to wandb.
    """
    def __init__(self, tasks, freq=2, threads=8, batch_size=4):
        self.tasks     = tasks
        self.freq_eval = freq
        self.kwargs    = dict(num_threads=threads, batch_size=batch_size)
        self.counter   = 0

    # called right after the normal HF evaluation loop
    def on_evaluate(self, args, state, control, **kwargs):
        self.counter += 1
        if self.counter % self.freq_eval:          # skip until it’s time
            return

        model, tokenizer = kwargs["model"], kwargs["tokenizer"]

        # —— 1. snapshot current weights to a temp dir ——
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            tokenizer.save_pretrained(tmp)

            # —— 2. run LightEval on that checkpoint ——
            res = evaluator.evaluate(
                model_id   = tmp,          # any transformers checkpoint path works
                tasks      = self.tasks,   # e.g. ["leaderboard|mmlu|0|true", ...]
                **self.kwargs
            )

        # —— 3. flatten & stream to W&B ——
        flat = {}
        for task, metric_dict in res["results"].items():   # 'results' key in Lighteval ≥0.9
            for k, v in metric_dict.items():
                flat[f"lighteval/{task}/{k}"] = v

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
callbacks = [LightEvalCallback(light_tasks, freq=500)]  # every 3rd eval ⇒ every 1500 steps



model_dir = "Qwen/Qwen2.5-3B-Instruct"
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


inputs = []
for file in os.listdir("data/"):
    with open("data/{}".format(file)) as f:
        inputs.append(f.read())
del inputs[6326] # broken file


def create_new_samples(text):
    tokenized_text = nltk.tokenize.sent_tokenize(text, language='english')
    return [[{"Question":"".join(t[:i]), "Response":"".join(t[:i])} for i in range(len(t))] for t in tokenized_text]




def _sample_generator(texts: List[str]) -> Iterator[Dict[str, str]]:
    for doc in texts:
        sents = sent_tokenize(doc)
        for k in range(1, len(sents) + 1):
            yield {
                "question": " ".join(sents[:k]),
                "answer":   " ".join(sents[k:]),
            }

def build_prefixqa_dataset(texts: List[str]) -> datasets.IterableDataset:
    features = datasets.Features({
        "question": datasets.Value("string"),
        "answer":   datasets.Value("string"),
    })
    return datasets.IterableDataset.from_generator(
        lambda: _sample_generator(texts),  # generator **factory**
        features=features,
    )


ds = build_prefixqa_dataset(inputs)

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

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

dataset = ds.map(
    formatting_prompts_func,
    batched=True,
)



data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)



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
steps = int(1000000/batch_size)

# Training Arguments
training_arguments = TrainingArguments(
    output_dir="output",
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
    evaluation_strategy="steps",
    eval_steps=0.01,
    save_steps=0.01
)

# Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset,
    peft_config=peft_config,
    data_collator=data_collator,
    callbacks = callbacks
)

gc.collect()
torch.cuda.empty_cache()
model.config.use_cache = False
trainer.train()
