import os
os.environ["WANDB_PROJECT"] = "llm-finetuning-skip"   # must come before Trainer is built
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

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from lighteval.models.transformers.transformers_model import TransformersModelConfig


def run_lighteval(checkpoint_path, tasks):
    tracker = EvaluationTracker(output_dir="./le_results", save_details=False)

    pipe_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        env_config=None,                 # uses HF_CACHE by default
    )

    model_cfg = TransformersModelConfig(
        model_name=checkpoint_path,      # local dir with snapshot from your callback
        dtype="float16",
        use_chat_template=True,
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
    return [[{"Question":"".join(t[:i]), "Response":"".join(t[:i])} for i in range(0,len(t),10)] for t in tokenized_text]




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
