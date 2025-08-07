import os


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

from dotenv import load_dotenv
from b2_uploader import B2Uploader
import time
from pathlib import Path

import gc, torch
import os, tempfile, wandb, json

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from lighteval.models.transformers.transformers_model import TransformersModelConfig
import argparse, pathlib, sys
import logging


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils import Config

config = Config("../configs/config_finetuning.json")
os.environ["WANDB_PROJECT"] = config["WANDB_PROJECT"]   # must come before Trainer is built
os.environ["WANDB_LOG_MODEL"] = config["WANDB_LOG_MODEL"]


path_to_out = Path(config["output_dir"])
path_to_out.mkdir(exist_ok=True)

def run_lighteval(checkpoint_path, tasks):
    """
    Run LightEval evaluation on a model checkpoint.
    
    Evaluates the model on specified tasks using the LightEval framework.
    The evaluation is performed using CPU to avoid GPU memory conflicts.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint to evaluate
        tasks (list): List of task specifications to evaluate on
        
    Returns:
        dict: Nested dictionary containing evaluation results for all tasks
        
    Examples:
        >>> tasks = ["leaderboard|gsm8k|0|true"]
        >>> results = run_lighteval("./checkpoint", tasks)
        >>> print(results["results"]["leaderboard"]["gsm8k"])
    """
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
    """
    Custom callback for running LightEval evaluations during training.
    
    This callback periodically evaluates the model on specified tasks using
    the LightEval framework and logs the results to Weights & Biases.
    
    Attributes:
        tasks (list): List of task specifications for evaluation
        freq (int): Frequency of evaluation (every N evaluations)
        count (int): Counter for tracking evaluation calls
    """
    
    def __init__(self, tasks, freq=2):
        """
        Initialize the LightEval callback.
        
        Args:
            tasks (list): List of task specifications for evaluation
            freq (int, optional): Frequency of evaluation. Defaults to 2.
        """
        self.tasks, self.freq = tasks, freq
        self.count = 0

    def on_evaluate(self, args, state, control, **kw):
        """
        Called after each evaluation step during training.
        
        Runs LightEval evaluation every N evaluations and logs results to W&B.
        
        Args:
            args: Training arguments
            state: Training state
            control: Training control object
            **kw: Additional keyword arguments including 'model' and 'tokenizer'
        """
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
# callbacks = [LightEvalCallback(light_tasks, freq=1000000)]
callbacks = []



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
    """
    Generate training samples from a list of texts.
    
    Creates question-answer pairs by splitting texts into sentences and
    using different portions as context and target.
    
    Args:
        texts (List[str]): List of input texts
        start (int): Starting index for sentence selection
        step (int): Step size for sentence selection
        
    Yields:
        Dict[str, str]: Dictionary with "question" and "answer" keys
        
    Examples:
        >>> texts = ["This is sentence one. This is sentence two. This is sentence three."]
        >>> for sample in _sample_generator(texts, 1, 2):
        ...     print(sample)
        {'question': 'This is sentence one.', 'answer': 'This is sentence two.'}
    """
    for doc in texts:
        sents = sent_tokenize(doc)
        for k in range(start, len(sents) + 1, step):
            yield {
                "question": " ".join(sents[:k]),
                "answer":   " ".join(sents[k:k + int(config["predict_sentences"])]),
            }

def build_prefixqa_dataset(texts: List[str], start:int, step:int) -> datasets.IterableDataset:
    """
    Build a prefix-QA dataset from a list of texts.
    
    Creates an iterable dataset for training where the model learns to
    predict the next sentence(s) given previous sentences as context.
    
    Args:
        texts (List[str]): List of input texts
        start (int): Starting index for sentence selection
        step (int): Step size for sentence selection
        
    Returns:
        datasets.IterableDataset: Dataset with question-answer pairs
        
    Examples:
        >>> dataset = build_prefixqa_dataset(texts, 1, 10)
        >>> for example in dataset:
        ...     print(example)
    """
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
    """
    Custom callback for logging sample predictions during training.
    
    This callback generates sample predictions during evaluation and logs
    them to Weights & Biases along with style similarity metrics.
    
    Attributes:
        sample_dataset: Dataset to sample from for generation
        model: The language model
        tokenizer: The tokenizer
        chunk_size (int): Batch size for generation
        gen_config: Generation configuration
    """
    
    def __init__(self, trainer, test_dataset, chunk_size=4,    num_samples = 32, max_new_tokens=256, log_model="checkpoint"):
        """
        Initialize the LLM sample callback.
        
        Args:
            trainer: The trainer object
            test_dataset: Dataset to sample from
            chunk_size (int, optional): Batch size for generation. Defaults to 4.
            num_samples (int, optional): Number of samples to generate. Defaults to 32.
            max_new_tokens (int, optional): Maximum new tokens to generate. Defaults to 256.
            log_model (str, optional): Model logging strategy. Defaults to "checkpoint".
        """
        super().__init__()
        # self._log_model = log_model
        self.sample_dataset = test_dataset.take(num_samples)
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.chunk_size = chunk_size
        self.gen_config = GenerationConfig.from_pretrained(trainer.model.name_or_path,
                                                           max_new_tokens=max_new_tokens)

    def generate(self, prompt):
        """
        Generate text from a single prompt.
        
        Args:
            prompt (str): Input prompt for generation
            
        Returns:
            str: Generated text
        """
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
        with torch.inference_mode():
            output = self.model.generate(tokenized_prompt, generation_config=self.gen_config)
        return self.tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)

    @torch.inference_mode()
    def _generate_chunk(self, prompts):
        """
        Generate text for a batch of prompts.
        
        Args:
            prompts (List[str]): List of input prompts
            
        Returns:
            List[str]: List of generated texts
        """
        tok = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        outs = self.model.generate(**tok, generation_config=self.gen_config)
        return self.tokenizer.batch_decode(
            outs[:, tok["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

    def samples_table(self):
        """
        Create a W&B table with sample predictions and style similarity metrics.
        
        Generates predictions for sample prompts, computes style similarity
        against a style bank, and creates a table for logging.
        
        Returns:
            wandb.Table: Table containing prompts, generations, and metrics
        """
        prompts = [ex["text"] for ex in self.sample_dataset]
        gens = []  # will collect all generations

        for start in range(0, len(prompts), self.chunk_size):
            sub = prompts[start:start + self.chunk_size]
            gens.extend(self._generate_chunk(sub))

        gen_emb = []
        for start in trange(0, len(gens), self.chunk_size):
            sub = gens[start:start + self.chunk_size]
            gen_emb.extend(
                style_encoder.encode(
                    sub,
                    batch_size=self.chunk_size,
                    normalize_embeddings=True,
                    device="cuda",
                )
            )
        gen_emb = torch.tensor(gen_emb, device="cuda")  # (N, 384)

        sims = util.cos_sim(gen_emb, torch.tensor(style_bank, device="cuda"))
        max_sims = sims.max(dim=1).values.cpu().numpy()

        cols = ["prompt", "generation"] \
               + list(self.gen_config.to_dict().keys()) \
               + ["style_sim_mean", "style_sim_std"]
        table = wandb.Table(columns=cols)
        cfg_vals = list(self.gen_config.to_dict().values())

        for p, g, s in zip(prompts, gens, max_sims):
            table.add_data(p, g, *cfg_vals, float(s), 0.0)

        return table

    def on_evaluate(self, args, state, control, **kwargs):
        """
        Called after each evaluation step to log sample predictions.
        
        Args:
            args: Training arguments
            state: Training state
            control: Training control object
            **kwargs: Additional keyword arguments
        """
        super().on_evaluate(args, state, control, **kwargs)
        self._wandb.log(
            {"sample_predictions": self.samples_table()},
            step=state.global_step,
        )
    # def on_evaluate(self, args, state, control, **kwargs):
    #     "Log the wandb.Table after calling trainer.evaluate"
    #     super().on_evaluate(args, state, control, **kwargs)
    #     records_table = self.samples_table(self.sample_dataset)
    #     self._wandb.log({"sample_predictions": records_table})




def formatting_prompts_func(examples):
    """
    Format examples into training prompts.
    
    Converts question-answer pairs into formatted prompts using the
    training prompt template and adds EOS tokens.
    
    Args:
        examples (dict): Dictionary containing "question" and "answer" keys
        
    Returns:
        dict: Dictionary with formatted "text" key
        
    Examples:
        >>> examples = {"question": ["Hello"], "answer": ["World"]}
        >>> result = formatting_prompts_func(examples)
        >>> print(result["text"][0])
    """
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


def truncate_long_prompts(batch):
    """
    Truncate prompts that exceed the maximum token limit.
    
    Args:
        batch (dict): Batch containing "text" key with prompts
        
    Returns:
        dict: Batch with truncated prompts
        
    Examples:
        >>> batch = {"text": ["very long prompt..."]}
        >>> result = truncate_long_prompts(batch)
    """
    trimmed = []
    for txt in batch["text"]:                 # txt is a string
        ids = tokenizer.encode(
            txt,
            add_special_tokens=False,
            truncation=False,
        )
        if len(ids) > config["max_eval_tok"]:
            ids = ids[:config["max_eval_tok"]]
            txt = tokenizer.decode(ids, skip_special_tokens=True)
        trimmed.append(txt)
    return {"text": trimmed}


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
eval_every = int(0.1 * steps)

output_dir = config["output_dir"]

# Training Arguments
training_arguments = TrainingArguments(
    output_dir=output_dir,
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
    save_steps= 1000, #eval_every,#0.01,
    disable_tqdm=False
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


# wandb_callback = LLMSampleCB(trainer, eval_dataset, chunk_size = 2, num_samples=16, max_new_tokens=256)
# trainer.add_callback(wandb_callback)

trainer.evaluate()

gc.collect()
torch.cuda.empty_cache()
model.config.use_cache = False
trainer.train()

final_model_path = "{}/final_model/".format(output_dir)
trainer.save_model(final_model_path)

output_dir_last = "{}/".format(output_dir)

load_dotenv()
uploader = B2Uploader()
uploader.upload_file(final_model_path, "{}_{}".format(config["model_dir"], int(time.time())))

wandb.finish()