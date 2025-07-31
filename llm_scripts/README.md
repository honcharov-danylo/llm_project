# llm_scripts

Scripts to **fine-tune** the base model and manage training runs (locally or via HTCondor).  
The current entrypoint is `finetuning_skip.py` which reads `../configs/config_finetuning.json` and logs to Weights & Biases.

---

## Contents

````

llm\_scripts/
├─ finetuning\_skip.py       # main fine-tuning script (uses utils.Config + W\&B)
├─ finetuning\_skip.sub      # HTCondor job file (parametric via EXTRA\_ARGS + LOGFILE)
├─ conda\_wrapper\_fine.sh    # Conda + WANDB\_PROJECT env wrapper for Condor
├─ run\_example.sh           # convenience: condor\_submit with overrides
└─ logs\_finetuning/

````

---

## Dependencies

```bash
conda create -n llm_finetuning python=3.10 -y
conda activate llm_finetuning

pip install -r requirements.txt
# plus training-time extras
pip install wandb lighteval  # lighteval is used by the script pipeline
# Install the right PyTorch for your CUDA
# e.g., pip install torch --index-url https://download.pytorch.org/whl/cu121
````

> `finetuning_skip.py` imports `wandb` and uses `transformers` (4.52.3), `peft` (0.15.2). If you plan to use 8-bit/4-bit loading, also install `bitsandbytes` compatible with your CUDA.

---

## Configuration

Read from `../configs/config_finetuning.json` (overrides allowed from CLI):

* `data_path`, `load_cleaned` – location of cleaned shards from `clean_scripts/`
* `model_dir` – base model id/path (e.g., `Qwen/Qwen2.5-3B-Instruct`)
* `output_dir` – training outputs/checkpoints
* `max_eval_tok`, `predict_sentences` – run behavior
* `WANDB_PROJECT`, `WANDB_LOG_MODEL` – W\&B logging

The script sets:

```python
os.environ["WANDB_PROJECT"]   = config["WANDB_PROJECT"]
os.environ["WANDB_LOG_MODEL"] = config["WANDB_LOG_MODEL"]
```

Make sure you’ve run `wandb login` or set `WANDB_API_KEY`.

---

## Run Locally

```bash
cd llm_scripts

python finetuning_skip.py \
  --model_dir Qwen/Qwen2.5-3B-Instruct \
  --output_dir ./runs/qwen25_ft \
  --data_path ../clean_scripts/cleaned_data
```

**Alternate override form:**

```bash
python finetuning_skip.py model_dir=Qwen/Qwen3-4B output_dir=./runs/qwen3_ft
```

---

## Run on HTCondor

`finetuning_skip.sub` supports two submit-time variables:

* `EXTRA_ARGS` → appended to `python finetuning_skip.py ...`
* `LOGFILE`    → used to name `logs_finetuning/<LOGFILE>.{out,err,log}`

**Quick example (provided):**

```bash
cd llm_scripts
bash run_example.sh
# Internally:
# condor_submit finetuning_skip.sub EXTRA_ARGS="--model_dir Qwen/Qwen3-4B --output_dir qwen3_out" LOGFILE="finetuning_qwen3"
```

**Manual submit:**

```bash
condor_submit finetuning_skip.sub \
  EXTRA_ARGS="--model_dir Qwen/Qwen2.5-3B-Instruct --output_dir runs/qwen25_ft" \
  LOGFILE="qwen25_ft"
```

The job requests one GPU and requires driver ≥ 12.0. Adjust resources in the `.sub` file if necessary.

---

## Outputs

* Training logs (and W\&B if enabled)
* Checkpoints/adapters under your `output_dir`