Evaluation pipelines and parameter sweeps for comparing **base** vs **finetuned** models across similarity metrics and stylometry signals. Outputs are written to `out/` and `out_params/`.

---

## Contents

````

eval\_scripts/
├─ eval.py             # main evaluation (base vs finetuned; writes out/out.json)
├─ eval\_params.py      # sweeps generation params; writes out\_params/\*.json
├─ eval\_logic.py       # logic-style dataset evaluation; writes out/out\_logic.json
├─ eval.sub            # HTCondor job for eval.py
├─ eval\_params.sub     # HTCondor job for eval\_params.py
├─ eval\_logic.sub      # HTCondor job for eval\_logic.py
├─ conda\_wrapper\_eval.sh    # Conda activation + WANDB\_PROJECT export
├─ analysis.ipynb
├─ logic\_analysis.ipynb
├─ logs\_eval/          # Condor stdout/err
└─ out/, out\_params/   # JSON results

````

---

## Dependencies

In addition to repo root requirements, install:

```bash
conda create -n eval_model python=3.10 -y
conda activate eval_model

pip install torch transformers peft datasets tqdm simphile faststylometry spacy nltk
python -m spacy download en_core_web_md
````

> The scripts import `faststylometry` (Burrows’ Delta), `simphile` (Jaccard/Compression similarity), `spacy` (semantic similarity), `peft` (adapter loading), and HF `datasets`.

---

## Configuration

These scripts read `../configs/config_eval.json` via `utils.Config`. Key fields used:

* `model_dir` (base model id/path)
* `finetuned_path` (adapter directory for PEFT)
* `data_path`, `logic_data` (input sets)
* `eval_*`, `batch_size`, `batch_size_eval`

You can **override** any field from the CLI (both `--k v` and `k=v` forms):

```bash
python eval.py --model_dir Qwen/Qwen2.5-3B-Instruct --finetuned_path ./trained_models/1
```

---

## Run Locally

**Main evaluation:**

```bash
cd eval_scripts
python eval.py \
  --model_dir Qwen/Qwen2.5-3B-Instruct \
  --finetuned_path ../trained_models/1 \
  --eval_size 256 --batch_size_eval 16
# → writes out/out.json
```

**Parameter sweep (e.g., temperature, repetition penalty):**

```bash
python eval_params.py \
  --model_dir Qwen/Qwen2.5-3B-Instruct \
  --finetuned_path ../trained_models/1
# → writes out_params/out_params_<temperature>_<repetition>.json
```

**Logic dataset evaluation:**

```bash
python eval_logic.py \
  --model_dir Qwen/Qwen2.5-3B-Instruct \
  --finetuned_path ../trained_models/1 \
  --logic_data ../logic_dataset/test_dataset.csv
# → writes out/out_logic.json
```

---

## Run on HTCondor

The job files request one GPU and require driver ≥ 12.0.

**Submit:**

```bash
cd eval_scripts
condor_submit eval.sub         # runs eval.py
condor_submit eval_params.sub  # runs eval_params.py
condor_submit eval_logic.sub   # runs eval_logic.py
```

**Customize:**

* If you need to change parameters, edit the Python command in the job file (or modify `config_eval.json`).
* Adjust `+request_gpus`, memory, or site `Requirements` as needed.

---

## Outputs

* `out/out.json` – includes inputs, base outputs, finetuned outputs, and metrics:

  * `jaccard`, `compression` (simphile)
  * `spacy_sim` (vector similarity using `en_core_web_md`)
  * `burrows` (Burrows’ Delta via `faststylometry`)

* `out_params/*.json` – same structure across parameter settings

* `out/out_logic.json` – raw fields for logic eval variants
