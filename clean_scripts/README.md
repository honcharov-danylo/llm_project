# clean_scripts

Data cleaning pipeline for the LLM fine-tuning/evaluation corpora.  
It normalizes Markdown, filters non-English, removes broken lines, optionally restores punctuation, and writes **partitioned** gzipped JSON outputs.

---

## Contents

````
clean\_scripts/
├─ clean.py                 # main cleaning job (partitioned over processes)
├─ clean.sub               # HTCondor job file (launches many partitions)
├─ conda\_wrapper.sh        # Conda activation wrapper used by Condor
├─ requirements.txt        # cleaning-specific deps
└─ logs/                   # Condor stdout/err
````
The cleaner expects raw Markdown/JSONL files to clean in `data/`. And then writes shards to:

```
clean\_scripts/cleaned\_data/data\_cleaned\_{PROC\_INDEX}.json.gz
```

---
## Dependencies

```bash
# create or use an existing env; example:
conda create -n data_cleaning python=3.10 -y
conda activate data_cleaning

# install cleaning deps
pip install -r clean_scripts/requirements.txt
# spaCy models (choose one; the code imports en_core_web_sm)
python -m spacy download en_core_web_sm
````

The script also uses:

* `langdetect`
* `symspellpy`
* `deepmultilingualpunctuation`
* `markdown-it-py`
* `mdformat`
* `tqdm`
* `spacy`

> `requirements.txt` includes the above (plus a direct wheel URL for `en_core_web_sm`); if the wheel URL fails, use `python -m spacy download en_core_web_sm`.

---

## Inputs & Outputs

* **Input:** by default the script expects raw Markdown/JSONL files under a folder named `data/` (relative to `clean_scripts/`). If your data lives elsewhere, edit the top of `clean.py` to point to your raw dataset folder/pattern.
* **Output:** `cleaned_data/data_cleaned_{i}.json.gz` with JSON mapping local indices to cleaned text.

> The script performs a simple **manual sharding**: each process receives `(proc_index, total_processes)` and slices the input accordingly.

---

## Run Locally (single process)

```bash
cd clean_scripts
python clean.py 0 1
# -> writes cleaned_data/data_cleaned_0.json.gz
```

## Run on HTCondor

The included `clean.sub` sets up a DAG of jobs that each run one shard:

```text
executable = conda_wrapper.sh
arguments  = python clean.py $(Process) 32

output     = logs/clean_$(Process).out
error      = logs/clean_$(Process).err
log        = logs/clean_$(Process).log
getenv     = True

request_cpus   = 1
request_memory = 16 GB

queue 31
# Example site constraint:
Requirements = TARGET.vm_name == "its-u20-nfs-20210413" && regexp("CRUSH", TARGET.name)
queue
```

**Submit:**

```bash
cd clean_scripts
condor_submit clean.sub
```

**Customize:**

* Change `32` in `arguments` to your `TOTAL` shard count (and `queue` to `TOTAL-1` if you use `$(Process)` starting at 0).
* Adjust `request_memory` based on data size.
* Remove or edit the site-specific `Requirements` line for your scheduler.
