# b2_uploader

Utilities for syncing files between your machine and **Backblaze B2** storage using `b2sdk`.

This directory is a small Python package  with a single public module that wraps `b2sdk`’s `Synchronizer`. 
---

## Contents

```

b2\_uploader/
├─ pyproject.toml              # build metadata (package name: b2\_uploader)
└─ src/
├─ **init**.py
└─ b2\_uploader.py           # B2Uploader class (auth + sync local <-> bucket)

````

## Installation

From the repo root:

```bash
pip install -e ./b2_uploader
# or build a wheel
pip wheel ./b2_uploader -w dist
````

**Requires:** Python ≥ 3.9 and `b2sdk`.

```bash
pip install b2sdk python-dotenv
```

---

## Authentication & Environment

Set these environment variables (or place them in a local `.env` at the repo root if you use `python-dotenv`):

* `B2_APPLICATION_KEY_ID` – your Backblaze Application Key ID
* `B2_APPLICATION_KEY` – your Backblaze Application Key
* `B2_BUCKET_NAME` – the B2 bucket to use (e.g., `my-llm-bucket`)

Example `.env`:

```dotenv
B2_APPLICATION_KEY_ID=your_key_id
B2_APPLICATION_KEY=your_key_secret
B2_BUCKET_NAME=my-llm-bucket
```

---

## Minimal Usage (Python)

```python
from b2_uploader import B2Uploader

# credentials are typically read from env vars
uploader = B2Uploader()

# Sync a local path to a bucket prefix (one-way local → B2)
# The module uses b2sdk.Synchronizer under the hood.
uploader.sync_directory(
    local_path="models/finetuned_smaller",
    bucket_name="my-llm-bucket",
    remote_prefix="adapters/finetuned_smaller"  # becomes b2://my-llm-bucket/adapters/finetuned_smaller
)
```
