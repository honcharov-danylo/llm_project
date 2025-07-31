# utils

Small shared utilities imported by other scripts.

---

## Contents

````

utils/
├─ **init**.py        # exposes `Config`
└─ config.py          # Config class (JSON config + CLI overrides + typed parsing)

````

---

## `Config` Class

`Config` wraps a JSON file and then applies CLI overrides. It’s used like:

```python
from utils import Config

config = Config("../configs/config_finetuning.json")
# access as dict or attributes:
model = config["model_dir"]
out   = config.output_dir
````

### CLI overrides

You can pass overrides in either `--key value` **or** `key=value` forms:

```bash
# Both forms are accepted:
python llm_scripts/finetuning_skip.py --model_dir Qwen/Qwen3-4B --output_dir runs/qwen3
python llm_scripts/finetuning_skip.py model_dir=Qwen/Qwen3-4B output_dir=runs/qwen3
```

`Config` attempts to **type-cast** values (numbers, booleans) so `"true"`, `"1"`, `"0.5"` become `True`, `1`, `0.5`, etc.

### Persisting changes

You can save a modified config back to disk:

```python
config.output_dir = "runs/new_place"
config.save()  # overwrites the original JSON
```

> The parser throws an `AttributeError` when you try to access a missing key as an attribute.