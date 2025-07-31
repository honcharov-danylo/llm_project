# configs

JSON configs used by training and evaluation. These are loaded via `utils.Config`, which supports overriding fields from the command line.

## Overriding Config Values from CLI

Scripts that do `config = Config("../configs/<file>.json")` support inline overrides. You can pass **either** `--key value` or `key=value` style (both are parsed).

Examples:

```bash
# finetuning: change base model + output dir
python llm_scripts/finetuning_skip.py --model_dir Qwen/Qwen3-4B --output_dir runs/qwen3_ft

# evaluation: override paths
python eval_scripts/eval.py model_dir=Qwen/Qwen2.5-3B-Instruct finetuned_path=./trained_models/run1
```

> Tip: to persist changes back to disk, use `Config.save()` inside a small helper or edit JSON files directly.

---
