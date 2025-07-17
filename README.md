# llm_project

## Fine-tuned model storage
Move the finetuned adapter into `models/`.

I've tried this code with `models/finetuned_smaller`

## Base model storage

Note: to avoid downloading the base model while running python, you can download the base model on the CLI using 
````bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct
````