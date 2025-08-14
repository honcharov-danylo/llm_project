# LLM Project: Fine-tuning and Evaluation Pipeline

A comprehensive pipeline for fine-tuning large language models (LLMs) and evaluating their performance using multiple similarity metrics and stylometric analysis. This project focuses on training models to continue text in a scientific writing style and evaluating the quality of generated text through various linguistic and semantic measures.

##  Project Overview

This project implements a complete workflow for:
- **Data Cleaning**: Preprocessing and normalizing text data for training
- **Model Fine-tuning**: Using PEFT (Parameter-Efficient Fine-Tuning) with LoRA
- **Comprehensive Evaluation**: Multiple evaluation metrics including stylometry, similarity, and logic reasoning
- **Parameter Optimization**: Systematic exploration of generation parameters

## Project Structure

```
llm_project/
├── clean_scripts/          # Data cleaning and preprocessing pipeline
├── llm_scripts/           # Model fine-tuning scripts
├── eval_scripts/          # Evaluation and analysis scripts
├── configs/               # Configuration files for different components
├── utils/                 # Utility functions and configuration management
├── b2_uploader/          # Backblaze B2 cloud storage integration
├── cluster_tools/         # HPC cluster management utilities
└── requirements.txt       # Main project dependencies
```

##  Key Features

- **Multi-stage Pipeline**: Clean → Fine-tune → Evaluate workflow
- **Parameter-Efficient Training**: LoRA-based fine-tuning to reduce memory requirements
- **Comprehensive Evaluation**: Jaccard similarity, compression similarity, spaCy semantic similarity, and Burrows' Delta stylometry
- **Cluster Integration**: HTCondor job submission for distributed processing
- **Cloud Storage**: Automatic model upload to Backblaze B2
- **Experiment Tracking**: Weights & Biases integration for training monitoring

##  Prerequisites

- Python 3.10+
- CUDA-compatible GPU (for training and inference)
- HTCondor cluster access (optional, for distributed processing)
- Backblaze B2 account (optional, for model storage)

##  Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd llm_project

# Create and activate conda environment
conda create -n llm_project python=3.10 -y
conda activate llm_project

# Install dependencies
pip install -r requirements.txt
```

### 2. Anaconda Setup
For the cluster applications anaconda must be installed in the home directory in the "anaconda3"
folder (path should be ~/anaconda3).

Each of the following directories - clean_scripts, llm_scripts, eval_scripts - contains a
corresponding **.sub** file which can be submitted as HTCondor job. Each of them also contains
name of the conda environment to be used.

By default, conda environments are expected to be:

| Cluster job    | Environment name |
|:---------------|-----------------:|
| clean_scripts/ |          llm_project |
| eval_scripts/  |          eval_model |
| llm_scripts/   |          llm_finetuning |



### 3. Data Preparation

```bash
cd clean_scripts
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run data cleaning (single process)
python clean.py 0 1

# Or submit to HTCondor for parallel processing
condor_submit clean.sub
```

### 4. Model Fine-tuning

```bash
cd llm_scripts
pip install -r requirements.txt
pip install wandb lighteval

# Run fine-tuning locally
python finetuning_skip.py \
  --model_dir Qwen/Qwen2.5-3B-Instruct \
  --output_dir ./runs/qwen25_ft \
  --data_path ../clean_scripts/cleaned_data

# Or submit to HTCondor
condor_submit finetuning_skip.sub \
  EXTRA_ARGS="--model_dir Qwen/Qwen2.5-3B-Instruct --output_dir runs/qwen25_ft" \
  LOGFILE="qwen25_ft"
```

### 5. Model Evaluation

```bash
cd eval_scripts
pip install torch transformers peft datasets tqdm simphile faststylometry spacy nltk
python -m spacy download en_core_web_md

# Run main evaluation
python eval.py \
  --model_dir Qwen/Qwen2.5-3B-Instruct \
  --finetuned_path ../trained_models/1

# Run parameter sweep
python eval_params.py \
  --model_dir Qwen/Qwen2.5-3B-Instruct \
  --finetuned_path ../trained_models/1
```

##  Configuration

### Fine-tuning Configuration (`configs/config_finetuning.json`)
Default config parameters:

```json
{
  "data_path": "/path/to/cleaned_data/",
  "load_cleaned": true,
  "model_dir": "Qwen/Qwen2.5-3B-Instruct",
  "max_eval_tok": 4096,
  "output_dir": "output_skip_5",
  "predict_sentences": 5,
  "WANDB_PROJECT": "llm-finetuning-skip-stylo",
  "WANDB_LOG_MODEL": "checkpoint"
}
```

### Evaluation Configuration (`configs/config_eval.json`)

```json
{
  "data_path": "/path/to/cleaned_data/",
  "model_dir": "Qwen/Qwen2.5-3B-Instruct",
  "finetuned_path": "/path/to/trained_models/1",
  "batch_size": 1,
  "eval_size": 256,
  "eval_length_prompt": 2048,
  "eval_length_response": 2048,
  "out_eval_file": "out/out.json"
}
```

## Core Components

### Data Cleaning (`clean_scripts/`)

The cleaning pipeline performs:
- Markdown normalization and formatting
- Language detection (English filtering)
- Text deduplication and cleaning
- Punctuation restoration
- Spell checking and word segmentation
- Output partitioning for parallel processing

**Key Features:**
- Uses spaCy for NLP processing
- SymSpell for spell checking
- Multilingual punctuation restoration
- Markdown formatting with mdformat

### Model Fine-tuning (`llm_scripts/`)

Fine-tuning implementation using:
- **PEFT (Parameter-Efficient Fine-Tuning)**: LoRA adapters for memory efficiency
- **4-bit Quantization**: BitsAndBytes integration for reduced memory usage
- **SFT Trainer**: Supervised Fine-Tuning with custom data formatting
- **Weights & Biases**: Training monitoring and experiment tracking

**Training Approach:**
- Next-sentence prediction task
- Scientific writing style continuation
- Custom prompt formatting with instruction-following structure

### Evaluation (`eval_scripts/`)

Comprehensive evaluation using multiple metrics:

1. **Similarity Metrics**:
   - Jaccard similarity (token overlap)
   - Compression similarity (information content)
   - spaCy semantic similarity (vector-based)

2. **Stylometric Analysis**:
   - Burrows' Delta for author attribution
   - Vocabulary analysis and comparison

3. **Parameter Sweeps**:
   - Temperature variation (0.5-1.0)
   - Repetition penalty exploration (1.0-2.0)

### Custom Model Integration

To use a different base model:

```bash
# Update configuration
python finetuning_skip.py --model_dir "your/model/path"

# Or edit config files directly
# configs/config_finetuning.json
# configs/config_eval.json
```

### Custom Evaluation Tasks

Add new evaluation metrics in `eval_scripts/`:

```python
# Example: Add custom similarity metric
def custom_similarity(generated, reference):
    # Your custom similarity calculation
    return similarity_score

# Integrate into evaluation pipeline
results["custom"] = [custom_similarity(x, inputs_out[i]) for i,x in enumerate(responses)]
```

### Cluster Deployment

For HTCondor cluster usage:

```bash
# Submit cleaning jobs
cd clean_scripts
condor_submit clean.sub

# Submit fine-tuning jobs
cd llm_scripts
condor_submit finetuning_skip.sub

# Submit evaluation jobs
cd eval_scripts
condor_submit eval.sub
```

### Evaluation Results
Default paths for the outputs:

- **`out/out.json`**: Main evaluation results with all metrics
- **`out_params/*.json`**: Parameter sweep results
- **`out/out_logic.json`**: Logic reasoning evaluation results

### Metrics Interpretation
- **Higher Jaccard/Compression**: Better lexical similarity
- **Higher spaCy similarity**: Better semantic similarity
- **Lower Burrows' Delta**: More similar writing style to reference corpus

## Support

For questions and support:
- Open an issue on GitHub
- Check the individual component READMEs for detailed usage
- Review configuration examples in the `configs/` directory

---

**Note**: This project is designed for research purposes. Ensure you have appropriate licenses and permissions for any models or datasets you use.