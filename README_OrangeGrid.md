# Setup

## Install conda (miniforge)

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/anaconda3
conda init
```
Note: To work with our scripts, the conda installation goes into `~/anaconda3` instead of the default `~/miniconda`

In order to be make Conda available automatically when you log into the cluster
you will also need to add the following to your `~/.bash_profile`

```bash
if [ -e ${HOME}/.bashrc ]
then
    source ${HOME}/.bashrc
fi
```

## Todos documentation
* how to download/use other models from huggingface hub
* seems like some parameters in config don't have a purpose. delete them?
    * config_finetuning.json
        * "env_path"

## Todos other
* finetuning_skip.py
    * check truncation in formatting_prompts_func(). Added special tokens to check limit of max_sequence length. "Token indices sequence length is longer than the specified maximum sequence length for this model"
    * logging doesn't seem to work. Don't see anything at stdout at any rate. Maybe no logger set up? 
* clean.py is pretty messy
* use "clean_scripts/logs_cleaning/" or reate clean_scripts/log/ if it doesn't exist. Cleaning script condor job requires clean_scripts/log/ directory, fails otherwise. But .gitignore removes log/. 
* make use of different models easier:
	* parameterize? different configs?
	* check example scripts don't use shell options that override the json config
	* do the example scripts even need these shell arguments about the model dir?

## Todos conda environments
* conda warpper use of `.bashrc` kosher?
* point condawrappers to default miniforge folder instead of ~/anaconda3
* finetuning_skip.py requires NLTK's Punkt model to be downloaded (can use shell command `python -c "import nltk; nltk.download('punkt_tab')"` and will save it to user folder). 
    * note: i also added some code to download the model to the script
