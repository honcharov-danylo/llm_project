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
* maybe use miniforge or mambaforge?
* how to download/use other models from huggingface hub

## Todos other
* 

## Todos conda environments
* "llm_project" environment name is used only for clean scripts, maybe rename
* conda warpper use of `.bashrc` kosher?
* make use of different models easier:
	* parameterize? different configs?
	* check example scripts don't use shell options that override the json config
	* do the example scripts even need these shell arguments about the model dir?