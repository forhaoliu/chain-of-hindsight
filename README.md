# CoH's Implementation

Authors' implementation of Chain-of-Hindsight which is proposed in [Chain of Hindsight aligns Language Models with Feedback](https://arxiv.org/abs/2302.02676) in Jax.

This implementation allows GPT model training and inference at both small (few TPUs/GPUs) and large (hundreds/thousands of TPUs/GPUs) scales through the use of data and model parallelism.

# Codebase structure

```bash
.
├── coh
│   ├── data
│   │   ├── hf_data.py # data loader for human feedback datasets
│   │   ├── pt_data.py # data loader for pretraining datasets
│   │   ├── templates.py # templates for generating chain of hindsight
│   ├── coh_train.py # main training script
│   ├── models
│   │   ├── gptj
│   │   │   ├── gptj.py # GPT-J model
│   │   │   ├── gptj_train.py # GPT-J training script
│   │   ├── opt
│   │   │   ├── opt.py # OPT model
│   │   │   ├── opt_train.py # OPT training script
│   ├── scripts
│   │   ├── lm_eval.py # evaluation script
│   │   ├── lm_serve.py # serving script
│   │   ├── serving.py # serving helper functions
│   ├── utils.py # helper functions
│   ├── jax_utils.py # jax helper functions
├── scripts
│   └── gpu_enviroment.yml  # conda environment for GPU hosts
│   └── tpu_vm_setup.sh  # script for installing dependencies on TPU hosts
│   └── tpu_util.sh  # script for starting/stopping TPU VMs
├── scripts  # scripts for evaluation
│    dataset_builders  # scripts used to generate some of the datasets
├── resources  # small, git-tracked files from which lists of words or prompts are loaded
└── train.py  # the main training script
```

# Installation
The installation method differs between GPU hosts and Cloud TPU hosts. Please follow the instructions below.
```shell
git clone https://github.com/lhao499/CoH.git CoH
cd CoH
export PYTHONPATH="${PWD}:$PYTHONPATH"
```
To install GPU dependencies, run:
```shell
conda env create -f scripts/gpu_enviroment.yml
```

To install TPU dependencies, run the following command on every TPU host:
```shell
sh ./scripts/tpu_vm_setup.sh
```
The tpu_util.sh is a useful script used to start a TPU VM and install dependencies. It is also used to stop the TPU VM and delete the TPU VM instance.

# Usage


## I want to train CoH

```shell
python3 -m coh.coh_train \
    --mp_mesh_dim=16 \
    --load_opt_config='huggingface::EleutherAI/gpt-j-6B' \
    --model='opt' \
    --pretrain_dataset.split='train' \
    --pretrain_dataset.path='c4' \
    --pretrain_dataset.seq_length=1024 \
    --pretrain_dataset.batch_size=512 \
    --feedback_dataset.tokenizer='EleutherAI/gpt-j-6B' \
    --feedback_dataset.split='train' \
    --feedback_dataset.seq_length=1024 \
    --feedback_dataset.batch_size=512 \
    --log_all_worker=False \
    --logger.online=False \
    --logger.project_id="" \
    --logger.experiment_id="" \
    --logger.experiment_note="" \
    --logger.gcs_output_dir="" \
    --logger.output_dir="$HOME/coh_output"
```

This will finetune a 6B model GPT-J on human feedback datasets.
You can choose model from GPT-J and OPT models to finetune, it loads pretrained models from HuggingFace. The default setting is to finetune on GPT-J.

## I want to train SFT

Just disable the usage of chain of hindsight, then it will finetune on the positive or negative feedback datasets without using chain of hindsight, i.e., the length is fixed to one.
You probably want to exclude negative feedback datasets from the training set, otherwise the model will be biased towards negative feedback.

## I want to train RLHF

Not being supported in this codebase, an implementation (in PyTorch) is available at [trlx](https://github.com/CarperAI/trlx)


## I want to do evaluation

```shell
python -m coh.scripts.lm_serve \
    --load_gptj_config='Your checkpoint path' \
    --load_checkpoint='Your checkpoint path' \
    --dtype='bf16' \
    --input_length=512 \
    --seq_length=2048 \
    --lm_server.host='127.0.0.1' \
    --lm_server.pre_compile='loglikelihood' \
    --lm_server.batch_size=2
```

The compiling process may take a while. The compiled model will be served at `127.0.0.1:5007`. Then, you can run the evaluation script:
```shell
python -m coh.scripts.lm_eval \
    --lm_server_url='http://localhost:5007/' \
    --tasks='wsc,winogrande' \
    --shots=0
```
Full list of tasks can be found at [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).


## I want to do human evaluation

Not being supported in this codebase, please refer to the paper (experiment settings and appendix) for more details of setting up the human evaluation.
Note to use pairwise comparisons which are more reliable than rating multiple methods at the same time.


## I wan to do pretraining / finetuning without human feedback / preference.

GPT-J
```shell
python -m coh.scripts.models.gptj.gptj_train
```

OPT
```shell
python -m coh.scripts.models.opt.opt_train
```

# Citation

If you use this codebase, please cite the following paper:

```bibtex
@article{liu2023languages,
  title={Chain of Hindsight aligns Language Models with Feedback},
  author={Liu, Hao and Sferrazza, Carmelo and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2302.02676},
  year={2023}
}
```


# Acknowledgement

This codebase is heavily built on top of [EasyLM](https://github.com/young-geng/EasyLM) using [Jax](https://github.com/google/jax) and [Flax](
https://github.com/google/flax). The evaluation code is heavily based on EleutherAI [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). The data loader and pretrained models weights are based on HuggingFace [Datasets](https://github.com/huggingface/datasets) and [Transformers](https://github.com/huggingface/transformers).
We thank the authors of these libraries for their great work without which this project would not be possible.


# Contact
If you have any questions, please contact hao.liu@cs.berkeley.edu.
