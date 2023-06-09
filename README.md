# CoH's Implementation

Authors' implementation of Chain-of-Hindsight which is proposed in [Chain of Hindsight aligns Language Models with Feedback](https://arxiv.org/abs/2302.02676) in Jax.

This implementation allows GPT model training at both small and large scales through the use of data and model parallelism.

*New in this release:*
- Added better support for feedback data
- Pre-generate feedback data for training to simplify the data pipeline
- Simplified data templates and retain similar performance
- Added better support for GPT-J and OPT models (including pretraining and finetuning)

# Codebase structure

```bash
.
├── coh
│   ├── data
│   │   ├── dataset.py # data loader for human feedback datasets and geneic pretraining datasets
│   │   ├── pack_hf.py # scripts for generating chain of hindsight data
│   ├── coh_train.py # main training script
│   ├── models
│   │   ├── gptj
│   │   │   ├── gptj_model.py # GPT-J model
│   │   │   ├── gptj_train.py # GPT-J training script
│   │   │   ├── gptj_serve.py # GPT-J serving script
│   │   ├── opt
│   │   │   ├── opt_model.py # OPT model
│   │   │   ├── opt_train.py # OPT training script
│   │   │   ├── opt_serve.py # OPT serving script
│   ├── scripts
│   │   ├── lm_eval_json.py # evaluation script
│   │   ├── lm_eval_harness.py # evaluation script for lm-eval-harness
│   ├── utils.py # helper functions
│   ├── jax_utils.py # jax helper functions
│   ├── checkpoint.py # checkpoint functions
│   ├── coh_train.py # main training script
│   ├── optimizers.py # optimizer functions
│   ├── serving.py # serving functions
├── scripts
│   └── gpu_enviroment.yml  # conda environment for GPU hosts
│   └── tpu_vm_setup.sh  # script for installing dependencies on TPU hosts
│   └── tpu_util.sh  # script for starting/stopping TPU VMs
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
conda env create -f scripts/gpu_environment.yml
```

To install TPU dependencies, run the following command on every TPU host:
```shell
sh ./scripts/tpu_vm_setup.sh
```
The tpu_util.sh is a useful script used to start a TPU VM and install dependencies. It is also used to stop the TPU VM and delete the TPU VM instance.

# Usage


**Run CoH training**

Prepare the feedback data for training. The data should be in the format of `jsonl` files.
The script `pack_hf.py` can be used to generate the data for training. It takes the raw feedback data and generates the chain of hindsight data.
An example that generates the data for training on GPT-J is as follows:
```shell
python -m coh.data.pack_hf \
    --output_dir='./local' \
    --dataset='dialogue,webgpt,summary' \
    --include_feedback='p,n,pn,np'
```
where `include_feedback` is the feedback types to include in the training data. The default is to include all feedback types. You can also include only positive feedback or negative feedback by setting `include_feedback='p'` or `include_feedback='n'`.
And you can also include auxiliary feedback types by setting `include_feedback='p,n,pn,np,aux'` which will result in a larger dataset that comes with more diverse feedback.

The generated data will be saved in the `output_dir` directory and you will need to specify the path to the data when running the training script.


Then, run the training script:
```shell
python -m coh.coh_train \
    --mp_mesh_dim=-1 \
    --load_opt_config='huggingface::facebook/opt-350m' \
    --model='opt' \
    --hf_train_dataset.type='feedback' \
    --hf_train_dataset.text_processor.fields_from_example='fields' \
    --hf_train_dataset.hf_dataset.path='/home/hao/research/coh/local/train.jsonl' \
    --hf_train_dataset.hf_dataset.seq_length=4 \
    --hf_train_dataset.hf_dataset.batch_size=2 \
    --pt_train_dataset.type='pretrain' \
    --pt_train_dataset.text_processor.fields='text' \
    --pt_train_dataset.pt_dataset.path='c4' \
    --pt_train_dataset.pt_dataset.split='train' \
    --pt_train_dataset.pt_dataset.seq_length=4 \
    --pt_train_dataset.pt_dataset.batch_size=2 \
    --hf_eval_dataset.type='feedback' \
    --hf_eval_dataset.text_processor.fields_from_example='fields' \
    --hf_eval_dataset.hf_dataset.path='/home/hao/research/coh/local/test.jsonl' \
    --hf_eval_dataset.hf_dataset.seq_length=4 \
    --hf_eval_dataset.hf_dataset.batch_size=2 \
    --pt_eval_dataset.type='pretrain' \
    --pt_eval_dataset.text_processor.fields='text' \
    --pt_eval_dataset.pt_dataset.path='c4' \
    --pt_eval_dataset.pt_dataset.split='validation' \
    --pt_eval_dataset.pt_dataset.seq_length=4 \
    --pt_eval_dataset.pt_dataset.batch_size=2 \
    --log_all_worker=False \
    --logger.online=False \
    --logger.project_id="" \
    --logger.experiment_id="" \
    --logger.experiment_note="" \
    --logger.gcs_output_dir="" \
    --logger.output_dir="/home/hao/experiment_output/coh_output"
```
Remeber to change the `hf_train_dataset.hf_dataset.path` and `hf_eval_dataset.hf_dataset.path` to your own path.
Please change the seq_length and batch_size according to your own GPU memory. The default one is using 1024 sequence length and 256 batch size.

You can switch between GPT-J and OPT by changing the `model` argument.
Loading pretrained model can be done by using the `load_gptj_config` and `load_checkpoint` arguments. It will load the pretrained model from the specified path, e.g., if you use `load_gptj_config='huggingface::EleutherAI/gpt-j-6B'`, it will load the pretrained model from HuggingFace.

As for CoH variant that conditions on feedback as input but not predicting sequence of outputs.
The script `pack_hf.py` can be used to generate the data for training. It takes the raw feedback data and generates the chain of hindsight data.

```shell
python -m coh.data.pack_hf \
    --output_dir='./local' \
    --dataset='dialogue,webgpt,summary' \
    --include_feedback='p,n'
```
where we specify `include_feedback='p,n'` to only include positive and negative feedback, but not interleave feedback.

The training follows the same procedure as CoH training.

**Run SFT training**

Standard SFT can be done by filtering the feedback data and then training the model on the filtered positive data. This is not currently supported in this codebase yet, but will be added soon. Here we provide a variant of SFT based on CoH. In addition to standard SFT that trains only on positive data, it takes into account the positive feedback as an input.

The script `pack_hf.py` can be used to generate the data for training. It takes the raw feedback data and generates the chain of hindsight data.

```shell
python -m coh.data.pack_hf \
    --output_dir='./local' \
    --dataset='dialogue,webgpt,summary' \
    --include_feedback='p'
```
where we specify `include_feedback='p'` to only include positive feedback.

The training follows the same procedure as CoH training.

**Run RLHF baseline**

Not being supported in this codebase, an implementation (in PyTorch) is available at [TRLX](https://github.com/CarperAI/trlx)


**Run qualtative evaluation**

You can manually check out the quality of finetuned models by running a server of the model:
For instance, to serve a GPT-J model, run:
```shell
python -m coh.models.gptj.gptj_serve \
    --load_gptj_config='Your checkpoint path' \
    --load_checkpoint='Your checkpoint path' \
    --mp_mesh_dim=-1 \
    --dtype='bf16' \
    --input_length=1024 \
    --seq_length=2048 \
    --do_sample=True \
    --temperature=1.0 \
    --lm_server.port=5007 \
    --lm_server.pre_compile='all' \
    --lm_server.chat_prepend_text='' \
    --lm_server.chat_lm_prefix='An helpful answer:' \
    --lm_server.chat_lm_suffix='</s>' \
    --lm_server.chat_user_prefix='User: ' \
    --lm_server.chat_user_suffix=' '
```
A chat interface will be served at `127.0.0.1:5007` and you can interact with the model by typing in the prompt.

This chat interface can also be used for preliminary human evaluation of dialogue and so on. You can also use the same interface to evaluate other models.


**Run quantitative evaluation**

Similarly, once you have a server ready, you can run the evaluation script:
```shell
python -m coh.scripts.lm_eval_harness \
    --lm_server_url='http://localhost:5007/' \
    --tasks='wsc,winogrande' \
    --shots=0
```
Full list of tasks can be found at [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).


**Human evaluation**

Not being supported in this codebase, please refer to the paper (experiment settings and appendix) for more details of setting up the human evaluation.
Note to use pairwise comparisons which are more reliable than rating multiple methods at the same time.


**Pretraining/finetuning without feedback**.

Please check out [gptj_train.py](./models/gptj/gptj_train.py) and [opt_train.py](./models/opt/opt_train.py) for the training scripts.
You can also use the same scripts to finetune the model on other datasets.

# Reference

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
https://github.com/google/flax).
The evaluation code is heavily based on EleutherAI [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). The data loader and pretrained models weights are based on HuggingFace [Datasets](https://github.com/huggingface/datasets) and [Transformers](https://github.com/huggingface/transformers).

We thank the authors of these libraries for their great work without which this project would not be possible.


# Contact
If you have any questions, please contact hao.liu@cs.berkeley.edu.
