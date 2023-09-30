# Chain of Hindsight aligns Language Models with Feedback

Hao Liu, Carmelo Sferrazza, Pieter Abbeel

Paper: https://arxiv.org/abs/2302.02676

Jax implementation of the Chain-of-Hindsight (CoH) idea proposed in [Chain of Hindsight aligns Language Models with Feedback](https://arxiv.org/abs/2302.02676).

*Updates Jun/14/2023*
- Improved sharding for large models
- Added support for Fully Sharded Data Parallelism (FSDP)
- Added support for LLaMA model and deprecated support for OPT model

*Updates Mar/29/2023*
- Added better support for feedback data
- Pre-generate feedback data for training to simplify the data pipeline
- Simplified data templates and retain similar performance
- Added better support for GPT-J and OPT models (including pretraining and finetuning)


# Installation
The installation method differs between GPU hosts and Cloud TPU hosts. Please follow the instructions below.
```shell
git clone https://github.com/lhao499/chain-of-hindsight.git chain-of-hindsight
cd chain-of-hindsight
export PYTHONPATH="${PWD}:$PYTHONPATH"
```
To install GPU dependencies, run:
```shell
conda env create -f gpu_requirement.yml
```

To install TPU dependencies, run the following command on every TPU host:
```shell
sh tpu_requirement.sh
```
The tpu_util.sh is a useful script used to start a TPU VM and install dependencies. It is also used to stop the TPU VM and delete the TPU VM instance.

# Usage


**Prepare data**

Prepare the feedback data for training. The data should be in the format of `jsonl` files.
The script `pack_hf.py` can be used to generate the data for training. It takes the raw feedback data and generates the chain of hindsight data.
An example that generates the data for training on GPT-J is as follows:
```shell
python -m coh.data.pack_hf \
    --output_dir='./' \
    --dataset='dialogue,webgpt,summary' \
    --include_feedback='p,n,pn,np'
```
where `include_feedback` is the feedback types to include in the training data. The default is to include all feedback types. You can also include only positive feedback or negative feedback by setting `include_feedback='p'` or `include_feedback='n'`.
And you can also include auxiliary feedback types by setting `include_feedback='p,n,pn,np,aux'` which will result in a larger dataset that comes with more diverse feedback.

As for CoH variant that conditions on feedback as input but not predicting sequence of outputs.
The script `pack_hf.py` can be used to generate the data for training. It takes the raw feedback data and generates the chain of hindsight data.

```shell
python -m coh.data.pack_hf \
    --output_dir='./' \
    --dataset='dialogue,webgpt,summary' \
    --include_feedback='p,n'
```
where we specify `include_feedback='p,n'` to only include positive and negative feedback, but not interleave feedback.
The training follows the same procedure as CoH training.

The generated data will be saved and you will need to specify the path to the data when running the training script.


*Note for PyTorch users*

For those interested in using PyTorch training code like [FastChat](https://github.com/lm-sys/FastChat), check out the [coh/data/pack_hf.py](https://github.com/lhao499/chain-of-hindsight/blob/main/coh/data/pack_hf.py) to converts human feedback data into JSONL format, suitable for integration into other codebases.
Refer to the [coh/data/doc.md](text preprocessing) for more details on the data processing.


**Run CoH training**

If using LLaMA, the first step is to prepare LLaMA Jax pretrained weights. You can either download official LLaMA weights and convert the official LLaMA checkpoint to Jax weights as the following:
``` shell
python3 -m coh.scripts.convert_checkpoint.py \
    --checkpoint_dir='path/to/pytorch/checkpoint' \
    --output_dir='path/to/output/checkpoint' \
    --streaming=True
```

Then, run the training script:
```shell
python3 -m coh.coh_train_llama \
    --load_llama_config='3b' \
    --load_checkpoint='' \ # path to the pretrained checkpoint if you want to finetune
    --tokenizer.vocab_file='path/to/tokenizer.model' \
    --hf_train_dataset.type='feedback' \
    --hf_train_dataset.text_processor.fields_from_example='fields' \
    --hf_train_dataset.hf_dataset.path='/home/hao/research/coh/local/train.jsonl' \ # path to the training human feedback data
    --hf_train_dataset.hf_dataset.seq_length=4 \
    --hf_train_dataset.hf_dataset.batch_size=2 \
    --pt_train_dataset.type='pretrain' \
    --pt_train_dataset.text_processor.fields='text' \
    --pt_train_dataset.pt_dataset.path='c4' \
    --pt_train_dataset.pt_dataset.split='train' \
    --pt_train_dataset.pt_dataset.streaming=False \ # Set to True then the dataset will be streamed from huggingface without downloading the whole dataset. This is useful when the dataset is large, but not recommended for large-scale training as it breaks down occasionally.
    --pt_train_dataset.pt_dataset.seq_length=4 \
    --pt_train_dataset.pt_dataset.batch_size=2 \
    --hf_eval_dataset.type='feedback' \
    --hf_eval_dataset.text_processor.fields_from_example='fields' \
    --hf_eval_dataset.hf_dataset.path='/home/hao/research/coh/local/test.jsonl' \ # path to the evaluation human feedback data
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
    --logger.output_dir="/home/hao/experiment_output/coh_output" \
    --logger.wandb_dir="/home/hao/experiment_output/coh_output"
```
Remeber to change the `hf_train_dataset.hf_dataset.path` and `hf_eval_dataset.hf_dataset.path` to your own path.
Please change the seq_length and batch_size according to your own GPU memory. The default one is using 1024 sequence length and 256 batch size.

If using GPT-J, it's similar command to run the training with the following changes:
```shell
python3 -m coh.coh_train_gptj \
    --load_gptj_config='huggingface::EleutherAI/gpt-j-6b' \
```
Based on our experiments, LLaMA performed better than GPT-J in terms of the final performance.

**Run SFT training**

Standard SFT can be done by filtering the feedback data and then training the model on the filtered positive data. This is not currently supported in this codebase yet, but will be added soon. Here we provide a variant of SFT based on CoH. In addition to standard SFT that trains only on positive data, it takes into account the positive feedback as an input.

The script `pack_hf.py` can be used to generate the data for training. It takes the raw feedback data and generates the chain of hindsight data.

```shell
python -m coh.data.pack_hf \
    --output_dir='./' \
    --dataset='dialogue,webgpt,summary' \
    --include_feedback='p'
```
where we specify `include_feedback='p'` to only include positive feedback.

The training follows the same procedure as CoH training.


**Run qualtative and qualtative evaluation**

You can manually check out the quality of finetuned models by running a server of the model:
For instance, to serve a GPT-J model, run:
```shell
python -m coh.coh_serve_llama \
    --load_llama_config='Your checkpoint path' \
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

Similarly, to serve a GPT-J model, run:
```shell
python -m coh.coh_serve_gptj \
    --load_gptj_config='Your checkpoint path' \
    --load_checkpoint='Your checkpoint path' \
```

Similarly, once you have a server ready, you can run the evaluation script:
```shell
python -m coh.scripts.lm_eval_harness \
    --lm_server_url='http://localhost:5007/' \
    --tasks='wsc,winogrande' \
    --shots=0
```
Full list of tasks can be found at [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness).


*Note for PyTorch users*

Alternatively, you can also convert the model to a huggingface model using [coh/scripts/convert_checkpoint.py](https://github.com/lhao499/chain-of-hindsight/blob/main/coh/scripts/convert_checkpoint.py), and then run the model using chat interface or evaluation code in huggingface ecosystem.


**Human evaluation**

Not being supported in this codebase, please refer to the paper (experiment settings and appendix) for more details of setting up the human evaluation.
Note to use pairwise comparisons which are more reliable than rating multiple methods at the same time.


# Reference

If you find our work relevant to your research, please cite:

```bibtex
@article{liu2023languages,
  title={Chain of Hindsight aligns Language Models with Feedback},
  author={Liu, Hao and Sferrazza, Carmelo and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2302.02676},
  year={2023}
}
```
