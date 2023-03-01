import dataclasses
import pprint
import random
import warnings
from collections import defaultdict
from functools import partial
from io import BytesIO

import gcsfs
import h5py
import numpy as np
import torch
from datasets import interleave_datasets, load_dataset
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from torch.utils.data import IterableDataset
from tqdm import tqdm, trange
from transformers import AutoTokenizer
from .templates import webgpt_template, webgpt_tie_template, summary_template, dialogue_template


class HumanFeedbackDataset(object):
    """ Human feedback dataset
    """
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.tokenizer = 'EleutherAI/gpt-j-6B'
        config.seq_length = 512
        config.split = 'train'
        config.batch_size = 8
        config.weight = ""

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer):
        self.config = self.get_default_config(config)
        self._tokenizer = tokenizer
        d1 = load_dataset('openai/webgpt_comparisons', split=self.config.split)
        d2 = load_dataset('Anthropic/hh-rlhf', split=self.config.split)
        d3 = load_dataset('openai/summarize_from_feedback', 'comparisons', split=self.config.split)
        if self.config.weight == "":
            p = [len(d1), len(d2), len(d3)]
        else:
            p = [int(x) for x in self.config.weight.split(',')]
            assert len(p) == 3, "weight length must be 3"
        p = [x / sum(p) for x in p]
        self._dataset = interleave_datasets([d1, d2, d3], p)

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        while True:
            tokens = []
            masks = []
            for sample in self._dataset:
                processed = self.format_to_sentence(sample)
                tokens.extend(processed['input_tokens'])
                masks.extend(processed['loss_masks'])
                tokens.append(self.tokenizer.eos_token_id)
                masks.append(1)
                while len(tokens) > chunk_size:
                    yield {
                        'tokens': np.array(tokens[:chunk_size], dtype=np.int32).reshape(
                            self.config.batch_size, -1
                        ),
                        'masks': np.array(masks[:chunk_size], dtype=np.int32).reshape(
                            self.config.batch_size, -1
                        ),
                    }
                    tokens = tokens[chunk_size:]
                    masks = masks[chunk_size:]

    def format_to_sentence(self, sample):
        if 'info' in sample and sample['info'] is not None and 'post' in sample['info']: # summarize_from_feedback
            prefix = sample['info']['post']
            pos_ind = int(sample['choice'])
            neg_ind = 1 - pos_ind
            pos = sample['summaries'][pos_ind]['text']
            neg = sample['summaries'][neg_ind]['text']
            format = random.choice(random.choice(summary_template))

        elif 'question' in sample and sample['question'] is not None and 'full_text' in sample['question']: # webgpt_comparisons
            score_0, score_1 = sample['score_0'], sample['score_1']
            prefix = sample['question']['full_text']
            if len(sample['quotes_0']['title']) != 0:
                token_0 = sample['quotes_0']['title'][0]
            if len(sample['quotes_0']['extract']) != 0:
                token_0 += sample['quotes_0']['extract'][0]
            if len(sample['answer_0']) != 0:
                token_0 += sample['answer_0']

            if len(sample['quotes_1']['title']) != 0:
                token_1 = sample['quotes_1']['title'][0]
            if len(sample['quotes_1']['extract']) != 0:
                token_1 += sample['quotes_1']['extract'][0]
            if len(sample['answer_1']) != 0:
                token_1 += sample['answer_1']

            if score_0 > score_1:
                pos, neg = token_0, token_1
                format = random.choice(random.choice(webgpt_template))
            elif score_0 < score_1:
                pos, neg = token_1, token_0
                format = random.choice(random.choice(webgpt_template))
            else:
                pos, neg = token_0, token_1
                format = random.choice(random.choice(webgpt_tie_template))
        else: # hh-rlhf
            pos = sample['chosen']
            neg = sample['rejected']
            prefix = ''
            format = random.choice(random.choice(dialogue_template))

        if format.endswith('{pos}') and '{neg}' in format:
            p1 = format.split('{neg}')[0]
            p2 = format.split('{neg}')[1].split('{pos}')[0]
            prefix = self._tokenizer.encode(prefix)
            p1 = self._tokenizer.encode(p1)
            neg = self._tokenizer.encode(neg)
            p2 = self._tokenizer.encode(p2)
            pos = self._tokenizer.encode(pos)
            input_tokens = prefix + p1 + neg + p2 + pos
            loss_masks = [0] * len(prefix) + [0] * len(p1) + [1] * len(neg) + [0] * len(p2) + [1] * len(pos)
        elif format.endswith('{neg}') and '{pos}' in format:
            p1 = format.split('{pos}')[0]
            p2 = format.split('{pos}')[1].split('{neg}')[0]
            prefix = self._tokenizer.encode(prefix)
            p1 = self._tokenizer.encode(p1)
            pos = self._tokenizer.encode(pos)
            p2 = self._tokenizer.encode(p2)
            neg = self._tokenizer.encode(neg)
            input_tokens = prefix + p1 + pos + p2 + neg
            loss_masks = [0] * len(prefix) + [0] * len(p1) + [1] * len(pos) + [0] * len(p2) + [1] * len(neg)
        elif format.endswith('{pos}'):
            p1 = format.split('{pos}')[0]
            prefix = self._tokenizer.encode(prefix)
            p1 = self._tokenizer.encode(p1)
            pos = self._tokenizer.encode(pos)
            input_tokens = prefix + p1 + pos
            loss_masks = [0] * len(prefix) + [0] * len(p1) + [1] * len(pos)
        elif format.endswith('{neg}'):
            p1 = format.split('{neg}')[0]
            prefix = self._tokenizer.encode(prefix)
            p1 = self._tokenizer.encode(p1)
            neg = self._tokenizer.encode(neg)
            input_tokens = prefix + p1 + neg
            loss_masks = [0] * len(prefix) + [0] * len(p1) + [1] * len(neg)
        elif format.endswith('{1st}'):
            p1 = format.split('{1st}')[0]
            p1 = p1.format(pos=pos, neg=neg)
            p2 = "the first one."
            prefix = self._tokenizer.encode(prefix)
            p1 = self._tokenizer.encode(p1)
            p2 = self._tokenizer.encode(p2)
            input_tokens = prefix + p1 + p2
            loss_masks = [0] * len(prefix) + [0] * len(p1) + [1] * len(p2)
        elif format.endswith('{2nd}'):
            p1 = format.split('{2nd}')[0]
            p1 = p1.format(pos=pos, neg=neg)
            p2 = "the second one."
            prefix = self._tokenizer.encode(prefix)
            p1 = self._tokenizer.encode(p1)
            p2 = self._tokenizer.encode(p2)
            input_tokens = prefix + p1 + p2
            loss_masks = [0] * len(prefix) + [0] * len(p1) + [1] * len(p2)
        else:
            raise ValueError('format: {}'.format(format))

        return {
            'input_tokens': input_tokens,
            'loss_masks': loss_masks
        }

    def __getstate__(self):
        return self.config, self.tokenizer

    def __setstate__(self, state):
        config, tokenizer = state
        self.__init__(config, tokenizer)

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def dataset(self):
        return self._dataset

    @property
    def vocab_size(self):
        return len(self._tokenizer)
