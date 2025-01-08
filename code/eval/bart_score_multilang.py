# This file is a modified version of code/eval/bart_score.py to support multi-lingual evaluation.
import torch
import torch.nn as nn
import traceback
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from typing import List
import numpy as np


class BARTScorer_multilang:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint='facebook/mbart-large-50-many-to-many-mmt'):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = MBart50TokenizerFast.from_pretrained(checkpoint)
        self.model = MBartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def score(self, srcs, tgts, src_lang="en_XX", tgt_lang="en_XX", batch_size=4): # for e.g. German to German src_lang="de_DE", tgt_lang="de_DE"
        """ Score a batch of examples """
        self.tokenizer.src_lang = src_lang
        tgt_lang_id = self.tokenizer.lang_code_to_id[tgt_lang]

        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list


    def test(self, batch_size=3):
        """ Test """
        src_list = [
            "This is a very good idea. Although simple, it is very insightful.",
            "I would like to know more about this concept.",
            "The cat sat on the mat."
        ]

        tgt_list = [
            "This is a good idea. Simple yet very insightful.",
            "Can you tell me more about this idea?",
            "The cat was sitting on the mat."
        ]

        print(self.score(src_list, tgt_list, src_lang="en_XX", tgt_lang="en_XX", batch_size=batch_size))