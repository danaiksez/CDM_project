import torch
import re

from data_loader.nlp import SPECIAL_TOKENS
from slp.util import mktensor
from nltk import WordPunctTokenizer


def remove_punctuation(txt):
    ch = "[.?:_'!,)(]"
    txt = re.sub(ch, '', txt)
    return txt 


class ToTokenIds(object):
    def __init__(self, word2idx, specials=SPECIAL_TOKENS):
        self.word2idx = word2idx
        self.specials = specials

    def __call__(self, x):
        return [self.word2idx[w]
                if w in self.word2idx
                else self.word2idx[self.specials.UNK.value]
                for w in x]


class ReplaceUnknownToken(object):
    def __init__(self, old_unk='<unk>', new_unk=SPECIAL_TOKENS.UNK.value):
        self.old_unk = old_unk
        self.new_unk = new_unk

    def __call__(self, x):
        return [w if w != self.old_unk else self.new_unk for w in x]


class ToTensor(object):
    def __init__(self, device='cpu', dtype=torch.long):
        self.device = device
        self.dtype = dtype

    def __call__(self, x):
        return mktensor(x, device=self.device, dtype=self.dtype)


class PunctTokenizer(object):
    def __init__(self,
                 lower=True,
                 prepend_cls=False,
                 prepend_bos=False,
                 append_eos=False,
                 stopwords=None,
                 specials=SPECIAL_TOKENS):
        self.lower = lower
        self.specials = SPECIAL_TOKENS
        self.pre_id = []
        self.post_id = []
        self.stopwords = stopwords
        if prepend_cls and prepend_bos:
            raise ValueError("prepend_bos and prepend_cls are"
                             " mutually exclusive")
        if prepend_cls:
            self.pre_id.append(self.specials.CLS.value)
        if prepend_bos:
            self.pre_id.append(self.specials.BOS.value)
        if append_eos:
            self.post_id.append(self.specials.EOS.value)
        self.punct = WordPunctTokenizer()

    def __call__(self, x):
        if self.lower:
            x = x.lower()

        x = (self.pre_id + self.punct.tokenize(x) + self.post_id)
        if self.stopwords:
            x = [w for w in x if w not in self.stopwords]
        return x

