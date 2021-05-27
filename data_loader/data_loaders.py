#from torchvision import datasets, transforms
from base import BaseDataLoader

import matplotlib.pyplot as plt
#adapted from https://github.com/Sanghoon94/DailyDialogue-Parser/blob/master/parser.py
#from base.base_data_loader import BaseDataLoader
import os, sys, torch, re
import numpy as np

from data_loader.transforms import ToTokenIds, ReplaceUnknownToken, ToTensor, PunctTokenizer
from slp.util.embeddings import EmbeddingsLoader
from slp.util import mktensor

from torchvision.transforms import Compose
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence as pad_sequence_torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'


def pad_sequence(sequences, batch_first=False, padding_len=None, padding_value=0):
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    if padding_len is not None:
        max_len = padding_len
    else:
        max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        if tensor.size(0) > padding_len:
            tensor = tensor[:padding_len]
        length = min(tensor.size(0), padding_len)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor
    return out_tensor


def remove_punctuation(txt):
    ch = "[.?:_'!,)(]"
    txt = re.sub(ch, '', txt)
    return txt

class DailyDialogue(Dataset):
    def __init__(self, input_dir, split='train', text_transforms=None):
        self.input_dir = input_dir
        self.split = split
        self.text_transforms = text_transforms
        self.sequences, self.emotions, self.actions, self.speakers = self.parse_data(self.input_dir, self.split)
        self.preprocessed = [self.preprocess(i) for i in range(len(self.sequences))]
        self.emotion_dataset = [self.emotion_dataset(i) for i in range(len(self.preprocessed))]
    def __len__(self):
        return len(self.emotions)

    def retrieve_str_label(self,labels):
        label_s = []
        for label in labels:
            if label == 0:
                label_s.append('no_emotion')
            elif label  == 1:
                label_s.append('anger')
            elif label  == 2:
                label_s.append('disgust')
            elif label  == 3:
                label_s.append('fear')
            elif label == 4:
                label_s.append('happiness')
            elif label  == 5:
                label_s.append('sadness')
            elif label == 6:
                label_s.append('surprise')
        return label_s

    def retrieve_emotions(self, dialogue):
        no_emotion = []
        anger = []
        disgust = []
        fear = []
        happiness = []
        sadness = []
        surprise = []
        for seqs in range(len(dialogue)):
            for seq in range(len(seqs)) :
                if label == 0:
                    no_emotion.append((dailogue[seqs][seq], self.speakers[seqs][seq], self.sequences[seqs][seq], self.retrieve_str_label(self.emotions[seqs][seq])))
                elif label  == 1:
                    langer.append(seq)
                elif label  == 2:
                    disgust.append(seq)
                elif label  == 3:
                    fear.append(seq)
                elif label == 4:
                    happiness.append(seq)
                elif label  == 5:
                    sadness.append(seq)
                elif label == 6:
                    surprise.append(seq)
            return no_emotion,anger,disgust,fear,happiness,sadness,surprise
    def preprocess(self, idx):
        sequences = self.sequences[idx]
        emotions = self.emotions[idx]
        actions = self.actions[idx]
        speakers = self.speakers[idx]

        segments = []
        if self.text_transforms is not None:
            for seq in sequences:
                seq = self.text_transforms(remove_punctuation(seq))
                segments.append(seq)
        #sequences = pad_sequence(segments, batch_first=True, padding_len=100)
        #emotions = torch.tensor(emotions)
            sequences = segments
        labels = self.retrieve_str_label(self.emotions[idx])
        seqs = []
         # ['no emotion','anger','disgust','fear','happiness','sadness','surprise']
        for seq in self.sequences[idx]:
            seqs.append(remove_punctuation(seq))
        return sequences, emotions, actions, speakers,seqs, labels

    def long_short_divsion(prepped):
        len_seq = []
        len_sen = []
        short_seq = []
        middle_seq = []
        long_seq = []
        short_sen = []
        middle_sen = []
        long_sen = []
        for (seq,x,y,z) in prepped:
            len_seq.append(len(seq))
            if len(seq) <= 5:
                short_seq.append(seq)
            elif len(seq) > 5 and len(seq) <= 9:
                middle_seq.append(seq)
            elif len(seq) > 9:
                long_seq.append(seq)

            for sentence in seq:
                if len(sentence) <= 7:
                    short_sen.append(sentence)
                elif len(sentence) > 7 and len(sentence)<= 12:
                    middle_sen.append(sentence)
                elif len(sentence) > 12:
                    long += 1
                    long_sen.append(sentence)
                len_sen.append(len(sentence))
        return short_seq, middle_seq, long_seq, short_sen, middle_sen, long_sen

    def parse_data(self, input_dir, split='train'):
        # Finding files
        dial_dir = os.path.join(input_dir, split, f'dialogues_{split}.txt')
        emo_dir = os.path.join(input_dir, split,  f'dialogues_emotion_{split}.txt')
        act_dir = os.path.join(input_dir, split, f'dialogues_act_{split}.txt')

        #open input
        in_dial = open(dial_dir, 'r'); in_emo = open(emo_dir, 'r'); in_act = open(act_dir, 'r')

        emotions = []; dacts = []; sequences = []; spkrs = []
        for line_count, (line_dial, line_emo, line_act) in enumerate(zip(in_dial, in_emo, in_act)):
            seqs = line_dial.split('__eou__')
            seqs = seqs[:-1] #remove newline char

            # add speaker info
            speakers = np.ones((len(seqs))).astype(int)
            mask = np.arange(0, len(seqs), 2)
            speakers[mask] = 0

            emos = np.asarray(line_emo.split(' ')[:-1]).astype(int)
            acts = np.asarray(line_act.split(' ')[:-1]).astype(int)

            seq_len = len(seqs); emo_len = len(emos); act_len = len(acts)
            try:
                assert seq_len == emo_len == act_len
            except:
                print('Line {line_count} has different lengths!')
                print('seq_len = {seq_len}, emo_len = {emo_len}, act_len = {act_len}')
                print('Skipping this entry.')

            # for loop over the utterances of the dialogue segment each time
            for seq in seqs:
                # Get rid of the blanks at the start & end of each turns
                if seq[0] == ' ':
                    seq = seq[1:]
                if seq[-1] == ' ':
                    seq = seq[:-1]
            sequences.append(seqs); emotions.append(emos); dacts.append(acts); spkrs.append(speakers)
        return sequences, emotions, dacts, spkrs

    def __getitem__(self, idx):
        return self.preprocessed[idx]


class DailyDialogDataloader(BaseDataLoader):
    def __init__(self, data_dir, split, batch_size=32, shuffle=False, validation_split=0.1, num_workers=2, collate_fn=None):
        cwd = os.getcwd()
        loader = EmbeddingsLoader(cwd + '/data/embeddings/glove.6B.300d.txt', 300)
        word2idx, idx2word, embeddings = loader.load()
        embeddings = torch.tensor(embeddings)
        self.collator = collator()
        self.batch_size = batch_size

        tokenizer = PunctTokenizer()
        replace_unknowns = ReplaceUnknownToken()
        to_token_ids = ToTokenIds(word2idx)
        to_tensor = ToTensor(device=DEVICE)
        self.text_transforms = Compose([
            tokenizer,
            replace_unknowns,
            to_token_ids,
            to_tensor])

        self.dataset = DailyDialogue(data_dir, split, self.text_transforms)
        #super().__init__(self.dataset, self.batch_size, shuffle, validation_split, num_workers, self.collator, drop_last=True)
        super().__init__(self.dataset, self.batch_size, shuffle, validation_split, num_workers, drop_last=True)


class collator(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.device = device
        self.pad_indx = pad_indx

    def __call__(self, batch):
        sequences, emotions, acts, speakers = map(list, zip(*batch))
        number_of_sentences = torch.tensor([len(s) for s in sequences], device=self.device)
        length_of_sentences = ([torch.tensor([torch.count_nonzero(s) for s in inp]) for inp in sequences])
        #sequences = [pad_sequence(i, padding_len=150, batch_first=True, padding_value=0) for i in sequences]

        sequences = (pad_sequence_torch(sequences,
                               batch_first=True,
                               padding_value=self.pad_indx)
                  .to(self.device))
        length_of_sentences = pad_sequence(length_of_sentences, padding_len=sequences.shape[1], batch_first=True, padding_value=1)

        emotions = pad_sequence(emotions, padding_len=len(emotions[0]), batch_first=True, padding_value=0)
        return sequences, length_of_sentences, emotions, number_of_sentences


if __name__ == '__main__':
    cwd = os.getcwd()
    input_dir = cwd + '/data/EMNLP_dataset/'
    # splits = ['train', 'test', 'validation']
    splits = ['test']
    # for split in splits:
    #     # import pdb; pdb.set_trace()
    #     print(split)
    #     dataloader = DailyDialogDataloader(input_dir, split= split,batch_size=32, shuffle=True, validation_split=0.2)
    dataloader = DailyDialogDataloader(input_dir,split= 'test')
    data = dataloader.dataset.sequences
    prepped = dataloader.dataset.preprocessed

    no_emotion = []
    anger = []
    disgust - []
    fear = []
    happiness = []
    sadness = []
    surprise = []
