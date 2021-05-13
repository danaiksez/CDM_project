#adapted from https://github.com/Sanghoon94/DailyDialogue-Parser/blob/master/parser.py
from base.base_data_loader import BaseDataLoader
import os, sys, torch, re
import numpy as np

from transforms import ToTokenIds, ReplaceUnknownToken, ToTensor, PunctTokenizer
from slp.util.embeddings import EmbeddingsLoader

from torchvision.transforms import Compose
from torch.utils.data import Dataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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

    def __len__(self):
        return len(self.emotions)

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

            segment = []
            # for loop over the utterances of the dialogue segment each time
            for seq, emo, act, speaker in zip(seqs, emos, acts, speakers):
                # Get rid of the blanks at the start & end of each turns
                utt = []
                if seq[0] == ' ':
                    seq = seq[1:]
                if seq[-1] == ' ':
                    seq = seq[:-1]

                if self.text_transforms is not None:
                    seq = self.text_transforms(remove_punctuation(seq))
                segment.append(seq)
            
            emotions.append(emos); dacts.append(acts); sequences.append(segment); spkrs.append(speakers)
        return sequences, emotions, dacts, spkrs

    def __getitem__(self, idx):
        return self.sequences[idx], self.emotions[idx], self.actions[idx], self.speakers[idx]

class DailyDialogDataloader(BaseDataLoader):
    def __init__(self, data_dir, split="train", batch_size=32, shuffle=False, validation_split=0.1, num_workers=2, collate_fn=None):
        loader = EmbeddingsLoader(cwd + '/data/embeddings/glove.840B.300d.txt', 300)
        word2idx, idx2word, embeddings = loader.load()
        embeddings = torch.tensor(embeddings)
        
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
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)



if __name__ == '__main__':
    cwd = os.getcwd()
    input_dir = cwd + '/data/EMNLP_dataset/'
    splits = ['train', 'test', 'validation']

    for split in splits:
        import pdb; pdb.set_trace()
        dataloader = DailyDialogDataloader(input_dir, batch_size=32, shuffle=True, validation_split=0.2)



