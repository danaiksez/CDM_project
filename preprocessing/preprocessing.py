#adapted from https://github.com/Sanghoon94/DailyDialogue-Parser/blob/master/parser.py
import os, sys
from torch.utils.data import Dataset


def check_dirs(dir_list):
    """Checks if the given directories in the list exist, if not directories are created."""
    
    for path in dir_list:
        try:
            os.stat(os.path.dirname(path))
        except:
            os.makedirs(os.path.dirname(path))


class DailyDialogue(Dataset):
    def __init__(self, input_dir, split='train', text_transforms=None):
        self.input_dir = input_dir
        self.split = split
        self.text_transforms = text_transforms
        self.sequences, self.emotions, self.actions = self.parse_data(self.input_dir, self.split)

    def __len__(self):
        return len(self.label)

    def parse_data(input_dir, split='train'):
        # Finding files
        dial_dir = os.path.join(input_dir, split, 'dialogues_{split}.txt')
        emo_dir = os.path.join(input_dir, split,  'dialogues_emotion_{split}.txt')
        act_dir = os.path.join(input_dir, split, 'dialogues_act_{split}.txt')

        #open input
        in_dial = open(dial_dir, 'r'); in_emo = open(emo_dir, 'r'); in_act = open(act_dir, 'r')
        emotions = []; dacts = []; sequences = []

        import pdb; pdb.set_trace()
        for line_count, (line_dial, line_emo, line_act) in enumerate(zip(in_dial, in_emo, in_act)):
            seqs = line_dial.split('__eou__')
            seqs = seqs[:-1] #remove newline char

            # add speaker info
            speakers = []
            for idx, seq in enumerate(seqs):
                if idx%2 == 0:
                    speaker = '0' # speaker A
                else:
                    speaker = '1' # speaker B
                speakers.append(speaker)

            emos = line_emo.split(' ')
            emos = emos[:-1]

            acts = line_act.split(' ')
            acts = acts[:-1]
            
            seq_len = len(seqs); emo_len = len(emos); act_len = len(acts)

            emotions.append(emos)
            dacts.append(acts)
            sequences.append(seqs)

            try:
                assert seq_len == emo_len == act_len
            except:
                print('Line {line_count} has different lengths!')
                print('seq_len = {seq_len}, emo_len = {emo_len}, act_len = {act_len}')
                print('Skipping this entry.')

            for seq, emo, act, speaker in zip(seqs, emos, acts, speakers):

                # Get rid of the blanks at the start & end of each turns
                if seq[0] == ' ':
                    seq = seq[1:]
                if seq[-1] == ' ':
                    seq = seq[:-1]

        return sequences, emotions, dacts

    def __getitem__(self, idx):
        return self.preprocessed[idx]


if __name__ == '__main__':
    cwd = os.getcwd()
    input_dir = cwd + '../data/EMNLP_dataset/'
    output_dir = cwd + '../data/EMNLP_dataset/'
    splits = ['train', 'test', 'validation']
    for split in splits:
        dataset = DailyDialogue(input_dir, split)
