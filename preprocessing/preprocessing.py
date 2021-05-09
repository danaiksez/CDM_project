#adapted from https://github.com/Sanghoon94/DailyDialogue-Parser/blob/master/parser.py
import os, sys

def check_dirs(dir_list):
    """Checks if the given directories in the list exist, if not directories are created."""
    
    for path in dir_list:
        try:
            os.stat(os.path.dirname(path))
        except:
            os.makedirs(os.path.dirname(path))



def parse_data(input_dir, output_dir, split='train'):

    # Finding files
    dial_dir = os.path.join(input_dir, split, f'dialogues_{split}.txt')
    emo_dir = os.path.join(input_dir, split,  f'dialogues_emotion_{split}.txt')
    act_dir = os.path.join(input_dir, split, f'dialogues_act_{split}.txt')

    out_dial_dir = os.path.join(output_dir, split, 'preprocessed', 'dialogues.txt')
    out_emo_dir = os.path.join(output_dir, split, 'preprocessed', 'emotion.txt')
    out_act_dir = os.path.join(output_dir, split, 'preprocessed', 'act.txt')
    out_speaker_dir = os.path.join(output_dir, split, 'preprocessed', 'speaker.txt')

    check_dirs([out_dial_dir, out_emo_dir, out_act_dir, out_speaker_dir])

    #open input
    in_dial = open(dial_dir, 'r')
    in_emo = open(emo_dir, 'r')
    in_act = open(act_dir, 'r')

    #open output
    out_dial = open(out_dial_dir, 'w')
    out_emo = open(out_emo_dir, 'w')
    out_act = open(out_act_dir, 'w')
    out_speaker = open(out_speaker_dir, 'w')

    
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
        
        seq_len = len(seqs)
        emo_len = len(emos)
        act_len = len(acts)
    
        try:
            assert seq_len == emo_len == act_len
        except:
            print(f'Line {line_count} has different lengths!')
            print(f'seq_len = {seq_len}, emo_len = {emo_len}, act_len = {act_len}')
            print('Skipping this entry.')

        for seq, emo, act, speaker in zip(seqs, emos, acts, speakers):

            # Get rid of the blanks at the start & end of each turns
            if seq[0] == ' ':
                seq = seq[1:]
            if seq[-1] == ' ':
                seq = seq[:-1]

            out_dial.write(seq)
            out_dial.write('\n')
            out_emo.write(emo)
            out_emo.write('\n')
            out_act.write(act)
            out_act.write('\n')
            out_speaker.write(speaker)
            out_speaker.write('\n')


    in_dial.close()
    in_emo.close()
    in_act.close()
    out_dial.close()
    out_emo.close()
    out_act.close()
    out_speaker.close()

if __name__ == '__main__':
    cwd = os.getcwd()
    input_dir = cwd + '/data/EMNLP_dataset/'
    output_dir = cwd + '/data/EMNLP_dataset/'
    splits = ['train', 'test', 'validation']
    for split in splits:
        parse_data(input_dir, output_dir, split)
