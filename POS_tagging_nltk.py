import nltk
import os
import numpy as np

def load_data(input_dir, output_dir, split='train'):

    # Finding files
    dial_dir = os.path.join(input_dir, split, 'preprocessed', 'dialogues.txt')
    out_dial_dir = os.path.join(output_dir, split, 'preprocessed', 'dialogues_pos_nltk.txt')
    out_dial = open(out_dial_dir, 'w')


    #open input
    in_dial = open(dial_dir, 'r')

    for line_count, line in enumerate(in_dial):

        tokens = nltk.word_tokenize(line)
        pos_tokens = nltk.pos_tag(tokens)
        pos_tags = np.array(pos_tokens)[:,1]

        try:
            assert len(pos_tags) == len(pos_tokens)
        except:
            print(f'Line {line_count} has different lengths!')

        pos_tags_str = " ".join(x for x in pos_tags)
        
        out_dial.write(pos_tags_str)
        out_dial.write('\n')
    
    in_dial.close()

if __name__ == '__main__':
    cwd = os.getcwd()
    input_dir = cwd + '/data/EMNLP_dataset/'
    output_dir = cwd + '/data/EMNLP_dataset/'
    splits = ['train', 'test', 'validation']
    for split in splits:
        load_data(input_dir, output_dir, split)