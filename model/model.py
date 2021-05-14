import os, torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

from slp.util.embeddings import EmbeddingsLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_embeddings():
    cwd = os.getcwd()
    loader = EmbeddingsLoader(cwd + '/data/embeddings/glove.6B.300d.txt', 300)
    word2idx, idx2word, embeddings = loader.load()
    embeddings = torch.tensor(embeddings)
    return embeddings

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class ThreeEncoders(BaseModel):
    def __init__(self, embedding_size, hidden_size, batch_size, num_classes, batch_first=False, bidirectional=False):
        super().__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.diction = load_embeddings()
        self.dict_size = len(self.diction)
        self.lookup = nn.Embedding(num_embeddings = self.dict_size, embedding_dim =300).from_pretrained(self.diction)

        self.speaker1 = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, batch_first=True, bidirectional=False)
        self.speaker2 = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, batch_first=True, bidirectional=False)
        self.context_encoder = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, batch_first=False, bidirectional=False)
        
        self.bidirectional = False
        if self.bidirectional:
            self.linear1 = nn.Linear(2 * hidden_size, self.num_classes)
            self.linear2 = nn.Linear(2 * hidden_size, self.num_classes)
        else:
            self.linear1 = nn.Linear(hidden_size, self.num_classes)
            self.linear2 = nn.Linear(hidden_size, self.num_classes)

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.speaker1_hidden_state = torch.zeros(batch_size, 1, self.hidden_size)
        self.speaker2_hidden_state = torch.zeros(batch_size, 1, self.hidden_size)
        self.context_hidden_state = torch.zeros(batch_size, 1, self.hidden_size)
        self.context_output = torch.zeros(1, batch_size, self.hidden_size)

        self.speaker1_hidden_state = self.speaker1_hidden_state.to(DEVICE)
        self.speaker2_hidden_state = self.speaker2_hidden_state.to(DEVICE)
        self.context_hidden_state = self.context_hidden_state.to(DEVICE)
        self.context_output = self.context_output.to(DEVICE)
        

    def forward(self, utterances, speakers):
        if not hasattr(self, "context_hidden_state"): #TODO: should we 'clear' the hidden state for each new dialog?
            self._init_hidden_state()

        outputs = []
        if speakers.shape[0] == 1:
            speakers = speakers[0]
        for (utterance, speaker) in zip(utterances, speakers):
            output_emb = self.lookup(utterance) # (B, S, 300)
            if speaker == 0:
                inputt = torch.cat((self.context_output, output_emb), dim=1) #(B, S+1, 300)
                output, self.speaker1_hidden_state = self.speaker1(inputt, self.speaker1_hidden_state)
                self.speaker1_hidden_state = repackage_hidden(self.speaker1_hidden_state)

                self.context_output, self.context_hidden_state = self.context_encoder(
                                                                    self.speaker1_hidden_state, 
                                                                    self.context_hidden_state)
                self.context_hidden_state = repackage_hidden(self.context_hidden_state)
                # context_output: (B, 1, 300), context_hidden_state: (1, B, 300)
                output = self.linear1(output)
            else:
                inputt = torch.cat((self.context_output, output_emb), dim=1)
                output, self.speaker2_hidden_state = self.speaker2(inputt, self.speaker2_hidden_state)
                self.speaker2_hidden_state= repackage_hidden(self.speaker2_hidden_state)
                self.context_output, self.context_hidden_state = self.context_encoder(
                                                                    self.speaker2_hidden_state, 
                                                                    self.context_hidden_state)
                self.context_hidden_state = repackage_hidden(self.context_hidden_state)
                output = self.linear2(output)
            outputs.append(output[0][-1])  # output: (B, S+1, classes)
        return torch.stack(outputs)
