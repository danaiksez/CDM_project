import os, torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

from slp.util.embeddings import EmbeddingsLoader
from slp.helpers import PackSequence, PadPackedSequence

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_embeddings():
    cwd = os.getcwd()
    loader = EmbeddingsLoader(cwd + '/data/embeddings/glove.6B.300d.txt', 300)
    word2idx, idx2word, embeddings = loader.load()
    embeddings = torch.tensor(embeddings)
    return embeddings.to(DEVICE)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class ThreeEncoders(BaseModel):
    def __init__(self, input_size, hidden, batch_size, num_layers, output_size, batch_first=False, bidirectional=False):
        super().__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        self.hidden = hidden
        self.num_classes = output_size

        self.attention = True
        self.diction = load_embeddings().to(DEVICE)
        self.dict_size = len(self.diction)
        self.lookup = nn.Embedding(num_embeddings = self.dict_size, embedding_dim =300).from_pretrained(self.diction).to(DEVICE)

        self.speaker1 = nn.GRU(input_size=input_size, hidden_size=hidden, batch_first=True, bidirectional=False)
        self.speaker2 = nn.GRU(input_size=input_size, hidden_size=hidden, batch_first=True, bidirectional=False)
        self.context_encoder = nn.GRU(input_size=input_size, hidden_size=hidden, batch_first=False, bidirectional=False)

        self.bidirectional = False
        hidden = 2 * self.hidden if self.bidirectional else self.hidden
        self.linear1 = nn.Linear(hidden, self.num_classes)
        self.linear2 = nn.Linear(hidden, self.num_classes)
        if self.attention:
            self.word1 = nn.Linear(hidden, hidden)
            self.context1 = nn.Linear(hidden, 1, bias=False)
            self.word2 = nn.Linear(hidden, hidden)
            self.context2 = nn.Linear(hidden, 1, bias=False)

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.speaker1_hidden_state = torch.zeros(batch_size, 1, self.hidden).to(DEVICE)
        self.speaker2_hidden_state = torch.zeros(batch_size, 1, self.hidden).to(DEVICE)
        self.context_hidden_state = torch.zeros(batch_size, 1, self.hidden).to(DEVICE)
        self.context_output = torch.zeros(1, batch_size, self.hidden).to(DEVICE)

    def forward(self, utterances, speakers):
        #if not hasattr(self, "context_hidden_state"):
        #    self._init_hidden_state()

        #import pdb; pdb.set_trace()
        outputs = None if self.attention else []
        if speakers.shape[0] == 1:
            speakers = speakers[0]
        for (utterance, speaker) in zip(utterances, speakers):
            output_emb = self.lookup(utterance).to(DEVICE) # (B, S, 300)
            if speaker == 0:
                inputt = torch.cat((self.context_output, output_emb), dim=1) #(B, S+1, 300)
                output, self.speaker1_hidden_state = self.speaker1(inputt, self.speaker1_hidden_state)
        #        self.speaker1_hidden_state = repackage_hidden(self.speaker1_hidden_state)

                self.context_output, self.context_hidden_state = self.context_encoder(
                                                                    self.speaker1_hidden_state,
                                                                    self.context_hidden_state)
        #        self.context_hidden_state = repackage_hidden(self.context_hidden_state)
                # context_output: (B, 1, 300), context_hidden_state: (1, B, 300)

                if self.attention:
                    output_a = self.word1(output)
                    output_a = self.context1(output_a)
                    output_a = F.softmax(output_a,dim=1)
                    output = (output * output_a).sum(1)

                output = self.linear1(output)
            else:
                inputt = torch.cat((self.context_output, output_emb), dim=1)
                output, self.speaker2_hidden_state = self.speaker2(inputt, self.speaker2_hidden_state)
        #        self.speaker2_hidden_state= repackage_hidden(self.speaker2_hidden_state)
                self.context_output, self.context_hidden_state = self.context_encoder(
                                                                    self.speaker2_hidden_state,
                                                                    self.context_hidden_state)
        #        self.context_hidden_state = repackage_hidden(self.context_hidden_state)

                if self.attention:
                    output_a = self.word2(output)
                    output_a = self.context2(output_a)
                    output_a = F.softmax(output_a,dim=1)
                    output = (output * output_a).sum(1)

                output = self.linear2(output)

            #import pdb; pdb.set_trace()
            if self.attention:
                if outputs == None:
                    outputs = output
                else:
                    outputs = torch.cat((outputs, output), dim=0)
            else:
                outputs.append(output[0][-1])  # output: (B, S+1, classes

        #self.speaker1_hidden_state = self.speaker1_hidden_state.detach()
        #self.speaker2_hidden_state = self.speaker2_hidden_state.detach()
        #self.context_hidden_state = self.context_hidden_state.detach()
        return outputs.to(DEVICE) if self.attention else torch.stack(outputs).to(DEVICE)


class GRU(nn.Module):
    def __init__(self, input_size, hidden, batch_size, num_layers,output_size, batch_first = True):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden = hidden
        self.batch_size = batch_size

        self.diction = load_embeddings()
        self.dict_size = len(self.diction)
        self.lookup = nn.Embedding(num_embeddings = self.dict_size, embedding_dim =300).from_pretrained(self.diction)
        self.gru = nn.GRU(input_size=input_size, hidden_size=self.hidden, batch_first=bool(batch_first), bidirectional=False)
        self.output = nn.Linear(hidden, output_size)
        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.h_0 = torch.zeros(1, batch_size, self.hidden).to(DEVICE)  # changed order
        

    def forward(self, input):
        output = []
        try:
            for utterance in input:
                if utterance.size(1) == 0:
                    continue
                output_emb = self.lookup(utterance) # (B, S, 30
                out, self.h_0 = self.gru(output_emb, self.h_0)
                out = out[:, -1,:]
                out = self.output(out)
                output.append(out)
            outputs = torch.stack(output, dim=0)
        except:
            import pdb; pdb.set_trace()
        return outputs.squeeze(1).to(DEVICE)