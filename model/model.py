import os, torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

from slp.util.embeddings import EmbeddingsLoader

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cuda'
print("using device", DEVICE)

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
    def __init__(self, embedding_size, hidden_size, batch_size, num_classes, batch_first=False, bidirectional=False):
        super().__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.attention = True
        self.diction = load_embeddings().to(DEVICE)
        self.dict_size = len(self.diction)
        self.lookup = nn.Embedding(num_embeddings = self.dict_size, embedding_dim =300).from_pretrained(self.diction).to(DEVICE)

        self.speaker1 = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, batch_first=True, bidirectional=False)
        self.speaker2 = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, batch_first=True, bidirectional=False)
        self.context_encoder = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, batch_first=False, bidirectional=False)

        self.bidirectional = False
        if self.bidirectional:
            self.linear1 = nn.Linear(2 * hidden_size, self.num_classes)
            self.linear2 = nn.Linear(2 * hidden_size, self.num_classes)
            if self.attention:
                self.word1 = nn.Linear(2 * hidden_size, 2 * hidden_size)
                self.context1 = nn.Linear(2 * hidden_size, 1, bias=False)
                self.word2 = nn.Linear(2 * hidden_size, 2 * hidden_size)
                self.context2 = nn.Linear(2 * hidden_size, 1, bias=False)

        else:
            self.linear1 = nn.Linear(hidden_size, self.num_classes)
            self.linear2 = nn.Linear(hidden_size, self.num_classes)
            if self.attention:
                self.word1 = nn.Linear(hidden_size, hidden_size)
                self.context1 = nn.Linear(hidden_size, 1, bias=False)
                self.word2 = nn.Linear(hidden_size, hidden_size)
                self.context2 = nn.Linear(hidden_size, 1, bias=False)


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

        outputs = None if self.attention else []
        if speakers.shape[0] == 1:
            speakers = speakers[0]
        for (utterance, speaker) in zip(utterances, speakers):
            output_emb = self.lookup(utterance).to(DEVICE) # (B, S, 300)
            if speaker == 0:
                inputt = torch.cat((self.context_output, output_emb), dim=1) #(B, S+1, 300)
                output, self.speaker1_hidden_state = self.speaker1(inputt, self.speaker1_hidden_state)
                self.speaker1_hidden_state = repackage_hidden(self.speaker1_hidden_state)

                self.context_output, self.context_hidden_state = self.context_encoder(
                                                                    self.speaker1_hidden_state,
                                                                    self.context_hidden_state)
                self.context_hidden_state = repackage_hidden(self.context_hidden_state)
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
                self.speaker2_hidden_state= repackage_hidden(self.speaker2_hidden_state)
                self.context_output, self.context_hidden_state = self.context_encoder(
                                                                    self.speaker2_hidden_state,
                                                                    self.context_hidden_state)
                self.context_hidden_state = repackage_hidden(self.context_hidden_state)

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

        self.speaker1_hidden_state = self.speaker1_hidden_state.detach()
        self.speaker2_hidden_state = self.speaker2_hidden_state.detach()
        self.context_hidden_state = self.context_hidden_state.detach()
        return outputs.to(DEVICE) if self.attention else torch.stack(outputs).to(DEVICE)

class LSTM(nn.Module):
    def __init__(self, input_size,  hidden, num_layers,output_size, batch_first = True):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden = hidden

        self.diction = load_embeddings()
        self.dict_size = len(self.diction)
        self.lookup = nn.Embedding(num_embeddings = self.dict_size, embedding_dim =300).from_pretrained(self.diction)
        # print('belangrijk', input_size, hidden, num_layers)
        self.lstm = nn.LSTM(input_size, hidden,num_layers, batch_first= True)
        self.output = nn.Linear(hidden, output_size)

    def forward(self, input):
        # output_emb = []
        output = []
        for utterance in input:
            output_emb = self.lookup(utterance) # (B, S, 300)
        # input = torch.cat(output_emb, dim=1)
            h_0 = torch.zeros(self.num_layers, 1, self.hidden).to(DEVICE)
            c_0 = torch.zeros(self.num_layers, 1, self.hidden).to(DEVICE)
            out,_ = self.lstm(output_emb, (h_0,c_0))
            out = out[:, -1,:]
            # out = self.output(out)
            print(out)
            output.append(out[0])
        # print("output voor engheid", output)
        outputs = torch.stack(output,dim=0)
        # self.output = outputs
        # print("this is the output", outputs)
        output_total = self.output(outputs)
        return output_total.to(DEVICE)
