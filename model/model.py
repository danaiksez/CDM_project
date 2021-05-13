import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class ThreeEncoders(BaseModel):
    def __init__(self, embedding_size, hidden_size, num_classes, batch_first=False, bidirectional=False):
        super().__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.speaker1 = nn.LSTM(embedding_size, hidden_size, batch_first=self.batch_first, bidirectional=self.bidirectional)
        self.speaker2 = nn.LSTM(embedding_size, hidden_size, batch_first=self.batch_first, bidirectional=self.bidirectional)
        self.context_encoder = nn.LSTM(embedding_size, hidden_size, batch_first=self.batch_first, bidirectional=self.bidirectional)
        self.linear = nn.Linear(hidden_size, num_classes)

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.speaker1_hidden_state = torch.zeros(2, batch_size, self.hidden_size)
        self.speaker2_hidden_state = torch.zeros(2, batch_size, self.hidden_size)
        self.context_hidden_state = torch.zeros(2, batch_size, self.hidden_size)
        self.speaker1_hidden_state = self.speaker1_hidden_state.to(DEVICE)
        self.speaker2_hidden_state = self.speaker2_hidden_state.to(DEVICE)
        self.context_hidden_state = self.context_hidden_state.to(DEVICE)

    def forward(self, utterances, speakers):
        if not hasattr(self, "self.context_hidden_state"):
            self._init_hidden_state()

        for (utterance, speaker) in zip(utterances, speakers):
            outputs = []
            if speaker == 0:
                inputt = torch.cat(self.context_output, utterance)
                output1, self.speaker1_hidden_state = self.speaker1(inputt)
                self.context_output, self.context_hidden_state = self.context_encoder(self.speaker1_hidden_state)
                outputs.append(output1)
            else:
                inputt = torch.cat(self.context_output, utterance)
                output2, self.speaker2_hidden_state = self.speaker2(inputt)
                self.context_output, self.context_hidden_state = self.context_encoder(self.speaker2_hidden_state)
                outputs.append(output2)
        return outputs
