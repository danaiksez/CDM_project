import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class ThreeEncoders(BaseModel):
    def __init__(self, embedding_size, hidden_size, num_classes, batch_first=False, bidirectional=False):
        super().__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.encoder1 = nn.LSTM(embedding_size, hidden_size, batch_first=self.batch_first, bidirectional=self.bidirectional)
        self.encoder2 = nn.LSTM(embedding_size, hidden_size, batch_first=self.batch_first, bidirectional=self.bidirectional)
        self.context_encoder = nn.LSTM(embedding_size, hidden_size, batch_first=self.batch_first, bidirectional=self.bidirectional)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x, turns):
        for (utt, turn) in zip(x, turns):
            inputs = torch.cat(self.context, utt)
            if turn == 0:
                output = self.encoder1(inputs))
            else:
                output = self.encoder2(inputs)
            self.context, (_, _) = self.context_encoder(output)
            output = self.linear(output)
        return output

"""
class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
"""
