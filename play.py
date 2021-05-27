from data_loader import data_loaders
import numpy as np
import torch, json
import torch.nn.functional as F
import torch.nn as nn
from model.model import ThreeEncoders
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from sklearn.metrics import precision_recall_fscore_support

torch.autograd.set_detect_anomaly(True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



if __name__ == '__main__':
    model = ThreeEncoders( 300, 300, 1, 1,7, batch_first=False, bidirectional=False)
    model.load_state_dict(torch.load('saved/models/Mnist_LeNet/0526_131232/checkpoint-epoch7.pth'))
    model.eval()
    # output, attention =
