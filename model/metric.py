import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support


def accuracy(output, target):
    with torch.no_grad():
        output = F.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def precision(output, target):
    with torch.no_grad():
        pred = F.softmax(output, dim=1)
        pred = torch.argmax(pred, dim=1)
        pred = pred.clone().cpu().numpy()
        labs = target.clone().cpu().numpy()
        prec, _, _, _ = precision_recall_fscore_support(labs, pred, labels=[0,1,2,3,4,5,6])
    return prec


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
