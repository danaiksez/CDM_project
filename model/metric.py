import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score


def accuracy(output, target):
    with torch.no_grad():
        output = F.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def precision(output,target):
    with torch.no_grad():
        #output = F.softmax(output, dim=1)
        #pred = torch.argmax(output, dim=1)
        pred = output
        assert pred.shape[0] == len(target)
        out = pred.cpu()
        trg = target.cpu()
        precision = precision_score(out, trg, average='macro')
        return precision

def recall(output,target):
    with torch.no_grad():
        #output = F.softmax(output, dim=1)
        #pred = torch.argmax(output, dim=1)
        pred = output
        assert pred.shape[0] == len(target)
        out = pred.cpu()
        trg = target.cpu()
        recall = recall_score(out, trg, average='macro')
        return recall


def f1_score(output,target):
    with torch.no_grad():
        import pdb; pdb.set_trace()
        output = F.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        out = pred.cpu()
        trg = target.cpu()
        f1 = f1_score(out, trg, average='macro')
        return f1


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
