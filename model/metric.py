import torch
import torch.nn.functional as F


def accuracy(output, target):
    with torch.no_grad():
        output = F.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def precision_0(score):
    return score[0][0]
def precision_1(score):
    return score[0][1]
def precision_2(score):
    return score[0][2]
def precision_3(score):
    return score[0][3]
def precision_4(score):
    return score[0][4]
def precision_5(score):
    return score[0][5]
def precision_6(score):
    return score[0][6]

def recall_0(score):
    return score[1][0]
def recall_1(score):
    return score[1][1]
def recall_2(score):
    return score[1][2]
def recall_3(score):
    return score[1][3]
def recall_4(score):
    return score[1][4]
def recall_5(score):
    return score[1][5]
def recall_6(score):
    return score[1][6]

def f1_0(score):
    return score[2][0]
def f1_1(score):
    return score[2][1]
def f1_2(score):
    return score[2][2]
def f1_3(score):
    return score[2][3]
def f1_4(score):
    return score[2][4]
def f1_5(score):
    return score[2][5]
def f1_6(score):
    return score[2][6]



def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
