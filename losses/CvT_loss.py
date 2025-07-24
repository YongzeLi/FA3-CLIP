import torch as th
import torch.nn as nn
import torch.nn.functional as F

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if isinstance(output, list):
        output = output[-1]

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res

def linear_combination(x, y, epsilon):
        return epsilon*x + (1-epsilon)*y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' \
            else loss.sum() if reduction == 'sum' else loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss/n, nll, self.epsilon)

class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()
    def forward(self, x, target):
        loss = th.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

def build_criterion(config, train=True):
    if config.AUG.MIXUP_PROB > 0.0 and config.LOSS.LOSS == 'softmax':
        criterion = SoftTargetCrossEntropy() if train else nn.CrossEntropyLoss()
    elif config.LOSS.LABEL_SMOOTHING > 0.0 and config.LOSS.LOSS == 'softmax':
        criterion = LabelSmoothingCrossEntropy(config.LOSS.LABEL_SMOOTHING)
    elif config.LOSS.LOSS == 'softmax':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError('Unkown loss {}'.format(config.LOSS.LOSS))
    return criterion
