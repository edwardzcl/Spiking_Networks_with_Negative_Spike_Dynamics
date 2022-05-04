import torch.nn as nn

class BasicCriterion(nn.Module):
    pass

def get_criterion_by_name(name,args=None):
    if name=='cross_entropy':
        return nn.CrossEntropyLoss()
    if name=='null':
        return BasicCriterion()