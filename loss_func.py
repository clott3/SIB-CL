import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def calpercloss(output,target):
    'Defines fractional loss of net output with data. This function is used for evaluation of bandstructures'
    loss=torch.abs(output - target)
    loss=torch.div(loss,target)
    loss[loss==float("Inf")] = 0 #ignore divide by 0 errors
    loss[torch.isnan(loss)]=0 #ignore nan errors
    loss=torch.mean(loss)
    return loss

def KLdiv(p,q):
    '''defines KL div loss for baseline_with_AE'''
    return torch.sum(p*(p/q).log())

def calDOSloss(output,target,startpt=100):
    '''Defines fractional loss of net output with data. This function is used for evaluation of DOS.
    param: startpt defines starting no. of points to not compute. We exclude first 100 data points since loss there almost 0'''
    ## TODO: To add in no. of points to compute as variable.
    diff = torch.abs(target[:,startpt:]-output[:,startpt:])
    numerator = torch.sum(diff,axis=1) # this has dim batchsize
    denominator = torch.sum(torch.abs(target[:,startpt:]),axis=1) # this has dim batchsize
    return torch.mean(numerator/denominator) # take the mean over all samples in batch

class LogLoss(nn.Module):
    '''Defines logloss as training criterion for DOS.
    A Class is used to maintain consistency with the other criterions'''
    def __init__(self):
        super(LogLoss,self).__init__()
    def forward(self,input,target):
        loss =torch.log(1 + F.l1_loss(input,target,reduction='none'))
        return torch.mean(loss)

class DOSLoss(nn.Module):
    def __init__(self,startpt=100):
        super(DOSLoss,self).__init__()
        self.startpt = startpt
    def forward(self,input,target):
        loss = calDOSloss(input,target,self.startpt)
        return loss

class FracLoss(nn.Module):
    def __init__(self):
        super(FracLoss,self).__init__()
    def forward(self,input,target):
        loss = calpercloss(input,target)
        return loss

class NTXentLoss(nn.Module):
    ''' this implements the Normalized Temperature Cross Entropy Loss use to compute Contrastive Loss in the SimCLR paper.
    This code is adapted from https://github.com/sthalles/SimCLR'''

    def __init__(self, device, batch_size, temperature, use_attention = False):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        self.mask_same = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.use_attention_bool = use_attention

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    def forward(self, zis, zjs=None):
        if not self.use_attention_bool:
            representations = torch.cat([zjs, zis], dim=0)
        else:
            representations = zis

        similarity_matrix = self.similarity_function(representations.unsqueeze(0), representations.unsqueeze(1))

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_same].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
