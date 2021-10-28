#!/usr/bin/env python3
"""
Created on Jun  9 2020

@author: charlotte
"""

import torch.nn as nn
import torch
import math
import numpy as np

class Encoder(nn.Module):
    def __init__(self, Lv, ks, dim = 2):
        super(Encoder, self).__init__()
        if dim == 2:
            self.enc_block2d = nn.Sequential(
                nn.Conv2d(1, Lv[0], kernel_size=ks,stride=1,padding=math.ceil((ks-1)/2)),
                nn.BatchNorm2d(Lv[0]),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                # nn.Dropout(p=0.2),
                nn.Conv2d(Lv[0], Lv[1], kernel_size=ks,stride=1,padding=math.ceil((ks-1)/2)),
                nn.BatchNorm2d(Lv[1]),
                nn.ReLU(),
                nn.MaxPool2d(4,4),
                # nn.Dropout(p=0.2),
                nn.Conv2d(Lv[1], Lv[2], kernel_size=ks,stride=1,padding=math.ceil((ks-1)/2)),
                nn.BatchNorm2d(Lv[2]),
                nn.ReLU(),
                nn.MaxPool2d(4,4)
                )
        elif dim == 3:
            assert(Lv[2]==Lv[3])
            self.enc_block3d = nn.Sequential(
                nn.Conv3d(1, Lv[0], kernel_size=ks,stride=1,padding=math.ceil((ks-1)/2)),
                nn.BatchNorm3d(Lv[0]),
                nn.ReLU(),
                nn.MaxPool3d(2,2),
                # nn.Dropout(p=0.2),
                nn.Conv3d(Lv[0], Lv[1], kernel_size=ks,stride=1,padding=math.ceil((ks-1)/2)),
                nn.BatchNorm3d(Lv[1]),
                nn.ReLU(),
                nn.MaxPool3d(2,2),
                # nn.Dropout(p=0.2),
                nn.Conv3d(Lv[1], Lv[2], kernel_size=ks,stride=1,padding=math.ceil((ks-1)/2)),
                nn.BatchNorm3d(Lv[2]),
                nn.ReLU(),
                nn.MaxPool3d(2,2),
                nn.Conv3d(Lv[2], Lv[3], kernel_size=ks,stride=1,padding=math.ceil((ks-1)/2)),
                nn.BatchNorm3d(Lv[3]),
                nn.ReLU(),
                nn.MaxPool3d(2,2)
                )
        self.avgpool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.avgpool2d = nn.AdaptiveAvgPool2d((1, 1))

        self.fcpart = nn.Sequential(
            nn.Linear(Lv[2] * 1 * 1, Lv[3]),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(Lv[3], Lv[4]),
            )
        self.Lv = Lv
        self.dim = dim

    def forward(self, x):
        if self.dim == 2:
            x = self.enc_block2d(x)
            x = self.avgpool2d(x)

        elif self.dim == 3:
            x = self.enc_block3d(x)
            x = self.avgpool3d(x)
        else:
            raise ValueError("dim has to be 2 or 3!")
        x = x.view(-1, self.Lv[2] * 1 * 1)
        x = self.fcpart(x)
        return x

class Projector(nn.Module):
    ''' Projector network accepts a variable number of layers indicated by depth.
    Option to include batchnorm after every layer.'''

    def __init__(self, Lvpj, hidden_dim, bnorm = False, depth = 2):
        super(Projector, self).__init__()
        print(f"Using projector; batchnorm {bnorm} with depth {depth}")
        nlayer = [nn.BatchNorm1d(Lvpj[0])] if bnorm else []
        list_layers = [nn.Linear(hidden_dim, Lvpj[0])] + nlayer + [nn.ReLU()]
        for _ in range(depth-2):
            list_layers += [nn.Linear(Lvpj[0], Lvpj[0])] + nlayer + [nn.ReLU()]
        list_layers += [nn.Linear(Lvpj[0],Lvpj[1])]
        self.proj_block = nn.Sequential(*list_layers)

    def forward(self, x):
        x = self.proj_block(x)
        return x


class PredictorBS(nn.Module):
    ''' This creates n separate branches that are initialized similarly
    for each of the n bands. Note n must be <= 10.
    If more than 10 bands needed, moduleDict need to cater for 2 digit index'''

    def __init__(self, num_bands, hidden_dim, num_predict, Lvp):
        super(PredictorBS, self).__init__()

        self.num_bands = num_bands
        self.num_predict = num_predict
        self.hidden_dim = hidden_dim
        self.Lvp = Lvp
        # we create a module dict with keys band0, band1, band2, etc for each branch
        # Initialize each branch with the same architecture
        self.branches = nn.ModuleDict()
        for i in range(self.num_bands):
            branch = 'band'+str(i)
            self.branches.update({branch: self.fc_block()})

    def fc_block(self):
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.Lvp[0]),
            nn.ReLU(),
            nn.Linear(self.Lvp[0], self.Lvp[1]),
            nn.ReLU(),
            nn.Linear(self.Lvp[1], self.Lvp[2]),
            nn.ReLU(),
            nn.Linear(self.Lvp[2], self.num_predict),
            )

    def forward(self, x):
        out = self.branches['band0'](x)
        out = out.unsqueeze_(2) # forward pass for first band
        if self.num_bands > 1:
            for i in range(1,self.num_bands):
                branch = 'band'+str(i)
                outband = self.branches[branch](x).unsqueeze_(2) # forward pass for each of the remaining bands
                out = torch.cat((out,outband),dim=2) # concatenate all bands
        return out ## This has shape (batchsize, 625, n) if n > 1 or (batchsize,625,1) if n == 1

class PredictorDOS(nn.Module):

    def __init__(self, num_dos, hidden_dim, Lvp):
        super(PredictorDOS, self).__init__()
        self.num_dos = num_dos
        self.hidden_dim = hidden_dim
        self.Lvp = Lvp
        self.DOSband = self.fc_block()

    def fc_block(self):
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.Lvp[0]),
            nn.ReLU(),
            nn.Linear(self.Lvp[0], self.Lvp[1]),
            nn.ReLU(),
            nn.Linear(self.Lvp[1], self.Lvp[2]),
            nn.ReLU(),
            nn.Linear(self.Lvp[2], self.num_dos),
            )

    def forward(self, x):
        x = self.DOSband(x)
        return x

class PredictorEig(nn.Module):

    def __init__(self,num_eval, hidden_dim, Lvp):
        super(PredictorEig, self).__init__()

        self.num_eval = num_eval
        self.hidden_dim = hidden_dim
        self.Lvp = Lvp
        self.Eigband = self.fc_block()

    def fc_block(self):
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.Lvp[0]),
            nn.ReLU(),
            nn.Linear(self.Lvp[0], self.Lvp[1]),
            nn.ReLU(),
            nn.Linear(self.Lvp[1], self.Lvp[2]),
            nn.ReLU(),
            nn.Linear(self.Lvp[2], self.num_eval)
            )

    def forward(self, x):
        x = self.Eigband(x)
        return x

class Net(nn.Module):
    def __init__(self,Lv=[],ks=7,Lvp=[],nbands=6,ndos=400,neval=1,predict='',dim=2):
        super(Net, self).__init__()
        self.enc = Encoder(Lv,ks,dim)
        if predict == 'bandstructures':
            self.predictor = PredictorBS(nbands, Lv[-1], 625, Lvp)
        elif predict == 'eigvec':
            self.predictor = PredictorBS(neval, Lv[-1],62**dim, Lvp)
        elif predict == 'oneband':
            self.predictor = PredictorBS(1,Lv[-1],Lvp) # only one branch
        elif predict == 'DOS':
            self.predictor = PredictorDOS(ndos,Lv[-1],Lvp)
        elif predict == 'eigval':
            self.predictor = PredictorEig(neval,Lv[-1],Lvp)
        else:
            raise ValueError("invalid prediction problem")
    def forward(self, x):
      x = self.enc(x)
      x = self.predictor(x)
      return x

# Define new sub module classes for CL in order to keep the same state dict keys
class Enc(nn.Module):
    def __init__(self,Lv,ks,dim):
        super(Enc, self).__init__()
        self.enc= Encoder(Lv,ks,dim)
    def forward(self, x):
      x = self.enc(x)
      return x

class Proj(nn.Module):
    def __init__(self,Lvpj,latent_dim, bnorm = False, depth = 2):
        super(Proj, self).__init__()
        self.projector = Projector(Lvpj,latent_dim, bnorm = bnorm, depth = depth)
    def forward(self, x):
      x = self.projector(x)
      return x

class Pred(nn.Module):
    def __init__(self,latent_dim=256,Lvp=[],nbands=6,ndos=400,neval=1,predict='',dim=2):
        super(Pred, self).__init__()
        if predict == 'bandstructures':
            self.predictor = PredictorBS(nbands,latent_dim,625,Lvp)
        elif predict == 'eigvec':
            self.predictor = PredictorBS(neval,latent_dim,62**dim,Lvp)
        elif predict == 'oneband':
            self.predictor = PredictorBS(1,latent_dim,Lvp) # only one branch
        elif predict == 'DOS':
            self.predictor = PredictorDOS(ndos,latent_dim,Lvp)
        elif predict == 'eigval':
            self.predictor = PredictorEig(neval,latent_dim,Lvp)
        else:
            raise ValueError("invalid predict problem")
    def forward(self, x):
      x = self.predictor(x)
      return x

if __name__ == '__main__':
    from prettytable import PrettyTable
    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params+=param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params
