import copy
import random
import torch
from torch import nn
import torch.nn.functional as F

from models import Enc, Proj # model classes
from loss_func import NTXentLoss

def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class SimCLR(nn.Module):

    def __init__(self, Lv,Lvpj,ks,ndim,device,batchsize_ptxt,temperature,
                    use_projector = False,bnorm = False, depth = 2):
        super().__init__()
        self.NTXent = NTXentLoss(device,batchsize_ptxt,temperature)
        self.projector = Proj(Lvpj,Lv[-1], bnorm = bnorm, depth = depth)
        self.use_projector = use_projector
        self.encoder = Enc(Lv,ks,ndim)

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        if self.use_projector:
            z1 = self.projector(z1)
            z2 = self.projector(z2)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        loss = self.NTXent(z1, z2)
        return loss

class byol_predictor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024, eps=1e-5, momentum=1-0.9),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(1024, 256)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class BYOL(nn.Module):
    def __init__(self, Lv,Lvpj,ks,ndim,use_projector = False, bnorm = False, depth = 2):
        super().__init__()

        self.projector = Proj(Lvpj,Lv[-1], bnorm = bnorm, depth = depth)
        self.use_projector = use_projector
        self.encoder = Enc(Lv,ks,ndim)
        self.target_enc = copy.deepcopy(self.encoder)
        if self.use_projector:
            self.target_encpj = copy.deepcopy(self.projector)

        if not use_projector:
            in_dim = Lv[-1]
        else:
            in_dim = Lvpj[-1]
        self.online_predictor = byol_predictor(in_dim)
        print("print byol predictor in dim", in_dim)

    def target_ema(self, k, K, base_ema=4e-3):
        # tau_base = 0.996
        # base_ema = 1 - tau_base = 0.996
        return 1 - base_ema * (cos(pi*k/K)+1)/2
        # return 1 - (1-self.tau_base) * (cos(pi*k/K)+1)/2

    @torch.no_grad()
    def update_moving_average(self, global_step, max_steps):
        tau = self.target_ema(global_step, max_steps)
        for online, target in zip(self.encoder.parameters(), self.target_enc.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data

    def forward(self, x1, x2):
        f_o, pj_o, h_o = self.encoder, self.projector, self.online_predictor

        f_t      = self.target_enc
        if self.use_projector:
            pj_t      = self.target_encpj

        z1_o = f_o(x1)
        z2_o = f_o(x2)
        if self.use_projector:
            z1_o = pj_o(z1_o)
            z2_o = pj_o(z2_o)
        p1_o = h_o(z1_o)
        p2_o = h_o(z2_o)

        with torch.no_grad():
            if self.use_projector:
                z1_t = pj_t(f_t(x1))
                z2_t = pj_t(f_t(x2))
            else:
                z1_t = f_t(x1)
                z2_t = f_t(x2)
        L = D(p1_o, z2_t) / 2 + D(p2_o, z1_t) / 2
        return L

if __name__ == "__main__":
    pass
