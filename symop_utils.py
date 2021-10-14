import numpy as np
import torch
import random
import torch.nn.functional as F

def translate_tensor(tensor,input_size=32, prob=None):
    '''Data augmentation function to enforce periodic boundary conditions. Inputs are arbitrarily translated in each dimension'''
    ndim = len(tensor[0,0,:].shape)
    p1 = np.random.choice(int(1/prob), len(tensor)) if prob is not None else np.zeros(len(tensor), dtype=int)

    for i in range(len(tensor)):
        tensor1 = tensor[i,0,:]
        if p1[i] == 0:
            if ndim == 2:
                tensor1 = torch.roll(tensor1,(np.random.choice(input_size),np.random.choice(input_size)),(0,1)) # translate by random no. of units (0-input_size) in each axis
            elif ndim == 3:
                tensor1 = torch.roll(tensor1,(np.random.choice(input_size),np.random.choice(input_size),np.random.choice(input_size)),(0,1,2))
        if i == 0:
            newtensor = tensor1.unsqueeze(0).unsqueeze(0) # add back channel dim and batch dim
        else:
            newtensor = torch.cat((newtensor,tensor1.unsqueeze(0).unsqueeze(0)),dim=0)
    return newtensor

def rotate_tensor(tensor, prob=None):

    ndim = len(tensor[0,0,:].shape)
    p1 = np.random.choice(int(1/prob), len(tensor)) if prob is not None else np.zeros(len(tensor), dtype=int)

    for i in range(len(tensor)):
        tensor1 = tensor[i,0,:]
        if p1[i] == 0:
            rottimes = np.random.choice(4) # 4-fold rotation; rotate by 0, 90, 280 or 270
            rotaxis = np.random.choice(ndim) # axis to rotate [0,1], [1,0] in 2D (double count negative rot is ok) and [0,1], [1,2], [2,0] in 3D (negative rotation covered by k = 3)
            tensor1 = torch.rot90(tensor1,k=rottimes,dims=[rotaxis,(rotaxis+1)%(ndim)])
        if i == 0:
            newtensor = tensor1.unsqueeze(0).unsqueeze(0)
        else:
            newtensor = torch.cat((newtensor,tensor1.unsqueeze(0).unsqueeze(0)),dim=0)
    return newtensor

def flip_tensor(tensor, prob=None):

    ndim = len(tensor[0,0,:].shape)
    p1 = np.random.choice(int(1/prob), len(tensor)) if prob is not None else np.zeros(len(tensor), dtype=int)

    for i in range(len(tensor)):
        tensor1 = tensor[i,0,:]
        if p1[i] == 0:
            if ndim == 2:
                flipaxis = random.choice([[0],[1],[]]) # flip hor, ver, or None (dont include Diagonals = flip + rot90)
            elif ndim == 3:
                flipaxis = random.choice([[0],[1],[2],[]]) # flip x, y, z or None (dont include Diagonals = flip + rotate)
            tensor1 = torch.flip(tensor1,flipaxis)
        if i == 0:
            newtensor = tensor1.unsqueeze(0).unsqueeze(0)
        else:
            newtensor = torch.cat((newtensor,tensor1.unsqueeze(0).unsqueeze(0)),dim=0)
    return newtensor


def scale_tensor(tensor, mineps = 1, maxeps = 20, prob=None):
    ndim = len(tensor[0,0,:].shape)
    p1 = np.random.choice(int(1/prob), len(tensor)) if prob is not None else np.zeros(len(tensor), dtype=int)

    for i in range(len(tensor)):
        tensor1 = tensor[i,0,:]
        if p1[i] == 0:
            tensornew1 = torch.zeros_like(tensor1)
            tensor1 = tensor1*(maxeps-mineps)+mineps # unstandardize input
            while not (torch.min(tensornew1)>mineps and torch.max(tensornew1)<maxeps): # we normalized input to be in 0 and 1
                factor = -random.uniform(-2,0) # uniform includes low but excludes high. want to include 1 but exclude 0
                tensornew1 = tensor1*factor
            tensornew1 = (tensornew1-mineps)/(maxeps-mineps)
        else:
            tensornew1 = tensor1

        if i == 0:
            newtensor = tensornew1.unsqueeze(0).unsqueeze(0)
        else:
            newtensor = torch.cat((newtensor,tensornew1.unsqueeze(0).unsqueeze(0)),dim=0)
    return newtensor # re-standardize input

def get_aug(x,translate=False,flip=False,rotate=False,scale=False,p=1,pg_uniform=False):
    ndim = len(x[0,0,:].shape)
    if translate:
        x = translate_tensor(x, prob = p)
    if pg_uniform:
        x = get_pg_uni(x, dim = ndim, prob = p)
    else:
        if flip:
            x = flip_tensor(x, prob = p)
        if rotate:
            x = rotate_tensor(x, prob = p)
    if scale:
        x = scale_tensor(x, prob = p)
    return x

## Define primitive operations
def iden(x):
    return x
def r2_001(x):
    x = torch.rot90(x,k=2,dims=[0,1]); return x
def r2_010(x):
    x = torch.rot90(x,k=2,dims=[2,0]); return x
def r2_100(x):
    x = torch.rot90(x,k=2,dims=[1,2]); return x
def r4_001p(x):
    x = torch.rot90(x,k=1,dims=[0,1]); return x
def r4_001m(x):
    x = torch.rot90(x,k=-1,dims=[0,1]); return x
def r4_010p(x):
    x = torch.rot90(x,k=1,dims=[2,0]); return x
def r4_010m(x):
    x = torch.rot90(x,k=-1,dims=[2,0]); return x
def r4_100p(x):
    x = torch.rot90(x,k=1,dims=[1,2]); return x
def r4_100m(x):
    x = torch.rot90(x,k=-1,dims=[1,2]); return x
def m_100(x):
    x = torch.flip(x,[0]); return x
def m_010(x):
    x = torch.flip(x,[1]); return x
def m_001(x):
    x = torch.flip(x,[2]); return x
def com2(f, g):
    return lambda x: f(g(x))
def com3(f, g, h):
    return lambda x: f(g(h(x)))
## define primitive operations in 2d
def r4p(x):
    x = torch.rot90(x,k=1); return x
def r4m(x):
    x = torch.rot90(x,k=-1); return x
def r2(x):
    x = torch.rot90(x,k=2); return x
def m_10(x):
    x = torch.flip(x,[0]); return x
def m_01(x):
    x = torch.flip(x,[1]); return x
def m_11(x):
    return com2(m_01,r4p)(x)
def m_m11(x):
    return com2(m_01,r4m)(x)


def get_pg_uni(x, dim = 3, prob=None):
    if dim == 3:
        oplist = [r2_001, r2_010, r2_100,
                com2(r4_001p,r4_100p), com2(r4_100m,r4_010p), com2(r4_001m,r4_100p),
                com2(r4_010m,r4_001p), com2(r4_010m,r4_100m),com2(r4_001p,r4_010p),
                com2(r4_010p,r4_100p),com2(r4_010m,r4_100p),
                com2(r4_001p,r2_100),com2(r4_001m,r2_100),
                r4_001m,r4_001p,r4_100m,
                com2(r4_100p,r2_010),com2(r4_100m,r2_010),
                r4_100p,r4_010p,com2(r4_010m,r2_100),
                r4_010m,com2(r4_010p,r2_100),
                com2(m_100,r2_100), m_001, m_010, m_100,
                com3(m_001,r4_001m,r4_100p),com3(m_100,r4_001p,r4_100p),
                com3(m_001,r4_001p,r4_100p),com3(m_010,r4_001p,r4_100p),
                com3(m_100,r4_010p,r4_100p),com3(m_001,r4_010p,r4_100p),
                com3(m_010,r4_010m,r4_100p),com3(m_010,r4_010p,r4_100p),
                com2(m_010,r4_001p),com2(m_100,r4_001p),com2(m_001,r4_001p),
                com2(m_001,r4_001m),com2(m_100,r4_100p),com2(m_001,r4_100p),
                com2(m_010,r4_100p),com2(m_100,r4_100m),com2(m_010,r4_010m),
                com2(m_100,r4_010p),com2(m_010,r4_010p),com2(m_001,r4_010p) ]
    elif dim == 2:
        oplist = [r2, r4p, r4m, m_10, m_01, m_11, m_m11]
    else:
        raise ValueError("only 2d or 3d")

    p1 = np.random.choice(int(1/prob), len(x)) if prob is not None else np.zeros(len(x), dtype=int)
    for i in range(len(x)):
        x1 = x[i,0,:]
        if p1[i] == 0:
            k = np.random.randint(len(oplist))
            x1 = oplist[k](x1)
        if i == 0:
            newx = x1.unsqueeze(0).unsqueeze(0)
        else:
            newx = torch.cat((newx,x1.unsqueeze(0).unsqueeze(0)),dim=0)
    return newx
