import numpy as np
from solveTISE import TISE
import matplotlib.pyplot as plt
import time
import h5py
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.sparse.linalg import eigs, eigsh
import scipy.sparse as sp
from scipy import interpolate
import math

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ngrid',type=int, help='number of grid points; default = 32', default=32)
parser.add_argument('--ndim',type=int, help='no. of dimensions, 2 or 3',default=2)
parser.add_argument('--nsam',type=int, help='number of samples',default=5000)
parser.add_argument('--seed',type=int, help='random seed',default=42)
parser.add_argument('--h5filename', required=True, type=str, help='REQUIRED: path to save generated h5 file, in format path/to/filename.h5')
parser.add_argument('--no_createh5', action='store_true', help='Set this flag to not create h5')
parser.add_argument('--xmax',type=float,default=5.)

args = parser.parse_args()

N = args.ngrid

h5out = args.h5filename
xmax = args.xmax
ndim = args.ndim

np.random.seed(args.seed)

if not args.no_createh5:
    with h5py.File(h5out,"a") as f:
        ## write universal (per h5 file) parameters
        f.create_dataset("universal/N",dtype='int',data=N)
        f.create_dataset("universal/gridmax",dtype='f',data=xmax)

for i in range(args.nsam):
    ## k in (0.01,0.5), c in (-2.5,2.5), pot <= 1

    kx = np.random.uniform(0.01,0.5)
    ky = np.random.uniform(0.01,0.5)
    kz = np.random.uniform(0.01,0.5)

    cx = np.random.uniform(-2.5,2.5)
    cy = np.random.uniform(-2.5,2.5)
    cz = np.random.uniform(-2.5,2.5)
    if args.ndim == 2:
        pot2d = np.abs(np.random.randn(N,N))
        sch = TISE(pot2d,dim=2, xmax=xmax)
        grid2d = sch.grid()
        pot = 0.5*kx*(grid2d[0]-cx)**2 + 0.5*ky*(grid2d[1]-cy)**2
        pot[pot > 1] = 1
        actuale = (np.sqrt(kx)+np.sqrt(ky))*0.5
        inputsolve = pot[1:-1,1:-1]
    elif args.ndim == 3:
        pot3d = np.abs(np.random.randn(N,N,N))
        sch = TISE(pot3d,dim=3,xmax=xmax)
        grid3d = sch.grid()
        pot = 0.5*kx*(grid3d[0]-cx)**2 + 0.5*ky*(grid3d[1]-cy)**2 +0.5*kz*(grid3d[2]-cz)**2
        pot[pot > 1] = 1
        actuale = (np.sqrt(kx)+np.sqrt(ky)+np.sqrt(kz))*0.5
        actuale = np.expand_dims(np.array(actuale),0)
        inputsolve = pot[1:-1,1:-1,1:-1] # ignore boundary points

    sch = TISE(inputsolve,dim=ndim,xmax=xmax)
    eigval, eigvec = sch.solve(num_eig=2)
    ind = np.argsort(eigval) #eigval and eigvec are already np arrays
    eigval = eigval[ind]
    eigvec = eigvec[:,ind]
    print(actuale)
    print(eigval)

    if not args.no_createh5:
        with h5py.File(h5out,"a") as f:
            f.create_dataset("unitcell/kx/"+str(i),dtype='f',data=kx)
            f.create_dataset("unitcell/ky/"+str(i),dtype='f',data=ky)
            f.create_dataset("unitcell/cx/"+str(i),dtype='f',data=cx)
            f.create_dataset("unitcell/cy/"+str(i),dtype='f',data=cy)
            if args.ndim == 3:
                f.create_dataset("unitcell/kz/"+str(i),dtype='f',data=kz)
                f.create_dataset("unitcell/cz/"+str(i),dtype='f',data=cz)
            f.create_dataset("unitcell/potential_fil/"+str(i),dtype='f',data=pot)
            f.create_dataset("grounde/"+str(i),dtype='f',data=actuale)
            f.create_dataset("eigval_fil/"+str(i),dtype='f',data=eigval)
            f.create_dataset("eigvec_fil/"+str(i),dtype='f',data=eigvec)
    # add fil here just to maintain consistency with dataset name of target set
    if i % 100 == 0:
        print(f"{i}samples done!")
