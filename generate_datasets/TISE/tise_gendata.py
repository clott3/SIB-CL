import numpy as np
from solveTISE import TISE
import time
import h5py
import sys
sys.path.append('./../PhC/')
from fourier_phc import FourierPhC
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.sparse.linalg import eigs, eigsh
import scipy.sparse as sp
from scipy import interpolate
import argparse
import math

"""instructions:
- to generate target for UC sampled at target res use defaults
- to generate target for UC sampled at some other res, specify --orires 1200 (e.g.)
- to generate lowres, specify --lowres flag. for UC sampled at target res add specify --orires 32 (target res)"""

parser = argparse.ArgumentParser()
parser.add_argument('--ngrid',type=int, help='number of grid points; default = 32', default=32)
parser.add_argument('--lrgrid',type=int, help='number of LR grid points; default = 5', default=5)

parser.add_argument('--ndim',type=int, help='no. of dimensions, 2 or 3',default=2)
parser.add_argument('--nsam',type=int, help='number of samples',default=5000)
parser.add_argument('--seed',type=int, help='random seed',default=42)

parser.add_argument('--h5filename', required=True, type=str, help='REQUIRED: path to save generated h5 file, in format path/to/filename.h5')
parser.add_argument('--maxF',type=int, help='maximum number of fourier components',default=2)
parser.add_argument('--no_createh5', action='store_true', help='Set this flag to not create h5')
parser.add_argument('--maxfill',type=float, help='maximum filling ratio',default=0.99)
parser.add_argument('--minfill',type=float, help='maximum filling ratio',default=0.01)

parser.add_argument('--sigmafactor',type=float, help='sigma for filtered',default=20.)
parser.add_argument('--lowres',action='store_true', help='create lowres and save high res input')
parser.add_argument('--xmax',type=float,default=5.)
parser.add_argument('--maxeps',type=float, default=1.0)
parser.add_argument('--orires',type=int, help ='to create high resolution of original, e.g. 1200, default not used', default=0)
# parser.add_argument('--no_orires', action='store_true')

# parser.add_argument('--refgrid',type=int, default=32)
parser.add_argument('--epsneg', action='store_true')
# parser.add_argument('--epszero', action='store_true')
parser.add_argument('--centre_min', action='store_true')

parser.add_argument('--use_fill', action='store_true')

args = parser.parse_args()

N = args.ngrid
if args.lowres:
    N = args.lrgrid
h5out = args.h5filename + ".h5"
xmax = args.xmax
maxF = args.maxF
orires = args.orires
ndim = args.ndim
maxeps=args.maxeps

np.random.seed(args.seed)

phc = FourierPhC(dim=ndim,maxF=maxF,maxeps=args.maxeps, mineps=0.,\
                minfill=args.minfill,maxfill=args.maxfill,\
                use_fill = args.use_fill)

if not args.no_createh5:
    with h5py.File(h5out,"a") as f:
        f.create_dataset("universal/N",dtype='int',data=N)
        f.create_dataset("universal/gridmax",dtype='f',data=xmax)
        f.create_dataset("universal/maxF",dtype='f',data=maxF)
        f.create_dataset("universal/maxeps",dtype='f',data=args.maxeps)
        f.create_dataset("universal/xmax",dtype='f',data=args.xmax)
        f.create_dataset("universal/usefill",dtype='f',data=args.use_fill)
    ## write universal (per h5 file) parameters

for i in range(args.nsam):
    uccoefs, ucgvecs, epsin, epsout, uclevel, filling = phc.get_random()
    # epslow = np.min([epsin,epsout])
    epshi = np.max([epsin,epsout])
    if args.epsneg: # if epsneg, we only sample 1 eps and take negative
        epslow = -epshi
    else:
        epslow = 0
    if args.orires != 0:
        totalres = int(orires/0.6) # this is the original sample, res 2000
    else:
        totalres = int(args.ngrid/0.6)

    input = phc.getunitcell(uccoefs, ucgvecs, epslow, epshi, uclevel,ucres=totalres)

    if args.ndim == 2:
        input = input[int(totalres/5):-int(totalres/5),int(totalres/5):-int(totalres/5)]
    elif args.ndim == 3:
        input = input[int(totalres/5):-int(totalres/5),int(totalres/5):-int(totalres/5),int(totalres/5):-int(totalres/5)]

    input = gaussian_filter(input,(totalres/args.sigmafactor)) # this is our high res input from [-xmax,xmax] inclusive of boundaries.

    if args.ndim == 2:
        if args.orires != 0:
            x = np.linspace(-xmax,xmax,input.shape[0])
            y = np.linspace(-xmax,xmax,input.shape[1])
            f = interpolate.interp2d(x,y, input)
            xnew = np.linspace(-xmax,xmax,N)
            ynew = np.linspace(-xmax,xmax,N)
            input = f(xnew,ynew)
        inputsolve = input[1:-1,1:-1] # Dirichlet BC: ignore boundary points

    elif args.ndim == 3:
        if args.orires != 0:
            x = np.linspace(-xmax,xmax,input.shape[0])
            y = np.linspace(-xmax,xmax,input.shape[1])
            z = np.linspace(-xmax,xmax,input.shape[2])

            f = interpolate.RegularGridInterpolator((x,y,z), input)
            xnew = np.linspace(-xmax,xmax,N)
            ynew = np.linspace(-xmax,xmax,N)
            znew = np.linspace(-xmax,xmax,N)
            newpoints = np.array(np.meshgrid(xnew,ynew,znew)).reshape(3,-1).T

            input = f(newpoints).reshape(N,N,N)
            input = np.swapaxes(input,0,1) # transposing swap axes.
        # Checked that agree at boundaries. i.e. for i = [0,1], input[i,:,:] == inputnew[i,:,:], input[:,i,:] == inputnew[:,i,:], etc
        inputsolve = input[1:-1,1:-1,1:-1] # ignore boundary points

    else:
        raise ValueError("only 2D or 3D")

    sch = TISE(inputsolve,dim=ndim,xmax=xmax)
    eigval, eigvec = sch.solve(num_eig=2)

    ind = np.argsort(eigval) #eigval and eigvec are already np arrays
    eigval = eigval[ind]
    eigvec = eigvec[:,ind]

    if args.lowres:
        xnew = np.linspace(-xmax,xmax,args.ngrid)
        ynew = np.linspace(-xmax,xmax,args.ngrid)
        inputsave = f(xnew,ynew) # want lowres input to be the same as target
    else:
        inputsave = input

    with h5py.File(h5out,"a") as f:

        if args.lowres:
            f.create_dataset("unitcell/small_potential_fil/"+str(i),dtype='f',data=input)

        # f.create_dataset("unitcell/potential_orires/"+str(i),dtype='f',data=input)
        f.create_dataset("unitcell/potential_fil/"+str(i),dtype='f',data=inputsave)
        f.create_dataset("unitcell/sigmafac/"+str(i),dtype='f',data=args.sigmafactor)
        f.create_dataset("unitcell/epslow/"+str(i),dtype='f',data=epslow)
        f.create_dataset("unitcell/epshi/"+str(i),dtype='f',data=epshi)
        f.create_dataset("unitcell/uccoefs/"+str(i),dtype='complex64',data=uccoefs)
        f.create_dataset("unitcell/ucgvecs/"+str(i),dtype='f',data=ucgvecs)
        f.create_dataset("unitcell/uclevel/"+str(i),dtype='f',data=uclevel)
        # f.create_dataset("unitcell/orires/"+str(i),dtype='f',data=args.orires)
        f.create_dataset("unitcell/filling/"+str(i),dtype='f',data=filling)

        f.create_dataset("eigval_fil/"+str(i),dtype='f',data=eigval)
        f.create_dataset("eigvec_fil/"+str(i),dtype='f',data=eigvec)

    if i % 50 == 0:
        print(f"{i}samples done!")
