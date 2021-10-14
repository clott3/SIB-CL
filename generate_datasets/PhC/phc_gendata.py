
import meep as mp
from meep import mpb
import numpy as np
import h5py
import time
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
from os import path
from fourier_phc import FourierPhC
from runmpb import MPB

import argparse
parser = argparse.ArgumentParser(description = 'Compute efreq and velocities for trigonal lattice')
parser.add_argument('--nbands',type=int, help='number of bands; default = 10', default=10)
parser.add_argument('--ndim',type=int, help='Dimension, 2 or 3. default = 2', default=2)
parser.add_argument('--sgnum', type=int, help='Space Group number; default = 1 (no symmetry)', default = 1)
parser.add_argument('--kgrid',type=int, help='number of kpoints per basis; default = 25',default=25)
parser.add_argument('--pol', required=True,type=str, help=' "tm", "te" or "all"; if "all", both pols will be computed for same unitcell')
parser.add_argument('--res', default=32,type=int, help='Resolution to compute; default = 32')
parser.add_argument('--nsam', required=True,default=1000,type=int, help='REQUIRED: number of samples')
parser.add_argument('--maxF', default=1,type=int, help='maximum no. of fourier components used to modulate unit cell; default = 2')
parser.add_argument('--h5filename', required=True, type=str, help='REQUIRED: h5 file will be saved in format: "<h5filename>-<pol>.h5" ')
parser.add_argument('--seed',type=int, help='random seed',default=42)
args = parser.parse_args()

np.random.seed(args.seed)

phc = FourierPhC(dim=args.ndim,maxF=args.maxF,maxeps=20., mineps=1.,use_uniform = False)

if args.sgnum == 1:
    uccoefs, ucgvecs, epsin, epsout, uclevel, filling = phc.get_random()
else:
    from fourier_genericsg import get_generic_fourier # this requires PyJuia installation
    uccoefs, ucgvecs = get_generic_fourier(sgnum=args.sgnum, D=args.ndim, maxF = args.maxF)
    filling = phc.sample_filling()
    uclevel = phc.getuclevel(ucgvecs,uccoefs,filling,dim=args.ndim)
    epsin, epsout = phc.sample_eps()

print("Now generating h5 file..")

mpb = MPB(dim=args.ndim, ucgvecs=ucgvecs) # ucgvecs is universal
kvec3 = mpb.get_kgrid(args.kgrid)
rvecs = (np.array([1.,0.]),np.array([0.,1.])) # square lattice
rvec3 = mpb.converttovec3(rvecs,args.ndim)
ucgvec3 = mpb.converttovec3(ucgvecs,args.ndim)

if args.pol == "all":
    suffix = ["-tm","-te"]
else:
    suffix = ["-"+args.pol]

h5out = args.h5filename

for suf in suffix:
    with h5py.File(h5out+suf+".h5","a") as f:
        ## write universal (per h5 file) parameters
        f.create_dataset("universal/rvecs",dtype='f',data=rvecs)
        f.create_dataset("universal/ucgvecs",dtype='f',data=ucgvecs)

for i in range(args.nsam):

    if args.sgnum == 1:
        uccoefs, ucgvecs, epsin, epsout, uclevel, filling = phc.get_random()
    else:
        uccoefs, ucgvecs = get_generic_fourier(sgnum=args.sgnum, D=args.ndim, maxF = args.maxF)
        filling = phc.sample_filling()
        uclevel = phc.getuclevel(ucgvecs,uccoefs,filling,dim=args.ndim)
        epsin, epsout = phc.sample_eps()

    epsavg = filling*epsout + (1-filling)*epsin

    for suf in suffix:
        efreq, gap, grpvel, defeps, mpbeps = mpb.run_mpb(run_type=suf[1:], \
                                        res=args.res, kvecs=kvec3, nbands=args.nbands,\
                                        rvecs=rvec3, uc_gvecs = ucgvec3, \
                                        uc_coefs = uccoefs, uc_level = uclevel, \
                                        eps_in = epsin, eps_out = epsout)

        with h5py.File(h5out+suf+".h5","a") as f:
            ## write unitcell parameters
            f.create_dataset("unitcell/inputepsimage/"+str(i),dtype='f',data=defeps)
            f.create_dataset("unitcell/mpbepsimage/"+str(i),dtype='f',data=mpbeps)
            f.create_dataset("unitcell/epsin/"+str(i),dtype='f',data=epsin)
            f.create_dataset("unitcell/epsout/"+str(i),dtype='f',data=epsout)
            f.create_dataset("unitcell/epsavg/"+str(i),dtype='f',data=epsavg)
            f.create_dataset("unitcell/filling/"+str(i),dtype='f',data=filling)
            f.create_dataset("unitcell/uclevel/"+str(i),dtype='f',data=uclevel)
            f.create_dataset("unitcell/uccoefs/"+str(i),dtype='complex',data=uccoefs)

            ## write mpbcalc outputs
            f.create_dataset("mpbcal/efreq/"+str(i),dtype='f',data=efreq)
            f.create_dataset("mpbcal/grpvel/"+str(i),dtype='f',data=grpvel)
            f.create_dataset("mpbcal/bandgap/"+str(i),dtype='f',data=gap)
