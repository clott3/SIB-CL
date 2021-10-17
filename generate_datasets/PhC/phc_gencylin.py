#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:39:46 2020

@author: charlotte
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--nbands',type=int, help='number of bands; default = 10', default=10)
parser.add_argument('--kgrid',type=int, help='number of kpoints per basis; default = 25',default=25)
parser.add_argument('--pol', required=True,type=str, help='tm or te; default: tm', default='tm')
parser.add_argument('--res', default=32,type=int, help='Resolution to compute; default = 32')
parser.add_argument('--nsam', required=True,default=1000,type=int, help='REQUIRED: number of samples')
parser.add_argument('--maxF', default=2,type=int, help='maximum no. of fourier components used to modulate unit cell; default = 2')
parser.add_argument('--h5filename', required=True, type=str, help='REQUIRED: path to save generated h5 file, in format path/to/filename.h5')

args = parser.parse_args()

import meep as mp
from meep import mpb
import numpy as np
import h5py
import time
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
from os import path
import math

Nk = args.kgrid
pol = args.pol
res = args.res
nbands = args.nbands
num_samples = args.nsam
h5out = args.h5filename

def converttovec3(klist,dim):
    'function converts list to a meep 3-vector'
    if dim == 2:
        kvec3=[]
        for i in range(len(klist)):
            kpoint = mp.Vector3(klist[i][0],klist[i][1])
            kvec3.append(kpoint)
    elif dim == 3:
        kvec3=[]
        for i in range(len(klist)):
            kpoint = mp.Vector3(klist[i][0],klist[i][1],klist[i][2])
            kvec3.append(kpoint)
    else:
        raise ValueError('Dimension must be 2 or 3')
    return kvec3
def convertfromvec3(vector3):
    """Convert Vector3 object to numpy array"""
    return np.array([vector3.x, vector3.y, vector3.z])

def runmpb(run_type="tm",radius=0.2, eps_in=20, eps_out=1, res=32,kvecs=None,nbands =10, rvecs=None):

    geometry = [mp.Cylinder(radius, material=mp.Medium(epsilon=eps_in))]
    geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1), basis1=rvecs[0], basis2=rvecs[1])

    grpvel_k =[]

    def grpvel(ms):
        gv3 = ms.compute_group_velocities()
        gv = []
        for gvband in gv3:
            gv.append(list(convertfromvec3(gvband)))
        grpvel_k.append(gv)

    ms = mpb.ModeSolver(num_bands=nbands,
                            k_points=kvecs,
                            geometry_lattice=geometry_lattice,
                            geometry=geometry,
                            resolution=res,
                            default_material=mp.Medium(epsilon=eps_out))

    if run_type == 'tm':
        ms.run_tm(grpvel)
    elif run_type == 'te':
        ms.run_te(grpvel)
    else:
        raise ValueError('Please specify polarization')

    efreq_k = ms.all_freqs
    gap_k = ms.gap_list
    mpbgeteps = ms.get_epsilon()

    return efreq_k, gap_k, grpvel_k, mpbgeteps

### SPECIFY LATTICE PARAMETERS AND RUN MPB TO GENERATE H5 ###
print("Now generating h5 file..")
## create uniform k-grid for sampling. k points are sampled in the middle of
## microcell
dk = 1/Nk #assume sample unit cell of unit length, Nkpoints = Nintervals
kx = np.linspace(-0.5+dk/2,0.5-dk/2,Nk)
ky = np.linspace(-0.5+dk/2,0.5-dk/2,Nk)
gridx, gridy = np.meshgrid(kx,ky)
gridx = np.ravel(gridx)
gridy = np.ravel(gridy)
kvecs = list()
for i in range(len(gridx)):
    kpoint = np.array((gridx[i],gridy[i]))
    kvecs.append(kpoint)

rvecs = (np.array([1.,0.]),np.array([0.,1.])) # square lattice
kvec3 = converttovec3(kvecs,2)
rvec3 = converttovec3(rvecs,2)

for i in range(num_samples):
    epsin = np.random.uniform(1,20)
    epsout = np.random.uniform(1,20)
    # epsin, epsout = sampleeps()
    #rad = np.random.uniform(0.0,0.5) # this includes 0 but excludes 0.5
    rad = np.abs(np.random.uniform(0.0,0.5)-0.5) # this excludes 0 (empty) but includes 0.5
    areacircle = math.pi*rad*rad
    epsavg = areacircle * epsin + (1-areacircle) * epsout
    # Run mpb and compute all desired quantities
    efreq, gap, grpvel, eps = runmpb(run_type=pol, radius = rad,
                                     eps_in=epsin, eps_out=epsout,
                                     res=res, kvecs=kvec3,
                                     nbands=nbands, rvecs=rvec3)

    with h5py.File(h5out,"a") as f:
        ## write unitcell parameters

        f.create_dataset("unitcell/mpbepsimage/"+str(i),dtype='f',data=eps)
        f.create_dataset("unitcell/epsin/"+str(i),dtype='f',data=epsin)
        f.create_dataset("unitcell/epsout/"+str(i),dtype='f',data=epsout)
        f.create_dataset("unitcell/epsavg/"+str(i),dtype='f',data=epsavg)


        ## write mpbcalc outputs
        f.create_dataset("mpbcal/efreq/"+str(i),dtype='f',data=efreq)
        f.create_dataset("mpbcal/grpvel/"+str(i),dtype='f',data=grpvel)
        f.create_dataset("mpbcal/bandgap/"+str(i),dtype='f',data=gap)
