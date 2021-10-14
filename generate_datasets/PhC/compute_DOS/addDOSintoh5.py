import glob
import os
import matplotlib.pyplot as plt
#import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d as gaussfil
import h5py
import math

import argparse
parser = argparse.ArgumentParser(description = 'Compute efreq and velocities for trigonal lattice')
parser.add_argument('--nbands',type=int, help='number of bands; default = 10', default=10)
parser.add_argument('--sgnum', type=int, help='Space Group number; default = 1 (no symmetry)', default = 1)
parser.add_argument('--kgrid',type=int, help='number of kpoints per basis; default = 25',default=25)
parser.add_argument('--nsam', required=True,type=int, help='input no. of samples')
parser.add_argument('--root_dir', type=str, default='./', help='input root directory if different from current path')
parser.add_argument('--h5prefix', required=True, type=str, help='input h5 file prefix in format: <prefix>-s<seed>-tm.h5')
# parser.add_argument('--seeds', required=True, nargs='+', type=int, help='input all the seeds of the h5 file, separated by a space')

args = parser.parse_args()

def calerror(sample,reference):
    if type(sample) != np.ndarray:
        sample = np.array(sample)
    if type(reference) != np.ndarray:
        reference = np.array(reference)
    errorw = abs(sample - reference)
    error = np.sum(errorw)/np.sum(reference)
    return errorw, error

def getDOS(txtfile):
    file = open(txtfile)
    wvec = []
    DOSvec = []
    for lines in file.readlines():
        line1 = lines.replace('n','').split(' ')
        wvec.append(float(line1[0]))
        DOSvec.append(float(line1[1]))
    freq = wvec[1:]
    refsignal = DOSvec[1:]
    return freq, refsignal

## Specify inputs here
sigma = 100 # bandwidth of filter
nsam = args.nsam # number of samples
wmax = 1.2 # Max frequency to truncate
nw = 500 # no. of frequency points to interpolate
sgnum = 1
# root_dir='/Users/charlotteloh/Desktop/gen_sg/'
# prefix = 'mf2-c0'
sgnum= args.sgnum
root_dir = args.root_dir
prefix = args.h5prefix

dosfolder = root_dir+f'DOS_{prefix}_sg{sgnum}/'
fcom = root_dir+f'{prefix}-sg{sgnum}-tm.h5'

# ii=1
# for part in [1,2,3,4,5,6,7]:
# # for part in [1,2]:
#     f1 = h5py.File(root_dir+f'{prefix}-s{part}-tm.h5',"r")
#     # rvecs = f1['universal/rvecs'][()]
#     # ucgvecs = f1['universal/ucgvecs'][()]

for ii in range(1,nsam+1):
    if len(str(ii)) == 1:
        dosii = '0'+str(ii)
    else:
        dosii = str(ii)
    # i+=1
    # try:
        # defeps = f1['unitcell/inputepsimage/'+str(i)][()]
    with h5py.File(fcom,"r") as f:
        mpbeps = f['unitcell/mpbepsimage/'+str(ii)][()]
        # epsavg = f['unitcell/epsavg/'+str(ii)][()]

        # epsin = f1['unitcell/epsin/'+str(i)][()]
        # epsout = f1['unitcell/epsout/'+str(i)][()]
        # filling = f1['unitcell/filling/'+str(i)][()]
        # uclevel = f1['unitcell/uclevel/'+str(i)][()]
        # uccoefs = f1['unitcell/uccoefs/'+str(i)][()]
        # efreq = f1['mpbcal/efreq/'+str(i)][()]
        # grpvel = f1['mpbcal/grpvel/'+str(i)][()]
        # gap = f1['mpbcal/bandgap/'+str(i)][()]

    dosfile = dosfolder+f"DOS_GRR_{dosii}.txt"
    freq, DOS = getDOS(dosfile)
    epsavg2 = np.mean(mpbeps)
        # old_w =np.array(freq32)*np.sqrt(epsavg)
    old_w2 =np.array(freq)*np.sqrt(epsavg2)
    dosfil = gaussfil(DOS,sigma)
    old_DOS = np.array(dosfil)
    new_w = np.linspace(0,wmax,nw)
    new_DOS2=np.interp(new_w,old_w2,old_DOS)
    el = new_w*2*math.pi*np.sqrt(epsavg2)
    eldiff = new_DOS2-el
    new_DOS2n = new_DOS2/np.sqrt(epsavg2)
    eln = el/np.sqrt(epsavg2)
    eldiffn = new_DOS2n-eln

    with h5py.File(fcom,"a") as f:
        # f.create_dataset("unitcell/inputepsimage/"+str(ii),dtype='f',data=defeps)
        # f.create_dataset("unitcell/mpbepsimage/"+str(ii),dtype='f',data=mpbeps)
        # f.create_dataset("unitcell/epsin/"+str(ii),dtype='f',data=epsin)
        # f.create_dataset("unitcell/epsout/"+str(ii),dtype='f',data=epsout)
        # f.create_dataset("unitcell/epsavg/"+str(ii),dtype='f',data=epsavg)
        # f.create_dataset("unitcell/filling/"+str(ii),dtype='f',data=filling)
        # f.create_dataset("unitcell/uclevel/"+str(ii),dtype='f',data=uclevel)
        # f.create_dataset("unitcell/uccoefs/"+str(ii),dtype='complex',data=uccoefs)
        # f.create_dataset("mpbcal/efreq/"+str(ii),dtype='f',data=efreq)
        # f.create_dataset("mpbcal/grpvel/"+str(ii),dtype='f',data=grpvel)
        # f.create_dataset("mpbcal/bandgap/"+str(ii),dtype='f',data=gap)
        f.create_dataset("mpbcal/DOS/"+str(ii),dtype='f',data=new_DOS2)
        f.create_dataset("mpbcal/DOSeldiff/"+str(ii),dtype='f',data=eldiff)
        f.create_dataset("mpbcal/DOSn/"+str(ii),dtype='f',data=new_DOS2n)
        f.create_dataset("mpbcal/DOSeldiffn/"+str(ii),dtype='f',data=eldiffn)
            # if ii == 1:
            #     f.create_dataset("universal/rvecs",dtype='f',data=rvecs)
            #     f.create_dataset("universal/ucgvecs",dtype='f',data=ucgvecs)
    if ii % 1000 == 0:
        print(str(ii)," DOS added!")
    # ii+=1

        # except KeyError:
        #     print(f"{i} samples loaded; total {ii} samples")
        #     f1.close()
        #     break

# f1.close()

    #     plt.figure()
    #     plt.plot(freq32,DOS32,label='original')
    #     plt.plot(freq32,fil32,label='filtered with $\sigma=100$')
    #     plt.legend()
    #     plt.ylabel('DOS')
    #     plt.xlabel('$w$')
    #     plt.savefig('plotsTM/sample{}.png'.format(sam))
    #     plt.close()

    #     plt.figure()
    #     plt.plot(np.array(freq32)*np.sqrt(epsavg),DOS32,label='original')
    #     plt.plot(np.array(freq32)*np.sqrt(epsavg),fil32,label='filtered with $\sigma=100$')
    #     plt.legend()
    #     plt.title('DOS')
    #     plt.ylabel('DOS')
    #     plt.xlabel('$w*\epsilon avg$')
    #     plt.savefig('plotsTM/scaled{}.png'.format(sam))
    #     plt.close()

    #     old_w =np.array(freq32)*np.sqrt(epsavg)
    #     old_DOS = np.array(fil32)

    #     wmax = 1.2
    #     nw = 500
    #     new_w = np.linspace(0,wmax,nw)
    #     new_DOS=np.interp(new_w,old_w,old_DOS)
    #     plt.figure()
    #     plt.plot(old_w,old_DOS,label='original; filtered with $\sigma=100$')
    #     plt.plot(new_w,new_DOS,label='interpolated 500; filtered with $\sigma=100$')
    #     plt.legend()
    #     plt.ylabel('DOS')
    #     plt.xlabel('$w*\epsilon avg$')
    #     plt.savefig('plotsTM/interpolated{}.png'.format(sam))
    #     plt.close()
