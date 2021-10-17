# import glob
import os
# import re
import numpy as np
import h5py
# import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser(description = 'Compute efreq and velocities for trigonal lattice')
parser.add_argument('--nbands',type=int, help='number of bands; default = 10', default=10)
parser.add_argument('--sgnum', type=int, help='Space Group number; default = 1 (no symmetry)', default = 1)
parser.add_argument('--kgrid',type=int, help='number of kpoints per basis; default = 25',default=25)
parser.add_argument('--nsam', default=5000,type=int, help='input max samples per seed if exceed 5k')
parser.add_argument('--root_dir', type=str, default='./', help='input root directory if different from current path')
parser.add_argument('--h5prefix', required=True, type=str, help='input h5 file prefix in format: <prefix>-s<seed>-tm.h5')
parser.add_argument('--seeds', required=True, nargs='+', type=int, help='input all the seeds of the h5 file, separated by a space')

args = parser.parse_args()

## Specify input parameters used ##
Nk = args.kgrid
numbands = args.nbands
nsam = args.nsam
sg= args.sgnum
root_dir = args.root_dir
prefix = args.h5prefix
txtfiledir = root_dir+f'txt_{prefix}'
fcom = root_dir+f'{prefix}-tm.h5'

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

if not os.path.exists(txtfiledir):
    os.makedirs(txtfiledir)

ii=1
for part in args.seeds:
# for part in [1,2]:
    f1 = h5py.File(root_dir+f'{prefix}-s{part}-tm.h5',"r")
    rvecs = f1['universal/rvecs'][()]
    ucgvecs = f1['universal/ucgvecs'][()]

    for i in range(nsam):
        # i+=1
        try:
            if len(str(ii)) == 1:
                fileno = '0'+str(ii)
            else:
                fileno = str(ii)

            defeps = f1['unitcell/inputepsimage/'+str(i)][()]
            mpbeps = f1['unitcell/mpbepsimage/'+str(i)][()]
            epsin = f1['unitcell/epsin/'+str(i)][()]
            epsout = f1['unitcell/epsout/'+str(i)][()]
            epsavg = f1['unitcell/epsavg/'+str(i)][()]
            filling = f1['unitcell/filling/'+str(i)][()]
            uclevel = f1['unitcell/uclevel/'+str(i)][()]
            uccoefs = f1['unitcell/uccoefs/'+str(i)][()]
            efreq = f1['mpbcal/efreq/'+str(i)][()]
            grpvel = f1['mpbcal/grpvel/'+str(i)][()]
            gap = f1['mpbcal/bandgap/'+str(i)][()]

            ## GENERATE TEXT FILES ... ##
            outvelGRR = open(txtfiledir+"/trivelGRR"+"_"+str(fileno)+".txt",'w+')
            outband = open(txtfiledir+"/outband"+"_"+str(fileno)+".txt","w+")
            #outfreqTr is the same as outband
            outfreqGRR = open(txtfiledir+"/trifreqGRR"+"_"+str(fileno)+".txt","w+")

            for line in range(Nk*Nk):
                outband.write(str(kvecs[line][0])+" "+str(kvecs[line][1])+" "+str(0.0)+" "+
                              str(efreq[line][0])+" "+
                              str(efreq[line][1])+" "+
                              str(efreq[line][2])+" "+
                              str(efreq[line][3])+" "+
                              str(efreq[line][4])+" "+
                              str(efreq[line][5])+" "+
                              str(efreq[line][6])+" "+
                              str(efreq[line][7])+" "+
                              str(efreq[line][8])+" "+
                              str(efreq[line][9])+"\n")
                for band in range(numbands):
                    outvelGRR.write(str(grpvel[line][band][0])+" "+
                                    str(grpvel[line][band][1])+" "+
                                    str(grpvel[line][band][2])+"\n")
                    outfreqGRR.write(str(efreq[line][band])+"\n")
            outband.close()
            outvelGRR.close()
            outfreqGRR.close()

            ### CREATE NEW COMBINED H5 FILE
            with h5py.File(fcom,"a") as f:
                f.create_dataset("unitcell/inputepsimage/"+str(ii),dtype='f',data=defeps)
                f.create_dataset("unitcell/mpbepsimage/"+str(ii),dtype='f',data=mpbeps)
                f.create_dataset("unitcell/epsin/"+str(ii),dtype='f',data=epsin)
                f.create_dataset("unitcell/epsout/"+str(ii),dtype='f',data=epsout)
                f.create_dataset("unitcell/epsavg/"+str(ii),dtype='f',data=epsavg)
                f.create_dataset("unitcell/filling/"+str(ii),dtype='f',data=filling)
                f.create_dataset("unitcell/uclevel/"+str(ii),dtype='f',data=uclevel)
                f.create_dataset("unitcell/uccoefs/"+str(ii),dtype='complex',data=uccoefs)
                f.create_dataset("mpbcal/efreq/"+str(ii),dtype='f',data=efreq)
                f.create_dataset("mpbcal/grpvel/"+str(ii),dtype='f',data=grpvel)
                f.create_dataset("mpbcal/bandgap/"+str(ii),dtype='f',data=gap)
                if ii == 1:
                    f.create_dataset("universal/rvecs",dtype='f',data=rvecs)
                    f.create_dataset("universal/ucgvecs",dtype='f',data=ucgvecs)
            ii+=1

        except KeyError:
            # print(f"{i} samples loaded; total {ii} samples")
            f1.close()
            break

f1.close()
outband.close()
outvelGRR.close()
outfreqGRR.close()
print(ii-1) # to input as nsam for bash script
