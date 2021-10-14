#!/usr/bin/env python3
"""
Created on Jun  9 2020

@author: charlotte
"""

import torch
from torch.utils import data
from torchvision.transforms import transforms

import numpy as np
import h5py
import os
import math
import matplotlib.pyplot as plt

from skimage.transform import resize
from scipy import interpolate

class PhC2D(data.Dataset):

    def __init__(self, path_to_h5_dir, trainsize, validsize=500, testsize=2000,
        predict = 'DOS', mode = 'ELdiff', domain = 'target', targetmode = 'none',
        split='train', nbands = 6, band = 0, eps = False, unnormDOS=False, ftsetind = None):

        self.mode = mode
        self.input_size = 32
        self.skipsize = 90
        self.to_load_eps = eps

        if domain == 'source':
            filename = 'cylin-tm-11k-3dosel-EL-max1p2.h5'
            nskip = 0

        elif domain == 'target':
            filename = 'mf1-tm-final-32k-3dosel-EL-max1p2.h5'
            nskip = self.skipsize # To pick good target samples for small sample size
        else:
            raise ValueError("Invalid domain. Either source or target.")

        # the following is to make a fix set of train-valid-test split
        totalstart = nskip + 1

        if split == 'train':
            indstart = totalstart
            indend = indstart + trainsize
        elif split == 'valid':
            indstart = totalstart + trainsize
            indend = indstart + validsize
        elif split == 'test':
            indstart = totalstart + trainsize + validsize
            indend = indstart + testsize

        indlist = range(indstart,indend)

        if ftsetind is not None:
            if split == 'train':
                indlist = ftsetind
            elif split == 'test':
                indlist = np.arange(18000,20000) # fixed test set

        self.len = len(indlist)

        ## initialize data lists
        self.x_data = []
        self.y_data = []
        self.EL_data = []
        self.ELd_data = []
        self.eps_data = []

        with h5py.File(os.path.join(path_to_h5_dir, filename), 'r') as f:
            for memb in indlist:

                inputeps = f['unitcell/mpbepsimage/'+str(memb)][()]
                epsavg = f["unitcell/epsavg/"+str(memb)][()]

                if predict == 'bandstructures':
                    y = f['mpbcal/efreq/'+str(memb)][()][:,:nbands]
                    el = f['mpbcal/emptylattice/'+str(memb)][()][:,:nbands]
                    eldiff = f['mpbcal/eldiff/'+str(memb)][()][:,:nbands]

                elif predict == 'oneband':
                    y = f['mpbcal/efreq/'+str(memb)][()][:,band]
                    el = f['mpbcal/emptylattice/'+str(memb)][()][:,band]
                    eldiff = f['mpbcal/eldiff/'+str(memb)][()][:,band]

                elif predict == 'DOS':
                    wvec = np.linspace(0,1.2,500)[:400]
                    if targetmode != 'unlabeled':
                        if unnormDOS:
                            y = f['mpbcal/DOS/'+str(memb)][()][:400]
                            eldiff = f['mpbcal/DOSeldiff/'+str(memb)][()][:400]
                            el = wvec*2*math.pi*np.sqrt(epsavg)
                        else:
                            y = f['mpbcal/DOS/'+str(memb)][()][:400]/np.sqrt(epsavg)
                            el = wvec*2*math.pi
                            eldiff = y-el
                    else:
                        y = [0.]
                        eldiff = [0.]
                        el = [0.]
                else:
                    raise ValueError("Invalid property to predict. either DOS or bandstructures or oneband")

                self.x_data.append(inputeps)
                self.y_data.append(y)
                self.EL_data.append(el)
                self.ELd_data.append(eldiff)
                self.eps_data.append(epsavg)

        # normalize x data
        self.x_data = (np.array(self.x_data).astype('float32')-1) / 19 # normalize
        self.x_data = np.expand_dims(self.x_data,1) # add 1 channel for CNN
        self.y_data = np.array(self.y_data).astype('float32')
        self.EL_data = np.array(self.EL_data).astype('float32')
        self.ELd_data = np.array(self.ELd_data).astype('float32')
        self.eps_data = np.array(self.eps_data).astype('float32')


    def __len__(self):
        return self.len

    def __getitem__(self, index):
        """
        :return: random sample for a task
        """
        ## input always first element in tuple and output always second element
        if not self.to_load_eps:
            if self.mode == 'raw':
                return self.x_data[index], self.y_data[index]
            elif self.mode == 'ELdiff':
                return self.x_data[index], self.ELd_data[index], self.y_data[index], self.EL_data[index]
        else:
            if self.mode == 'raw':
                return self.x_data[index], self.y_data[index], self.eps_data[index]
            elif self.mode == 'ELdiff':
                return self.x_data[index], self.ELd_data[index], self.y_data[index], self.EL_data[index], self.eps_data[index]

class TISEdata(data.Dataset):

    def __init__(self, path_to_h5_dir, trainsize, validsize=500, testsize=2000, lowres =5,
        predict = 'eigval', ndim = 3, neval = 1, domain = 'target', split='train', downres = 32, \
        tisesource = 'lr', ftsetind = None):

        self.input_size = 64
        self.maxeps = me
        rr = f'_n{downres}' if ndim == 3 else ''

        if domain == 'source':
            if tisesource == 'lr':
                filename = f'tise_{ndim}d_mf2_sigfac20_xmax5_me1.0_e0{rr}_LR{lowres}.h5'

            elif tisesource == 'sho':
                if ndim == 3:
                    filename = f'tise_{ndim}d_sho_v1_n{downres}.h5'
                elif ndim == 2:
                    filename = f'tise_{ndim}d_sho_v3.h5'


        elif domain == 'target':
            filename = f'tise_{ndim}d_mf2_sigfac20_xmax5_me1.0_e0{rr}_res64.h5'

        elif domain == 'nolabel':
            filename = f'tise_{ndim}d_mf2_sigfac20_xmax5_me1.0_e0{rr}_LR4_35k.h5'

        else:
            raise ValueError("Invalid domain. Either source or target.")

        # the following is to make a fix set of train-valid-test split
        print("loaded file: ", filename)
        totalstart = 1

        if split == 'train':
            indstart = totalstart
            indend = indstart + trainsize
        elif split == 'valid':
            indstart = totalstart + trainsize
            indend = indstart + validsize
        elif split == 'test':
            indstart = totalstart + trainsize + validsize
            indend = indstart + testsize

        indlist = range(indstart,indend)

        if ftsetind is not None:
            if split == 'train':
                indlist = ftsetind
            elif split == 'test':
                indlist = np.arange(10000,12000)

        self.len = len(indlist)

        ## initialize data lists
        self.x_data = []
        self.y_data = []

        with h5py.File(os.path.join(path_to_h5_dir, filename), 'r') as f:
            for memb in indlist:
                if ndim == 2:
                    input = f['unitcell/potential_fil/'+str(memb)][()]

                elif ndim == 3: #downsize input to fit on memory
                    input = f['unitcell/potential_fil_'+str(downres)+'/'+str(memb)][()]

                if predict == 'eigval':
                    y = f['eigval_fil/'+str(memb)][()][:neval]
                    y = np.real(y)

                    if domain == 'source' and tisesource == 'sho':
                        y = f['grounde/'+str(memb)][()]
                        y = np.expand_dims(np.real(y),1)

                elif predict == 'eigvec':
                    y = f['eigvec_fil/'+str(memb)][()][:,:neval]
                    if domain == 'source' and tisesource == 'lr':
                        y = np.real(y).reshape(int(lowres-2),int(lowres-2))
                        y = resize(y,(int(input_size-2),int(input_size-2))) # interpolate
                        y = y.reshape(-1,1)
                    y = np.real(y)**2 # predict probability distribution
                else:
                    raise ValueError("Invalid property to predict. either eigval or eigvec")

                self.x_data.append(input)
                self.y_data.append(y)

        # normalize x data
        mineps = 0
        self.x_data = (np.array(self.x_data).astype('float32')-mineps) / (self.maxeps-mineps) # normalize
        self.x_data = np.expand_dims(self.x_data,1) # add 1 channel for CNN
        self.y_data = np.array(self.y_data).astype('float32')


    def __len__(self):
        return self.len

    def __getitem__(self, index):
        """
        :return: random sample for a task
        """
        return self.x_data[index], self.y_data[index]


if __name__ == '__main__':
    pass
