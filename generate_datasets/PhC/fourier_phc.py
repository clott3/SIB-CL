#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 17:29:01 2020

@author: charlotte
"""

from scipy.stats import truncnorm
import numpy as np

class FourierPhC:

    def __init__(self,dim,maxF,mineps=1.,maxeps=20.,samplesig = 10., minfill=0.01,maxfill=0.99, \
                use_fill = False):
        """
        :param dim: 2 or 3 for 2D or 3D
        :param maxF: maximum no. of fourier components to include in sum
        :param mineps: min value for eps/potential
        :param maxeps: maximum value for eps/potential

        """
        self.dim = dim
        self.maxF = maxF
        self.mineps = float(mineps)
        self.maxeps = float(maxeps)
        self.minfill = minfill
        self.maxfill = maxfill
        self.samplesig = samplesig
        self.use_fill = use_fill

    def sample_eps(self):
        eps1 = np.random.uniform(self.mineps,self.maxeps)
        eps2 = np.random.uniform(self.mineps,self.maxeps)
        return eps1, eps2

    def sample_filling(self, meanfill = 0.75, std = 0.2):
        if self.use_fill:
            filling = 0.
            while not (self.minfill <= filling <= self.maxfill):
                filling = np.random.normal(loc=meanfill,scale=std)
        else:
            filling = np.random.uniform(self.minfill,self.maxfill)
        return filling

    def generate_coefs(self):
        '''Returns the random coefficients and the G vectors in the fourier sum'''
        nvec = np.arange(-self.maxF,self.maxF+1,1)
        if self.dim == 2:
            xy = np.meshgrid(nvec,nvec)
            ucgvecs = list(zip(np.ravel(xy[0]),np.ravel(xy[1])))
        elif self.dim == 3:
            xyz = np.meshgrid(nvec,nvec,nvec)
            ucgvecs = list(zip(np.ravel(xyz[0]),np.ravel(xyz[1]),np.ravel(xyz[2])))
        ucgvecs = np.array(ucgvecs)
        r, phi = np.random.uniform(low=0.,high=1.,size=len(ucgvecs)), 2*np.pi*np.random.uniform(low=0.,high=1.,size=len(ucgvecs))
        uccoefs = r*np.exp(phi*1j)
        return uccoefs, ucgvecs

    def calc_fourier_sum(self,r, uc_gvecs, uc_coefs):
        q = 2 * np.pi * np.matmul(r.T, uc_gvecs.T)
        fvec = np.real(uc_coefs) * np.cos(q) + -np.imag(uc_coefs) * np.sin(q)
        return np.sum(fvec, axis = -1)

    def getuclevel(self, ucgvecs,uccoefs,filling,dim=2,nsam=50):
        '''Returns the isoval ("threshold for the fourier sum") that meets the
        filling requirement of the unit cell'''
        x = np.linspace(-0.5,0.5,nsam)
        y = np.linspace(-0.5,0.5,nsam)
        z = np.linspace(-0.5,0.5,nsam)
        if self.dim == 2:
            r = np.array(np.meshgrid(x,y))
        if self.dim == 3:
            r = np.array(np.meshgrid(x,y,z))
        fsum = self.calc_fourier_sum(r,ucgvecs,uccoefs)
        fsum = np.ravel(fsum)
        return np.quantile(fsum,filling)

    def get_random(self):
        '''Returns a tuple containing all the parameters required to specify a random unit cell'''
        uccoefs, ucgvecs = self.generate_coefs()
        filling = self.sample_filling()
        uclevel = self.getuclevel(ucgvecs,uccoefs,filling,dim=self.dim)
        epsin, epsout = self.sample_eps()

        return uccoefs, ucgvecs, epsin, epsout, uclevel, filling

    def getunitcell(self,uccoefs, ucgvecs, eps_in, eps_out, uclevel,ucres=32):
        '''Returns an array with the 'image' of the eps profile or potential'''
        X = np.linspace(-0.5,0.5,ucres)
        Y = np.linspace(-0.5,0.5,ucres)
        Z = np.linspace(-0.5,0.5,ucres)
        if self.dim == 3:
            r = np.array(np.meshgrid(X,Y,Z))
        elif self.dim == 2:
            r = np.array(np.meshgrid(X,Y))

        val = self.calc_fourier_sum(r,ucgvecs,uccoefs)
        fsum = np.zeros_like(val)
        fsum[val>uclevel] = eps_in
        fsum[val<=uclevel] = eps_out

        return fsum



if __name__ == '__main__':

    import matplotlib.pyplot as plt

    phc = FourierPhC(dim=2,maxF=1)
    ## Generate 5 random unit cells to visualize ##
    for _ in range(5):
        uccoefs, ucgvecs, epsin, epsout, uclevel, filling = phc.get_random()
        phc2d = phc.getunitcell(uccoefs, ucgvecs, epsin, epsout, uclevel)
        plt.figure()
        plt.imshow(phc2d, cmap='Greys')
        plt.colorbar()
