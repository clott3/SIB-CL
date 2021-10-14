import numpy as np
from scipy.sparse.linalg import eigs, eigsh
import matplotlib.pyplot as plt
import scipy.sparse as sp

class TISE:
    def __init__(self,pot=None,dim = 2, ngrid=None, xmax = 5):
        self.pot = pot
        self.dim = dim
        if pot is not None:
            assert(dim==len(pot.shape))
            self.ngrid = pot.shape[0] ## assume equal grid in x,y,z
        else:
            self.ngrid = ngrid
        self.x_min = -xmax
        self.x_max = xmax
        self.y_min = -xmax
        self.y_max = xmax
        self.z_min = -xmax
        self.z_max = xmax

    def grid(self):
        xvec = np.linspace(start=self.x_min, stop=self.x_max, num=self.ngrid)
        yvec = np.linspace(start=self.y_min, stop=self.y_max, num=self.ngrid)
        zvec = np.linspace(start=self.z_min, stop=self.z_max, num=self.ngrid)
        if self.dim == 1:
            grid = xvec
        elif self.dim == 2:
            grid = np.meshgrid(xvec,yvec)
        elif self.dim == 3:
            grid = np.meshgrid(xvec,yvec,zvec)
        else:
            raise ValueError("dim must be 1, 2 or 3")
        return grid

    def kinetic(self):
        # d = (self.x_max - self.x_min)/float(self.ngrid - 1) ## assume equal grid
        # Dirichlet BC: we solve for x = xmin+d, 2d, .... xmax-d for ngrid points. pad the end points with 0
        d = (self.x_max - self.x_min)/float(self.ngrid + 1) ## assume equal grid

        if self.dim == 1:
            dx_stencil = -2*np.diag(np.ones(self.ngrid)) \
                        + 1*np.diag(np.ones(self.ngrid-1), k=-1) \
                        + 1*np.diag(np.ones(self.ngrid-1), k=1)
            return -0.5*dx_stencil/ d**2 ## in units hbar=1 mass=1

        elif self.dim == 2:
            shiftedy = np.ones(self.ngrid * (self.ngrid-1))
            shiftedx = np.ones(self.ngrid-1)

            sdy_stencil = -2*sp.diags(np.ones(self.ngrid * self.ngrid)) + 1*sp.diags(shiftedy, offsets=-self.ngrid) + 1*sp.diags(shiftedy, offsets=self.ngrid)
            sdx_stencil = -2*sp.diags(np.ones(self.ngrid)) + 1*sp.diags(shiftedx, offsets=-1) + 1*sp.diags(shiftedx, offsets=1)
            sdx_stencil = sp.kron(sp.eye(self.ngrid),sdx_stencil)
            com_stencil = (sdx_stencil + sdy_stencil)

            return -0.5*com_stencil/ d**2 ## in units hbar=1 mass=1

        elif self.dim == 3:

            shifted = np.ones(self.ngrid-1)
            shiftedy = np.ones(self.ngrid * (self.ngrid-1))
            shiftedz = np.ones(self.ngrid*self.ngrid*(self.ngrid-1))

            sdx_stencil = -2*sp.diags(np.ones(self.ngrid)) + 1*sp.diags(shifted, offsets=-1) + 1*sp.diags(shifted, offsets=1)
            sdx_stencil = sp.kron(sp.eye(self.ngrid*self.ngrid),sdx_stencil)
            sdy_stencil = -2*sp.diags(np.ones(self.ngrid*self.ngrid)) + 1*sp.diags(shiftedy, offsets=-self.ngrid) + 1*sp.diags(shiftedy, offsets=self.ngrid)
            sdy_stencil = sp.kron(sp.eye(self.ngrid),sdy_stencil)
            sdz_stencil = -2*sp.diags(np.ones(self.ngrid*self.ngrid*self.ngrid)) + 1*sp.diags(shiftedz, offsets=-self.ngrid*self.ngrid) + 1*sp.diags(shiftedz, offsets=self.ngrid*self.ngrid)
            com_stencil = (sdx_stencil + sdy_stencil + sdz_stencil)
            return -0.5*com_stencil/ d**2 ## in units hbar=1 mass=1

        else:
            raise ValueError("dim must be 2 or 3")

    def solve(self, num_eig = 1):
        if not self.dim == 1:
            V = sp.diags(self.pot.flatten())
            T = self.kinetic()
            H = T + V
        else:
            V = np.diag(self.pot.flatten())
            T = self.kinetic()
            H = T + V
            H = sp.csc_matrix(H)

        eigval, eigvec = eigsh(H,k=num_eig, which='SA')
        return eigval, eigvec


if __name__ == '__main__':

    from fourier_phc import FourierPhC
    orires = 50
    xmax = 0.5
    ndim = 2
    phc = FourierPhC(dim=ndim,maxF=2,maxeps=0.5, mineps=0.,\
                    minfill=0.01,maxfill=0.99,\
                    use_fill = False, \
                    use_uniform = True, use_eps2=False, use_eps3=False)
    uccoefs, ucgvecs, epsin, epsout, uclevel, filling = phc.get_random()
    epslow = np.min([epsin,epsout])
    epshi = np.max([epsin,epsout])
    epslow = -epshi
    totalres = int(orires/0.6) # this is the original sample, res 2000
    input = phc.getunitcell(uccoefs, ucgvecs, epslow, epshi, uclevel,ucres=totalres)

    time_start = time.time()
    sch = TISE(input,dim=ndim,xmax=xmax)
    eigval, eigvec = sch.solve(num_eig=2)
    print(time.time()-time_start)
    print(eigval)

    time_start = time.time()
    sch = TISE(input,dim=ndim,xmax=xmax,use_sparse=True)
    eigval, eigvec = sch.solve(num_eig=2)
    print(time.time()-time_start)
    print(eigval)
