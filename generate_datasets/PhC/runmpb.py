import meep as mp
from meep import mpb
import numpy as np

class MPB:
    def __init__(self,dim=2,ucgvecs=None):
        self.ndim = dim
        self.ucgvecs = ucgvecs #this is a np array

    def converttovec3(self,klist,dim):
        'Converts list to a meep Vector3 object'
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

    def convertfromvec3(self, vector3):
        """Convert Vector3 object to numpy array"""
        return np.array([vector3.x, vector3.y, vector3.z])

    def get_kgrid(self,Nk):
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
        kvec3 = self.converttovec3(kvecs,self.ndim)
        return kvec3

    def calc_fourier_sum(self,r, uc_gvecs, uc_coefs, level_set = False):
        if level_set:
            r = self.convertfromvec3(r)
            uc_gvecs = np.concatenate([uc_gvecs,np.zeros((uc_gvecs.shape[0],1))],axis =1)
        q = 2 * np.pi * np.matmul(r.T, uc_gvecs.T)
        fvec = np.real(uc_coefs) * np.cos(q) + -np.imag(uc_coefs) * np.sin(q)
        return np.sum(fvec, axis = -1)

    def getunitcell(self, uccoefs, ucgvecs, eps_in, eps_out, uclevel,ucres=32):
        '''Returns an array with the 'image' of the eps profile or potential'''
        X = np.linspace(-0.5,0.5,ucres)
        Y = np.linspace(-0.5,0.5,ucres)
        Z = np.linspace(-0.5,0.5,ucres)
        if self.ndim == 3:
            r = np.array(np.meshgrid(X,Y,Z))
        elif self.ndim == 2:
            r = np.array(np.meshgrid(X,Y))
        val = self.calc_fourier_sum(r,ucgvecs,uccoefs)
        fsum = np.zeros_like(val)
        fsum[val>uclevel] = eps_in
        fsum[val<=uclevel] = eps_out
        return fsum

    def run_mpb(self, run_type="tm", res=32, prefix="", kvecs=None, nbands=10, rvecs=None,
          uc_gvecs=None, uc_coefs=None, uc_level=0., eps_in=20, eps_out=1):
        """
        Keyword arguments:
        run_type: string "te" or "tm" that describe which solver to use
        dim:    dimensionality of the problem (2 or 3)
        sgnum:  space group number
        res:    resolution
        kvecs:  list of k-vectors to evaluate at
        nbands: number of bands to compute
        rvecs:  (dim, dim) array of lattice basis vectors
        uc_gvecs: (N, dim) array of g vectors
        uc_coefs: (N) array of complex coefficients
        """

        if self.ndim == 2:
            geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1), basis1=rvecs[0], basis2=rvecs[1])
        elif self.ndim == 3:
            geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1, 1), basis1=rvecs[0], basis2=rvecs[1], basis3=rvecs[2])
        else:
            raise ValueError('Missing argument for dimension')

        # Instead of defining the geometry, we define the default material using a function that returns the value of
        # epsilon given the position

        def level_set(r):
            """Calculate epsilon profile based on level-set of the Fourier sum"""
            val = self.calc_fourier_sum(r,self.ucgvecs,uc_coefs,level_set=True)
            return mp.Medium(epsilon=eps_in) if val > uc_level else mp.Medium(epsilon=eps_out)

        grpvel_k =[]

        def grpvel(ms):
            gv3 = ms.compute_group_velocities()
            gv = []
            for gvband in gv3:
                gv.append(list(self.convertfromvec3(gvband)))
            grpvel_k.append(gv)

        ms = mpb.ModeSolver(num_bands=nbands,
                                k_points=kvecs,
                                geometry_lattice=geometry_lattice,
                                resolution=res,
                                default_material=level_set)
        if run_type == 'tm':
            ms.run_tm(grpvel)
        elif run_type == 'te':
            ms.run_te(grpvel)
        else:
            raise ValueError('Specify polarization')

        efreq_k = ms.all_freqs
        gap_k = ms.gap_list
        mpbgeteps = ms.get_epsilon()
        defeps = self.getunitcell(uc_coefs,self.ucgvecs, eps_in, eps_out, uc_level, ucres=res)

        return efreq_k, gap_k, grpvel_k, defeps, mpbgeteps
