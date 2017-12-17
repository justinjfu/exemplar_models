"""
Featurize/Discretize the input space for debugging purposes.

Default models use the Discretizer() class which does a no-op transformation.
"""
import numpy as np
from scipy.stats import norm


class Discretizer():
    """ No-op discretizer """
    def __init__(self, dX):
        self.dX = dX

    def transform(self, data):
        data = np.array(data)
        # N, d = data.shape
        # assert d == self.dX
        return data

    @property
    def dim_transformed(self):
        return self.dX


class GaussKernelDiscretizer(Discretizer):
    def __init__(self, bounds, nkern, wratio=2.0):
        super(GaussKernelDiscretizer, self).__init__(dX=len(bounds))
        self.kernels = []
        self.dim = len(bounds)
        self.nkern = nkern
        for dim, bound in enumerate(bounds):
            means = np.linspace(bound[0], bound[1], num=nkern)
            diff = means[1]-means[0]
            width = diff/wratio
            self.kernels.append(
                {'means': means,
                 'std': width}
            )

    def transform(self, data):
        data = np.array(data)
        N, d = data.shape
        assert d == self.dim
        transformed = np.zeros((N, self.dim*self.nkern))
        for dim in range(self.dim):
            data_vals = data[:, dim]
            kernels = self.kernels[dim]
            std = kernels['std']

            kernelized = []
            for mean in kernels['means']:
                probs = norm.pdf((data_vals - mean)/std)
                kernelized.append(probs)
            kernelized = np.array(kernelized).T

            transformed[:, dim*self.nkern:(dim+1)*self.nkern] = kernelized
        return transformed

    @property
    def dim_transformed(self):
        return  self.nkern * self.dim


class HardDiscretizer(Discretizer):
    def __init__(self, bounds, resolution):
        super(HardDiscretizer, self).__init__(dX=len(bounds))
        self.kernels = []
        self.dim = len(bounds)
        self.resolution = resolution
        self.bounds = bounds

    def transform(self, data):
        data = np.array(data)
        N, d = data.shape

        transformed = np.zeros((N, self.dim*self.resolution))
        for dim in range(self.dim):
            bounds = self.bounds[dim]
            bins = np.linspace(bounds[0], bounds[1], self.resolution)
            inds = np.digitize(data[:,dim], bins)-1
            inds[inds>=self.resolution]=self.resolution-1
            one_hots = np.zeros((N, self.resolution))
            one_hots[np.arange(N), inds] = 1.0
            transformed[:, dim * self.resolution:(dim + 1) * self.resolution] = \
                one_hots
        return transformed

    @property
    def dim_transformed(self):
        return self.resolution * self.dim


def env_spec_bounds(env_spec, bound_cap=None):
    bounds = np.array(list(zip(*env_spec.observation_space.bounds)))
    if bound_cap:
        bounds[bounds>bound_cap] = bound_cap
        bounds[bounds<-bound_cap] = -bound_cap
    return bounds

if __name__ == "__main__":
    disc = GaussKernelDiscretizer(bounds=[(-1,1), (-1,1)], nkern=2)
    disc2 = HardDiscretizer(bounds=[(-1,1), (-1,1)], resolution=4)

    data = np.array([
        [0,0],
        [1,1],
        [-1,-1],
    ])

    print(disc.transform(data))
    print(disc2.transform(data))
