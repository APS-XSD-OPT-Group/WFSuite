import numba as nb
from numba import prange
import numpy as np

def dist_numpy(v1, arry2):
    return np.negative(np.sum((v1-arry2)**2, axis=2))


@nb.jit('float32[:,:](float32[:], float32[:,:,:])', nopython=True)
def dist_numba(v1, arry2):
    c = np.empty(arry2.shape[0:2], dtype=np.float32)
    for ii in prange(arry2.shape[0]):
        for jj in range(arry2.shape[1]):
            s = 0
            for kk in range(arry2.shape[2]):
                s += (v1[kk]-arry2[ii, jj, kk])**2
            c[ii, jj] = - s

    return c.astype(np.float32)

