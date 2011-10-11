import numpy as np
from ..psht import *
from matplotlib import pyplot as plt
from nose.tools import ok_, eq_

def test_basic():
    nmaps = 1
    nside = 16
    lmax = 2 * nside
    alm = np.zeros((((lmax + 1)*(lmax+2))//2, nmaps), dtype=np.complex128)
    alm[lm_to_idx_mmajor(1, 0, lmax), :] = 1
    alm[lm_to_idx_mmajor(0, 0, lmax), :] = 3
    T = PshtMmajorHealpix(nside=16, lmax=32, alm=alm)
    map = T.alm2map()
    yield ok_, map[0, 0] != map[-1, 0]
    alm = np.zeros((((lmax + 1)*(lmax+2))//2), dtype=np.complex128)
    alm[lm_to_idx_mmajor(1, 0, lmax)] = 1
    alm[lm_to_idx_mmajor(0, 0, lmax)] = 3
    T = PshtMmajorHealpix(nside=16, lmax=32, alm=alm)
    nmap = T.alm2map()
    nmap = np.reshape(nmap, map.shape)
    yield ok_, np.all(nmap == map)

    alm = np.zeros((((lmax + 1)*(lmax+2))//2, nmaps), dtype=np.complex128)
    alm[lm_to_idx_mmajor(0, 0, lmax), :] = 3
    T = PshtMmajorHealpix(nside=16, lmax=32, alm=alm)
    map = T.alm2map()
    yield eq_, map[0, 0],  map[-1, 0]

    nmaps = 2
    nside = 16
    lmax = 2 * nside
    alm = np.zeros((((lmax + 1)*(lmax+2))//2, nmaps), dtype=np.complex128)
    alm[lm_to_idx_mmajor(1, 0, lmax), :] = 1
    alm[lm_to_idx_mmajor(0, 0, lmax), :] = 3
    alm[:, 1] *= -1
    T = PshtMmajorHealpix(nside=16, lmax=32, alm=alm)
    map = T.alm2map()
    yield ok_, np.all(map[:, 0] == -map[:, 1])

    #polarization tests
    nmaps = 3
    alm = np.zeros((((lmax + 1)*(lmax+2))//2, nmaps), dtype=np.complex128)
    alm[lm_to_idx_mmajor(1, 1, lmax), :] = 1 + 4j
    alm[lm_to_idx_mmajor(0, 0, lmax), :] = 3
    alm[lm_to_idx_mmajor(5, 4, lmax), :] = 2 + 5j
    
    alm[:, 0] *= 1
    alm[:, 1] *= 1
    alm[:, 2] *= -1
    T = PshtMmajorHealpix(nside=16, lmax=32, alm=alm, alm_polarization=1, map_polarization=1)
    map = T.alm2map()

    nmaps = 3
    alm = np.zeros((((lmax + 1)*(lmax+2))//2, nmaps), dtype=np.complex128)
    alm[lm_to_idx_mmajor(1, 1, lmax), :] = 1 + 4j
    alm[lm_to_idx_mmajor(0, 0, lmax), :] = 3
    alm[lm_to_idx_mmajor(5, 4, lmax), :] = 2 + 5j
    
    alm[:, 0] *= 1
    alm[:, 1] *= 1
    alm[:, 2] *= -1
    alm = np.reshape(alm, 3*alm.shape[0])
    T = PshtMmajorHealpix(nside=16, lmax=32, alm=alm, alm_polarization='interleave', map_polarization=1)
    nmap = T.alm2map()
    yield ok_, np.all(map == nmap)

    alm = np.zeros((nmaps, ((lmax + 1)*(lmax+2))//2), dtype=np.complex128)
    alm[:, lm_to_idx_mmajor(1, 1, lmax)] = 1 + 4j
    alm[:, lm_to_idx_mmajor(0, 0, lmax)] = 3
    alm[:, lm_to_idx_mmajor(5, 4, lmax)] = 2 + 5j
    alm[0, :] *= 1
    alm[1, :] *= 1
    alm[2, :] *= -1
    alm = np.reshape(alm, 3*alm.shape[1])
    T = PshtMmajorHealpix(nside=16, lmax=32, alm=alm, 
                          alm_polarization='stack', map_polarization=1)
    nmap = T.alm2map()
    yield ok_, np.all(map == nmap)

    alm = np.zeros((((lmax + 1)*(lmax+2))//2, 5), dtype=np.complex128)
    alm[lm_to_idx_mmajor(1, 1, lmax), :] = 1 + 4j
    alm[lm_to_idx_mmajor(0, 0, lmax), :] = 3
    alm[lm_to_idx_mmajor(5, 4, lmax), :] = 2 + 5j
    
    alm[:, 0] *= 1
    alm[:, 1] *= 1
    alm[:, 2] *= -1
    alm[:, 3] *= 1
    alm[:, 4] *= -1
    T = PshtMmajorHealpix(nside=16, lmax=32, alm=alm)
    map = T.alm2map()
    yield ok_, np.all(map[:, 0] == map[:, 1])
    yield ok_, np.all(map[:, 0] == -map[:, 2])
    yield ok_, np.all(map[:, 2] == map[:, 4])
    yield ok_, np.all(map[:, 1] == map[:, 3])
