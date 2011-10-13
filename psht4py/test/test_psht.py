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
    yield ok_, not np.all(map == 0)
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
    alm = np.zeros((3, ((lmax + 1)*(lmax+2))//2, 5), dtype=np.complex128)
    alm[:, lm_to_idx_mmajor(1, 1, lmax), :] = 1 + 4j
    alm[:, lm_to_idx_mmajor(0, 0, lmax), :] = 3
    alm[:, lm_to_idx_mmajor(5, 4, lmax), :] = 2 + 5j
    alm[:, :, 0] *= 1
    alm[:, :, 1] *= 1
    alm[:, :, 2] *= -1
    alm[:, :, 3] *= 1
    alm[:, :, 4] *= -1

    T = PshtMmajorHealpix(nside=16, lmax=32, alm=alm, alm_polarization=0, 
                          map_polarization=0, map_axis=1, alm_axis=1)
    nmap = T.alm2map()
    yield ok_, np.all(nmap[:, :, 0] == nmap[:, :, 1])
    yield ok_, np.all(nmap[:, :, 0] == -nmap[:, :, 2])
    yield ok_, np.all(nmap[:, :, 2] == nmap[:, :, 4])
    yield ok_, np.all(nmap[:, :, 1] == nmap[:, :, 3])
    yield ok_, np.all(nmap[0] == map)

    #Testing map2alm
    npix = 12 * nside ** 2
    map = np.zeros(npix)
    map[:npix // 2] = 1
    T = PshtMmajorHealpix(nside=16, lmax=32, map=map)
    alm = T.map2alm()
    yield ok_, alm[0] != alm[1] 
    yield ok_, not np.all(alm == 0)
    map = np.zeros((npix, 1))
    map[:npix // 2] = 1
    T = PshtMmajorHealpix(nside=16, lmax=32, map=map)
    nalm = T.map2alm()
    yield ok_, np.all(nalm[:, 0] == alm)

    map = np.zeros((npix, 2))
    map[:npix //2] = 1
    map[:, 1] *= -1
    T = PshtMmajorHealpix(nside=16, lmax=32, map=map)
    nalm = T.map2alm()
    yield ok_, np.all(nalm[:, 0] == -nalm[:, 1])
    map = np.zeros((npix, 3))
    map[:npix // 2] = 1
    map[:, 1] *=-1
    map[:, 2] *= 0.5
    T = PshtMmajorHealpix(nside=16, lmax=32, map=map, map_polarization=1,
            alm_polarization=1)
    alm = T.map2alm()
    map = np.reshape(map, npix * 3)
    T = PshtMmajorHealpix(nside=16, lmax=32, map=map, map_polarization='interleave', alm_polarization=1)
    nalm = T.map2alm()
    yield ok_, np.all(nalm == alm)
    map = np.zeros((3, npix))
    map[:, :npix // 2] = 1
    map[1, :] *=-1
    map[2, :] *= 0.5
    map = np.reshape(map, npix * 3)
    T = PshtMmajorHealpix(nside=16, lmax=32, map=map, map_polarization='stack', alm_polarization=1)
    nalm = T.map2alm()
    yield ok_, np.all(nalm == alm)
    T = PshtMmajorHealpix(nside=16, lmax=32, map=map, map_polarization='stack', alm_polarization='interleave')
    alm = T.map2alm()
    yield ok_, np.all(np.reshape(nalm, nalm.shape[0]*3) == alm)
    T = PshtMmajorHealpix(nside=16, lmax=32, map=map, map_polarization='stack', alm_polarization=0, alm_axis=1)
    nalm = T.map2alm()
    T = PshtMmajorHealpix(nside=16, lmax=32, map=map, map_polarization='stack', alm_polarization='stack')
    alm = T.map2alm()
    yield ok_, np.all(np.reshape(nalm, nalm.shape[1] * 3) == alm)
    map = np.zeros((5, npix, 2, 3, 6))
    map[:, :npix // 2] = 1
    map[:, :, :, 1] *= -1
    map[:, :, :, 2] *= 0.5
    T = PshtMmajorHealpix(nside=16, lmax=32, map=map, map_polarization=3, alm_polarization=3, map_axis=1, alm_axis=1)
    alm = T.map2alm()
    yield ok_, np.all(alm[0, :, 0, 0, 0] == nalm[0])
    yield ok_, np.all(alm[0, :, 0, 1, 0] == nalm[1])
    yield ok_, np.all(alm[0, :, 0, 2, 0] == nalm[2])
    yield ok_, not np.all(alm[0, :, 0, 2, 0] == 0)
