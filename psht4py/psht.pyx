"""
Interface to libpsht, for use while benchmarking.
"""
from libc.stdlib cimport malloc, free
cimport numpy as np
import numpy as np

cdef extern from "stddef.h":
    ctypedef int ptrdiff_t

cdef extern from "psht.h":
    
    ctypedef struct psht_alm_info:
        pass

    ctypedef struct psht_geom_info:
        pass

    ctypedef struct pshtd_joblist:
        pass

    ctypedef struct pshtd_cmplx:
        pass


    void psht_make_rectangular_alm_info(int lmax, int mmax, int stride,
                                        psht_alm_info **alm_info)

    void psht_make_general_alm_info(int lmax, int nm, int stride, int *mval,
                                    ptrdiff_t *mstart, psht_alm_info **alm_info)
    void psht_destroy_alm_info(psht_alm_info *info)
    void psht_destroy_geom_info(psht_geom_info *info)
    void pshtd_make_joblist (pshtd_joblist **joblist)
    void pshtd_clear_joblist (pshtd_joblist *joblist)
    void pshtd_destroy_joblist (pshtd_joblist *joblist)
    void pshtd_add_job_alm2map (pshtd_joblist *joblist, pshtd_cmplx *alm,
                                double *map, int add_output)
    void pshtd_add_job_alm2map_pol ( pshtd_joblist *joblist, pshtd_cmplx *almT, 
                                pshtd_cmplx *almG, pshtd_cmplx *almC,
                                double *mapT, double *mapQ, double *mapU, 
                                int add_output)
    void pshtd_add_job_map2alm (pshtd_joblist *joblist, double *map, 
                                pshtd_cmplx *alm, int add_output)
    void pshtd_add_job_map2alm_pol ( pshtd_joblist *joblist, double *mapT, 
                                     double *mapQ, double *mapU, 
                                     pshtd_cmplx *almT, pshtd_cmplx *almG, 
                                     pshtd_cmplx *almC, int add_output)
    void pshtd_execute_jobs (pshtd_joblist *joblist,
                             psht_geom_info *geom_info, psht_alm_info *alm_info)
    

cdef extern from "psht_geomhelpers.h":
    void psht_make_healpix_geom_info (int nside, int stride,
                                      psht_geom_info **geom_info)
    void psht_make_weighted_healpix_geom_info (int nside, int stride,
                                               double *weight,
                                               psht_geom_info **geom_info)
        
def lm_to_idx_mmajor(l, m, lmax):
    # broadcasts
    return m * (2 * lmax - m + 3) // 2 + (l - m)

if sizeof(ptrdiff_t) == 4:
    ptrdiff_dtype = np.int32
elif sizeof(ptrdiff_t) == 8:
    ptrdiff_dtype = np.int64
else:
    assert False

cdef class PshtMmajorHealpix:
    cdef psht_alm_info *alm_info
    cdef psht_geom_info *geom_info
    cdef pshtd_joblist *joblist
    cdef readonly int Nside, nmaps, lmax
    
    def __cinit__(self, lmax, Nside, nmaps, 
                  np.ndarray[double, mode='c'] weights=None):
        # fill in alm_info
        cdef np.ndarray[int, mode='c'] mval = np.arange(lmax + 1, dtype=np.intc)
        # mvstart is the "hypothetical" a_{0m}
        cdef np.ndarray[ptrdiff_t, mode='c'] mstart = \
             lm_to_idx_mmajor(0, mval, lmax).astype(ptrdiff_dtype)
        mstart *= nmaps
        self.nmaps = nmaps
        psht_make_general_alm_info(lmax, lmax + 1, nmaps,
                                   <int*>mval.data,
                                   <ptrdiff_t*>mstart.data,
                                   &self.alm_info)
        if weights is None:
            psht_make_healpix_geom_info(Nside, 1, &self.geom_info)
        else:
            psht_make_weighted_healpix_geom_info(Nside, 1,  
                                                 <double*>weights.data,
                                                 &self.geom_info)
        pshtd_make_joblist(&self.joblist)
        self.Nside = Nside
        self.lmax = lmax
        
    def __dealloc__(self):
        if self.alm_info != NULL:
            psht_destroy_alm_info(self.alm_info)
        if self.geom_info != NULL:
            psht_destroy_geom_info(self.geom_info)
        if self.joblist != NULL:
            pshtd_destroy_joblist(self.joblist)

    #For now, only supports single map or polarization
    def alm2map(self,
                np.ndarray[double complex, ndim=2, mode='c'] alm,
                np.ndarray[double, ndim=2, mode='c'] map=None,
                int repeat=1):
        cdef int j
        if alm.shape[1] != ((self.lmax + 1) * (self.lmax + 2)) // 2:
            raise ValueError('alm.shape does not match lmax')
        if map is None:
            map = np.zeros((alm.shape[0], 12 * self.Nside**2), 
                            np.double, order='c')
#        if alm.shape[1] != self.nmaps:
#            raise ValueError('Number of maps does not match nmaps passed in constructor')
        if not (alm.shape[0] == map.shape[0]):
            raise ValueError('Arrays not conforming')
        if map.shape[1] != 12 * self.Nside**2:
            raise ValueError('map must have shape (nmaps, npix)')
        if alm.shape[0] == 1:
            pshtd_add_job_alm2map(self.joblist,
                                  <pshtd_cmplx*>alm.data,
                                  <double*>map.data, 0)
        elif alm.shape[0] == 3:
            pshtd_add_job_alm2map_pol(self.joblist, 
                                      <pshtd_cmplx*>alm.data,
                                      <pshtd_cmplx*>alm.data + alm.shape[1],
                                      <pshtd_cmplx*>alm.data + 2 * alm.shape[1],
                                      <double*>map.data,
                                      <double*>map.data + map.shape[1],
                                      <double*>map.data + 2 * map.shape[1], 0)
        for i in range(repeat):
            pshtd_execute_jobs(self.joblist, self.geom_info, self.alm_info)
        pshtd_clear_joblist(self.joblist)
        return map

    def map2alm(self, 
                np.ndarray[double, ndim=2, mode='c'] map,
                np.ndarray[double complex, ndim=2, mode='c'] alm = None,
                int repeat=1):
        cdef int j
        if alm is None:
            alm = np.zeros((map.shape[0], (self.lmax + 1) * (self.lmax + 2) 
                            // 2, ), np.complex, order='c')
        if not (alm.shape[0] == map.shape[0]):
            raise ValueError('Arrays not conforming')
        if map.shape[1] != 12 * self.Nside**2:
            raise ValueError('map must have shape (npix, nmaps)')
        if map.shape[0] == 1:
            pshtd_add_job_map2alm(self.joblist,
                                  <double*>map.data,
                                  <pshtd_cmplx*>alm.data, 0)
        elif map.shape[0] == 3:
            pshtd_add_job_map2alm_pol(self.joblist,
                                     <double*>map.data,
                                     <double*>map.data + map.shape[1],
                                     <double*>map.data + 2 * map.shape[1],
                                     <pshtd_cmplx*>alm.data,
                                     <pshtd_cmplx*>alm.data + alm.shape[1],
                                     <pshtd_cmplx*>alm.data + 2 * alm.shape[1],
                                     0)
        for i in range(repeat):
            pshtd_execute_jobs(self.joblist, self.geom_info, self.alm_info)
        pshtd_clear_joblist(self.joblist)
        return alm


def alm2map_mmajor(alm, lmax, map=None, Nside=None, repeat=1):
    nmaps = alm.shape[1]
    if map is not None:
        assert Nside is None
        Nside = int(np.sqrt(map.shape[0] // 12))
    return PshtMmajorHealpix(lmax, Nside, nmaps).alm2map(alm, map, repeat)

        
