"""
Interface to libpsht, for use while benchmarking.
"""
from libc.stdlib cimport malloc, free
cimport numpy as cnp
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
    cdef cnp.ndarray map
    cdef cnp.ndarray alm
    cdef int alm_pol_offset, map_pol_offset
    
    def __cinit__(self, nside, lmax, map=None, alm=None, mmax=None, 
                  weights=None, map_polarization=None,
                  alm_polarization=None, axis=0, should_accumulate=False):
        if mmax is None:
            mmax = lmax
        if (map_polarization not in (None, 'interleave', 'stack') and not
                isinstance(map_polarization, int)):
            raise ValueError("map_polarization has incorrect value")
        if (alm_polarization not in (None, 'interleave', 'stack') and not
                isinstance(alm_polarization, int)):
            raise ValueError("alm_polarization has incorrect value")
        if (map_polarization is None and alm_polarization is not None or 
                alm_polarization is None and map_polarization is not None):
            raise ValueError("Cannot do polarization for map and not for alms"
                             " or vice versa")
                    
        if map is None and alm is None:
            raise ValueError("Either map or alms must be non-None")
        if map is None:
            map = np.zeros((alm.shape[i] if i != axis else 12 *nside ** 2 
                            for i in range(len(alm.shape))), dtype=np.double)
        elif alm is None:
            alm = np.zeros((map.shape[i] if i != axis else 
                            lmax * (lmax + 1) // 2 + mmax + 1
                            for i in range(len(alm.shape))), 
                            dtype=np.complex128)


        if map.shape != tuple(alm.shape[i] if i != axis else 12 * nside ** 2
                            for i in range(len(alm.shape))):
            raise ValueError("Map shape incompatible with alm shape")

        cdef cnp.ndarray map_ = map
        cdef cnp.ndarray alm_ = alm

        if map_.stride[axis] % sizeof(map_.dtype) != 0:
            raise ValueError("Map stride not a multiple of map dtype")
        if alm_.stride[axis] % sizeof(alm_.dtype) != 0:
            raise ValueError("Alm stride not a multiple of alm dtype")

        #Determine strides and offsets
        map_stride = map_.stride[axis] / sizeof(map_.dtype)
        alm_stride = alm_.stride[axis] / sizeof(alm_.dtype)

        if map_polarization == 'interleave':
            map_stride *= 3
            self.map_pol_offset = 1
        elif map_polarization == 'stack':
            self.map_pol_offset = 12 * nside ** 2

        if alm_polarization == 'interleave':
            alm_stride *= 3
            self.alm_pol_offset = 1
        elif alm_polarization == 'stack':
            self.alm_pol_offset = lmax * (lmax + 1) // 2 + mmax + 1

        nm = mmax
        cdef cnp.ndarray mval = np.arange(mmax, dtype=np.intc)
        # mvstart is the "hypothetical" a_{0m}
        cdef cnp.ndarray[ptrdiff_t, mode='c'] mvstart = \
             lm_to_idx_mmajor(0, mval, lmax).astype(ptrdiff_dtype)
        mvstart *= alm_stride
        psht_make_general_alm_info(lmax, mmax + 1, alm_stride,
                                   <int*>mval.data,
                                   <ptrdiff_t*>mvstart.data,
                                   &self.alm_info)

        cdef cnp.ndarray weights_ 
        if weights is None:
            psht_make_healpix_geom_info(nside, map_stride, &self.geom_info)
        else:
            weights_ = weights
            psht_make_weighted_healpix_geom_info(nside, map_stride,  
                                                 <double*>weights_.data,
                                                 &self.geom_info)
        pshtd_make_joblist(&self.joblist)
        self.alm = alm_
        self.map = map_
        self.axis = axis
        self.map_polarization = map_polarization
        self.alm_polarization = alm_polarization
        if should_accumulate:
            self.add_output = 1
        else:
            self.add_output = 0

    def __dealloc__(self):
        if self.alm_info != NULL:
            psht_destroy_alm_info(self.alm_info)
        if self.geom_info != NULL:
            psht_destroy_geom_info(self.geom_info)
        if self.joblist != NULL:
            pshtd_destroy_joblist(self.joblist)

    def alm2map(self):
        cdef int axis = self.axis
        if isinstance(self.alm_polarization, int):
            in_itT = cnp.PyArray_IterAllButAxis(self.alm[(slice(None),) * 
                                            self.alm_polarization + (0,) + 
                                            (slice(None),) * (self.alm.ndim -
                                            self.alm_polarization - 1)], &axis)
            in_itE = cnp.PyArray_IterAllButAxis(self.alm[(slice(None),) * 
                                            self.alm_polarization + (1,) + 
                                            (slice(None),) * (self.alm.ndim -
                                            self.alm_polarization - 1)], &axis)
            in_itB = cnp.PyArray_IterAllButAxis(self.alm[(slice(None),) * 
                                            self.alm_polarization + (2,) + 
                                            (slice(None),) * (self.alm.ndim -
                                            self.alm_polarization - 1)], &axis)
            not_done = cnp.PyArray_ITER_NOTDONE(in_itT)
        else:
            in_it = cnp.PyArray_IterAllButAxis(self.alm, &axis)
            not_done = cnp.PyArray_ITER_NOTDONE(in_it)

        if isinstance(self.map_polarization, int):
            out_itT = cnp.PyArray_IterAllButAxis(self.map[(slice(None),) * 
                                            self.map_polarization + (0,) + 
                                            (slice(None),) * (self.map.ndim -
                                            self.map_polarization - 1)], &axis)
            out_itQ = cnp.PyArray_IterAllButAxis(self.map[(slice(None),) * 
                                            self.map_polarization + (1,) + 
                                            (slice(None),) * (self.map.ndim -
                                            self.map_polarization - 1)], &axis)
            out_itU = cnp.PyArray_IterAllButAxis(self.map[(slice(None),) * 
                                            self.map_polarization + (2,) + 
                                            (slice(None),) * (self.map.ndim -
                                            self.map_polarization - 1)], &axis)
        else:
            out_it = cnp.PyArray_IterAllButAxis(self.map, &axis)

        while not_done:
            if self.map_polarization is None:
                #alm_polarization must also be None in this case
                pshtd_add_job_alm2map(self.joblist,
                                    <pshtd_cmplx*>cnp.PyArray_ITER_DATA(in_it),
                                    <double*>cnp.PyArray_ITER_DATA(out_it),
                                    self.add_output)
                cnp.PyArray_ITER_NEXT(in_it)
                cnp.PyArray_ITER_NEXT(out_it)
                not_done = cnp.PyArray_ITER_NOTDONE(in_it)
            elif (self.map_polarization in ('interleave', 'stack') and
                    self.alm_polarization in ('interleave', 'stack')):
                pshtd_add_job_alm2map_pol(self.joblist,
                                        <pshtd_cmplx*>cnp.PyArray_ITER_DATA(in_it),
                                        <pshtd_cmplx*>cnp.PyArray_ITER_DATA(in_it) + self.alm_pol_offset,
                                        <pshtd_cmplx*>cnp.PyArray_ITER_DATA(in_it) + 2 * self.alm_pol_offset,
                                        <double*>cnp.PyArray_ITER_DATA(out_it),
                                        <double*>cnp.PyArray_ITER_DATA(out_it) + self.map_pol_offset,
                                        <double*>cnp.PyArray_ITER_DATA(out_it) + 2 * self.map_pol_offset,
                                        self.add_output)
                cnp.PyArray_ITER_NEXT(in_it)
                cnp.PyArray_ITER_NEXT(out_it)
                not_done = cnp.PyArray_ITER_NOTDONE(in_it)
            elif (self.map_polarization in ('interleave', 'stack') and 
                    isinstance(self.alm_polarization, int)):
                pshtd_add_job_alm2map_pol(self.joblist,
                                        <pshtd_cmplx*>cnp.PyArray_ITER_DATA(in_itT),
                                        <pshtd_cmplx*>cnp.PyArray_ITER_DATA(in_itE),
                                        <pshtd_cmplx*>cnp.PyArray_ITER_DATA(in_itB),
                                        <double*>cnp.PyArray_ITER_DATA(out_it),
                                        <double*>cnp.PyArray_ITER_DATA(out_it) + self.map_pol_offset,
                                        <double*>cnp.PyArray_ITER_DATA(out_it) + 2 * self.map_pol_offset,
                                        self.add_output)
                cnp.PyArray_ITER_NEXT(in_itT)
                cnp.PyArray_ITER_NEXT(in_itB)
                cnp.PyArray_ITER_NEXT(in_itE)
                cnp.PyArray_ITER_NEXT(out_it)
                not_done = cnp.PyArray_ITER_NOTDONE(in_itT)
            elif (isinstance(self.map_polarization, int) and 
                    self.alm_polarization in ('interleave', 'stack')):
                pshtd_add_job_alm2map_pol(self.joblist,
                                        <pshtd_cmplx*>cnp.PyArray_ITER_DATA(in_it),
                                        <pshtd_cmplx*>cnp.PyArray_ITER_DATA(in_it) + self.alm_pol_offset,
                                        <pshtd_cmplx*>cnp.PyArray_ITER_DATA(in_it) + 2 * self.alm_pol_offset,
                                        <double*>cnp.PyArray_ITER_DATA(out_itT),
                                        <double*>cnp.PyArray_ITER_DATA(out_itQ),
                                        <double*>cnp.PyArray_ITER_DATA(out_itU),
                                        self.add_output)
                cnp.PyArray_ITER_NEXT(in_it)
                cnp.PyArray_ITER_NEXT(out_itT)
                cnp.PyArray_ITER_NEXT(out_itQ)
                cnp.PyArray_ITER_NEXT(out_itU)
                not_done = cnp.PyArray_ITER_NOTDONE(in_it)
            elif (isinstance(self.map_polarization, int) and
                    isinstance(self.alm_polarization, int)):
                pshtd_add_job_alm2map_pol(self.joblist,
                                        <pshtd_cmplx*>cnp.PyArray_ITER_DATA(in_itT),
                                        <pshtd_cmplx*>cnp.PyArray_ITER_DATA(in_itE),
                                        <pshtd_cmplx*>cnp.PyArray_ITER_DATA(in_itB),
                                        <double*>cnp.PyArray_ITER_DATA(out_itT),
                                        <double*>cnp.PyArray_ITER_DATA(out_itQ),
                                        <double*>cnp.PyArray_ITER_DATA(out_itU),
                                        self.add_output)
                cnp.PyArray_ITER_NEXT(in_itT)
                cnp.PyArray_ITER_NEXT(in_itB)
                cnp.PyArray_ITER_NEXT(in_itE)
                cnp.PyArray_ITER_NEXT(out_itT)
                cnp.PyArray_ITER_NEXT(out_itQ)
                cnp.PyArray_ITER_NEXT(out_itU)
                not_done = cnp.PyArray_ITER_NOTDONE(in_itT)

        pshtd_execute_jobs(self.joblist, self.geom_info, self.alm_info)

#    def map2alm(self, 
#                np.ndarray[double, ndim=2, mode='c'] map,
#                np.ndarray[double complex, ndim=2, mode='c'] alm = None,
#                int repeat=1):
#        cdef int j
#        if alm is None:
#            alm = np.zeros((map.shape[0], (self.lmax + 1) * (self.lmax + 2) 
#                            // 2, ), np.complex, order='c')
#        if not (alm.shape[0] == map.shape[0]):
#            raise ValueError('Arrays not conforming')
#        if map.shape[1] != 12 * self.nside**2:
#            raise ValueError('map must have shape (npix, nmaps)')
#        if map.shape[0] == 1:
#            pshtd_add_job_map2alm(self.joblist,
#                                  <double*>map.data,
#                                  <pshtd_cmplx*>alm.data, 0)
#        elif map.shape[0] == 3:
#            pshtd_add_job_map2alm_pol(self.joblist,
#                                     <double*>map.data,
#                                     <double*>map.data + map.shape[1],
#                                     <double*>map.data + 2 * map.shape[1],
#                                     <pshtd_cmplx*>alm.data,
#                                     <pshtd_cmplx*>alm.data + alm.shape[1],
#                                     <pshtd_cmplx*>alm.data + 2 * alm.shape[1],
#                                     0)
#        for i in range(repeat):
#            pshtd_execute_jobs(self.joblist, self.geom_info, self.alm_info)
#        pshtd_clear_joblist(self.joblist)
#        return alm
#
#
#def alm2map_mmajor(alm, lmax, map=None, nside=None, repeat=1):
#    nmaps = alm.shape[1]
#    if map is not None:
#        assert nside is None
#        nside = int(np.sqrt(map.shape[0] // 12))
#    return PshtMmajorHealpix(lmax, nside, nmaps).alm2map(alm, map, repeat)

        
