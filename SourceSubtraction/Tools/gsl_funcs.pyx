cimport cython
from cython.view cimport array as cvarray
from cython_gsl cimport *

from cython.parallel import prange, parallel
cimport openmp
import numpy as np
cimport numpy as np

cimport openmp

from libc.math cimport sqrt, acos, sin, cos, floor
from libc.stdio cimport printf
from libc.stdlib cimport malloc, free

cdef extern from "gsl/gsl_sf_legendre.h":
    int gsl_sf_legendre_sphPlm_array(const int lmax, int m, const double x, double * result_array) nogil
    int gsl_sf_legendre_array_size(const int lmax, const int m) nogil
    ctypedef enum gsl_sf_legendre_t:
        GSL_SF_LEGENDRE_SCHMIDT,
        GSL_SF_LEGENDRE_SPHARM,
        GSL_SF_LEGENDRE_FULL,
        GSL_SF_LEGENDRE_NONE

    int gsl_sf_legendre_array(const gsl_sf_legendre_t norm, const int lmax, const double x, double * result_array) nogil
    int gsl_sf_legendre_array_n(const int lmax) nogil
    int gsl_sf_legendre_array_index(const int l, const int m) nogil

    int gsl_sf_legendre_Pl_array(const int lmax, const double x, double * result_array) nogil
    int gsl_sf_legendre_Pl_deriv_array(const int lmax, const double x, double * result_array, double * result_deriv) nogil

cdef void progress(long position, long length) nogil:
    """
    Prints a progress bar for cython.
    """

    # Do every per cent step, add to the progress bar
    cdef double steps = floor(<double>position / <double> length / 0.1)
    cdef long isteps = <long>steps
    cdef long i

    # Clear the line
    printf("\33[2K\r")
    for i in range(isteps):
        printf(':')



@cython.boundscheck(False)
@cython.wraparound(False)
def main(double[:,:] coord,long[:] idx,long[:] lval,long[:] mval):
    """
    
    """

    cdef int lmax = np.max(lval)
    cdef long N = idx.size
    cdef long Nsrc = coord.shape[0]
    cdef long i,j
    cdef int step, k, pos, last
    cdef double x, Pl
    cdef np.ndarray[np.float64_t, ndim=1] steps = np.linspace(-1,1,N,dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] output = np.zeros((N,2),dtype=np.float64)
    cdef double[:,:] output_view = output
    cdef double[:] result = np.zeros(N, dtype=np.float64)
    cdef double[:,:] coord_view  = coord
    cdef long[:] lval_view = lval
    cdef long[:] mval_view = mval 

    cdef char* c_str = '%d'
    # We chunk the parallel data over sources 
    # (may not be most efficient for small number of sources)
    # (but we generall have 10^4 sources)
    for j in prange(0,Nsrc,nogil=True):
        for i in range(N):
            Pl = gsl_sf_legendre_sphPlm(lval_view[i],mval_view[i],cos(coord_view[j,1]))
            output_view[i,0] += coord_view[j,2]*Pl*cos(mval_view[i]*coord_view[j,0]) * sin(coord_view[j,1])
            output_view[i,1] += coord_view[j,2]*Pl*sin(mval_view[i]*coord_view[j,0]) * sin(coord_view[j,1])

    return output


@cython.boundscheck(False)
@cython.wraparound(False)
def precomp_harmonics(double[:,:] coord,long N, long lmax): #,long[:] idx,long[:] lval,long[:] mval):
    cdef long Nsrc = coord.shape[0]
    cdef long i,j,m
    cdef np.ndarray[np.float64_t, ndim=2] output = np.zeros((N,2),dtype=np.float64)

    cdef int step, k, pos

    # Each thread needs its own counter (last), and buffer (result)
    cdef int * last
    cdef double * result

    # Enter parallised code and release GIL
    with nogil, parallel():

        # Declare memory needed for each thread...
        last = <int *> malloc(sizeof(int))
        result = <double *>malloc(sizeof(double) * N)

        # Now we loop over all the sources (these will be distributed equally between threads)
        for j in prange(0,Nsrc):
            last[0] = 0 # reset local counter
            for m in range(lmax):
                gsl_sf_legendre_sphPlm_array(lmax,m , cos(coord[j,1]), result)
                step = gsl_sf_legendre_array_size(lmax,m)
                for k in range(step):
                    pos = last[0] + k
                    output[pos,0] += result[k]*coord[j,2]*cos(m*coord[j,0]) 
                    output[pos,1] += result[k]*coord[j,2]*sin(m*coord[j,0]) 
                last[0] += step
        free(last)
        free(result)

    return output

@cython.boundscheck(False)
@cython.wraparound(False)
def precomp_harmonics_onethread(double[:,:] coord,long N, long lmax): #,long[:] idx,long[:] lval,long[:] mval):
    cdef long Nsrc = coord.shape[0]
    cdef long i,j,m
    cdef np.ndarray[np.float64_t, ndim=2] output = np.zeros((N,2),dtype=np.float64)

    cdef int step, k, pos

    # Each thread needs its own counter (last), and buffer (result)
    cdef int * last
    cdef double * result

    # Enter parallised code and release GIL
    with nogil:

        # Declare memory needed for each thread...
        last = <int *> malloc(sizeof(int))
        result = <double *>malloc(sizeof(double) * N)

        # Now we loop over all the sources (these will be distributed equally between threads)
        for j in range(0,Nsrc):
            last[0] = 0 # reset local counter
            for m in range(lmax):
                gsl_sf_legendre_sphPlm_array(lmax,m , cos(coord[j,1]), result)
                step = gsl_sf_legendre_array_size(lmax,m)
                for k in range(step):
                    pos = last[0] + k
                    output[pos,0] += result[k]*coord[j,2]*cos(m*coord[j,0]) 
                    output[pos,1] += result[k]*coord[j,2]*sin(m*coord[j,0]) 
                last[0] += step
        free(last)
        free(result)

    return output


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_transfer_function(double[:] theta, double[:] beam, int lmax, int lt, int mt):#,long N, long lmax): 
    cdef long i,j,l
    cdef int array_length, theta_length

    # Each thread needs its own counter (last), and buffer (result)
    cdef int * last
    cdef double * result
    cdef int index
    cdef int num_threads = openmp.omp_get_max_threads()
    cdef int itask
    theta_length = len(theta)
    cdef np.ndarray[np.float64_t, ndim=2] window = np.zeros((num_threads,lmax+1),dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] Pl = np.zeros((theta_length),dtype=np.float64)

    cdef double dx
    dx = (theta[0] - theta[1])**2
    dx = dx**0.5
    
    array_length = lmax + 1 #gsl_sf_legendre_array_n(lmax)
    with nogil, parallel():
        
        for i in prange(theta_length):
            itask = openmp.omp_get_thread_num()
            #printf("%d %d\n",itask,num_threads)
            result = <double *>malloc(sizeof(double) * array_length)
            gsl_sf_legendre_Pl_array(lmax, cos(theta[i]), result)
            for j in range(array_length):
                window[itask,j] += beam[i]*result[j]*sin(theta[i])*dx
            free(result)

    return np.sum(window,axis=0)
