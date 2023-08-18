#!python3
# -*-coding:Utf-8 -*
# distutils: language = c++


########################################################################################################
########################################################################################################

#
# %%%% !!! IMPORTANT NOTE !!! %%%%
# At the end of the fast_ms_ne.py file, a demo presents how this python code can be used. Running this file (python fast_ms_ne.py) will run the demo. Importing this module will not run the demo. The demo takes a few minutes. To be able to use the code in fast_ms_ne.py, do not forget to first compile the Cython file 'cython_implem.pyx'; check the instructions below for explanations on the required compilation steps. 
# %%%% !!!                !!! %%%%

#     cython_implem.pyx

# This project and the codes in this repository implement fast multi-scale neighbor embedding algorithms for nonlinear dimensionality reduction (DR). 
# The fast algorithms which are implemented are described in the article "Fast Multiscale Neighbor Embedding", from Cyril de Bodt, Dounia Mulders, Michel Verleysen and John A. Lee, published in IEEE Transactions on Neural Networks and Learning Systems, in 2020. 
# The implementations are provided using the python programming language, but involve some C and Cython codes for performance purposes. 

# Link to retrieve the article: https://ieeexplore.ieee.org/document/9308987

# If you use the codes in this repository or the article, please cite as: 
# - C. de Bodt, D. Mulders, M. Verleysen and J. A. Lee, "Fast Multiscale Neighbor Embedding," in IEEE Transactions on Neural Networks and Learning Systems, 2020, doi: 10.1109/TNNLS.2020.3042807.
# - BibTeX entry:
# @article{CdB2020FMsNE,
#  author={C. {de Bodt} and D. {Mulders} and M. {Verleysen} and J. A. {Lee}},
#  journal={{IEEE} Trans. Neural Netw. Learn. Syst.},
#  title={{F}ast {M}ultiscale {N}eighbor {E}mbedding}, 
#  year={2020},
#  volume={},
#  number={},
#  pages={1-15},
#  doi={10.1109/TNNLS.2020.3042807}}

# The files contained in this repository are:
# - fast_ms_ne.py: main python code to employ. At the end of the fast_ms_ne.py file, a demo presents how this python code can be used. Running this file (python fast_ms_ne.py) will run the demo. Importing this module will not run the demo. The demo takes a few minutes. To be able to use the code in fast_ms_ne.py, the Cython file 'cython_implem.pyx' must first be compiled; check the instructions below for explanations on the required compilation steps, as well as for more information on the content of the 'fast_ms_ne.py' file. The tested versions of the imported packages are also specified hereunder. 
# - cython_implem.pyx: Cython implementations of the multi-scale and fast multi-scale neighbor embedding algorithms.
# - setup.py: this python file can be used to compile the Cython file 'cython_implem.pyx', which is necessary to be able to run the functions implemented in the python 'fast_ms_ne.py' file.
# - arithmetic_ansi.h, arithmetic_sse_double.h, arithmetic_sse_float.h, lbfgs.c, lbfgs.h, vptree.h: code files employed in 'cython_implem.pyx'.

# To compile the Cython file 'cython_implem.pyx' before using the python 'fast_ms_ne.py' file, perform the following steps:
# - Make sure to have Cython ('https://cython.org/', last consulted on May 30, 2020) installed on your system. For instructions, check 'https://cython.readthedocs.io/en/latest/src/quickstart/install.html' (last consulted on May 30, 2020). Note that this web link mentions that Cython requires a C compiler to be present on the system, and provides further information to get such a C compiler according to your system. Note also that Cython is available from the Anaconda Python distribution. 
# - Run the command 'python setup.py build_ext --inplace' in the folder in which you downloaded the code files provided in this repository. Running this command may for instance be done from your 'Anaconda Prompt', if you are using the Anaconda Python distribution. Check 'https://cython.readthedocs.io/en/latest/src/quickstart/build.html' (last consulted on May 30, 2020) if you wish to have more information on this step. 
# - You can now use the functions provided in the Python 'fast_ms_ne.py' file as you would normally do in python. You can also now run the demo of the 'fast_ms_ne.py' file, simply by running this file (python fast_ms_ne.py). The demo takes a few minutes. 

# The main functions of the fast_ms_ne.py file are:
# - 'mssne': nonlinear dimensionality reduction through multi-scale SNE (Ms SNE), as presented in the reference [2] below and summarized in [1]. This function enables reducing the dimension of a data set. Given a data set with N samples, the 'mssne' function has O(N**2 log(N)) time complexity. It can hence run on databases with up to a few thousands of samples. This function is based on the Cython implementations in 'cython_implem.pyx'.
# - 'mstsne': nonlinear dimensionality reduction through multi-scale t-SNE (Ms t-SNE), as presented in the reference [6] below and summarized in [1]. This function enables reducing the dimension of a data set. Given a data set with N samples, the 'mstsne' function has O(N**2 log(N)) time complexity. It can hence run on databases with up to a few thousands of samples. This function is based on the Cython implementations in 'cython_implem.pyx'.
# - 'fmssne': nonlinear dimensionality reduction through fast multi-scale SNE (FMs SNE), as presented in the reference [1] below. This function enables reducing the dimension of a data set. Given a data set with N samples, the 'fmssne' function has O(N (log(N))**2) time complexity. It can hence run on very large-scale databases. This function is based on the Cython implementations in 'cython_implem.pyx'.
# - 'fmstsne': nonlinear dimensionality reduction through fast multi-scale t-SNE (FMs t-SNE), as presented in the reference [1] below. This function enables reducing the dimension of a data set. Given a data set with N samples, the 'fmstsne' function has O(N (log(N))**2) time complexity. It can hence run on very large-scale databases. This function is based on the Cython implementations in 'cython_implem.pyx'.
# - 'eval_dr_quality': unsupervised evaluation of the quality of a low-dimensional embedding, as introduced in [3, 4] and employed and summarized in [1, 2, 5]. This function enables computing DR quality assessment criteria measuring the neighborhood preservation from the high-dimensional space to the low-dimensional one. The documentation of the function explains the meaning of the criteria and how to interpret them. Given a data set with N samples, the 'eval_dr_quality' function has O(N**2 log(N)) time complexity. It can hence run using databases with up to a few thousands of samples. This function is not based on the Cython implementations in 'cython_implem.pyx'.
# - 'red_rnx_auc': this function is similar to the 'eval_dr_quality' function, but given a data set with N samples, the 'red_rnx_auc' function has O(N*Kup*log(N)) time complexity, where Kup is the maximum neighborhood size accounted when computing the quality criteria. This function can hence run using much larger databases than 'eval_dr_quality', provided that Kup is small compared to N. This function is based on the Cython implementations in 'cython_implem.pyx'.
# - 'viz_2d_emb' and 'viz_qa': visualization of a 2-D embedding and of the quality criteria. These functions respectively enable to: 
# ---> 'viz_2d_emb': plot a 2-D embedding. This function is not based on the Cython implementations in 'cython_implem.pyx'.
# ---> 'viz_qa': depict the quality criteria computed by 'eval_dr_quality' and 'red_rnx_auc'. This function is not based on the Cython implementations in 'cython_implem.pyx'.
# The documentations of the functions describe their parameters. The demo shows how they can be used. 

# Notations:
# - DR: dimensionality reduction.
# - HD: high-dimensional.
# - LD: low-dimensional.
# - HDS: HD space.
# - LDS: LD space.
# - SNE: stochastic neighbor embedding.
# - t-SNE: t-distributed SNE.
# - Ms SNE: multi-scale SNE.
# - Ms t-SNE: multi-scale t-SNE.
# - BH t-SNE: Barnes-Hut t-SNE.

# References:
# [1] C. de Bodt, D. Mulders, M. Verleysen and J. A. Lee, "Fast Multiscale Neighbor Embedding," in IEEE Transactions on Neural Networks and Learning Systems, 2020, doi: 10.1109/TNNLS.2020.3042807.
# [2] Lee, J. A., Peluffo-Ordóñez, D. H., & Verleysen, M. (2015). Multi-scale similarities in stochastic neighbour embedding: Reducing dimensionality while preserving both local and global structure. Neurocomputing, 169, 246-261.
# [3] Lee, J. A., & Verleysen, M. (2009). Quality assessment of dimensionality reduction: Rank-based criteria. Neurocomputing, 72(7-9), 1431-1443.
# [4] Lee, J. A., & Verleysen, M. (2010). Scale-independent quality criteria for dimensionality reduction. Pattern Recognition Letters, 31(14), 2248-2257.
# [5] Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013). Type 1 and 2 mixtures of Kullback–Leibler divergences as cost functions in dimensionality reduction based on similarity preservation. Neurocomputing, 112, 92-108.
# [6] de Bodt, C., Mulders, D., Verleysen, M., & Lee, J. A. (2018). Perplexity-free t-SNE and twice Student tt-SNE. In ESANN (pp. 123-128).
# [7] van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of Machine Learning Research, 9(Nov), 2579-2605.
# [8] van der Maaten, L. (2014). Accelerating t-SNE using tree-based algorithms. Journal of Machine Learning Research, 15(1), 3221-3245.

# author: Cyril de Bodt (Human Dynamics - MIT Media Lab, and ICTEAM - UCLouvain)
# @email: cdebodt __at__ mit __dot__ edu, or cyril __dot__ debodt __at__ uclouvain.be
# Last modification date: Jan 21th, 2021
# Copyright (c) 2021 Université catholique de Louvain (UCLouvain), ICTEAM. All rights reserved.

# The codes in this repository were tested with Python 3.6.5 (Anaconda distribution, Continuum Analytics, Inc.). They use the following modules:
# - numpy: version 1.14.2 tested
# - numba: version 0.37.0 tested
# - scipy: version 1.0.1 tested
# - matplotlib: version 2.2.2 tested
# - scikit-learn: version 0.19.1 tested
# - Cython: version 0.28.1 tested

# You can use, modify and redistribute this software freely, but not for commercial purposes. 
# The use of this software is at your own risk; the authors are not responsible for any damage as a result from errors in the software.


########################################################################################################
########################################################################################################


#cython: binding=False
#cython: wraparound=False
#cython: boundscheck=False
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: profile=False
#cython: linetrace=False
#cython: infer_types=False
#cython: embedsignature=False
#cython: cdivision=True 
#cython: cdivision_warnings=False 
#cython: overflowcheck=False 
#cython: overflowcheck.fold=False 
#cython: language_level=3 
#cython: always_allow_keywords=False 
#cython: type_version_tag=True 
#cython: iterable_coroutine=False 
#cython: optimize.use_switch=True 
#cython: optimize.unpack_method_calls=True 
#cython: warn.undeclared=False 
#cython: warn.unreachable=False 
#cython: warn.maybe_uninitialized=False 
#cython: warn.unused=False 
#cython: warn.unused_arg=False 
#cython: warn.unused_result=False 
#cython: warn.multiple_declarators=False 


#######################################################
####################################################### Imports 
#######################################################


# Numpy is needed to define FLOAT64_EPS. 'cimport' is used to import compile-time information about the numpy module. 
import numpy as np
cimport numpy as np
# Importing some functions from the C math library
from libc.math cimport sqrt, log, exp, fabs, round, log2, pow, int, abs
# Import Python C-API memory allocation functions
from cpython.mem cimport PyMem_Malloc, PyMem_Free, PyMem_Realloc
# Sorting function
from libcpp.algorithm cimport sort
# Random number generation and exit function
from libc.stdlib cimport rand, srand, exit, EXIT_FAILURE
# Print
from libc.stdio cimport printf
# Limits of the floating-point values
from libc.float cimport DBL_MIN, DBL_MAX
# Memory copy function
from libc.string cimport memcpy, memset

import scipy.signal

import time


DTYPE = np.double
ctypedef np.double_t DTYPE_t

DTYPE_int = np.int32
ctypedef np.int32_t DTYPE_int_t


#######################################################
####################################################### Global variables
#######################################################


# If some double is smaller than EPSILON_DBL in magnitude, it is considered as close to zero.
cdef double EPSILON_DBL = 1e-8

# To avoid dividing by zeros in similarity-related quantities.  
cdef double FLOAT64_EPS = np.finfo(dtype=np.float64).eps


#######################################################
####################################################### Minimum and maximum functions
#######################################################


cdef inline double min_arr_ptr(const double* x, Py_ssize_t m) nogil:
    """
    Return the minimum value in a one-dimensional array, assuming the latter has at least one element.
    m is the size of the array x. 
    """
    cdef Py_ssize_t i
    cdef double v = x[0]
    for i in range(1, m, 1):
        if x[i] < v:
            v = x[i]
    return v


cdef inline double max_arr_ptr(const double* x, Py_ssize_t m) nogil:
    """
    Return the maximum value in a one-dimensional array, assuming the latter has at least one element.
    m is the size of the array x. 
    """
    cdef Py_ssize_t i
    cdef double v = x[0]
    for i in range(1, m, 1):
        if x[i] > v:
            v = x[i]
    return v


cdef inline Py_ssize_t max_arr_ptr_Pysst(const Py_ssize_t* x, Py_ssize_t m) nogil:
    """
    Return the maximum value in a one-dimensional array, assuming the latter has at least one element.
    m is the size of the array x. 
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t v = x[0]
    for i in range(1, m, 1):
        if x[i] > v:
            v = x[i]
    return v


cdef inline double max_arr2d_col(double** x, Py_ssize_t m, Py_ssize_t c) nogil:
    """
    Search the maximum of some column in a 2d array. m is the number of rows, c is the column to search. 
    """
    cdef Py_ssize_t i
    cdef double v = x[0][c]
    for i in range(1, m, 1):
        if x[i][c] > v:
            v = x[i][c]
    return v


cdef inline double min_arr_ptr_step(const double* x, Py_ssize_t m, Py_ssize_t start, Py_ssize_t step) nogil:
    """
    Similar to min_arr_ptr, but with start and step parameters. 
    """
    cdef double v = x[start]
    start += step
    while start < m:
        if x[start] < v:
            v = x[start]
        start += step
    return v


cdef inline double max_arr_ptr_step(const double* x, Py_ssize_t m, Py_ssize_t start, Py_ssize_t step) nogil:
    """
    Similar to max_arr_ptr, but with start and step parameters. 
    """
    cdef double v = x[start]
    start += step
    while start < m:
        if x[start] > v:
            v = x[start]
        start += step
    return v


#######################################################
####################################################### Euclidean distance function
#######################################################


cdef inline double sqeucl_dist_ptr(const double* x, const double* y, Py_ssize_t m) nogil:
    """
    Computes the squared Euclidean distance between x and y, which are assumed to be one-dimensional and containing the same number m of elements. 
    In:
    - x, y: two one-dimensional arrays with the same number of elements.
    - m: size of x and y. 
    Out:
    The squared Euclidean distance between x and y.
    """
    cdef Py_ssize_t i
    cdef double d = 0.0
    cdef double v
    for i in range(m):
        v = x[i] - y[i]
        d += v*v
    return d


#######################################################
####################################################### Infinite distance function
#######################################################


cdef inline double inf_dist_ptr(const double* x, const double* y, Py_ssize_t m) nogil:
    """
    Computes the infinite distance (i.e. the distance based on the infinite norm) between x and y, which are assumed to be one-dimensional and with the same number of elements. x and y are assumed to have at least one element. 
    In:
    - x, y: pointers to two one-dimensional arrays with the same number of elements. They are assumed to have at least one element.
    - m: size of x and y.
    Out:
    The infinite distance between x and y.
    """
    cdef Py_ssize_t i
    cdef double d = fabs(x[0] - y[0])
    cdef double v
    for i in range(1, m, 1):
        v = fabs(x[i] - y[i])
        if v > d:
            d = v
    return d


#######################################################
####################################################### Mean of an array
#######################################################


cdef inline double mean_arr_ptr_step(const double* x, Py_ssize_t m, Py_ssize_t start, Py_ssize_t step, double N) nogil:
    """
    Return the mean of the elements pointed by x, at the indexes start, start+step, start+2*step, ..., until start+m-1.
    m is the total size of x.
    N is the number of elements over which we compute the mean.
    """
    cdef double v = x[start]
    start += step
    while start < m:
        v += x[start]
        start += step
    return v/N


#######################################################
####################################################### Variance of an array
#######################################################


cdef inline double var_arr_ptr_step(const double* x, Py_ssize_t m, Py_ssize_t start, Py_ssize_t step, double N, double den_var) nogil:
    """
    Computes the variance of the elements pointed by x, at the indexes start, start + step, start + 2*step, ..., until start+m-1. 
    m is the total size of x.
    m must be at least 2.
    den_var can be set to N-1.0
    """
    cdef double mu = mean_arr_ptr_step(x, m, start, step, N)
    cdef double diff = x[start] - mu
    cdef double v = diff * diff
    start += step
    while start < m:
        diff = x[start] - mu
        v += diff * diff
        start += step
    return v/den_var


#######################################################
####################################################### Allocation functions. The returned values must be freed. 
#######################################################


cdef inline void free_int_2dmat(int** arr, Py_ssize_t M):
    """
    """
    cdef Py_ssize_t m
    for m in range(M):
        PyMem_Free(arr[m])
    PyMem_Free(arr)


cdef inline void free_int_3dmat(int*** arr, Py_ssize_t M, Py_ssize_t N):
    """
    """
    cdef Py_ssize_t m, n
    for m in range(M):
        for n in range(N):
            PyMem_Free(arr[m][n])
        PyMem_Free(arr[m])
    PyMem_Free(arr)


cdef inline void free_dble_2dmat(double** arr, Py_ssize_t M):
    """
    """
    cdef Py_ssize_t m
    for m in range(M):
        PyMem_Free(arr[m])
    PyMem_Free(arr)


cdef inline void free_dble_3dmat(double*** arr, Py_ssize_t M, Py_ssize_t N):
    """
    """
    cdef Py_ssize_t m, n
    for m in range(M):
        for n in range(N):
            PyMem_Free(arr[m][n])
        PyMem_Free(arr[m])
    PyMem_Free(arr)


cdef inline void free_Pysst_2dmat(Py_ssize_t** arr, Py_ssize_t M):
    """
    """
    cdef Py_ssize_t m
    for m in range(M):
        PyMem_Free(arr[m])
    PyMem_Free(arr)


cdef inline void free_Pysst_3dmat(Py_ssize_t*** arr, Py_ssize_t M, Py_ssize_t N):
    """
    """
    cdef Py_ssize_t m, n
    for m in range(M):
        for n in range(N):
            PyMem_Free(arr[m][n])
        PyMem_Free(arr[m])
    PyMem_Free(arr)


cdef inline int* seq_1step(Py_ssize_t N):
    """
    """
    cdef int* all_ind = <int*> PyMem_Malloc(N*sizeof(int))
    if all_ind is NULL:     
        return NULL
    cdef int i
    for i in range(N):
        all_ind[i] = i
    return all_ind


cdef inline int** calloc_int_2dmat(Py_ssize_t M, Py_ssize_t N):
    """
    """
    cdef int** mat_ret = <int**> PyMem_Malloc(M*sizeof(int*))
    if mat_ret is NULL:     
        return NULL
    cdef Py_ssize_t m
    cdef size_t shdp = N*sizeof(int)
    for m in range(M):
        mat_ret[m] = <int*> PyMem_Malloc(shdp)
        if mat_ret is NULL:
            free_int_2dmat(mat_ret, m)
            return NULL
        # Setting the elements of mat_ret[m] to zero.
        memset(mat_ret[m], 0, shdp)
    return mat_ret


cdef inline int*** alloc_int_3dmat(Py_ssize_t M, Py_ssize_t N, Py_ssize_t K):
    """
    """
    cdef int*** mat_ret = <int***> PyMem_Malloc(M*sizeof(int**))
    if mat_ret is NULL:     
        return NULL
    cdef Py_ssize_t m, n
    for m in range(M):
        mat_ret[m] = <int**> PyMem_Malloc(N*sizeof(int*))
        if mat_ret[m] is NULL:     
            free_int_3dmat(mat_ret, m, N)
            return NULL
        for n in range(N):
            mat_ret[m][n] = <int*> PyMem_Malloc(K*sizeof(int))
            if mat_ret[m][n] is NULL:     
                free_int_2dmat(mat_ret[m], n)
                free_int_3dmat(mat_ret, m, N)
                return NULL
    return mat_ret


cdef inline double** alloc_dble_2dmat(Py_ssize_t M, Py_ssize_t N):
    """
    """
    cdef double** mat_ret = <double**> PyMem_Malloc(M*sizeof(double*))
    if mat_ret is NULL:     
        return NULL
    cdef Py_ssize_t m
    for m in range(M):
        mat_ret[m] = <double*> PyMem_Malloc(N*sizeof(double))
        if mat_ret[m] is NULL:     
            free_dble_2dmat(mat_ret, m)
            return NULL
    return mat_ret


cdef inline double** alloc_dble_2dmat_varKpysst(Py_ssize_t M, Py_ssize_t* N):
    """
    """
    cdef double** mat_ret = <double**> PyMem_Malloc(M*sizeof(double*))
    if mat_ret is NULL:     
        return NULL
    cdef Py_ssize_t m
    for m in range(M):
        mat_ret[m] = <double*> PyMem_Malloc(N[m]*sizeof(double))
        if mat_ret[m] is NULL:     
            free_dble_2dmat(mat_ret, m)
            return NULL
    return mat_ret


cdef inline double*** alloc_dble_3dmat(Py_ssize_t M, Py_ssize_t N, Py_ssize_t K):
    """
    """
    cdef double*** mat_ret = <double***> PyMem_Malloc(M*sizeof(double**))
    if mat_ret is NULL:     
        return NULL
    cdef Py_ssize_t m, n
    for m in range(M):
        mat_ret[m] = <double**> PyMem_Malloc(N*sizeof(double*))
        if mat_ret[m] is NULL:     
            free_dble_3dmat(mat_ret, m, N)
            return NULL
        for n in range(N):
            mat_ret[m][n] = <double*> PyMem_Malloc(K*sizeof(double))
            if mat_ret[m][n] is NULL:     
                free_dble_2dmat(mat_ret[m], n)
                free_dble_3dmat(mat_ret, m, N)
                return NULL
    return mat_ret


cdef inline double*** alloc_dble_3dmat_varK(Py_ssize_t M, Py_ssize_t N, int** K):
    """
    Same as alloc_dble_3dmat, but the size of the third dimension may change. 
    """
    cdef double*** mat_ret = <double***> PyMem_Malloc(M*sizeof(double**))
    if mat_ret is NULL:     
        return NULL
    cdef Py_ssize_t m, n
    for m in range(M):
        mat_ret[m] = <double**> PyMem_Malloc(N*sizeof(double*))
        if mat_ret[m] is NULL:     
            free_dble_3dmat(mat_ret, m, N)
            return NULL
        for n in range(N):
            mat_ret[m][n] = <double*> PyMem_Malloc(K[m][n]*sizeof(double))
            if mat_ret[m][n] is NULL:     
                free_dble_2dmat(mat_ret[m], n)
                free_dble_3dmat(mat_ret, m, N)
                return NULL
    return mat_ret


cdef inline Py_ssize_t** alloc_Pysst_2dmat_varN(Py_ssize_t M, Py_ssize_t* N):
    cdef Py_ssize_t** mat_ret = <Py_ssize_t**> PyMem_Malloc(M*sizeof(Py_ssize_t*))
    if mat_ret is NULL:     
        return NULL
    cdef Py_ssize_t m
    for m in range(M):
        mat_ret[m] = <Py_ssize_t*> PyMem_Malloc(N[m]*sizeof(Py_ssize_t))
        if mat_ret[m] is NULL:     
            free_Pysst_2dmat(mat_ret, m)
            return NULL
    return mat_ret


cdef inline Py_ssize_t*** alloc_Pysst_3dmat_varK(Py_ssize_t M, Py_ssize_t N, int** K):
    """
    Same as alloc_dble_3dmat, but the size of the third dimension may change. 
    """
    cdef Py_ssize_t*** mat_ret = <Py_ssize_t***> PyMem_Malloc(M*sizeof(Py_ssize_t**))
    if mat_ret is NULL:     
        return NULL
    cdef Py_ssize_t m, n
    for m in range(M):
        mat_ret[m] = <Py_ssize_t**> PyMem_Malloc(N*sizeof(Py_ssize_t*))
        if mat_ret[m] is NULL:     
            free_Pysst_3dmat(mat_ret, m, N)
            return NULL
        for n in range(N):
            mat_ret[m][n] = <Py_ssize_t*> PyMem_Malloc(K[m][n]*sizeof(Py_ssize_t))
            if mat_ret[m][n] is NULL:     
                free_Pysst_2dmat(mat_ret[m], n)
                free_Pysst_3dmat(mat_ret, m, N)
                return NULL
    return mat_ret


cdef inline Py_ssize_t*** alloc_Pysst_3dmat_varK_3dK(Py_ssize_t M, Py_ssize_t N, Py_ssize_t*** K, Py_ssize_t idk):
    """
    Same as alloc_dble_3dmat, but the size of the third dimension may change. 
    """
    cdef Py_ssize_t*** mat_ret = <Py_ssize_t***> PyMem_Malloc(M*sizeof(Py_ssize_t**))
    if mat_ret is NULL:     
        return NULL
    cdef Py_ssize_t m, n
    for m in range(M):
        mat_ret[m] = <Py_ssize_t**> PyMem_Malloc(N*sizeof(Py_ssize_t*))
        if mat_ret[m] is NULL:     
            free_Pysst_3dmat(mat_ret, m, N)
            return NULL
        for n in range(N):
            mat_ret[m][n] = <Py_ssize_t*> PyMem_Malloc(K[m][n][idk]*sizeof(Py_ssize_t))
            if mat_ret[m][n] is NULL:     
                free_Pysst_2dmat(mat_ret[m], n)
                free_Pysst_3dmat(mat_ret, m, N)
                return NULL
    return mat_ret


#######################################################
####################################################### L-BFGS optimization (C library)
#######################################################


cdef extern from "lbfgs.h":
    ctypedef double lbfgsfloatval_t
    ctypedef lbfgsfloatval_t* lbfgsconst_p "const lbfgsfloatval_t *"
    
    ctypedef lbfgsfloatval_t (*lbfgs_evaluate_t)(void *, lbfgsconst_p, lbfgsfloatval_t *, int, lbfgsfloatval_t)
    ctypedef int (*lbfgs_progress_t)(void *, lbfgsconst_p, lbfgsconst_p, lbfgsfloatval_t, lbfgsfloatval_t, lbfgsfloatval_t, lbfgsfloatval_t, int, int, int)
    
    cdef enum LineSearchAlgo:
        LBFGS_LINESEARCH_DEFAULT,
        LBFGS_LINESEARCH_MORETHUENTE,
        LBFGS_LINESEARCH_BACKTRACKING_ARMIJO,
        LBFGS_LINESEARCH_BACKTRACKING_WOLFE,
        LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE
    
    cdef enum ReturnCode:
        LBFGS_SUCCESS,
        LBFGS_ALREADY_MINIMIZED,
        LBFGSERR_UNKNOWNERROR,
        LBFGSERR_LOGICERROR,
        LBFGSERR_OUTOFMEMORY,
        LBFGSERR_CANCELED,
        LBFGSERR_INVALID_N,
        LBFGSERR_INVALID_N_SSE,
        LBFGSERR_INVALID_X_SSE,
        LBFGSERR_INVALID_EPSILON,
        LBFGSERR_INVALID_TESTPERIOD,
        LBFGSERR_INVALID_DELTA,
        LBFGSERR_INVALID_LINESEARCH,
        LBFGSERR_INVALID_MINSTEP,
        LBFGSERR_INVALID_MAXSTEP,
        LBFGSERR_INVALID_FTOL,
        LBFGSERR_INVALID_WOLFE,
        LBFGSERR_INVALID_GTOL,
        LBFGSERR_INVALID_XTOL,
        LBFGSERR_INVALID_MAXLINESEARCH,
        LBFGSERR_INVALID_ORTHANTWISE,
        LBFGSERR_INVALID_ORTHANTWISE_START,
        LBFGSERR_INVALID_ORTHANTWISE_END,
        LBFGSERR_OUTOFINTERVAL,
        LBFGSERR_INCORRECT_TMINMAX,
        LBFGSERR_ROUNDING_ERROR,
        LBFGSERR_MINIMUMSTEP,
        LBFGSERR_MAXIMUMSTEP,
        LBFGSERR_MAXIMUMLINESEARCH,
        LBFGSERR_MAXIMUMITERATION,
        LBFGSERR_WIDTHTOOSMALL,
        LBFGSERR_INVALIDPARAMETERS,
        LBFGSERR_INCREASEGRADIENT
    
    ctypedef struct lbfgs_parameter_t:
        int m
        lbfgsfloatval_t epsilon
        int past
        lbfgsfloatval_t delta
        int max_iterations
        int linesearch
        int max_linesearch
        lbfgsfloatval_t min_step
        lbfgsfloatval_t max_step
        lbfgsfloatval_t ftol
        lbfgsfloatval_t wolfe
        lbfgsfloatval_t gtol
        lbfgsfloatval_t xtol
        lbfgsfloatval_t orthantwise_c
        int orthantwise_start
        int orthantwise_end
    
    int lbfgs(int, lbfgsfloatval_t *, lbfgsfloatval_t *, lbfgs_evaluate_t, lbfgs_progress_t, void *, lbfgs_parameter_t *)
    void lbfgs_parameter_init(lbfgs_parameter_t *)
    lbfgsfloatval_t *lbfgs_malloc(int)
    void lbfgs_free(lbfgsfloatval_t *)


#######################################################
####################################################### Multi-scale SNE residues
#######################################################


cdef inline int ms_def_n_scales(double Nd, int K_star, int L_min, bint isLmin1) nogil:
    """
    """
    if isLmin1:
        return <int> round(log2(Nd/(<double> K_star)))
    else:
        return (<int> round(log2(Nd/(<double> K_star)))) + 1 - L_min


cdef inline int ms_def_shift_Lmin(bint isnotLmin1, Py_ssize_t L_min) nogil:
    """
    """
    cdef int shift_L_min = 1
    cdef Py_ssize_t h
    if isnotLmin1:
        for h in range(L_min-1):
            shift_L_min *= 2
    return shift_L_min


cdef inline int* ms_def_Kh(int K_star, bint isnotLmin1, int shift_L_min, Py_ssize_t L):
    """
    The returned value must be freed. 
    """
    cdef int* K_h = <int*> PyMem_Malloc(L*sizeof(int))
    if K_h is NULL:     
        return NULL
    K_h[0] = K_star
    if isnotLmin1:
        K_h[0] *= shift_L_min
    cdef Py_ssize_t h
    for h in range(1, L, 1):
        K_h[h] = K_h[h-1]*2
    return K_h 


cdef inline void sne_hdpinn_nolog(const double* ds_nn, double tau, Py_ssize_t nnn, double* pinn) nogil:
    """
    Computes SNE sim, without their log.
    ds_nn is assumed to contain the minimum squared distance - the squared distance. 
    tau is the denominator of the exponentials
    nnn is the number of neighbors
    pinn is the location at which the similarities will be stored. 
    """
    cdef double den = 0.0
    cdef Py_ssize_t i
    for i in range(nnn):
        pinn[i] = exp(ds_nn[i]/tau)
        den += pinn[i]
    for i in range(nnn):
        pinn[i] /= den


cdef inline double sne_densim(const double* ds_nn, double tau, Py_ssize_t nnn) nogil:
    """
    Computes the denominator of the similarities
    ds_nn is assumed to contain the minimum squared distance - the squared distance. 
    tau is the denominator of the exponentials
    nnn is the number of neighbors
    """
    cdef double den = 0.0
    cdef Py_ssize_t i
    for i in range(nnn):
        den += exp(ds_nn[i]/tau)
    return den


cdef inline double sne_binsearch_fct(const double* ds_nn, double tau, Py_ssize_t nnn, double log_perp) nogil:
    """
    Computes the entropry of the similarities minus the logarithm of the perplexity
    ds_nn is assumed to contain the minimum squared distance - the squared distance. 
    tau is the denominator of the exponentials
    nnn is the number of neighbors
    """
    cdef double a, b, v, den
    v = 0.0
    den = 0.0
    cdef Py_ssize_t i
    for i in range(nnn):
        a = ds_nn[i]/tau
        b = exp(a)
        v -= a*b
        den += b
    return v/den + log(den) - log_perp


cdef inline double sne_binsearch_bandwidth_fit(const double* ds_nn, Py_ssize_t nnn, double log_perp, double tau) nogil:
    """
    Tune the bandwidths of HD SNE similarities. 
    ds_nn is assumed to contain the minimum squared distance - the squared distance. 
    nnn is the number of neighbors
    Returns the bandwidths. 
    The 4th parameter, tau, is the starting point for the binary search. It can be set to 1.0 if no prior guess is known. 
    """
    cdef double f_tau = sne_binsearch_fct(ds_nn, tau, nnn, log_perp)
    if fabs(f_tau) <= EPSILON_DBL:
        return tau
    cdef double tau_up, tau_low
    if f_tau > 0:
        tau_low = tau*0.5
        if (tau_low < DBL_MIN) or (fabs(sne_densim(ds_nn, tau_low, nnn)) < DBL_MIN):
            # Binary search failed. The root is too close from 0 for the numerical precision: the denominator of the similarities is almost 0.
            return tau
        f_tau = sne_binsearch_fct(ds_nn, tau_low, nnn, log_perp)
        if fabs(f_tau) <= EPSILON_DBL:
            return tau_low
        tau_up = tau
        while f_tau > 0:
            tau_up = tau_low
            tau_low *= 0.5
            if (tau_low < DBL_MIN) or (fabs(sne_densim(ds_nn, tau_low, nnn)) < DBL_MIN):
                # Binary search failed. The root is too close from 0 for the numerical precision.
                return tau_up
            f_tau = sne_binsearch_fct(ds_nn, tau_low, nnn, log_perp)
            if fabs(f_tau) <= EPSILON_DBL:
                return tau_low
    else:
        tau_up = 2.0*tau
        if fabs(sne_densim(ds_nn, tau_up, nnn)-nnn) <= EPSILON_DBL:
            # Binary search failed. The root is too big for the numerical precision of the exponentials of the similarities. All the exponentials at the denominator = 1 and hence, the denominator = nnn. 
            return tau
        f_tau = sne_binsearch_fct(ds_nn, tau_up, nnn, log_perp)
        if fabs(f_tau) <= EPSILON_DBL:
            return tau_up
        tau_low = tau
        while f_tau < 0:
            tau_low = tau_up
            tau_up *= 2.0
            if fabs(sne_densim(ds_nn, tau_up, nnn)-nnn) <= EPSILON_DBL:
                # Binary search failed. The root is too big for the numerical precision of the exponentials of the similarities.
                return tau_low
            f_tau = sne_binsearch_fct(ds_nn, tau_up, nnn, log_perp)
            if fabs(f_tau) <= EPSILON_DBL:
                return tau_up
    cdef Py_ssize_t nit = 0
    cdef Py_ssize_t nit_max = 1000
    while nit < nit_max:
        tau = (tau_up+tau_low)*0.5
        f_tau = sne_binsearch_fct(ds_nn, tau, nnn, log_perp)
        if fabs(f_tau) <= EPSILON_DBL:
            return tau
        elif f_tau > 0:
            tau_up = tau
        else:
            tau_low = tau
        nit += 1
    # Binary search failed
    return tau 


cdef inline lbfgsfloatval_t* init_lbfgs_var(size_t shdp, int prod_N_nc, double* xlds):
    """
    """
    # Variables for the optimization. We must use lbfgs_malloc to benefitt from SSE2 optimization. 
    cdef lbfgsfloatval_t* xopt = lbfgs_malloc(prod_N_nc)
    if xopt is NULL:
        return NULL
    # Initializing the the variables to the current LDS. We can use memcpy as lbfgsfloatval_t is, in our case, strictly equivalent to a double. 
    memcpy(xopt, xlds, shdp)
    # Returning
    return xopt 


cdef inline int* f_def_n_ds_h(bint isLmin1, int N, int shift_L_min, double Nd, Py_ssize_t L):
    """
    """
    cdef int* n_ds_h = <int*> PyMem_Malloc(L*sizeof(int))
    if n_ds_h is NULL:     
        return NULL
    # Multiplication factor to determine the elements of n_ds_h
    cdef double mf
    if isLmin1:
        mf = 1.0
        n_ds_h[0] = N
    else:
        mf = 1.0/(<double> shift_L_min)
        n_ds_h[0] = <int> round(Nd*mf)
    # Filling n_ds_h
    cdef Py_ssize_t h
    for h in range(1, L, 1):
        mf *= 0.5
        n_ds_h[h] = <int> round(Nd*mf)
    return n_ds_h


cdef inline int* f_def_nnn_h(Py_ssize_t L, int* K_h, int* n_ds_h, bint cperp):
    """
    """
    cdef int* nnn_h = <int*> PyMem_Malloc(L*sizeof(int))
    if nnn_h is NULL:     
        return NULL
    cdef Py_ssize_t h
    # Filling nnn_h
    if cperp:
        nnn_h[0] = 3*K_h[0]
        if nnn_h[0] > n_ds_h[0]:
            nnn_h[0] = n_ds_h[0]
        for h in range(1, L, 1):
            if nnn_h[0] > n_ds_h[h]:
                nnn_h[h] = n_ds_h[h]
            else:
                nnn_h[h] = nnn_h[0]
    else:
        for h in range(L):
            nnn_h[h] = 3*K_h[h]
            if nnn_h[h] > n_ds_h[h]:
                nnn_h[h] = n_ds_h[h]
    return nnn_h


cdef inline int f_nnn_tot(int* nnn_h, Py_ssize_t L) nogil:
    """
    """
    # Sum of the elements of nnn_h
    cdef int nnn_tot = 0
    cdef Py_ssize_t h
    for h in range(L):
        nnn_tot += nnn_h[h]
    return nnn_tot


cdef inline bint f_nn_ds_hdprec(int d_hds, int* K_h, int N, Py_ssize_t L, int* n_ds_h, int* all_ind, int* nnn_h, bint isLmin1, double* X_hds, Py_ssize_t n_rs, int*** arr_nn_i_rs, int** nnn_i_rs, double*** ds_nn_i_rs, double*** tau_h_i_rs, int nnn_tot, bint sym_nn_set):
    """
    Return False if everything is ok, True if memory problem. 
    """
    # Defining some variables
    cdef Py_ssize_t rs, i, j, nsr, isa, k, last, h, nrs_loop
    cdef VpTree* vpt
    cdef bint build_vp, in_cur_ds
    cdef double* Xhd_cur
    cdef double* x
    cdef int* i_sds
    cdef int nnn, nnn_ub, nnn_cpy
    # Number of bytes of an HD data point
    cdef size_t shdp = d_hds*sizeof(double)
    # Logarithm of the considered perplexity and temporary variable
    cdef double log_perp, min_ds
    cdef bint clogp
    if K_h[0] == 2:
        log_perp = log(2.0)
        clogp = False
    else:
        clogp = True
    # For each scale
    for h in range(L):
        nnn_ub = nnn_h[h]+1
        if nnn_ub > n_ds_h[h]:
            nnn_ub = n_ds_h[h]
        if (h == 0) and isLmin1:
            # Vantage-point tree for the complete data set. No need to use the cython vantage-point tree class: we can directly call the C code! But a del statement (below) is necessary to avoid a memory leak. 
            vpt = new VpTree(X_hds, N, d_hds)
            # The vantage-point tree must not be build anymore
            build_vp = False
            # Indicates that the data point for which the neighbors are searched is in the currently considered subsampled data set
            in_cur_ds = True
            # Number of random samplings over which we need to iterate. Only 1 since, in this case, all the random samplings leads to the same results for the first scale. 
            nrs_loop = 1
        else:
            # The vantage-point tree must be created
            build_vp = True
            # Number of random samplings over which we need to iterate. 
            nrs_loop = n_rs
            # Allocating memory for the subsampled data sets at scale h
            Xhd_cur = <double*> PyMem_Malloc(n_ds_h[h]*d_hds*sizeof(double))
            if Xhd_cur is NULL:     
                return True
            # Allocating memory to store the indexes of the data points in the subsampled data set
            i_sds = <int*> PyMem_Malloc(n_ds_h[h]*sizeof(int))
            if i_sds is NULL:     
                PyMem_Free(Xhd_cur)
                return True
        # For each random sampling
        for rs in range(nrs_loop):
            # Subsampling the data set and building the vantage-point tree
            if build_vp:
                # Subsampling the data set without replacement
                nsr = N
                j = 0
                for i in range(n_ds_h[h]):
                    isa = rand()%nsr
                    # Storing the sampled index
                    i_sds[j] = all_ind[isa]
                    # Making sure that the further samplings will be made without replacement
                    nsr -= 1
                    if isa != nsr:
                        all_ind[isa] = all_ind[nsr]
                        all_ind[nsr] = i_sds[j]
                    j += 1
                # Sorting i_sds, to be able to check whether the considered data points lie in the subsampled data set
                sort(i_sds, i_sds + n_ds_h[h])
                # Constructing Xhd_cur
                nsr = 0
                for i in range(n_ds_h[h]):
                    isa = i_sds[i]*d_hds
                    memcpy(&Xhd_cur[nsr], &X_hds[isa], shdp)
                    nsr += d_hds
                # Building the vantage-point tree for the subsampled data set. No need to call the cython vantage-point tree class: we can directly call the C code! But a del statement is necessary to avoid a memory leak. 
                vpt = new VpTree(Xhd_cur, n_ds_h[h], d_hds)
                # Setting nsr back to 0 as it will be used to check whether the considered data point lie in the subsampled data set
                nsr = 0
            # Searching the nearest neighbors of all data points in the subsampled data set
            for i in range(N):
                # Checking whether the considered data point is in the currently considered subsampled data set
                if build_vp:
                    if (nsr < n_ds_h[h]) and (i == i_sds[nsr]):
                        nsr += 1
                        nnn = nnn_ub
                        in_cur_ds = True
                    else:
                        nnn = nnn_h[h]
                        in_cur_ds = False
                else:
                    # Number of neighbors to search in the vantage-point tree. Need to define it here because nnn is modified in the loop. 
                    nnn = nnn_ub
                isa = nnn_i_rs[rs][i]
                # Searching the nnn nearest neighbors of i in vpt
                x = &X_hds[i*d_hds]
                vpt.search(x, nnn, &arr_nn_i_rs[rs][i][isa])
                # Converting the indexes in the range of the full data set instead of the subsampled one
                if build_vp:
                    for j in range(nnn):
                        arr_nn_i_rs[rs][i][isa] = i_sds[arr_nn_i_rs[rs][i][isa]]
                        isa += 1
                # Removing the considered data point from its nearest neighbor if it belongs to the subsampled data set
                if in_cur_ds:
                    isa = nnn_i_rs[rs][i]
                    nnn -= 1
                    for j in range(nnn):
                        if arr_nn_i_rs[rs][i][isa] == i:
                            arr_nn_i_rs[rs][i][isa] = arr_nn_i_rs[rs][i][nnn_i_rs[rs][i]+nnn]
                            break
                        isa += 1
                
                # Computing the squared euclidean distance between the considered data point and its nearest neighbors
                isa = nnn_i_rs[rs][i]
                min_ds = DBL_MAX
                for j in range(nnn):
                    ds_nn_i_rs[rs][i][isa] = sqeucl_dist_ptr(x, &X_hds[arr_nn_i_rs[rs][i][isa]*d_hds], d_hds)
                    if min_ds > ds_nn_i_rs[rs][i][isa]:
                        min_ds = ds_nn_i_rs[rs][i][isa]
                    isa += 1
                
                # Substracting the minimum squared distance and changing the sign, to avoid to do it during the computation of the bandwidths
                isa = nnn_i_rs[rs][i]
                for j in range(nnn):
                    ds_nn_i_rs[rs][i][isa] = min_ds - ds_nn_i_rs[rs][i][isa]
                    isa += 1
                
                # Logarithm of the current perplexity
                if clogp:
                    log_perp = log(<double> min(K_h[0], nnn - 1))
                
                # Computing the HD bandwith of the similarities at scale h, in random sampling rs and with respect to data point i
                tau_h_i_rs[h][rs][i] = sne_binsearch_bandwidth_fit(&ds_nn_i_rs[rs][i][nnn_i_rs[rs][i]], nnn, log_perp, 1.0)
                
                # Only adding the new neighbors to arr_nn_i_rs
                k = nnn_i_rs[rs][i]
                nnn_cpy = nnn
                for j in range(nnn_cpy):
                    for isa in range(nnn_i_rs[rs][i]):
                        if arr_nn_i_rs[rs][i][isa] == arr_nn_i_rs[rs][i][k]:
                            nnn -= 1
                            last = nnn_i_rs[rs][i]+nnn
                            if last > k:
                                arr_nn_i_rs[rs][i][k] = arr_nn_i_rs[rs][i][last]
                                ds_nn_i_rs[rs][i][k] = ds_nn_i_rs[rs][i][last]
                            break
                    else:
                        # If no break in inner loop, set the squared distance back to its value and increment k
                        ds_nn_i_rs[rs][i][k] = min_ds - ds_nn_i_rs[rs][i][k]
                        k += 1
                # Updating the number of considered neighbors
                nnn_i_rs[rs][i] += nnn
                # Updating for the other random samplings if they all use the same vantage-point tree at the first scale
                if not build_vp:
                    for isa in range(1, n_rs, 1):
                        tau_h_i_rs[h][isa][i] = tau_h_i_rs[h][rs][i]
                        nnn_i_rs[isa][i] = nnn
                        for j in range(nnn):
                            arr_nn_i_rs[isa][i][j] = arr_nn_i_rs[rs][i][j]
                            ds_nn_i_rs[isa][i][j] = ds_nn_i_rs[rs][i][j]
            if build_vp:
                # Call the destructor of the tree and free the ressources allocated for the object
                del vpt
        if build_vp:
            # Free the memory for the subsampled data set at the current scale
            PyMem_Free(Xhd_cur)
            PyMem_Free(i_sds)
        else:
            # Call the destructor of the tree and free the ressources allocated for the object
            del vpt
    cdef int* n_nnn
    if sym_nn_set:
        # Intermediate variable to store the new number of nearest neighbors for each data point
        shdp = N*sizeof(int)
        n_nnn = <int*> PyMem_Malloc(shdp)
        if n_nnn is NULL:
            return True
        # Symmetrizing the nearest neighbors sets
        for rs in range(n_rs):
            memcpy(n_nnn, nnn_i_rs[rs], shdp)
            # Computing the new number of neighbors to consider for each data point
            for i in range(N):
                for isa in range(nnn_i_rs[rs][i]):
                    j = arr_nn_i_rs[rs][i][isa]
                    for nsr in range(nnn_i_rs[rs][j]):
                        if arr_nn_i_rs[rs][j][nsr] == i:
                            break
                    else:
                        # i is not in the neighbors of j: we must add it
                        n_nnn[j] += 1
            # Updating the memory allocated to arr_nn_i_rs and ds_nn_i_rs according to the new number of neighbors
            for i in range(N):
                if n_nnn[i] < nnn_tot:
                    # Reallocating arr_nn_i_rs to the considered number of neighbors
                    i_sds = <int*> PyMem_Realloc(<void*> arr_nn_i_rs[rs][i], n_nnn[i]*sizeof(int))
                    if i_sds is NULL:
                        PyMem_Free(n_nnn)
                        return True
                    arr_nn_i_rs[rs][i] = i_sds
                    # Reallocating ds_nn_i_rs to the considered number of neighbors
                    Xhd_cur = <double*> PyMem_Realloc(<void*> ds_nn_i_rs[rs][i], n_nnn[i]*sizeof(double))
                    if Xhd_cur is NULL:
                        PyMem_Free(n_nnn)
                        return True
                    ds_nn_i_rs[rs][i] = Xhd_cur
                elif n_nnn[i] > nnn_tot:
                    # Allocating some space to store the new number of neighbors
                    i_sds = <int*> PyMem_Malloc(n_nnn[i]*sizeof(int))
                    if i_sds is NULL:
                        PyMem_Free(n_nnn)
                        return True
                    memcpy(i_sds, arr_nn_i_rs[rs][i], nnn_i_rs[rs][i]*sizeof(int))
                    PyMem_Free(arr_nn_i_rs[rs][i])
                    arr_nn_i_rs[rs][i] = i_sds
                    # Allocating some space to store the distances to the new number of neighbors
                    Xhd_cur = <double*> PyMem_Malloc(n_nnn[i]*sizeof(double))
                    if Xhd_cur is NULL:
                        PyMem_Free(n_nnn)
                        return True
                    memcpy(Xhd_cur, ds_nn_i_rs[rs][i], nnn_i_rs[rs][i]*sizeof(double))
                    PyMem_Free(ds_nn_i_rs[rs][i])
                    ds_nn_i_rs[rs][i] = Xhd_cur
            # Adding the new considered neighbors
            memcpy(n_nnn, nnn_i_rs[rs], shdp)
            for i in range(N):
                for isa in range(nnn_i_rs[rs][i]):
                    j = arr_nn_i_rs[rs][i][isa]
                    for nsr in range(nnn_i_rs[rs][j]):
                        if arr_nn_i_rs[rs][j][nsr] == i:
                            break
                    else:
                        # i is not in the neighbors of j: we must add it
                        arr_nn_i_rs[rs][j][n_nnn[j]] = <int> i
                        ds_nn_i_rs[rs][j][n_nnn[j]] = ds_nn_i_rs[rs][i][isa]
                        n_nnn[j] += 1
            # Substracting the minimum from the squared distances and changing the sign, to avoid doing it when computing the similarities
            for i in range(N):
                min_ds = min_arr_ptr(ds_nn_i_rs[rs][i], n_nnn[i])
                for j in range(n_nnn[i]):
                    ds_nn_i_rs[rs][i][j] = min_ds - ds_nn_i_rs[rs][i][j]
            # Updating nnn_i_rs with the new number of neighbors
            memcpy(nnn_i_rs[rs], n_nnn, shdp)
        PyMem_Free(n_nnn)
    else:
        # Reallocating arr_nn_i_rs and ds_nn_i_rs to reserve the exact amount of memory which is needed, and removing the minimum squared distances and changing their signs, to avoid doing it when computing the HD similarities. 
        for rs in range(n_rs):
            for i in range(N):
                if nnn_i_rs[rs][i] < nnn_tot:
                    # Reallocating arr_nn_i_rs to the considered number of neighbors
                    i_sds = <int*> PyMem_Realloc(<void*> arr_nn_i_rs[rs][i], nnn_i_rs[rs][i]*sizeof(int))
                    if i_sds is NULL:
                        return True
                    arr_nn_i_rs[rs][i] = i_sds
                    # Reallocating ds_nn_i_rs to the considered number of neighbors
                    Xhd_cur = <double*> PyMem_Realloc(<void*> ds_nn_i_rs[rs][i], nnn_i_rs[rs][i]*sizeof(double))
                    if Xhd_cur is NULL:
                        return True
                    ds_nn_i_rs[rs][i] = Xhd_cur
                # Substracting the minimum from the squared distances and changing the sign, to avoid doing it when computing the similarities
                min_ds = min_arr_ptr(ds_nn_i_rs[rs][i], nnn_i_rs[rs][i])
                for j in range(nnn_i_rs[rs][i]):
                    ds_nn_i_rs[rs][i][j] = min_ds - ds_nn_i_rs[rs][i][j]
    # Everything ok -> return False
    return False


cdef inline double f_update_mso_step(Py_ssize_t k, Py_ssize_t h, Py_ssize_t n_rs, Py_ssize_t N, int** nnn_i_rs, double*** ds_nn_i_rs, double*** tau_h_i_rs, double*** simhd_ms_nn_i_rs, double*** simhd_h_nn_i_rs) nogil:
    """
    k refers to the number of currently considered scales, between 1 and the number of scales. 
    h is the index of the current scale. 
    """
    cdef Py_ssize_t rs, i, j
    cdef double kd, ikd
    # Computing the multi-scale similarities for the current multi-scale optimization step
    if k == 1:
        # Computing the similarities at the last scale and storing them in simhd_ms_nn_i_rs
        for rs in range(n_rs):
            for i in range(N):
                sne_hdpinn_nolog(ds_nn_i_rs[rs][i], tau_h_i_rs[h][rs][i], nnn_i_rs[rs][i], simhd_ms_nn_i_rs[rs][i])
        return 1.0
    else:
        # Storing the current value of k, in double
        kd = <double> k
        # Inverse of k
        ikd = 1.0/kd
        # Value of kd at the previous step
        kd -= 1.0
        # Computing the similarities at the current scale and updating simhd_ms_nn_i_rs
        for rs in range(n_rs):
            for i in range(N):
                sne_hdpinn_nolog(ds_nn_i_rs[rs][i], tau_h_i_rs[h][rs][i], nnn_i_rs[rs][i], simhd_h_nn_i_rs[rs][i])
                for j in range(nnn_i_rs[rs][i]):
                    simhd_ms_nn_i_rs[rs][i][j] = (kd*simhd_ms_nn_i_rs[rs][i][j] + simhd_h_nn_i_rs[rs][i][j])*ikd
        return ikd


#######################################################
####################################################### Vantage-point trees
#######################################################


cdef extern from "vptree.h":
    cdef cppclass VpTree:
        VpTree(const double* X, int N, int D)
        void search(const double* x, int k, int* idx)


#######################################################
####################################################### Fast multi-scale t-SNE 
#######################################################


cdef inline Py_ssize_t*** fms_sym_nn_match(Py_ssize_t n_rs, Py_ssize_t N_1, int*** arr_nn_i_rs, int** nnn_i_rs, Py_ssize_t n_components):
    """
    This assumes that the nearest neighbor sets are symmetric: if i is in the neighbors of j (ie in arr_nn_i_rs[rs][j]), then j must be in the neighbors of i (ie in arr_nn_i_rs[rs][i]).
    Return NULL if problem.  
    """
    cdef Py_ssize_t rs, i, j, idj, k, nnn
    # Temporarily modifying nnn_i_rs
    for rs in range(n_rs):
        for i in range(N_1):
            nnn_i_rs[rs][i] = 4*nnn_i_rs[rs][i] + 2
    # Allocate memory
    cdef Py_ssize_t*** m_nn = alloc_Pysst_3dmat_varK(n_rs, N_1, nnn_i_rs)
    if m_nn is NULL:
        return NULL
    # Setting nnn_i_rs back to its value
    for rs in range(n_rs):
        for i in range(N_1):
            nnn_i_rs[rs][i] = (nnn_i_rs[rs][i] - 2)/4
    # Filling m_nn
    cdef Py_ssize_t* tmp
    for rs in range(n_rs):
        for i in range(N_1):
            nnn = 2
            for idj in range(nnn_i_rs[rs][i]):
                j = arr_nn_i_rs[rs][i][idj]  #current considered neighbour of i
                if j > i:
                    for k in range(nnn_i_rs[rs][j]):  #number of considered neighbours of j, i.e. the current considered neighbour of i
                        if arr_nn_i_rs[rs][j][k] == i:
                            m_nn[rs][i][nnn] = idj #idj's considered neigh for i -> to find index in arr_nn_i_rs (position de j dans liste arr_nn_i_rs i)
                            nnn += 1
                            m_nn[rs][i][nnn] = j*n_components #indice du voisin consideré pour i * n_components
                            nnn += 1
                            m_nn[rs][i][nnn] = k #k_ième voisin considéré pour j (position de i dans liste arr_nn_i_rs de j)
                            nnn += 1
                            m_nn[rs][i][nnn] = j #voisin considéré pour l'instant pour i cad j
                            nnn += 1
                            break
                    else:  #will be activated if the if condition in for loop is never activated i.e. i is not considered by j as neighbour
                        free_Pysst_3dmat(m_nn, n_rs, N_1)
                        return NULL
            m_nn[rs][i][0] = nnn #nombre de cellules utilisées dans cette ligne
            # m_nn[rs][i][1] = number of j > i in the neighbors of i in random sampling rs
            m_nn[rs][i][1] = (nnn-2)/4
            if nnn < nnn_i_rs[rs][i]:
                tmp = <Py_ssize_t*> PyMem_Realloc(<void*> m_nn[rs][i], nnn*sizeof(Py_ssize_t))
                if tmp is NULL:
                    free_Pysst_3dmat(m_nn, n_rs, N_1)
                    return NULL
                m_nn[rs][i] = tmp #on enlève de l'espace de mémoire alloué en trop
    return m_nn


cdef inline Py_ssize_t** gather_nn_all_rs(Py_ssize_t* nnn_all_rs, Py_ssize_t n_rs, Py_ssize_t N_1, Py_ssize_t*** m_nn):
    """
    """
    # Will store the indexes of the neighbors j of i over all random samplings, such that j>i 
    cdef Py_ssize_t** inn_all_rs = <Py_ssize_t**> PyMem_Malloc(N_1*sizeof(Py_ssize_t*))
    if inn_all_rs is NULL:
        return NULL
    cdef Py_ssize_t* tmp
    cdef Py_ssize_t i, rs, j, k, n, nel
    for i in range(N_1):
        # Counting the total number of neighbors across all random samplings (possibly with duplicates)
        nel = 0
        for rs in range(n_rs):
            nel += m_nn[rs][i][1]
        # Allocate memory
        inn_all_rs[i] = <Py_ssize_t*> PyMem_Malloc(nel*sizeof(Py_ssize_t))
        if inn_all_rs[i] is NULL:
            free_Pysst_2dmat(inn_all_rs, i)
            return NULL
        # Add the neighbors of the first random sampling
        nnn_all_rs[i] = m_nn[0][i][1]
        k = 0
        for j in range(3, m_nn[0][i][0], 4):
            inn_all_rs[i][k] = m_nn[0][i][j]
            k += 1
        # Adding the neighbors of the other random samplings
        for rs in range(1, n_rs, 1):
            n = nnn_all_rs[i]
            for j in range(3, m_nn[rs][i][0], 4):
                for k in range(nnn_all_rs[i]):
                    if inn_all_rs[i][k] == m_nn[rs][i][j]:
                        break
                else:
                    inn_all_rs[i][n] = m_nn[rs][i][j]
                    n += 1
            nnn_all_rs[i] = n
        # Reallocating inn_all_rs[i]
        if nnn_all_rs[i] < nel:
            tmp = <Py_ssize_t*> PyMem_Realloc(<void*> inn_all_rs[i], nnn_all_rs[i]*sizeof(Py_ssize_t))
            if tmp is NULL:
                free_Pysst_2dmat(inn_all_rs, i)
                return NULL
            inn_all_rs[i] = tmp
    return inn_all_rs


cdef inline Py_ssize_t*** fms_nn_rs_match_all_rs(Py_ssize_t n_rs, Py_ssize_t N_1, Py_ssize_t** inn_all_rs, Py_ssize_t* nnn_all_rs, Py_ssize_t*** m_nn):
    """
    """
    cdef Py_ssize_t*** idnn_in_ars = alloc_Pysst_3dmat_varK_3dK(n_rs, N_1, m_nn, 1)
    if idnn_in_ars is NULL:
        return NULL
    cdef Py_ssize_t rs, i, j, k, n
    for rs in range(n_rs):
        for i in range(N_1):
            k = 0
            for j in range(3, m_nn[rs][i][0], 4):
                for n in range(nnn_all_rs[i]):
                    if inn_all_rs[i][n] == m_nn[rs][i][j]:
                        idnn_in_ars[rs][i][k] = n
                        k += 1
                        break
                else:
                    free_Pysst_3dmat(idnn_in_ars, n_rs, N_1)
                    return NULL
    return idnn_in_ars


cdef inline void fmstsne_symmetrize(Py_ssize_t n_rs, Py_ssize_t N_1, double*** sim_hd, Py_ssize_t*** m_nn, double*** sim_hd_sym) nogil:
    """
    This assumes that the nearest neighbor sets are symmetric: if i is in the neighbors of j (ie in arr_nn_i_rs[rs][j]), then j must be in the neighbors of i (ie in arr_nn_i_rs[rs][i]).
    Be careful that only the similarities between i and j such that j>i are actually symmetrized, since only these are used in the evaluate function. 
    """
    cdef double tot
    cdef Py_ssize_t rs, i, idj, inn
    for rs in range(n_rs):
        tot = 0.0
        for i in range(N_1):
            for inn in range(2, m_nn[rs][i][0], 4):
                idj = m_nn[rs][i][inn]
                sim_hd_sym[rs][i][idj] = sim_hd[rs][i][idj] + sim_hd[rs][m_nn[rs][i][inn+3]][m_nn[rs][i][inn+2]]
                tot += sim_hd_sym[rs][i][idj]
        tot = 1.0/(2.0*tot) #somme de toutes les similaritées, on multiplie par 2 comme juste la moitié est considerée et tant donné que c'est symmétrique
        for i in range(N_1):
            for inn in range(2, m_nn[rs][i][0], 4):
                sim_hd_sym[rs][i][m_nn[rs][i][inn]] *= tot


cdef struct Opfmstsne:
    Py_ssize_t N                # Number of data points
    Py_ssize_t N_1              # N-1
    Py_ssize_t n_components     # Dimension of the LDS
    size_t sstx                 # Size, in bytes, of the vector of variables and hence, of the gradient
    Py_ssize_t n_rs             # Number of random samplings
    bint n_rs_geq1              # n_rs > 1
    double n_rs_f               # Number of random samplings in double
    double inv_n_rs_2f          # 2.0/n_rs_f
    double inv_n_rs_4f          # 4.0/n_rs_f
    double theta_s              # The square of the threshold parameter for the Barnes-Hut algorithm
    double*** sim_hd_ms         # Multi-scale HD similarities
    Py_ssize_t*** m_nn          # As returned by fms_sym_nn_match
    Py_ssize_t* nnn_all_rs      # nnn_all_rs[i] contains the total number of neighbors considered for i over all random samplings.
    Py_ssize_t** inn_all_rs     # inn_all_rs[i] contains the neighbors considered for i over all random samplings.
    double* dsld_all_rs         # Allows to store the squared LD distances between a data point and its considered neighbors across all random samplings.
    Py_ssize_t*** idnn_in_ars   # idnn_in_ars[rs][i][j] contains the index of the distance between i and its jth neighbor in random sampling rs in dsld_all_rs.
    int inter_fct               # Interaction function to employ in the space-partitioning tree
    double* qdiff               # Array with n_components elements to store intermediate computations when traversing the tree
    int M                       #resolution of the grid
    char* grid_meth             #discretization method used


cdef struct computed_Z:
    double PM_Z


cdef double[:,:] _kernel(double[:] x, double[:] y):
    """
    Kernel function used in order to compute the repulsive force
    
    ----------
    Parameters
    ----------
    * x : Array of double, of shape [M]
        Particular x-coordinates.
        
    * y : Array of double, of shape [M]
        Particular y-coordinates.
        
    -------
    Returns
    -------
    * response : Array of double, of shape [M,M]
        Response of the filter on the grid defined by x and y.
    """
    
    cdef:
        int i, j
        int M = x.shape[0]
        double[:,:] response = np.empty((M,M), dtype=DTYPE)

    for i in range(M):
        for j in range(M):
            response[j,i] = pow(1.0 + pow(x[j], 2.0) + pow(y[i], 2.0), -1.0)
        
    return response


cdef double[:,:] _compute_Repu_NGP(const double* Y_1D, int M, int n):
    """
    This function computes the repulsive term with respect to the NGP method
    
    ----------
    Parameters
    ----------
    * Y : Array of double, of shape [n_instances, 2]
        Embedding.

    * M : integer
        Resolution of the grid in both x and y direction.        
        
    * n : integer
        Number of instances.
        
    -------
    Returns
    -------
    * dY : Array of double, of shape [n_instances, 2]
        Repulsive force on each instance.
    """

    cdef:
        int i, j
        double boundary = 0.0
        double dx
        double Z
        int[:,:] hist = np.zeros((M, M), dtype=DTYPE_int)
        double[:,:] values
        np.ndarray[np.int_t, ndim=1] idx, idy
        np.ndarray[np.double_t, ndim=1] coord
        np.ndarray[np.double_t, ndim=2] pot
        np.ndarray[np.double_t, ndim=2] dY = np.empty((n, 2), dtype=DTYPE)
        np.ndarray[np.double_t, ndim=3] grad
        double[:,:] Y = np.empty((n, 2), dtype=DTYPE)

    for i in range(n):
        Y[i,0] = Y_1D[2*i]
        Y[i,1] = Y_1D[2*i+1]
        
    for i in range(n):
        for j in range(2):
            if abs(Y[i,j]) > boundary:
                boundary = abs(Y[i,j])
                
    dx = (2.0 * boundary) / (M - 1)
    boundary = boundary + dx
    
    coord = np.linspace(-1.0 * boundary, 1.0 * boundary, M, endpoint=True, dtype='float')

    idx = np.searchsorted(coord[:], Y[:,0], side='left').astype(int) - 1
    idy = np.searchsorted(coord[:], Y[:,1], side='left').astype(int) - 1
    dx = coord[1] - coord[0]
    
    for i in range(n):
        if abs(coord[idx[i]] - Y[i,0]) > (dx / 2.0):
                idx[i] += 1
        if abs(coord[idy[i]] - Y[i,1]) > (dx / 2.0):
                idy[i] += 1
    
    for i in range(n):
        hist[idx[i],idy[i]] += 1
            
    values = _kernel(coord, coord)
    pot = scipy.signal.fftconvolve(hist, values, mode='same')
    grad = np.gradient(pot) / np.float64(dx)
    
    Z = np.sum(pot[idx[:], idy[:]]) / 2.0
    
    dY[:,0] = grad[0,idx[:],idy[:]]
    dY[:,1] = grad[1,idx[:],idy[:]]
    
    return dY / Z


cdef np.ndarray[np.double_t, ndim=2] _compute_Repu_CIC(const double* Y_1D, int M, int n, computed_Z* Z_fin):
    """
    This function computes the repulsive term with respect to the CIC method
    
    ----------
    Parameters
    ----------
    * Y : Array of double, of shape [n_instances, 2]
        Embedding.

    * M : integer
        Resolution of the grid in both x and y direction.        
        
    * n : integer
        Number of instances.
        
    -------
    Returns
    -------
    * dY : Array of double, of shape [n_instances, 2]
        Repulsive force on each instance.
    """

    cdef:
        int i,j
        double boundary = 0.0
        double dx
        double Z = 0.0
        double[:,:] hist = np.zeros((M, M), dtype=DTYPE)
        double[:,:] values
        np.ndarray[np.int_t, ndim=1] idx, idy
        np.ndarray[np.double_t, ndim=1] coord
        np.ndarray[np.double_t, ndim=2] pot
        np.ndarray[np.double_t, ndim=2] dY = np.empty((n, 2), dtype=DTYPE)
        np.ndarray[np.double_t, ndim=3] grad

        double[:,:] Y = np.empty((n, 2), dtype=DTYPE)

    for i in range(n):
        Y[i,0] = Y_1D[2*i]
        Y[i,1] = Y_1D[2*i+1]

    for i in range(n):
        for j in range(2):
            if abs(Y[i,j]) > boundary:
                boundary = abs(Y[i,j])
                
    dx = (2.0 * boundary) / (M - 1)
    boundary = boundary + dx
    
    coord = np.linspace(-1.0 * boundary, 1.0 * boundary, M, endpoint=True)

    idx = np.searchsorted(coord[:], Y[:,0], side='left').astype(int) - 1
    idy = np.searchsorted(coord[:], Y[:,1], side='left').astype(int) - 1
    dx = coord[1] - coord[0]
    
    for i in range(n):
        if abs(coord[idx[i]] - Y[i,0]) > (dx / 2.0):
                idx[i] += 1
        if abs(coord[idy[i]] - Y[i,1]) > (dx / 2.0):
                idy[i] += 1
    
    for i in range(n):
        hist[idx[i],idy[i]] += pow(dx, -2.0) * abs((coord[idx[i]+1] - Y[i,0]) * (coord[idy[i]+1] - Y[i,1]))
        hist[idx[i]+1,idy[i]] += pow(dx, -2.0) * abs((coord[idx[i]] - Y[i,0]) * (coord[idy[i]+1] - Y[i,1]))
        hist[idx[i],idy[i]+1] += pow(dx, -2.0) * abs((coord[idx[i]+1] - Y[i,0]) * (coord[idy[i]] - Y[i,1]))
        hist[idx[i]+1,idy[i]+1] += pow(dx, -2.0) * abs((coord[idx[i]] - Y[i,0]) * (coord[idy[i]] - Y[i,1]))
            
    values = _kernel(coord, coord)
    pot = scipy.signal.fftconvolve(hist, values, mode='same')
    grad = np.gradient(pot) / np.float64(dx)
    
    for i in range(n):
        dY[i,0] = grad[0,idx[i],idy[i]] + ((grad[0,idx[i]+1,idy[i]] - grad[0,idx[i],idy[i]]) / dx) * (Y[i,0] - coord[idx[i]]) + \
                    ((grad[0,idx[i],idy[i]+1] - grad[0,idx[i],idy[i]]) / dx) * (Y[i,1] - coord[idy[i]]) + \
                    ((grad[0,idx[i]+1,idy[i]+1] - grad[0,idx[i],idy[i]]) / (dx**2)) * (Y[i,0] - coord[idx[i]]) * (Y[i,1] - coord[idy[i]])
        dY[i,1] = grad[1,idx[i],idy[i]] + ((grad[1,idx[i]+1,idy[i]] - grad[1,idx[i],idy[i]]) / dx) * (Y[i,0] - coord[idx[i]]) + \
                    ((grad[1,idx[i],idy[i]+1] - grad[1,idx[i],idy[i]]) / dx) * (Y[i,1] - coord[idy[i]]) + \
                    ((grad[1,idx[i]+1,idy[i]+1] - grad[1,idx[i],idy[i]]) / (dx**2)) * (Y[i,0] - coord[idx[i]]) * (Y[i,1] - coord[idy[i]])
        Z = Z + (pot[idx[i],idy[i]] + ((pot[idx[i]+1,idy[i]] - pot[idx[i],idy[i]]) / dx) * (Y[i,0] - coord[idx[i]]) + \
                    ((pot[idx[i],idy[i]+1] - pot[idx[i],idy[i]]) / dx) * (Y[i,1] - coord[idy[i]]) + \
                    ((pot[idx[i]+1,idy[i]+1] - pot[idx[i],idy[i]]) / (dx**2)) * (Y[i,0] - coord[idx[i]]) * (Y[i,1] - coord[idy[i]]))

    Z_fin[0].PM_Z = Z

    return dY / (Z / 2.0)


cdef inline lbfgsfloatval_t fmstsne_evaluate(void* instance, const lbfgsfloatval_t* x, lbfgsfloatval_t* g, const int n, const lbfgsfloatval_t step):
    """
    Computes cost function and gradient for the current LD coordinates.
    See documentation on the web. 
    n stores the number of variables
    We exploit the fact that the nearest neighbor sets are symmetric. 
    """
    cdef Opfmstsne* popt = <Opfmstsne*> instance
    # Initializing the gradient to 0
    memset(g, 0, popt.sstx)

    cdef double[:,:] dY_repu = np.empty((popt.N,2), dtype=DTYPE)
    cdef computed_Z* Z_CIC = <computed_Z*> PyMem_Malloc(sizeof(computed_Z))
    # Index variables
    cdef Py_ssize_t i, rs, idx, k, inn, j, idj
    # Intermediate variables. Z will store the denominator of the LD similarities, as computed by the Barnes-Hut algorithm
    cdef double Z, cfx, a
    # Stores the index of the currently considered data point in x
    idx = 0
    # For each data point
    dY_repu = _compute_Repu_CIC(x, popt.M, popt.N, Z_CIC)
    Z = Z_CIC.PM_Z

    # Check whether Z is not too small
    if Z < FLOAT64_EPS:
        Z = FLOAT64_EPS
    cfx = log(Z)

    for i in range(popt.N):
        g[2*i] = dY_repu[i,0]
        g[2*i+1] = dY_repu[i,1]

    # Pointer toward the considered data point
    cdef const double* xi
    # Cost function value to return
    cdef lbfgsfloatval_t fx = 0.0
    # For each data point
    idx = 0
    
    for i in range(popt.N_1):
        # Currently considered data point
        xi = &x[idx]
        for inn in range(2, popt.m_nn[0][i][0], 4):
            idj = popt.m_nn[0][i][inn+1]
            a = 1.0 + sqeucl_dist_ptr(xi, &x[idj], popt.n_components)
            Z = popt.sim_hd_ms[0][i][popt.m_nn[0][i][inn]]
            # Updating the cost function
            fx += Z * log(a)
            # Updating the gradient
            Z /= a
            for j in range(popt.n_components):
                a = Z * (xi[j] - x[idj])
                g[idx+j] += 4*a
                g[idj] -= 4*a
                idj += 1
        idx += popt.n_components

    # Returning the cost function value

    return fx + cfx


cpdef inline void fmstsne_implem(double[::1] X_hds, double[::1] X_lds, int N, int d_hds, int n_components, bint cperp, int n_rs, double ms_thetha, int nit_max, double gtol, double ftol, int maxls, int maxcor, int L_min, int rseed, int M):
    """
    Cython implementation of FMs t-SNE.
    L_min is provided in argument. 
    X_hds and X_lds must both be in a 1d array
    """
    # Fix the random seed
    srand(rseed)
    # Number of data points in double
    cdef double Nd = <double> N
    
    #####
    ##### Perplexity-related quantities
    #####
    
    cdef int K_star = 2
    cdef bint isLmin1 = L_min == 1
    cdef bint isnotLmin1 = not isLmin1
    # Number of scales
    cdef int L = ms_def_n_scales(Nd, K_star, L_min, isLmin1)
    
    # Just a shift for the perplexity at first scale when L_min != 1
    cdef int sLm_nt = ms_def_shift_Lmin(isnotLmin1, L_min)
    
    # Perplexity at each scale
    cdef int* K_h = ms_def_Kh(K_star, isnotLmin1, sLm_nt, L)
    if K_h is NULL:     
        printf("Error in fmssne_implem function of cython_implem.pyx: out of memory for K_h.")
        exit(EXIT_FAILURE)
    
    #####
    ##### Computing the size of the subsampled data set at each scale
    #####
    # Size of the subsampled data set at each scale (except the first scale if L_min==1)
    cdef int* n_ds_h = f_def_n_ds_h(isLmin1, N, sLm_nt, Nd, L)
    if n_ds_h is NULL:     
        PyMem_Free(K_h)
        printf("Error in fmssne_implem function of cython_implem.pyx: out of memory for n_ds_h.")
        exit(EXIT_FAILURE)
    
    #####
    ##### Indexes of all the examples in the data set
    #####
    
    cdef int* all_ind = seq_1step(N)
    if all_ind is NULL:     
        PyMem_Free(K_h)
        PyMem_Free(n_ds_h)
        printf("Error in fmssne_implem function of cython_implem.pyx: out of memory for all_ind.")
        exit(EXIT_FAILURE)
    
    #####
    ##### Number of neighbors to compute per data point for each scale
    #####
    cdef int* nnn_h = f_def_nnn_h(L, K_h, n_ds_h, cperp)
    if nnn_h is NULL:     
        PyMem_Free(K_h)
        PyMem_Free(n_ds_h)
        PyMem_Free(all_ind)
        printf("Error in fmssne_implem function of cython_implem.pyx: out of memory for nnn_h.")
        exit(EXIT_FAILURE)
    # Sum of the elements of nnn_h
    sLm_nt = f_nnn_tot(nnn_h, L)
    #####
    ##### Computing the considered neighbors of each data point, for each scale and random sampling
    #####
    
    # Allocating memory to store the indexes of the considered neighbors for each data point, for each random sampling. In function f_nn_ds_hdprec, arr_nn_i_rs will be reallocated so that its third dimension may be smaller than sLm_nt.
    cdef int*** arr_nn_i_rs = alloc_int_3dmat(n_rs, N, sLm_nt)
    if arr_nn_i_rs is NULL:     
        PyMem_Free(K_h)
        PyMem_Free(n_ds_h)
        PyMem_Free(all_ind)
        PyMem_Free(nnn_h)
        printf("Error in fmssne_implem function of cython_implem.pyx: out of memory for arr_nn_i_rs.")
        exit(EXIT_FAILURE)
    
    # Allocating memory to store the number of considered neighbors for each data point, for each random sampling
    cdef int** nnn_i_rs = calloc_int_2dmat(n_rs, N)
    if nnn_i_rs is NULL:     
        PyMem_Free(K_h)
        PyMem_Free(n_ds_h)
        PyMem_Free(all_ind)
        PyMem_Free(nnn_h)
        free_int_3dmat(arr_nn_i_rs, n_rs, N)
        printf("Error in fmssne_implem function of cython_implem.pyx: out of memory for nnn_i_rs.")
        exit(EXIT_FAILURE)
    
    # Allocating memory to store the squared distances between the considered neighbors and each data point, for each random sampling. In fact, for each random sampling rs, data point i and neighbor j, ds_nn_i_rs[rs][i][j] will contain the minimum squared distance between i and all its neighbors in random sampling rs minus the squared distance between i and j. In function f_nn_ds_hdprec, ds_nn_i_rs will be reallocated so that its third dimension may be smaller than sLm_nt.
    cdef double*** ds_nn_i_rs = alloc_dble_3dmat(n_rs, N, sLm_nt)
    if ds_nn_i_rs is NULL:     
        PyMem_Free(K_h)
        PyMem_Free(n_ds_h)
        PyMem_Free(all_ind)
        PyMem_Free(nnn_h)
        free_int_3dmat(arr_nn_i_rs, n_rs, N)
        free_int_2dmat(nnn_i_rs, n_rs)
        printf("Error in fmssne_implem function of cython_implem.pyx: out of memory for ds_nn_i_rs.")
        exit(EXIT_FAILURE)
    
    # Allocating memory to store the HD bandwidths for each scale, data point and random sampling
    cdef double*** tau_h_i_rs = alloc_dble_3dmat(L, n_rs, N)
    if tau_h_i_rs is NULL:     
        PyMem_Free(K_h)
        PyMem_Free(n_ds_h)
        PyMem_Free(all_ind)
        PyMem_Free(nnn_h)
        free_int_3dmat(arr_nn_i_rs, n_rs, N)
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        printf("Error in fmssne_implem function of cython_implem.pyx: out of memory for tau_h_i_rs.")
        exit(EXIT_FAILURE)
    
    print("I entered")
    # Computing the considered nearest neighbors of each data point for each random sampling and filling arr_nn_i_rs, nnn_i_rs, ds_nn_i_rs and tau_h_i_rs. The considered nearest neighbors of each data point are also symmetrized for each random sampling (i.e. if i is in the considered nearest neighbors of j, than j must also be in the considered nearest neighbors of i).
    if f_nn_ds_hdprec(d_hds, K_h, N, L, n_ds_h, all_ind, nnn_h, isLmin1, &X_hds[0], n_rs, arr_nn_i_rs, nnn_i_rs, ds_nn_i_rs, tau_h_i_rs, sLm_nt, True):
        PyMem_Free(K_h)
        PyMem_Free(n_ds_h)
        PyMem_Free(all_ind)
        PyMem_Free(nnn_h)
        free_int_3dmat(arr_nn_i_rs, n_rs, N)
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        free_dble_3dmat(tau_h_i_rs, L, n_rs)
        printf("Error in fmssne_implem function of cython_implem.pyx: out of memory in function f_nn_ds_hdprec.")
        exit(EXIT_FAILURE)

    print("I quitted")
    
    # Free stuffs which will not be used anymore
    PyMem_Free(K_h)
    PyMem_Free(n_ds_h)
    PyMem_Free(all_ind)
    PyMem_Free(nnn_h)
    
    #####
    ##### Data structure facilitating the symmetrization of the HD similarities
    #####
    # sLm_nt now refers to N-1
    sLm_nt = N-1
    cdef Py_ssize_t*** m_nn = fms_sym_nn_match(n_rs, sLm_nt, arr_nn_i_rs, nnn_i_rs, n_components)
    if m_nn is NULL:
        free_int_3dmat(arr_nn_i_rs, n_rs, N)
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        free_dble_3dmat(tau_h_i_rs, L, n_rs)
        printf('Error in function fmstsne_implem of module cython_implem.pyx: out of memory in function fms_sym_nn_match.')
        exit(EXIT_FAILURE)
    
    # Free resources which are not needed anymore
    free_int_3dmat(arr_nn_i_rs, n_rs, N)
    
    #####
    ##### Allocating memory to store the HD similarities
    #####
    
    # Array storing the multi-scale HD similarities, as computed during the multi-scale optimization
    cdef double*** simhd_ms_nn_i_rs = alloc_dble_3dmat_varK(n_rs, N, nnn_i_rs)
    if simhd_ms_nn_i_rs is NULL:
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        free_dble_3dmat(tau_h_i_rs, L, n_rs)
        free_Pysst_3dmat(m_nn, n_rs, sLm_nt)
        printf("Error in fmssne_implem function of cython_implem.pyx: out of memory for simhd_ms_nn_i_rs.")
        exit(EXIT_FAILURE)
    
    # Array storing the HD similarities at some scale h, during the multi-scale optimization
    cdef double*** simhd_h_nn_i_rs = alloc_dble_3dmat_varK(n_rs, N, nnn_i_rs)
    if simhd_h_nn_i_rs is NULL:
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
        free_dble_3dmat(tau_h_i_rs, L, n_rs)
        free_Pysst_3dmat(m_nn, n_rs, sLm_nt)
        printf("Error in fmssne_implem function of cython_implem.pyx: out of memory for simhd_h_nn_i_rs.")
        exit(EXIT_FAILURE)
    
    #####
    ##### Data structures to compute the LD distances when evaluating the cost function and its gradient
    #####
    
    # isLmin1 now refers to n_rs > 1
    isLmin1 = n_rs > 1
    
    # nnn_all_rs[i] will contain the total number of neighbors considered for i over all random samplings.
    cdef Py_ssize_t* nnn_all_rs
    # inn_all_rs[i] will contain the neighbors considered for i over all random samplings.
    cdef Py_ssize_t** inn_all_rs
    # dsld_all_rs will allow to store the squared LD distances between a data point and its neighbors over all random samplings.
    cdef double* dsld_all_rs
    # idnn_in_ars[rs][i][j] contains the index of the distance between i and its jth neighbor in random sampling rs in dsld_all_rs.
    cdef Py_ssize_t*** idnn_in_ars
    if isLmin1:
        # nnn_all_rs will be filled in function gather_nn_all_rs
        nnn_all_rs =  <Py_ssize_t*> PyMem_Malloc(sLm_nt*sizeof(Py_ssize_t))
        if nnn_all_rs is NULL:
            free_int_2dmat(nnn_i_rs, n_rs)
            free_dble_3dmat(ds_nn_i_rs, n_rs, N)
            free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
            free_dble_3dmat(simhd_h_nn_i_rs, n_rs, N)
            free_dble_3dmat(tau_h_i_rs, L, n_rs)
            free_Pysst_3dmat(m_nn, n_rs, sLm_nt)
            printf('Error in function fmstsne_implem of module cython_implem.pyx: out of memory for nnn_all_rs.')
            exit(EXIT_FAILURE)
        inn_all_rs = gather_nn_all_rs(nnn_all_rs, n_rs, sLm_nt, m_nn)
        if inn_all_rs is NULL:
            free_int_2dmat(nnn_i_rs, n_rs)
            free_dble_3dmat(ds_nn_i_rs, n_rs, N)
            free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
            free_dble_3dmat(simhd_h_nn_i_rs, n_rs, N)
            free_dble_3dmat(tau_h_i_rs, L, n_rs)
            free_Pysst_3dmat(m_nn, n_rs, sLm_nt)
            PyMem_Free(nnn_all_rs)
            printf('Error in function fmstsne_implem of module cython_implem.pyx: out of memory in function gather_nn_all_rs.')
            exit(EXIT_FAILURE)
        dsld_all_rs = <double*> PyMem_Malloc(max_arr_ptr_Pysst(nnn_all_rs, sLm_nt)*sizeof(double))
        if dsld_all_rs is NULL:
            free_int_2dmat(nnn_i_rs, n_rs)
            free_dble_3dmat(ds_nn_i_rs, n_rs, N)
            free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
            free_dble_3dmat(simhd_h_nn_i_rs, n_rs, N)
            free_dble_3dmat(tau_h_i_rs, L, n_rs)
            free_Pysst_3dmat(m_nn, n_rs, sLm_nt)
            PyMem_Free(nnn_all_rs)
            free_Pysst_2dmat(inn_all_rs, sLm_nt)
            printf('Error in function fmstsne_implem of module cython_implem.pyx: out of memory for dsld_all_rs.')
            exit(EXIT_FAILURE)
        idnn_in_ars = fms_nn_rs_match_all_rs(n_rs, sLm_nt, inn_all_rs, nnn_all_rs, m_nn)
        if idnn_in_ars is NULL:
            free_int_2dmat(nnn_i_rs, n_rs)
            free_dble_3dmat(ds_nn_i_rs, n_rs, N)
            free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
            free_dble_3dmat(simhd_h_nn_i_rs, n_rs, N)
            free_dble_3dmat(tau_h_i_rs, L, n_rs)
            free_Pysst_3dmat(m_nn, n_rs, sLm_nt)
            PyMem_Free(nnn_all_rs)
            free_Pysst_2dmat(inn_all_rs, sLm_nt)
            PyMem_Free(dsld_all_rs)
            printf('Error in function fmstsne_implem of module cython_implem.pyx: out of memory in function fms_nn_rs_match_all_rs.')
            exit(EXIT_FAILURE)
    else:
        nnn_all_rs = NULL
        inn_all_rs = NULL
        dsld_all_rs = NULL
        idnn_in_ars = NULL
    
    #####
    ##### Multi-scale optimization
    #####
    # Pointer toward the start of the LDS
    cdef double* xlds = &X_lds[0]
    cdef int prod_N_nc = N*n_components
    # Number of bytes of the array for the optimization
    cdef size_t shdp = prod_N_nc*sizeof(double)
    # Variables for the optimization, initialized to the current LDS. 
    cdef lbfgsfloatval_t* xopt = init_lbfgs_var(shdp, prod_N_nc, xlds)
    if xopt is NULL:
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
        free_dble_3dmat(simhd_h_nn_i_rs, n_rs, N)
        free_dble_3dmat(tau_h_i_rs, L, n_rs)
        free_Pysst_3dmat(m_nn, n_rs, sLm_nt)
        if isLmin1:
            PyMem_Free(nnn_all_rs)
            free_Pysst_2dmat(inn_all_rs, sLm_nt)
            PyMem_Free(dsld_all_rs)
            free_Pysst_3dmat(idnn_in_ars, n_rs, sLm_nt)
        printf('Out of memory for xopt')
        exit(EXIT_FAILURE)
    
    # Structure gathering the data which are necessary to evaluate the cost function and the gradient
    cdef Opfmstsne* popt = <Opfmstsne*> PyMem_Malloc(sizeof(Opfmstsne))
    if popt is NULL:
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
        free_dble_3dmat(simhd_h_nn_i_rs, n_rs, N)
        free_dble_3dmat(tau_h_i_rs, L, n_rs)
        free_Pysst_3dmat(m_nn, n_rs, sLm_nt)
        if isLmin1:
            PyMem_Free(nnn_all_rs)
            free_Pysst_2dmat(inn_all_rs, sLm_nt)
            PyMem_Free(dsld_all_rs)
            free_Pysst_3dmat(idnn_in_ars, n_rs, sLm_nt)
        lbfgs_free(xopt)
        printf("Out of memory for popt")
        exit(EXIT_FAILURE)
    # Filling popt
    popt.N = N
    popt.N_1 = sLm_nt
    popt.n_components = n_components
    popt.sstx = shdp
    popt.n_rs = n_rs
    popt.n_rs_geq1 = isLmin1
    popt.n_rs_f = <double> n_rs
    popt.inv_n_rs_2f = 2.0/popt.n_rs_f
    popt.inv_n_rs_4f = 2.0*popt.inv_n_rs_2f
    popt.sim_hd_ms = simhd_h_nn_i_rs
    popt.m_nn = m_nn
    popt.nnn_all_rs = nnn_all_rs
    popt.inn_all_rs = inn_all_rs
    popt.dsld_all_rs = dsld_all_rs
    popt.idnn_in_ars = idnn_in_ars
    popt.M = int(pow(2.0, M))
    popt.grid_meth = "CIC"
    # Space-partitioning trees are working with the squared threshold to save the computation time of computing the square root for the Euclidean distance
    popt.theta_s = ms_thetha*ms_thetha
    popt.inter_fct = 7
    K_star = n_components*sizeof(double)
    popt.qdiff = <double*> PyMem_Malloc(K_star)
    if popt.qdiff is NULL:
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
        free_dble_3dmat(simhd_h_nn_i_rs, n_rs, N)
        free_dble_3dmat(tau_h_i_rs, L, n_rs)
        free_Pysst_3dmat(m_nn, n_rs, sLm_nt)
        if isLmin1:
            PyMem_Free(nnn_all_rs)
            free_Pysst_2dmat(inn_all_rs, sLm_nt)
            PyMem_Free(dsld_all_rs)
            free_Pysst_3dmat(idnn_in_ars, n_rs, sLm_nt)
        lbfgs_free(xopt)
        PyMem_Free(popt)
        printf("Out of memory for popt.qdiff.\n")
        exit(EXIT_FAILURE)
    # Parameters of the L-BFGS optimization
    cdef lbfgs_parameter_t param
    cdef lbfgs_parameter_t* pparam = &param
    # Initializing param with default values
    lbfgs_parameter_init(pparam)
    # Updating some parameters
    param.m = maxcor
    param.epsilon = gtol
    param.delta = ftol
    param.max_iterations = nit_max
    param.max_linesearch = maxls
    param.past = 1
    # We modify the default values of the minimum and maximum step sizes of the line search because the problem is badly scaled
    param.max_step = DBL_MAX
    param.min_step = DBL_MIN
    
    # k refers to the number of currently considered scales and h to the index of the current scale. Nd will store the inverse of the number of currently considered scales. 
    cdef Py_ssize_t k, h
    h = L-1

    print("Number of perpl: " + str(L))
    for k in range(1, L+1, 1):
        # Updates related to the current multi-scale optimization step
        f_update_mso_step(k, h, n_rs, N, nnn_i_rs, ds_nn_i_rs, tau_h_i_rs, simhd_ms_nn_i_rs, simhd_h_nn_i_rs)
        # Symmetrizing the multi-scale HD similarities. Be careful that only the similarities between i and j such that j>i are actually symetrized, since only these are used in the evaluate function. 
        fmstsne_symmetrize(n_rs, sLm_nt, simhd_ms_nn_i_rs, m_nn, simhd_h_nn_i_rs)
        # Performing the optimization
        print("IN" + str(k))
        lbfgs(prod_N_nc, xopt, NULL, fmstsne_evaluate, NULL, popt, pparam)
        h -= 1
        print("OUT" + str(k))
    
    # Gathering the optimized LD coordinates
    memcpy(xlds, xopt, shdp)
    
    # Free the ressources
    free_int_2dmat(nnn_i_rs, n_rs)
    free_dble_3dmat(ds_nn_i_rs, n_rs, N)
    free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
    free_dble_3dmat(simhd_h_nn_i_rs, n_rs, N)
    free_dble_3dmat(tau_h_i_rs, L, n_rs)
    free_Pysst_3dmat(m_nn, n_rs, sLm_nt)
    if isLmin1:
        PyMem_Free(nnn_all_rs)
        free_Pysst_2dmat(inn_all_rs, sLm_nt)
        PyMem_Free(dsld_all_rs)
        free_Pysst_3dmat(idnn_in_ars, n_rs, sLm_nt)
    lbfgs_free(xopt)
    PyMem_Free(popt.qdiff)
    PyMem_Free(popt)


#######################################################
####################################################### Quality criteria Q_NX and R_NX 
#######################################################


cdef struct nnRank:
    int nn                # Index of a sample
    int rank              # Rank of the sample

cdef inline bint sortByInd(const nnRank v, const nnRank w) nogil:
    """
    Returns True when v.nn is < than w.nn.
    """
    return v.nn < w.nn

cpdef inline double drqa_qnx_rnx_auc(double[::1] X_hds, double[::1] X_lds, int N, int d_hds, int d_lds, int Kup, double[::1] qnxk, double[::1] rnxk, int rnxk_size):
    """
    Compute the quality criteria curves Q_NX(K) and R_NX(K) with the neighborhood size K ranging from 1 to Kup. The AUC of the reduced R_NX(K) curve is returned. 
    In:
    - X_hds: one-dimensional array with the HD samples stacked one after the other. 
    - X_lds: one-dimensional array with the LD samples stacked one after the other. 
    - N: number of samples.
    - d_hds: dimension of the HDS.
    - d_lds: dimension of the LDS.
    - Kup: greatest neighborhood size to consider.
    - qnxk: array to store the Q_NX(K) values for K = 1, ..., Kup.
    - rnxk: array to store the R_NX(K) values for K = 1, ..., min(N-2, Kup).
    - rnxk_size: min(N-2, Kup).
    This function modifies the arrays qnxk and rnxk.
    Out: 
    - A double being the AUC of the reduced R_NX curve. 
    Remark: 
    - the time complexity to compute these criteria scales as O(N*Kup*log(N)).
    - the Euclidean distance is employed to compute the quality criteria. 
    """
    # Initializing qnxk to zero
    memset(&qnxk[0], 0, Kup*sizeof(double))
    
    # Constructing the VP trees in the HDS and LDS
    cdef VpTree* vpt_hd = new VpTree(&X_hds[0], N, d_hds)
    cdef VpTree* vpt_ld = new VpTree(&X_lds[0], N, d_lds)
    
    # Kup + 1
    cdef int Kupadd = Kup + 1
    
    # Allocating an array to store the Kupadd nearest HD neighbor of a data point
    cdef int* nn_hd = <int*> PyMem_Malloc(Kupadd*sizeof(int))
    if nn_hd is NULL:     
        del vpt_hd
        del vpt_ld
        printf("Error in drqa_qnx_rnx_auc function of cython_implem.pyx: out of memory for nn_hd.")
        exit(EXIT_FAILURE)
    # Allocating an array to store the Kupadd nearest LD neighbor of a data point
    cdef int* nn_ld = <int*> PyMem_Malloc(Kupadd*sizeof(int))
    if nn_ld is NULL:     
        del vpt_hd
        del vpt_ld
        PyMem_Free(nn_hd)
        printf("Error in drqa_qnx_rnx_auc function of cython_implem.pyx: out of memory for nn_ld.")
        exit(EXIT_FAILURE)
    
    # Allocating an array of structure to store the indexes of the HD neighbors and their ranks
    cdef nnRank* nnrk_hd = <nnRank*> PyMem_Malloc(Kup*sizeof(nnRank))
    if nnrk_hd is NULL:
        del vpt_hd
        del vpt_ld
        PyMem_Free(nn_hd)
        PyMem_Free(nn_ld)
        printf("Error in drqa_qnx_rnx_auc function of cython_implem.pyx: out of memory for nnrk_hd.")
        exit(EXIT_FAILURE)
    
    # Variable to iterate over the samples
    cdef Py_ssize_t i, ihd, ild, j, lb, ub, mid
    ihd = 0
    ild = 0
    cdef int jr, Kupsub
    Kupsub = Kup - 1
    
    # For each data point
    for i in range(N):
        # Searching the Kupadd nearest neighbors of sample i in HDS and LDS
        vpt_hd.search(&X_hds[ihd], Kupadd, nn_hd)
        vpt_ld.search(&X_lds[ild], Kupadd, nn_ld)
        
        # Filling nnrk_hd
        jr = 0
        for j in range(Kupadd):
            if nn_hd[j] != i:
                nnrk_hd[jr].nn = nn_hd[j]
                nnrk_hd[jr].rank = jr
                jr += 1
        
        # Sorting nnrk_hd according to the nn keys
        sort(nnrk_hd, nnrk_hd + Kup, sortByInd)
        
        # LD rank
        jr = 0
        # For each LD neighbor
        for j in range(Kupadd):
            if nn_ld[j] != i:
                # If nn_ld[j] is in the range of nnrk_hd
                if (nn_ld[j] >= nnrk_hd[0].nn) and (nn_ld[j] <= nnrk_hd[Kupsub].nn):
                    # Searching for nn_ld[j] in nnrk_hd using binary search
                    lb = 0
                    ub = Kup
                    while ub - lb > 1:
                        mid = (ub + lb)//2
                        if nn_ld[j] == nnrk_hd[mid].nn:
                            lb = mid
                            break
                        elif nn_ld[j] < nnrk_hd[mid].nn:
                            ub = mid
                        else:
                            lb = mid + 1
                    # Updating qnxk only if nn_ld[j] == nnrk_hd[lb].nn
                    if nn_ld[j] == nnrk_hd[lb].nn:
                        # Updating at the biggest rank between the HD and LD ones
                        if jr < nnrk_hd[lb].rank:
                            qnxk[nnrk_hd[lb].rank] += 1.0
                        else:
                            qnxk[jr] += 1.0
                # Incrementing the LD rank
                jr += 1
        
        # Updating ihd and ild
        ihd += d_hds
        ild += d_lds
    
    # Free the ressources
    del vpt_hd
    del vpt_ld
    PyMem_Free(nn_hd)
    PyMem_Free(nn_ld)
    PyMem_Free(nnrk_hd)
    
    # Computing the cumulative sum of qnxk and normalizing it
    cdef double cs = 0.0
    cdef double Nd = <double> N
    for i in range(Kup):
        cs += qnxk[i]
        qnxk[i] = cs/Nd
        Nd += N
    
    # Computing rnxk and its AUC
    Nd = <double> (N-1)
    cs = Nd - 1.0
    cdef double K = 1.0
    cdef double iK = 1.0
    cdef double siK = 0.0
    cdef double auc = 0.0
    for i in range(rnxk_size):
        siK += iK
        rnxk[i] = (Nd*qnxk[i] - K)/cs
        auc += (rnxk[i]*iK)
        K += 1.0
        iK = 1.0/K
        cs -= 1.0
    
    # Normalizing the AUC
    auc /= siK
    
    # Returning the AUC
    return auc