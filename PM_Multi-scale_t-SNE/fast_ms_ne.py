#! python3
# -*-coding:Utf-8 -*

########################################################################################################
########################################################################################################

#
# %%%% !!! IMPORTANT NOTE !!! %%%%
# At the end of the fast_ms_ne.py file, a demo presents how this python code can be used. Running this file (python fast_ms_ne.py) will run the demo. Importing this module will not run the demo. The demo takes a few minutes. To be able to use the code in fast_ms_ne.py, do not forget to first compile the Cython file 'cython_implem.pyx'; check the instructions below for explanations on the required compilation steps. 
# %%%% !!!                !!! %%%%

#     fast_ms_ne.py

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

import numpy as np, numba, sklearn.decomposition, cython_implem, scipy.spatial.distance, matplotlib.pyplot as plt, time, os, sklearn.manifold, sklearn.datasets
from load_coil_20 import load_COIL_20_data

# Name of this file
module_name = "fast_ms_ne.py"

##############################
############################## 
# General functions used by others in the code. 
####################

def init_lds(X_hds, N, init='pca', n_components=2, rand_state=None, var=10.0**(-4)):
    """
    Initialize the LD embedding.
    In:
    - X_hds: numpy.ndarray with shape (N, M), containing the HD data set, with one example per row and one dimension per column, or None. If X_hds is set to None, init cannot be equal to 'pca', otherwise an error is raised. 
    - N: number of examples in the data set. If X_hds is not None, N must be equal to X_hds.shape[0]. 
    - init: determines the initialization of the LD embedding. 
    ---> If isinstance(init, str) is True:
    ------> If init is equal to 'pca', the LD embedding is initialized with the first n_components principal components of X_hds. X_hds cannot be None in this case, otherwise an error is raised. 
    ------> If init is equal to 'random', the LD embedding is initialized randomly, using a uniform Gaussian distribution with a variance equal to var. X_hds may be set to None in this case.
    ------> Otherwise an error is raised.
    ---> If isinstance(init, np.ndarray) is True:
    ------> init must in this case be a 2-D numpy array, with N rows and n_components columns. It stores the LD positions to use for the initialization, with one example per row and one LD dimension per column. init[i,:] contains the initial LD coordinates for the HD sample X_hds[i,:]. X_hds may be set to None in this case. If init.ndim != 2 or init.shape[0] != N or init.shape[1] != n_components, an error is raised.
    ---> Otherwise, an error is raised.
    - n_components: number of dimensions in the LD space.
    - rand_state: random state to use. Such a random state can be created using the function 'np.random.RandomState'. If it is None, it is set to np.random. 
    - var: variance employed when init is equal to 'random'. 
    Out:
    A numpy ndarray with shape (N, n_components), containing the initialization of the LD data set, with one example per row and one LD dimension per column.
    """
    global module_name
    if rand_state is None:
        rand_state = np.random
    if isinstance(init, str):
        if init == "pca":
            if X_hds is None:
                raise ValueError("Error in function init_lds of module {module_name}: init cannot be set to 'pca' if X_hds is None.".format(module_name=module_name))
            return sklearn.decomposition.PCA(n_components=n_components, whiten=False, copy=True, svd_solver='auto', iterated_power='auto', tol=0.0, random_state=rand_state).fit_transform(X_hds)
        elif init == 'random':
            return var * rand_state.randn(N, n_components)
        else:
            raise ValueError("Error in function init_lds of module {module_name}: unknown value '{init}' for init parameter.".format(module_name=module_name, init=init))
    elif isinstance(init, np.ndarray):
        if init.ndim != 2:
            raise ValueError("Error in function init_lds of module {module_name}: init must be 2-D.".format(module_name=module_name))
        if init.shape[0] != N:
            raise ValueError("Error in function init_lds of module {module_name}: init must have {N} rows, but init.shape[0] = {v}.".format(module_name=module_name, N=N, v=init.shape[0]))
        if init.shape[1] != n_components:
            raise ValueError("Error in function init_lds of module {module_name}: init must have {n_components} columns, but init.shape[1] = {v}.".format(module_name=module_name, n_components=n_components, v=init.shape[1]))
        return init
    else:
        raise ValueError("Error in function init_lds of module {module_name}: unknown type value '{v}' for init parameter.".format(module_name=module_name, v=type(init)))

def eucl_dist_matr(X):
    """
    Compute the pairwise Euclidean distances in a data set. 
    In:
    - X: a 2-D np.ndarray with shape (N,M) containing one example per row and one feature per column.
    Out:
    A 2-D np.ndarray dm with shape (N,N) containing the pairwise Euclidean distances between the data points in X, such that dm[i,j] stores the Euclidean distance between X[i,:] and X[j,:].
    """
    return scipy.spatial.distance.squareform(X=scipy.spatial.distance.pdist(X=X, metric='euclidean'), force='tomatrix')

##############################
##############################
# Nonlinear dimensionality reduction through fast multi-scale t-SNE (FMs t-SNE) [1]. 
# See the documentation of the 'fmstsne' function for details. 
# The demo at the end of this file presents how it can be used. 
# Given a data set with N samples, the 'fmstsne' function has O(N (log(N))**2) time complexity. It can hence run on very large-scale databases. 
####################

def fmstsne(X_hds, n_components=2, init='pca', rand_state=None, nit_max=50, gtol=10.0**(-5), ftol=2.2204460492503131e-09, maxls=50, maxcor=10, bht=0.75, fseed=1):
    """
    Apply fast multi-scale t-SNE on a data set X_hds to reduce its dimension, as presented in [1].
    In:
    - X_hds: 2-D numpy.ndarray with shape (N, M), containing the HD data set, with one example per row and one feature per column. It is assumed that it does not contain duplicated examples. 
    - n_components: number of dimensions of the low-dimensional embedding of X_hds. 
    - init: sets the initialization of the LD embedding as described in the function init_lds. 
    - rand_state: random state to use in init_lds. See init_lds for documentation. 
    - nit_max: maximum number of L-BFGS steps at each stage of the multi-scale optimization.
    - gtol: tolerance for the infinite norm of the gradient in the L-BFGS algorithm. The L-BFGS iterations hence stop when max{|g_i | i = 1, ..., n} <= gtol where g_i is the i-th component of the gradient. 
    - ftol: tolerance for the relative updates of the cost function value in L-BFGS.
    - maxls: maximum number of line search steps per L-BFGS-B iteration.
    - maxcor: the maximum number of variable metric corrections used to define the limited memory matrix in L-BFGS. 
    - bht: a float strictly between 0 and 1 and which is the Barnes-Hut threshold to employ. If it is not strictly between 0 and 1, an error is raised. 
    - fseed: a strictly positive integer being the random seed used in Cython to perform the random sampling of the HD data set at the different scales. If it is not an integer >=1, an error is raised. 
    Out:
    A 2-D numpy.ndarray X_lds with shape (N, n_components), containing the low dimensional data set representing X_hds. It contains one example per row and one feature per column. X_lds[i,:] contains the LD coordinates of the HD sample X_hds[i,:]. 
    Remarks:
    - L-BFGS algorithm is used, as in [1].
    - Multi-scale optimization is performed, as presented in [2].
    - Euclidean distances are employed to evaluate the pairwise similarities in both the HD and LD spaces. 
    """
    global module_name
    # Checking bht
    if (bht <= 0) or (bht >= 1):
        raise ValueError("Error in function fmstsne of module {module_name}: bht={bht} while it should be a float strictly between 0 and 1.".format(module_name=module_name, bht=bht))
    else:
        bht = np.float64(bht)
    # Checking fseed
    if (not isinstance(fseed, int)) or (fseed <=0):
        raise ValueError("Error in function fmstsne of module {module_name}: fseed={fseed} while it should be an integer >= 1.".format(module_name=module_name, fseed=fseed))
    # Number of samples and dimension of the HDS
    N, M = X_hds.shape
    # Constructing a 1-D numpy array with C-order containing the HDS
    X_hds_1D = np.ascontiguousarray(a=np.reshape(a=X_hds, newshape=N*M, order='C'), dtype=np.float64)
    # Initializing the LD embedding.
    X_lds = np.ascontiguousarray(a=np.reshape(a=init_lds(X_hds=X_hds, N=N, init=init, n_components=n_components, rand_state=rand_state, var=1.0), newshape=N*n_components, order='C'), dtype=np.float64)
    # Applying the fast multi-scale t-SNE algorithm
    cython_implem.fmstsne_implem(X_hds_1D, X_lds, N, M, n_components, True, 1, bht, nit_max, gtol, ftol, maxls, maxcor, 1, fseed, 9)
    # Reshaping the LDS
    X_lds = np.reshape(a=X_lds, newshape=(N, n_components), order='C')
    # Returning
    return X_lds

##############################
############################## 
# Unsupervised DR quality assessment: rank-based criteria measuring the HD neighborhood preservation in the LD embedding [3, 4]. 
# These criteria are used in the experiments reported in [1]. 
# The main functions are 'eval_dr_quality' and 'red_rnx_auc'. 
# See their documentations for details. They explain the meaning of the quality criteria and how to interpret them. 
# The demo at the end of this file presents how to use the 'eval_dr_quality' and 'red_rnx_auc' functions. 
# Given a data set with N samples, the 'eval_dr_quality' function has O(N**2 log(N)) time complexity. It can hence run using databases with up to a few thousands of samples. 
# On the other hand, given a data set with N samples, the 'red_rnx_auc' function has O(N*Kup*log(N)) time complexity, where Kup is the maximum neighborhood size accounted when computing the quality criteria. This function can hence run using much larger databases than 'eval_dr_quality', provided that Kup is small compared to N. 
####################

def coranking(d_hd, d_ld):
    """
    Computation of the co-ranking matrix, as described in [4]. 
    The time complexity of this function is O(N**2 log(N)), where N is the number of data points.
    In:
    - d_hd: 2-D numpy array representing the redundant matrix of pairwise distances in the HDS.
    - d_ld: 2-D numpy array representing the redundant matrix of pairwise distances in the LDS.
    Out:
    The (N-1)x(N-1) co-ranking matrix, where N = d_hd.shape[0].
    """
    # Computing the permutations to sort the rows of the distance matrices in HDS and LDS. 
    perm_hd = d_hd.argsort(axis=-1, kind='mergesort')
    perm_ld = d_ld.argsort(axis=-1, kind='mergesort')
    
    N = d_hd.shape[0]
    i = np.arange(N, dtype=np.int64)
    # Computing the ranks in the LDS
    R = np.empty(shape=(N,N), dtype=np.int64)
    for j in range(N):
        R[perm_ld[j,i],j] = i
    # Computing the co-ranking matrix
    Q = np.zeros(shape=(N,N), dtype=np.int64)
    for j in range(N):
        Q[i,R[perm_hd[j,i],j]] += 1
    # Returning
    return Q[1:,1:]

@numba.jit(nopython=True)
def eval_auc(arr):
    """
    Evaluates the AUC, as defined in [2].
    In:
    - arr: 1-D numpy array storing the values of a curve from K=1 to arr.size.
    Out:
    The AUC under arr, as defined in [2], with a log scale for K=1 to arr.size. 
    """
    i_all_k = 1.0/(np.arange(arr.size)+1.0)
    return np.float64(np.dot(arr, i_all_k))/(i_all_k.sum())

@numba.jit(nopython=True)
def eval_rnx(Q):
    """
    Evaluate R_NX(K) for K = 1 to N-2, as defined in [5]. N is the number of data points in the data set.
    The time complexity of this function is O(N^2).
    In:
    - Q: a 2-D numpy array representing the (N-1)x(N-1) co-ranking matrix of the embedding. 
    Out:
    A 1-D numpy array with N-2 elements. Element i contains R_NX(i+1).
    """
    N_1 = Q.shape[0]
    N = N_1 + 1
    # Computing Q_NX
    qnxk = np.empty(shape=N_1, dtype=np.float64)
    acc_q = 0.0
    for K in range(N_1):
        acc_q += (Q[K,K] + np.sum(Q[K,:K]) + np.sum(Q[:K,K]))
        qnxk[K] = acc_q/((K+1)*N)
    # Computing R_NX
    arr_K = np.arange(N_1)[1:].astype(np.float64)
    rnxk = (N_1*qnxk[:N_1-1]-arr_K)/(N_1-arr_K)
    # Returning
    return rnxk

def eval_dr_quality(d_hd, d_ld):
    """
    Compute the DR quality assessment criteria R_{NX}(K) and AUC, as defined in [2, 3, 4, 5] and as employed in the experiments reported in [1].
    These criteria measure the neighborhood preservation around the data points from the HDS to the LDS. 
    Based on the HD and LD distances, the sets v_i^K (resp. n_i^K) of the K nearest neighbors of data point i in the HDS (resp. LDS) can first be computed. 
    Their average normalized agreement develops as Q_{NX}(K) = (1/N) * \sum_{i=1}^{N} |v_i^K \cap n_i^K|/K, where N refers to the number of data points and \cap to the set intersection operator. 
    Q_{NX}(K) ranges between 0 and 1; the closer to 1, the better.
    As the expectation of Q_{NX}(K) with random LD coordinates is equal to K/(N-1), which is increasing with K, R_{NX}(K) = ((N-1)*Q_{NX}(K)-K)/(N-1-K) enables more easily comparing different neighborhood sizes K. 
    R_{NX}(K) ranges between -1 and 1, but a negative value indicates that the embedding performs worse than random. Therefore, R_{NX}(K) typically lies between 0 and 1. 
    The R_{NX}(K) values for K=1 to N-2 can be displayed as a curve with a log scale for K, as closer neighbors typically prevail. 
    The area under the resulting curve (AUC) is a scalar score which grows with DR quality, quantified at all scales with an emphasis on small ones.
    The AUC lies between -1 and 1, but a negative value implies performances which are worse than random. 
    In: 
    - d_hd: 2-D numpy array of floats with shape (N, N), representing the redundant matrix of pairwise distances in the HDS.
    - d_ld: 2-D numpy array of floats with shape (N, N), representing the redundant matrix of pairwise distances in the LDS.
    Out: a tuple with
    - a 1-D numpy array with N-2 elements. Element i contains R_{NX}(i+1).
    - the AUC of the R_{NX}(K) curve with a log scale for K, as defined in [2].
    Remark:
    - The time complexity to evaluate the quality criteria is O(N**2 log(N)). It is the time complexity to compute the co-ranking matrix. R_{NX}(K) can then be evaluated for all K=1, ..., N-2 in O(N**2). 
    """
    # Computing the co-ranking matrix of the embedding, and the R_{NX}(K) curve.
    rnxk = eval_rnx(Q=coranking(d_hd=d_hd, d_ld=d_ld))
    # Computing the AUC, and returning.
    return rnxk, eval_auc(rnxk)

def red_rnx_auc(X_hds, X_lds, Kup=10000):
    """
    This 'red_rnx_auc' function is similar to 'eval_dr_quality', as it computes the DR quality assessment criteria R_{NX}(K) and AUC, as employed in the experiments of [1], but it evaluates these criteria only for the neighborhood sizes K up to Kup, at the opposite of the 'eval_dr_quality' function which considers all possible neighborhood sizes. 
    For a description of the quality criteria and how they can be interpreted, check the documentation of the 'eval_dr_quality' function. 
    While the 'eval_dr_quality' function has a O(N**2 log(N)) time complexity when the considered data set has N samples, this 'red_rnx_auc' function has a O(Kup * N * log(N)) time complexity. Provided that Kup is small compared to N, it can hence be employed on much larger databases than 'eval_dr_quality', which is limited to data sets with a few thousands samples. 
    At the opposite of the 'eval_dr_quality' function, which can be employed using any types of distances in the HDS and the LDS, this 'red_rnx_auc' function is only considering Euclidean distances, in both the HDS and the LD embedding. 
    The R_{NX}(K) values computed by this function can be displayed as a curve for K=1 to Kup, with a log scale for K, as closer neighbors typically prevail. 
    The area under the resulting reduced R_{NX} curve (AUC) is a scalar score which grows with DR quality for neighborhood sizes up to Kup.
    R_{NX}(K) ranges between -1 and 1, but a negative value indicates that the embedding performs worse than random. Therefore, R_{NX}(K) typically lies between 0 and 1. 
    The AUC lies between -1 and 1, but a negative value implies performances which are worse than random for the neighborhood sizes smaller than Kup. 
    In: 
    - X_hds: 2-D numpy.ndarray with N rows, containing the HD data set, with one example per row and one feature per column.
    - X_lds: 2-D numpy.ndarray with N rows, containing the LD data set representing X_hds. It contains one example per row and one feature per column. X_lds[i,:] contains the LD coordinates of the HD sample X_hds[i,:]. If X_lds.shape[0] is not equal to X_hds.shape[0], an error is raised. 
    - Kup: largest neighborhood size to consider when computing the quality criteria. It must be an integer >= 1 and <= X_hds.shape[0]-1, otherwise an error is raised. 
    Out: a tuple with
    - a 1-D numpy array with min(Kup, N-2) elements. Element at index i, starting from 0, contains R_{NX}(i+1).
    - a scalar being the AUC of the R_{NX}(K) curve with a log scale for K, with K ranging from 1 to Kup, as defined in [1].
    Remark:
    - The time complexity of this function is O(Kup*N*log(N)). 
    - Euclidean distances are employed in both the HDS and the LDS. 
    """
    global module_name
    # Number N of examples in the data set and number of HD dimensions
    N, M = X_hds.shape
    # Number of LD dimensions
    P = X_lds.shape[1]
    # Checking that X_lds also has N rows
    if not np.isclose(N, X_lds.shape[0]):
        raise ValueError("Error in function red_rnx_auc of module {module_name}: X_hds.shape[0]={N} whereas X_lds.shape[0]={M}.".format(module_name=module_name, N=N, M=X_lds.shape[0]))
    # Checking that Kup is an integer >=1 and <= N-1
    if (not isinstance(Kup, int)) or (Kup < 1) or (Kup > N-1):
        raise ValueError("Error in function red_rnx_auc of module {module_name}: Kup={Kup} whereas it should be an integer >= 1 and <={v}.".format(module_name=module_name, v=N-1, Kup=Kup))
    # Initializing the arrays to store the reduced Q_NX and R_NX curves.
    qnx = np.empty(shape=Kup, dtype=np.float64)
    rnx_size = min(Kup, N-2)
    rnx = np.empty(shape=rnx_size, dtype=np.float64)
    # Reshaping X_hds and X_lds
    X_hds_1D = np.ascontiguousarray(a=np.reshape(a=X_hds, newshape=N*M, order='C'), dtype=np.float64)
    X_lds_1D = np.ascontiguousarray(a=np.reshape(a=X_lds, newshape=N*P, order='C'), dtype=np.float64)
    # Computing the reduced quality criteria
    auc = cython_implem.drqa_qnx_rnx_auc(X_hds_1D, X_lds_1D, N, M, P, Kup, qnx, rnx, rnx_size)
    # Returning
    return rnx, auc

##############################
##############################
# Plot functions. 
# The main functions are 'viz_2d_emb' and 'viz_qa'.
# Their documentations detail their parameters. 
# The demo at the end of this file presents how to use these functions. 
####################

def rstr(v, d=2):
    """
    Rounds v with d digits and returns it as a string. If it starts with 0, it is omitted. 
    In:
    - v: a number. 
    - d: number of digits to keep.
    Out:
    A string representing v rounded with d digits. If it starts with 0, it is omitted. 
    """
    p = 10.0**d
    v = str(int(round(v*p))/p)
    if v[0] == '0':
        v = v[1:]
    elif (len(v) > 3) and (v[:3] == '-0.'):
        v = "-.{a}".format(a=v[3:])
    return v

def check_create_dir(path):
    """
    Create a directory at the specified path only if it does not already exist.
    """
    dir_path = os.path.dirname(path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def save_show_fig(fname=None, f_format=None, dpi=100):
    """
    Save or show a figure.
    In:
    - fname: filename to save the figure, without the file extension. If None, the figure is shown.
    - f_format: format to save the figure. If None, set to pdf. 
    - dpi: DPI to save the figure.
    Out: 
    A figure is shown if fname is None, and saved otherwise.
    """
    if fname is None:
        plt.show()
    else:
        if f_format is None:
            f_format = 'pdf'
        # Checking whether a folder needs to be created
        # check_create_dir(fname)
        # Saving the figure
        plt.savefig('C:\\Users\\cedri\\Downloads\\PM Ms TSNE SVHN.pdf', dpi=400, format='pdf', bbox_inches='tight', facecolor='w', edgecolor='w', orientation='portrait', transparent=False, pad_inches=0.1)

def viz_2d_emb(X, vcol, tit='', fname=None, f_format=None, cmap='turbo', sdot=10, marker='o', a_scat=0.8, edcol_scat='face', stit=15, lw=2.0):
    """
    Plot a 2-D embedding of a data set.
    In:
    - X: a 2-D numpy array with shape (N, 2), where N is the number of data points to represent in the 2-D embedding.
    - vcol: a 1-D numpy array with N elements, indicating the colors of the data points in the colormap.
    - tit: title of the figure.
    - fname, f_format: path. Same as in save_show_fig.
    - cmap: colormap.
    - sdot: size of the dots.
    - marker: marker.
    - a_scat: alpha used to plot the data points.
    - edcol_scat: edge color for the points of the scatter plot. From the official documentation: "If None, defaults to (patch.edgecolor). If 'face', the edge color will always be the same as the face color. If it is 'none', the patch boundary will not be drawn. For non-filled markers, the edgecolors kwarg is ignored; color is determined by c.".
    - stit: fontsize of the title of the figure.
    - lw: linewidth for the scatter plot.
    Out:
    Same as save_show_fig.
    """  
    global module_name
    
    # Checking X
    if X.ndim != 2:
        raise ValueError("Error in function viz_2d_emb of {module_name}: X must be a numpy array with shape (N, 2), where N is the number of data points to plot in the 2-D embedding.".format(module_name=module_name))
    if X.shape[1] != 2:
        raise ValueError("Error in function viz_2d_emb of {module_name}: X must have 2 columns.".format(module_name=module_name))
    
    # Computing the limits of the axes
    xmin = X[:,0].min()
    xmax = X[:,0].max()
    ev = (xmax-xmin)*0.05
    x_lim = np.asarray([xmin-ev, xmax+ev])
    
    ymin = X[:,1].min()
    ymax = X[:,1].max()
    ev = (ymax-ymin)*0.05
    y_lim = np.asarray([ymin-ev, ymax+ev])
    
    vmc = vcol.min()
    vMc = vcol.max()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Setting the limits of the axes
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    
    # Plotting the data points
    ax.scatter(X[:,0], X[:,1], c=vcol, cmap=cmap, s=sdot, marker=marker, alpha=a_scat, edgecolors=edcol_scat, vmin=vmc, vmax=vMc, linewidths=lw)
    
    # Removing the ticks
    ax.set_xticks([], minor=False)
    ax.set_xticks([], minor=True)
    ax.set_xticklabels([], minor=False)
    ax.set_yticks([], minor=False)
    ax.set_yticks([], minor=True)
    ax.set_yticklabels([], minor=False)
    
    ax.set_title(tit, fontsize=stit)
    plt.tight_layout()
    
    # Saving or showing the figure, and closing
    save_show_fig(fname=fname, f_format=f_format)
    plt.close()

def viz_qa(Ly, fname=None, f_format=None, ymin=None, ymax=None, Lmarkers=None, Lcols=None, Lleg=None, Lls=None, Lmedw=None, Lsdots=None, lw=2, markevery=0.1, tit='', xlabel='', ylabel='', alpha_plot=0.9, alpha_leg=0.8, stit=25, sax=20, sleg=15, zleg=1, loc_leg='best', ncol_leg=1, lMticks=10, lmticks=5, wMticks=2, wmticks=1, nyMticks=11, mymticks=4, grid=True, grid_ls='solid', grid_col='lightgrey', grid_alpha=0.7, xlog=True):
    """
    Plot the DR quality criteria curves. 
    In: 
    - Ly: list of 1-D numpy arrays. The i^th array gathers the y-axis values of a curve from x=1 to x=Ly[i].size, with steps of 1. 
    - fname, f_format: path. Same as in save_show_fig.
    - ymin, ymax: minimum and maximum values of the y-axis. If None, ymin (resp. ymax) is set to the smallest (resp. greatest) value among [y.min() for y in Ly] (resp. [y.max() for y in Ly]).
    - Lmarkers: list with the markers for each curve. If None, some pre-defined markers are used.
    - Lcols: list with the colors of the curves. If None, some pre-defined colors are used.
    - Lleg: list of strings, containing the legend entries for each curve. If None, no legend is shown.
    - Lls: list of the linestyles ('solid', 'dashed', ...) of the curves. If None, 'solid' style is employed for all curves. 
    - Lmedw: list with the markeredgewidths of the curves. If None, some pre-defined value is employed. 
    - Lsdots: list with the sizes of the markers. If None, some pre-defined value is employed.
    - lw: linewidth for all the curves. 
    - markevery: approximately 1/markevery markers are displayed for each curve. Set to None to mark every dot.
    - tit: title of the plot.
    - xlabel, ylabel: labels for the x- and y-axes.
    - alpha_plot: alpha for the curves.
    - alpha_leg: alpha for the legend.
    - stit: fontsize for the title.
    - sax: fontsize for the labels of the axes. 
    - sleg: fontsize for the legend.
    - zleg: zorder for the legend. Set to 1 to plot the legend behind the data, and to None to keep the default value.
    - loc_leg: location of the legend ('best', 'upper left', ...).
    - ncol_leg: number of columns to use in the legend.
    - lMticks: length of the major ticks on the axes.
    - lmticks: length of the minor ticks on the axes.
    - wMticks: width of the major ticks on the axes.
    - wmticks: width of the minor ticks on the axes.
    - nyMticks: number of major ticks on the y-axis (counting ymin and ymax).
    - mymticks: there are 1+mymticks*(nyMticks-1) minor ticks on the y axis.
    - grid: True to add a grid, False otherwise.
    - grid_ls: linestyle of the grid.
    - grid_col: color of the grid.
    - grid_alpha: alpha of the grid.
    - xlog: True to produce a semilogx plot and False to produce a plot. 
    Out:
    A figure is shown. 
    """
    # Number of curves
    nc = len(Ly)
    # Checking the parameters
    if ymin is None:
        ymin = np.min(np.asarray([arr.min() for arr in Ly]))
    if ymax is None:
        ymax = np.max(np.asarray([arr.max() for arr in Ly]))
    if Lmarkers is None:
        Lmarkers = ['x']*nc
    if Lcols is None:
        Lcols = ['blue']*nc
    if Lleg is None:
        Lleg = [None]*nc
        add_leg = False
    else:
        add_leg = True
    if Lls is None:
        Lls = ['solid']*nc
    if Lmedw is None:
        Lmedw = [float(lw)/2.0]*nc
    if Lsdots is None:
        Lsdots = [12]*nc
    
    # Setting the limits of the y-axis
    y_lim = [ymin, ymax]
    
    # Defining the ticks on the y-axis
    yMticks = np.linspace(start=ymin, stop=ymax, num=nyMticks, endpoint=True, retstep=False)
    ymticks = np.linspace(start=ymin, stop=ymax, num=1+mymticks*(nyMticks-1), endpoint=True, retstep=False)
    yMticksLab = [rstr(v) for v in yMticks]
    
    # Initial values for xmin and xmax
    xmin, xmax = 1, -np.inf
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if xlog:
        fplot = ax.semilogx
    else:
        fplot = ax.plot
    
    # Plotting the data
    for id, y in enumerate(Ly):
        x = np.arange(start=1, step=1, stop=y.size+0.5, dtype=np.int64)
        xmax = max(xmax, x[-1])
        fplot(x, y, label=Lleg[id], alpha=alpha_plot, color=Lcols[id], linestyle=Lls[id], lw=lw, marker=Lmarkers[id], markeredgecolor=Lcols[id], markeredgewidth=Lmedw[id], markersize=Lsdots[id], dash_capstyle='round', solid_capstyle='round', dash_joinstyle='round', solid_joinstyle='round', markerfacecolor=Lcols[id], markevery=markevery)
    
    # Setting the limits of the axes
    ax.set_xlim([xmin, xmax])
    ax.set_ylim(y_lim)
    
    # Setting the major and minor ticks on the y-axis 
    ax.set_yticks(yMticks, minor=False)
    ax.set_yticks(ymticks, minor=True)
    ax.set_yticklabels(yMticksLab, minor=False, fontsize=sax)
    
    # Defining the legend
    if add_leg:
        leg = ax.legend(loc=loc_leg, fontsize=sleg, markerfirst=True, fancybox=True, framealpha=alpha_leg, ncol=ncol_leg)
        if zleg is not None:
            leg.set_zorder(zleg)
    
    # Setting the size of the ticks labels on the x axis
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(sax)
    
    # Setting ticks length and width
    ax.tick_params(axis='both', length=lMticks, width=wMticks, which='major')
    ax.tick_params(axis='both', length=lmticks, width=wmticks, which='minor')
    
    # Setting the positions of the labels
    ax.xaxis.set_tick_params(labelright=False, labelleft=True)
    ax.yaxis.set_tick_params(labelright=False, labelleft=True)
    
    # Adding the grids
    if grid:
        ax.xaxis.grid(True, linestyle=grid_ls, which='major', color=grid_col, alpha=grid_alpha)
        ax.yaxis.grid(True, linestyle=grid_ls, which='major', color=grid_col, alpha=grid_alpha)
    ax.set_axisbelow(True)
    
    ax.set_title(tit, fontsize=stit)
    ax.set_xlabel(xlabel, fontsize=sax)
    ax.set_ylabel(ylabel, fontsize=sax)
    plt.tight_layout()
    
    # Saving or showing the figure, and closing
    save_show_fig(fname=fname, f_format=f_format)
    plt.close()

##############################
############################## 
# Demo presenting how to use the main functions of this file.
####################

if __name__ == '__main__':
    print("==============================================")
    print("===== Starting the demo of fast_ms_ne.py =====")
    print("==============================================")
    
    # List of tuples. There is one tuple per considered data set in this demo. The first element of each tuple is a function enabling to load the data set, while the second element of each tuple is a string storing a name for the associated data set. 
    L_data = [(sklearn.datasets.load_digits, 'Digits'), (lambda: sklearn.datasets.make_blobs(n_samples=11000, n_features=12, centers=22, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=3), 'Blobs')]
    n_data = len(L_data)
    
    # Boolean. Whether or not to plot the LD embeddings as they are computed in the demo. 
    plot_emb = True
    
    # Largest neighborhood size to consider when employing the 'red_rnx_auc' function for the reduced quality assessment in the demo. 
    Kup = 5000
    
    # For each data set
    for idx_data, data_t in enumerate(L_data):
        ###
        ###
        ###
        # Load function and name of the current data set
        data_load, data_name = data_t
        print("=====")
        print("==== Data set #{i}/{n_data}: {data_name}".format(i=idx_data+1, n_data=n_data, data_name=data_name))
        print("=====")
        
        ###
        ###
        ###
        print('- Loading the HD data set')
        # TIP: to change the employed data set, you just need to modify the next code line to provide different values for X_hds and labels. Afterwards, only X_hds is employed to compute the LD embeddings. The labels are only used to plot the obtained LD embeddings using colors. 
        
        ################################################

        # D_data = data_load()
        # if isinstance(D_data, dict):
        #     X_hds, labels = D_data['data'], D_data['target']
        # elif isinstance(D_data, tuple):
        #     X_hds, labels = D_data
        # else:
        #     raise ValueError("Error in the demo of module {module_name}: the data set '{data_name}' #{i}/{n_data} cannot be loaded.".format(module_name=module_name, data_name=data_name, i=idx_data+1, n_data=n_data))

        ################################################

        X_hds_1 = np.load('.//SVHN_64_features_1.npy', allow_pickle=True)
        labels_1 = np.load('.//SVHN_labels_1.npy', allow_pickle=True)
        X_hds_2 = np.load('.//SVHN_64_features_2.npy', allow_pickle=True)
        labels_2 = np.load('.//SVHN_labels_2.npy', allow_pickle=True)

        X_hds = np.concatenate((X_hds_1, X_hds_2), axis=0)
        labels = np.concatenate((labels_1, labels_2))
        labels = labels.astype(float)

        ################################################

        # X_hds = np.load('.//MNIST_data.npy', allow_pickle=True)
        # labels = np.load('.//MNIST_labels.npy', allow_pickle=True)
        # labels = np.array(labels).astype(float)

         ################################################

        # X_hds, labels, object_names = load_COIL_20_data(path_data='../coil-20-proc')

         ################################################

        # Number of samples and dimension of the HD data set
        N_samp, M_HD = X_hds.shape
        print("Number of data samples: {N_samp}".format(N_samp=N_samp))
        print("HDS dimension: {M_HD}".format(M_HD=M_HD))
        # Targeted dimension of the LD embeddings
        dim_LDS = 2
        print("Targeted LDS dimension: {dim_LDS}".format(dim_LDS=dim_LDS))
        # Whether the currently considered data set is big in terms of its number of samples or not. 
        big_data = (N_samp >= 10000)
        if big_data:
            print('The data set is big in terms of its number of samples.')
            print('Multi-scale SNE, multi-scale t-SNE and t-SNE are hence not applied; only their fast versions are employed (fast multi-scale SNE, fast multi-scale t-SNE and Barnes-Hut t-SNE).')
            print('The reduced DR quality is evaluated; it means that the R_{NX}(K) curve is computed only for K=1 to Kup={Kup}, and that the AUC refers to the area under this reduced curve, with a log scale for K, instead of the full one for K=1 to N-2={v}, with N being the number of data samples.'.format(Kup=Kup, v=N_samp-2, NX='{NX}'))
        else:
            print('The data set is moderate in terms of its number of samples.')
            print('Multi-scale SNE, multi-scale t-SNE and t-SNE are hence applied, as well as their fast versions (fast multi-scale SNE, fast multi-scale t-SNE and Barnes-Hut t-SNE).')
            print('The DR quality is completely evaluated; the R_{NX}(K) curve is computed for K=1 to N-2={v}, with N being the number of data samples, and the AUC refers to the area under this curve with a log scale for K.'.format(v=N_samp-2, NX='{NX}'))
        print('===')
        print('===')
        print('===')
        
        ###
        ###
        ###
        if not big_data:
            # Function to compute a 2-D numpy array containing the pairwise distances in a data set, if it is not too big in terms of its number of samples. This function is used to compute the HD distances for the DR quality assessment when the data set is of moderate size.
            compute_dist_HD_qa = eucl_dist_matr
            # Function to compute a 2-D numpy array containing the pairwise distances in a data set, if it is not too big in terms of its number of samples. This function is used to compute the LD distances for the DR quality assessment when the data set is of moderate size. Note that in all DR methods employed in this code (multi-scale SNE, multi-scale t-SNE, t-SNE, fast multi-scale SNE, fast multi-scale t-SNE, Barnes-Hut t-SNE), the LD embedding is computed using Euclidean distances in the LD space, independently of the value of compute_dist_LD_qa.
            compute_dist_LD_qa = eucl_dist_matr
        # Lists to provide as parameters to viz_qa, to visualize the DR quality assessment as conducted in [1].
        L_rnx, Lmarkers, Lcols, Lleg_rnx, Lls, Lmedw, Lsdots = [], [], [], [], [], [], []
        
        ###
        ###
        ###
        # If the data set is not too big, we can compute all the pairwise HD distances between its samples. 
        if not big_data:
            print('- Computing the pairwise Euclidean distances in the HD data set')
            t0 = time.time()
            dm_hd = compute_dist_HD_qa(X_hds)
            t = time.time() - t0
            print('Done. It took {t} seconds.'.format(t=rstr(t)))
            print('===')
            print('===')
            print('===')
        
        ###
        ###
        ###
        # Initialization type of the LD embedding. Check the 'init_ld_emb' function for details. Note that you can provide the LD coordinates to use for the initialization by setting init_ld_emb to a 2-D numpy.ndarray containing the initial LD positions, with one example per row and one LD dimension per column, init_ld_emb[i,:] containing the initial LD coordinates related to the HD sample X_hds[i,:].
        init_ld_emb = 'pca'
        
        ###
        ###
        ###
        # Fast multi-scale t-SNE can be employed on very large-scale databases. 
        print('- Applying fast multi-scale t-SNE on the data set to obtain a {dim_LDS}-D embedding'.format(dim_LDS=dim_LDS))
        if data_name == 'Blobs':
            print('This takes a few seconds (i.e., around 32 seconds with a processor Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz 2.21GHz).')
        elif data_name == 'Digits':
            print('This takes a few seconds (i.e., around 3 seconds with a processor Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz 2.21GHz).')
        t0 = time.time()
        X_ld_fmstsne = fmstsne(X_hds=X_hds, n_components=dim_LDS, init=init_ld_emb, rand_state=np.random.RandomState(2), bht=0.75, fseed=1, nit_max=50)
        np.save('C:\\Users\\cedri\\Downloads\\PM_Ms_TSNE_new.npy', X_ld_fmstsne)
        t = time.time() - t0
        print('Done. It took {t} seconds.'.format(t=rstr(t)))
        
        ###
        ###
        ###
        if big_data:
            print('- Evaluating the reduced DR quality of the LD embedding obtained using fast multi-scale t-SNE')
            if data_name == 'Blobs':
                print('This takes a few seconds (i.e., around 34 seconds with a processor Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz 2.21GHz).')
            t0 = time.time()
            rnx_fmstsne, auc_fmstsne = red_rnx_auc(X_hds=X_hds, X_lds=X_ld_fmstsne, Kup=Kup)
            t = time.time() - t0
            print('Done. It took {t} seconds.'.format(t=rstr(t)))
            print('AUC: {v}'.format(v=rstr(auc_fmstsne, 4)))
        else:
            print('- Evaluating the DR quality of the LD embedding obtained using fast multi-scale t-SNE')
            t0 = time.time()
            rnx_fmstsne, auc_fmstsne = eval_dr_quality(d_hd=dm_hd, d_ld=compute_dist_LD_qa(X_ld_fmstsne))
            t = time.time() - t0
            print('Done. It took {t} seconds.'.format(t=rstr(t)))
            print('AUC: {v}'.format(v=rstr(auc_fmstsne, 4)))
        
        # Updating the lists for viz_qa
        L_rnx.append(rnx_fmstsne)
        Lmarkers.append('s')
        Lcols.append('cyan')
        Lleg_rnx.append('FMs $t$-SNE')
        Lls.append('solid')
        Lmedw.append(0.5)
        Lsdots.append(10)


        ###
        ###
        ###
        if plot_emb:
            print('- Plotting the LD embedding obtained using fast multi-scale t-SNE')
            print('If a figure is shown, close it to continue.')
            # TIP: you can save the produced plot by specifying a path for the figure in the fname parameter of the following line. The format of the figure can be specified through the f_format parameter. Check the documentation of the save_show_fig function for more information. 
            viz_2d_emb(X=X_ld_fmstsne, vcol=labels, fname=None, f_format=None, sdot=4)
            print('===')
            print('===')
            print('===')
        
        ###
        ###
        ###
    ###
    ###
    ###
    print('*********************')
    print('***** Done! :-) *****')
    print('*********************')