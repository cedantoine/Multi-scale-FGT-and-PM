#! python3
# -*-coding:Utf-8 -*

########################################################################################################
########################################################################################################

import scipy.misc, numpy as np, imageio.v2 as imageio

def pseudo_random_permutation(stop, start=0, step=1, dtype=np.int64, seed=1):
    """
    This function allows to create a pseudo random permutation of np.arange(start=start, step=step, stop=stop, dtype=dtype). With fixed seed, the permutation remains the same. 
    Useful to permute the rows of a data set in a 'random' order, while keeping the same permutation at each run of the program.
    np.random.seed(seed) could be used, but this would cause the subsequent calls to np.random to be fixed. Instead, we instantiate the class np.random.RandomState(seed) and use it to randomly shuffle the data set. Doing so, the permutation of the rows of the data set will remain the same at each run with fixed seed, but the subsequent calls to np.random will not be affected and will remain pseudo-random.
    
    In:
    - start, step, stop, dtype: arguments of np.arange
    - seed: seed value 
    Out:
    A permutation of np.arange(start=start, step=step, stop=stop, dtype=dtype).
    """
    perm = np.arange(start=start, step=step, stop=stop, dtype=dtype)
    np.random.RandomState(seed).shuffle(perm)
    return perm

def load_COIL_20_data(path_data, nobj=20, nim=72):
    """    
    Load the processed COIL-20 data set. 
    
    This function assumes that there exists a folder '{p}/coil-20-proc'.format(p=path_data) containing the COIL-20 image bank which can be downloaded on https://cave.cs.columbia.edu/repository/COIL-20 (consulted on Feb 23, 2023).
    These files are assumed to be the same as in the archive which can be downloaded on this website. 
    
    In:
    - path_data: string indicating the path of the data set. 
    - nobj: number of object to consider in the data set. The first nobj objects are considered. If nobj = 20, then all objects are considered. If nobj is < 1, > 20 or if it is not an integer, an error is raised.
    - nim: number of versions of each object to consider in the data set. The first nim versions are considered. If nobj = 72, then all versions of the objects are considered. If nobj is < 1, > 72 or if it is not an integer, an error is raised.
    Out:
    A tuple, with:
    - X: a numpy.ndarray of floats with shape (nobj*nim, 128**2). It contains one image per row. It stores an image in a row by packing it in the vector row by row. The images corresponding to the same objects are not necessarily in consecutive lines in X.
    - t: a numpy.array of int, with size nobj*nim, giving the labels of the images (the label of an image is the number of the corresponding object in the COIL-20 image bank). Element i of this array gives the label (int between 0 and nobj-1) of the image stored at the ith row of the data set stored in X. 
    - object_names: a nobj-elements list of strings which gives meaningful names to the numeric labels in t. Element i gives the name of the objects with numeric label i.
    """
    if (nobj < 1) or (nobj > 20) or (not isinstance(nobj, int)):
        raise ValueError("Error in function load_COIL_20_data: nobj must be an integer between 1 and 20.")
    
    if (nim < 1) or (nim > 72) or (not isinstance(nim, int)):
        raise ValueError("Error in function load_COIL_20_data: nim must be an integer between 1 and 72.")
    
    seed_shuffle_coil20_data = 5
    
    # Number of examples in the data set
    nex = nim * nobj
    # The images have 128 x 128 pixels
    npx = 128
    # Dimension of each image 
    dim = npx**2
    
    # Hard-coding the object names
    object_names = ['duck', 'form', 'car', 'cat', 'anacin', 'car2', 'form2', 'powder', 'tylenol', 'vaseline', 'mushroom', 'cup', 'pig', 'cylinder', 'vase', 'conditioner', 'cup2', 'cup3', 'car3', 'philadelphia']
    
    # Allocating output matrix 
    X = np.zeros(shape=(nex, dim), dtype=np.float64)
    
    # Labels
    t = np.zeros(shape=nex, dtype=np.int64)
    
    # Loading the images 
    for i in range(nobj):
        for j in range(nim):
            X[i*nim+j,:]=imageio.imread('{p}/coil-20-proc/obj{ob}__{nr}.png'.format(p=path_data, ob=i+1, nr=j)).astype(np.float64).reshape(dim)
        t[i*nim:(i+1)*nim] = i
    
    # Shuffling the data set so that consecutive rows are not necessarily related to the same image
    perm = pseudo_random_permutation(stop=nex, start=0, step=1, dtype=np.int64, seed=seed_shuffle_coil20_data)
    X = X[perm]
    t = t[perm]
    
    return X, t, object_names[:nobj]

def load_COIL_20_data_test():
    """
    Simple test of load_COIL_20_data function.
    """
    print("==== load_COIL_20_data_test ====")
    X, t, object_names = load_COIL_20_data(path_data='./coil-20-proc')
    
    print("Data set: ", X)
    print("Labels : ", t)
    print("Data set shape: ", X.shape)
    print("Labels shape : ", t.shape)
    print("Object names  : ", object_names)
    print("Are the data all equal 0? ", np.all(np.isclose(X,0)))
    print("Are the labels all equal 0? ", np.all(np.isclose(t,0)))
    print("Type of the data: ", type(X), ', ', X.dtype)
    print("Type of the labels: ", type(t), ', ', t.dtype)

# load_COIL_20_data_test()
