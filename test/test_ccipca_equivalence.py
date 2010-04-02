from csc.divisi.forgetful_ccipca import CCIPCA
import numpy as np

def normalize(array):
    return array / (np.sqrt(np.sum(array*array, axis=1))[:,np.newaxis])

def rmse(array):
    return np.sqrt(np.sum(array*array)/array.shape[0]/array.shape[1])

def test_ccipca_equivalence():
    array = np.random.random(size=(10,10))
    col_mean = np.mean(array, axis=1)
    row_mean = np.mean(array, axis=0)
    overall_mean = np.mean(array)
    zero_mean_array = array - col_mean[:, np.newaxis] - row_mean[np.newaxis,:] + overall_mean
    print zero_mean_array
    print np.mean(zero_mean_array, axis=0)
    print np.mean(zero_mean_array, axis=1)

    norm_array = normalize(zero_mean_array)

    U, Sigma, Vt = np.linalg.svd(norm_array)
    
    ccipca = CCIPCA(6, vector_size=10, amnesia=3.0)
    for iteration in xrange(500):
        for column in xrange(10):
            ccipca.fixed_iteration(norm_array[:,column], learn=True)

        a = ccipca._v[1:6] * 10
        b = (U * Sigma * Sigma).T[:5]
        print (np.abs(a/b))
        print Sigma[:5]
        print rmse(np.abs(a/b)-1)

test_ccipca_equivalence()

