import numpy as np
import random
from csc.divisi.tensor import Tensor, DenseTensor, data
from csc.divisi.labeled_view import LabeledView
from csc.util.persist import get_picklecached_thing

def _getLandmarkPoints(size):
    numpoints = int(np.sqrt(size))
    return random.sample(xrange(size), numpoints)

def normalize(matrix):
    norms = np.sqrt(np.sum(matrix * matrix, axis=-1))
    return matrix / norms[:,np.newaxis]

def compute_distances(matrix1, matrix2):
    nmatrix1 = normalize(matrix1)
    nmatrix2 = normalize(matrix2)
    distances = np.arccos(np.maximum(-1, np.minimum(1, np.dot(nmatrix1, nmatrix2.T))))
    return distances * distances

def mds(matrix, k=2):
    if isinstance(matrix, Tensor):
        matrix = data(matrix.to_dense())

    # Find Landmark points
    N = matrix.shape[0]
    landmarks = _getLandmarkPoints(N)
    num_landmarks = len(landmarks)
    
    landmark_matrix = matrix[landmarks]
    sqdistances = compute_distances(landmark_matrix, landmark_matrix)

    # Do normal MDS on landmark points
    means = np.mean(sqdistances, axis=1)      # this is called mu_n in the paper
    global_mean = np.mean(means)              # this is called mu in the paper

    # this is called B in the paper
    distances_balanced = -(sqdistances - means[np.newaxis,:] - means[:,np.newaxis] + global_mean)/2

    # find the eigenvectors and eigenvalues with our all-purpose hammer
    # for the paper, Lambda = lambda, Q = V
    Q, Lambda, Qt = np.linalg.svd(distances_balanced)
    k = min(k, len(Lambda))
    mdsarray_sharp = Q[:,:k]                       # called L^sharp transpose in the paper
    mdsarray = mdsarray_sharp * np.sqrt(Lambda)[np.newaxis,:k]  # called L transpose in the paper

    # Make Triangulation Object
    return MDSProjection(landmark_matrix, mdsarray_sharp, means)

class MDSProjection(object):
    def __init__(self, landmarks, mdsarray_sharp, means):
        self.landmarks = landmarks
        self.mdsarray_sharp = mdsarray_sharp
        self.means = means
        self.N, self.k = self.mdsarray_sharp.shape
    def project(self, vector):
        # vector can in fact be a matrix of many vectors

        # Dimensions:
        # vector = (m x ndim) or possibly just (ndim),
        #   with ndim = K from the SVD
        # dist = (m x N)
        # means = (N)
        # mdsarray_sharp = (N x k)
        dist = (compute_distances(vector, self.landmarks) - self.means)/2
        return np.dot(dist, -self.mdsarray_sharp)

def aspace_mds():
    from csc.conceptnet.analogyspace import conceptnet_2d_from_db
    cnet = conceptnet_2d_from_db('en')
    aspace = cnet.normalized().svd(k=100)
    labels = cnet.label_list(0)

    ptmatrix = data(aspace.u)
    ptmatrix *= data(aspace.svals)

    proj = mds(ptmatrix)

    result = proj.project(data(aspace.u))
    return LabeledView(DenseTensor(result), [labels, None])

if __name__ == '__main__':
    aspace_mds = get_picklecached_thing("/Users/rspeer/code/luminoso/mds.pickle", aspace_mds)

