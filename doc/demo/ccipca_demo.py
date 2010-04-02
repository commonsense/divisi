from csc.divisi.ccipca import CCIPCA, zerovec
from csc.divisi.labeled_view import make_dense_labeled_tensor
from random import random
from math import sqrt
import numpy

VEC_LEN = 16
VEC_COUNT = 4
ITERS = 1024
MAG_MAX = 100.0
K=6

# Prepare test vectors
vecs = [numpy.array([random() for i in range(VEC_LEN)]) for j in range(VEC_COUNT)]
vecs = [make_dense_labeled_tensor(v, None).to_sparse() for v in vecs]
mags = sorted([MAG_MAX * random() for i in range(VEC_COUNT)])

print "Vectors prepared:"
vecs = [v.hat() for v in vecs]
for v in vecs: print v.values()

print "\nMags selected:"
print mags
print "\n"

# Prepare the CCIPCA instance
ccipca = CCIPCA(k=K)

# Prepare samples
for i in range(ITERS):
    w = [random() * mags[j] for j in range(VEC_COUNT)]
    v = zerovec()
    for weight, vec in zip(w,vecs):
        v += vec * weight
    print "Training with vector: ", v.values()

    # Train
    #mags_res = ccipca.iteration([v], learn=True)
    #print "Mags:", mags_res
    v_out = ccipca.smooth(v, k_max=(VEC_COUNT+1), learn=True)
    v_err = v - v_out
    print v_err.norm()
    print "Err%:", 100.0 * v_err.norm() / v.norm()

# Show eigenvectors
print "\nGot eigenvectors:\n"
for v in ccipca._v: print v.values()

# Show eigenvector magnitudes
print "\nEigenvector magnitudes:\n"
print [v.norm() for v in ccipca._v]
