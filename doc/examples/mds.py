import orange
import orngMDS
import numpy as np
from math import acos as _acos
from csc.divisi.tensor import DenseTensor
from csc.divisi.view import LabeledView

from csc.util.persist import get_picklecached_thing
cnet = get_picklecached_thing('cnet.pickle.gz')
aspace = cnet.normalized().svd()
n = aspace.u.shape[0]

wut = aspace.weighted_u.tensor
vecs = (wut[i,:] for i in xrange(n))
normalized_vecs = [vec.hat() for vec in vecs]

def acos(x):
    if x > 1: return _acos(1)
    if x < -1: return _acos(-1)
    return _acos(x)

concept_labels = aspace.weighted_u.label_list(0)

print 'dist'
distance = orange.SymMatrix(n)
for i in range(n):
    for j in range(i+1):
        distance[i, j] = acos(normalized_vecs[i]*normalized_vecs[j])

print 'setup'
mds=orngMDS.MDS(distance)
print 'run'
mds.run(100) 

def get_mds_matrix():
    array = np.array(mds.points)
    matrix = LabeledView(DenseTensor(array), [aspace.u.label_list(0), None])
    return matrix

mds_matrix = get_picklecached_thing('aspace_mds.pickle', get_mds_matrix)

