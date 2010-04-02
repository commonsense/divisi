from csc.divisi import *
t = SparseLabeledTensor(ndim=2)

# Fruit    property
t['apple',      'red']    = 1
t['apple',      'orange'] = -1
t['strawberry', 'red']    = 1
t['strawberry', 'sweet'] = 1
t['orange',     'orange'] = 1
t['orange',     'citrus'] = 1
t['orange',     'sour'] = 1
t['lemon',      'citrus'] = 1
t['lemon',      'sour'] = 2
t['lemon',      'sweet'] = -1
t['lemon',      'yellow'] = 1
#t['apple', 'sweet'] = 1

# Compute SVD
svd_result = t.svd(k=2)

rec = svd_result.u * svd_result.core * svd_result.v.T

print rec['apple', 'sweet']
print rec['apple', 'citrus']
print rec['strawberry', 'sour']

similar_to_apple = svd_result.u['apple',:]* (svd_result.core*svd_result.core) * svd_result.u.T
dict(similar_to_apple)

from csc.divisi.cnet import *
concept_similarity(svd_result, 'apple').top_items()
