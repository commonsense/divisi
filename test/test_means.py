import numpy as np
import unittest
from nose.tools import eq_, raises
from math import sqrt
from csc.divisi.tensor import DictTensor
from csc.divisi.util import nested_list_to_dict
from tensor_util import assertTensorEqual, zeros_removed

data = np.array([[1, 2, 3, 4],
                 [-1,2, 3, 4],
                 [0, 1, -1,0]])

tensor = DictTensor(2)
tensor.update(zeros_removed(nested_list_to_dict(data)))
eq_(len(tensor), 10)

# For NumPy, "along an axis" means something different.
ms_data = data - data.mean(1)[:,np.newaxis]
ms_tensor = DictTensor(2)
ms_tensor.update(nested_list_to_dict(ms_data))

def test_means():
    means = tensor.means()
    eq_(len(means), 2)
    assert np.allclose(means[0], [(1+2+3+4)/4., (-1+2+3+4)/4., (0+1+-1+0)/4.])
    assert np.allclose(means[1], [0, (2+2+1)/3., (3+3-1)/3., (4+4+0)/3.])

def test_mean_subtracted():
    mean_subtracted = tensor.mean_subtracted()
    m = np.zeros(data.shape)
    for (r, c), v in mean_subtracted.iteritems():
        m[r, c] = v
    assert np.allclose(m, ms_data)
    assert np.allclose(m.mean(1), 0)

def test_mean_subtracted_svd():
    plain_svd = ms_tensor.svd(k=2)
    offset_svd = tensor.mean_subtracted().svd(k=2)
    assert np.allclose(plain_svd.u.unwrap(), offset_svd.u.unwrap())
    assert np.allclose(plain_svd.v.unwrap(), offset_svd.v.unwrap())
    assert np.allclose(plain_svd.svals.unwrap(), offset_svd.svals.unwrap())

@raises(TypeError)
def test_blend_mean_subtracted():
    from csc.divisi.blend import Blend
    Blend([tensor.mean_subtracted()])
