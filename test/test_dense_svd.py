from csc.divisi.tensor import DenseTensor, data
import numpy as np

def test_dense_svd():
    dense = DenseTensor(np.random.random(size=(10, 15)))
    sparse = dense.to_sparse()

    print repr(data(dense.svd().u))
    dense_svd_u = np.abs(data(dense.svd().u))
    sparse_svd_u = np.abs(data(sparse.svd().u))

    assert np.all(np.abs(dense_svd_u - sparse_svd_u) < 0.0001)
