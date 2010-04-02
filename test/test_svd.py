import numpy
import unittest
from math import sqrt
from csc.divisi.tensor import DictTensor
from csc.divisi.util import nested_list_to_dict
from tensor_util import assertTensorEqual


svd_2d_test_matrix = numpy.zeros((4, 5))
svd_2d_test_matrix[0, 0] = 1
svd_2d_test_matrix[0, 4] = 2
svd_2d_test_matrix[1, 2] = 3
svd_2d_test_matrix[3, 1] = 4

class SVD2DTest(unittest.TestCase):
    def setUp(self):
        self.tensor = DictTensor(2)
        # Note: this command actually puts 20 values in tensor!
        self.tensor.update(nested_list_to_dict(svd_2d_test_matrix))
        self.svd = self.tensor.svd(k=3)
        self.incremental = self.tensor.incremental_svd(k=3, niter=200)
        self.u, self.svals, self.v = self.svd.u, self.svd.svals, self.svd.v

    def test_incremental(self):
        self.assertEqual(self.incremental.u.shape[0], self.tensor.shape[0])
        self.assertEqual(len(self.incremental.svals), self.incremental.u.shape[1])
        self.assertEqual(len(self.incremental.svals), self.incremental.v.shape[1])
        self.assertEqual(self.incremental.v.shape[0], self.tensor.shape[1])

        assertTensorEqual(self.incremental.u,
                               [[0, 0, 1],
                                [0, 1, 0],
                                [0, 0, 0],
                                [1, 0, 0]])

        assertTensorEqual(self.incremental.v,
                               [[0, 0, sqrt(.2)],
                                [1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0],
                                [0, 0, sqrt(.8)]])

        assertTensorEqual(self.incremental.svals,
                               [4, 3, sqrt(5)])

    def test_decomposition(self):
        self.assertEqual(self.u.shape[0], self.tensor.shape[0])
        self.assertEqual(len(self.svals), self.u.shape[1])
        self.assertEqual(len(self.svals), self.v.shape[1])
        self.assertEqual(self.v.shape[0], self.tensor.shape[1])

        assertTensorEqual(self.u,
                               [[0, 0, 1],
                                [0, -1, 0],
                                [0, 0, 0],
                                [-1, 0, 0]], abs=True)

        assertTensorEqual(self.v,
                               [[0, 0, sqrt(.2)],
                                [-1, 0, 0],
                                [0, -1, 0],
                                [0, 0, 0],
                                [0, 0, sqrt(.8)]], abs=True)

        assertTensorEqual(self.svals,
                               [4, 3, sqrt(5)])

    def test_reconstructed(self):
        assertTensorEqual(self.svd.reconstructed,
                               [[1, 0, 0, 0, 2],
                                [0, 0, 3, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 4, 0, 0, 0]])
        assertTensorEqual(self.svd.reconstructed[1,:],
                                [0, 0, 3, 0, 0])
        assertTensorEqual(self.svd.reconstructed[:,2],
                               [0, 3, 0, 0])

    def test_orthonormality(self):
        identity = [[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]]
        assertTensorEqual(self.u.T * self.u,
                               identity)

        assertTensorEqual(self.v.T * self.v,
                               identity)

    def test_variance(self):
        # Assert that the SVD explained some of the variance.
        diff_k3 = self.tensor - self.svd.reconstructed
        tensor_mag = self.tensor.magnitude()
        diff_k3_mag = diff_k3.magnitude()
        self.assert_(tensor_mag > diff_k3_mag)

        # Check that a smaller SVD explains less of the variance, but still some.
        svd_k1 = self.tensor.svd(k=1)
        diff_k1 = self.tensor - svd_k1.reconstructed
        diff_k1_mag = diff_k1.magnitude()
        self.assert_(tensor_mag > diff_k1_mag > diff_k3_mag)
