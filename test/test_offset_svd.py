'''
Test the SVD with row and column offsets.
'''

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

# Fake some offsets.
svd_2d_test_matrix[0, :] -= 1
svd_2d_test_matrix[1, :] += 3
svd_2d_test_matrix[2, :] += 5
svd_2d_test_matrix[:, 1] -= 4
svd_2d_test_matrix[:, 3] += 2
offset_for_row = [1, -3, -5, 0]
offset_for_col = [0, 4, 0, -2, 0]

class SVD2DTest(unittest.TestCase):
    def setUp(self):
        self.tensor = DictTensor(2)
        # Note: this command actually puts 20 values in tensor!
        self.tensor.update(nested_list_to_dict(svd_2d_test_matrix))
        self.svd = self.tensor.svd(k=3, offset_for_row=offset_for_row, offset_for_col=offset_for_col)
        self.u, self.svals, self.v = self.svd.u, self.svd.svals, self.svd.v


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
