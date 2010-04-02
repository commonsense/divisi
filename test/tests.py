import unittest
from csc.divisi.tensor import DictTensor, DenseTensor
from csc.divisi.normalized_view import NormalizedView, TfIdfView
from csc.divisi.labeled_view import LabeledView, make_sparse_labeled_tensor
from csc.divisi.ordered_set import OrderedSet, IdentitySet
from csc.divisi.unfolded_set import UnfoldedSet
from csc.divisi.labeled_tensor import SparseLabeledTensor
from csc.divisi.util import nested_list_to_dict
from math import sqrt

import numpy
import random

from tensor_util import assert_dims_consistent, assertTensorEqual, assertTensorEqualDict, nones_removed

class DictTensorTest(unittest.TestCase):
    slice_testcase = [[1,    None, None],
                       [None, 2,    3   ],
                       [4,    None, None],
                       [None, 5,    None]]

    def test_initial(self):
        self.assertEqual(len(self.tensor), 0)
        self.assertEqual(len(self.tensor.keys()), 0)
        assert_dims_consistent(self.tensor)
        self.assertEqual(self.tensor.shape, (0, 0))
        assert isinstance(self.tensor[4, 5], (float, int, long))
        self.assertEqual(self.tensor[5, 5], 0)
        self.assertEqual(self.tensor[2, 7], 0)

    def test_storage(self):
        self.tensor[5, 5] = 1
        self.tensor[2, 7] = 2

        assertTensorEqual(self.tensor,
                          [[None]*8,
                           [None]*8,
                           [None]*7 + [2],
                           [None]*8,
                           [None]*8,
                           [None]*5 + [1, None, None]])

    def test_slice(self):
        self.tensor.update(nones_removed(nested_list_to_dict(self.slice_testcase)))

        # Test end conditions: start index
        # is included in slice, end index is not
        slice = self.tensor[1:3, 0:2]
        assertTensorEqual(slice,
                          [[None, 2],
                           [4, None]])

        # Test that slicing on some dims correctly
        # reduces the dimensionality of the tensor
        slice = self.tensor[3, :]
        assertTensorEqual(slice, [None, 5, None])

        # Test the step parameter
        slice = self.tensor[1:4:2, :]
        assertTensorEqual(slice,
                               [[None, 2, 3],
                                [None, 5, None]])

    def test_transpose(self):
        self.tensor[0, 0] = 1
        self.tensor[1, 2] = 3
        self.tensor[2, 0] = 4
        self.tensor[3, 1] = 5

        t = self.tensor.transpose()
        assertTensorEqual(t,
                          [[1, None, 4, None],
                           [None, None, None, 5],
                           [None, 3, None, None]])

    def test_delete(self):
        self.tensor.update(nones_removed(nested_list_to_dict(self.slice_testcase)))
        assertTensorEqual(self.tensor, self.slice_testcase)

        del self.tensor[0,0]
        assertTensorEqual(self.tensor,
                               [[None, None, None],
                                [None, 2,    3   ],
                                [4,    None, None],
                                [None, 5,    None]])

    def test_contains(self):
        self.tensor[1,2] = 1
        self.tensor[4,5] = 2
        self.assertTrue((1,2) in self.tensor)
        self.assertTrue(self.tensor.has_key((1,2)))
        self.assertFalse((4,2) in self.tensor)
        self.assertFalse((1,5) in self.tensor)


    def setUp(self):
        self.tensor = DictTensor(2)

    def test_1D(self):
        tensor_1D = DictTensor(1)
        tensor_1D[2] = 1

        assertTensorEqual(tensor_1D,
                               [None, None, 1])

    def test_combine_by_element(self):
        t1 = DictTensor(2)
        t2 = DictTensor(2)
        t1[1, 1] = 1
        t1[1, 0] = 2
        t2[1, 1] = 4
        t2[0, 1] = 5

        t3 = t1.combine_by_element(t2, lambda x, y: x + (2*y))
        assertTensorEqual(t3,
                               [[None, 10],
                                [2, 9]])

        # Make sure errors are raised when the tensors don't have the
        # same shape or number of dimensions
        t4 = DictTensor(2)
        t4[0, 2] = 3
        t4[1, 0] = 5
        self.assertRaises(IndexError, lambda: t1.combine_by_element(t4, lambda x, y: x + y))
        t4 = DictTensor(3)
        self.assertRaises(IndexError, lambda: t1.combine_by_element(t4, lambda x, y: x + y))

    def testAdd(self):
        t1 = DictTensor(2)
        t2 = DictTensor(2)
        t1[0, 0] = 1
        t1[1, 1] = 1
        t1[1, 0] = 2
        t2[2, 1] = 4
        t2[1, 0] = 5

        t3 = t1 + t2
        assertTensorEqual(t3,
                               [[1, None],
                                [7, 1],
                                [None, 4]])

    def testICmul(self):
        t1 = tensor_from_nested_list([[1, 2], [3, 4]])
        assertTensorEqual(t1, [[1, 2], [3, 4]])
        t1 *= 2
        assertTensorEqual(t1, [[2, 4], [6, 8]])

    def testICdiv(self):
        t1 = tensor_from_nested_list([[2, 4], [6, 8]])
        t1 /= 2
        assertTensorEqual(t1, [[1, 2], [3, 4]])

    def testReprOfEmpty(self):
        repr(self.tensor)
        self.tensor.example_key()

    def testNorm(self):
        norm_test = [[0,0,0],
                    [0,1,0],
                    [0,5.0,0]]
        self.tensor.update(nested_list_to_dict(norm_test))
        self.assertEqual(self.tensor.norm(), sqrt(26.0))
        self.assertEqual(self.tensor.magnitude(), sqrt(26.0))


from numpy import zeros
class DenseTensorTest(unittest.TestCase):
    def setUp(self):
        self.tensor = DenseTensor(zeros((3,4)))

    def test_initial(self):
        self.assertEqual(len(self.tensor), 3*4)
        self.assertEqual(len(self.tensor.keys()), 3*4)
        assert_dims_consistent(self.tensor)
        self.assert_(isinstance(self.tensor[2, 3], (float, int, long)))
        self.assertEqual(self.tensor.shape, (3, 4))
        self.assertEqual(self.tensor[1, 2], 0)
        self.assertEqual(self.tensor[0, 3], 0)

    def test_container(self):
        self.tensor[0, 0] = 1
        self.tensor[2, 3] = 2
        self.tensor[0, -1] = 3

        assertTensorEqual(self.tensor,
                               [[1, 0, 0, 3],
                                [0, 0, 0, 0],
                                [0, 0, 0, 2]])


class LabeledDictTensorTest(unittest.TestCase):
    def setUp(self):
        self.tensor = make_sparse_labeled_tensor(2)

    def test_storage(self):
#        self.assertEqual(len(self.tensor), 0)
        self.assertEqual(len(self.tensor.keys()), 0)
        self.assertEqual(self.tensor.ndim, 2)
        self.assertEqual(self.tensor.shape, (0, 0))

        # Unknown keys are 0 (or default_value).
        self.assertEqual(self.tensor['banana', 'yellow'], 0)
        self.assertEqual(self.tensor['apple', 'red'], 0)

        self.tensor['banana', 'yellow'] = 1
        self.tensor['apple', 'red'] = 2
        self.tensor['apple', 'blue'] = 3
        self.tensor['orange', 'yellow'] = 4

        expected = {('banana', 'yellow') : 1,
                    ('apple', 'red') : 2,
                    ('apple', 'blue') : 3,
                    ('orange', 'yellow') : 4}
        assertTensorEqualDict(self.tensor, expected)


    def test_slicing(self):
        for row in ['x', 'y', 'z']:
            for col in ['a', 'b']:
                self.tensor[row, col] = ord(row)*100+ord(col)

        slice = self.tensor['x', :]
        self.assertEqual(slice.shape, (2,))
        self.assertEqual(slice['a'], ord('x')*100+ord('a'))

        slice = self.tensor.slice(0, 'x')
        self.assertEqual(slice.shape, (2,))
        self.assertEqual(slice['a'], ord('x')*100+ord('a'))

        slice = self.tensor[:, 'a']
        self.assertEqual(slice.shape, (3,))
        self.assertEqual(slice['x'], ord('x')*100+ord('a'))

        slice = self.tensor.slice(1, 'a')
        self.assertEqual(slice.shape, (3,))
        self.assertEqual(slice['x'], ord('x')*100+ord('a'))


    def test_wrapped(self):
        self.tensor['a', '1'] = 2
        self.assertEqual(self.tensor['a','1'], 2)

        indices = self.tensor.indices(('a','1'))
        self.assertEqual(self.tensor.tensor[indices], 2)

        self.assert_(self.tensor._label_dims_correct())

    def test_contains(self):
        self.tensor['1','2'] = 1
        self.tensor['4','5'] = 2
        self.assertTrue(('1','2') in self.tensor)
        self.assertTrue(self.tensor.has_key(('1','2')))
        self.assertFalse(('4','2') in self.tensor)
        self.assertFalse(('1','5') in self.tensor)
        self.assertFalse(self.tensor.has_key(('1','5')))


    def test_add(self):
        t1 = make_sparse_labeled_tensor(2)
        t2 = make_sparse_labeled_tensor(2)

        t1['cat', 'mammal'] = 1
        t1['cat', 'pet'] = 1
        t1['panda', 'mammal'] = 1
        t2['cat', 'pet'] = 1
        t2['bear', 'pet'] = -1

        t3 = t1 + t2
        # FIXME: FINISH THIS!


class TfIdfTest(unittest.TestCase):
    def test(self):
        '''Run the testcase from the Wikipedia article (in comments)'''
        tensor = DictTensor(2)
        # Consider a document containing 100 words wherein the word cow appears 3 times.
        # [specifically, let there be a document where 'cow' appears 3 times
        #  and 'moo' appears 97 times]
        doc = 0
        cow = 1
        moo = 2
        tensor[cow, doc] = 3
        tensor[moo, doc] = 97
        # Following the previously defined formulas, the term frequency (TF) for cow is then 0.03 (3 / 100).
        tfidf = TfIdfView(tensor) # (can't create it earlier b/c it's read-only)
        self.assertEqual(tfidf.counts_for_document[doc], 100)
        self.assertAlmostEqual(tfidf.tf(cow, doc), 0.03)

        # Now, assume we have 10 million documents and cow appears in one thousand of these.
        #  [specifically, let 'cow' appear in documents 0 and 10,000,000-1000+1 till 10,000,000
        for doc in xrange(10000000-1000+1,10000000):
            tensor[cow, doc] = 1

        # Then, the inverse document frequency is calculated as ln(10 000 000 / 1 000) = 9.21.
        tfidf = TfIdfView(tensor) # (have to update after adding the other docs)
        self.assertEqual(tfidf.num_documents, 10000000)
        self.assertEqual(tfidf.num_docs_that_contain_term[cow], 1000)
        self.assertAlmostEqual(tfidf.idf(cow), 9.21, 2)

        # The TF-IDF score is the product of these quantities: 0.03 * 9.21 = 0.28.
        score = tfidf[cow, 0]
        self.assertEqual(len(getattr(score, 'shape', ())), 0)
        self.assertAlmostEqual(score, 0.28, 2)

    def test_transposed(self):
        '''Run the same testcase, but with the matrix transposed.'''
        tensor = DictTensor(2)
        # Consider a document containing 100 words wherein the word cow appears 3 times.
        # [specifically, let there be a document where 'cow' appears 3 times
        #  and 'moo' appears 97 times]
        doc = 0
        cow = 1
        moo = 2
        tensor[doc, cow] = 3
        tensor[doc, moo] = 97
        # Following the previously defined formulas, the term frequency (TF) for cow is then 0.03 (3 / 100).
        tfidf = TfIdfView(tensor, transposed=True) # (can't create it earlier b/c it's read-only)
        self.assertEqual(tfidf.counts_for_document[doc], 100)
        self.assertAlmostEqual(tfidf.tf(cow, doc), 0.03)

        # Now, assume we have 10 million documents and cow appears in one thousand of these.
        #  [specifically, let 'cow' appear in documents 0 and 10,000,000-1000+1 till 10,000,000
        for doc in xrange(10000000-1000+1,10000000):
            tensor[doc, cow] = 1

        # Then, the inverse document frequency is calculated as ln(10 000 000 / 1 000) = 9.21.
        tfidf = TfIdfView(tensor, transposed=True) # (have to update after adding the other docs)
        self.assertEqual(tfidf.num_documents, 10000000)
        self.assertEqual(tfidf.num_docs_that_contain_term[cow], 1000)
        self.assertAlmostEqual(tfidf.idf(cow), 9.21, 2)

        # The TF-IDF score is the product of these quantities: 0.03 * 9.21 = 0.28.
        score = tfidf[0, cow]
        self.assertEqual(len(getattr(score, 'shape', ())), 0)
        self.assertAlmostEqual(score, 0.28, 2)

class UnfoldedSparseTensorTest(unittest.TestCase):
    def setUp(self):
        self.raw = DictTensor(3)
        for x1 in range(2):
            for x2 in range(3):
                for x3 in range(4):
                    self.raw[x1, x2, x3] = x1*100+x2*10+x3

    def test_unfold0(self):
        uf = self.raw.unfolded(0)
        self.assertEqual(uf.shape, (2, 3*4))
        self.assertEqual(len(uf), 2*3*4)
        for x1 in range(2):
            for x2 in range(3):
                for x3 in range(4):
                    self.assertEqual(uf[x1, (x2, x3)], x1*100+x2*10+x3)

    def test_unfold1(self):
        uf = self.raw.unfolded(1)
        self.assertEqual(uf.shape, (3, 2*4))
        for x1 in range(2):
            for x2 in range(3):
                for x3 in range(4):
                    self.assertEqual(uf[x2, (x1, x3)], x1*100+x2*10+x3)

    def test_unfold2(self):
        uf = self.raw.unfolded(2)
        self.assertEqual(uf.shape, (4, 2*3))
        for x1 in range(2):
            for x2 in range(3):
                for x3 in range(4):
                    self.assertEqual(uf[x3, (x1, x2)], x1*100+x2*10+x3)

    def test_compact0(self):
        uf = self.raw.unfolded(0)
        compact = DictTensor(2)
        uf.compact_to(compact)
        self.assertEqual(len(compact), 2*3*4)
        for x1 in range(2):
            for x2 in range(3):
                for x3 in range(4):
                    self.assertEqual(compact[x1, x2*4+x3], x1*100+x2*10+x3)


class UnfoldErrors(unittest.TestCase):
# 1D is just stupid, not an error.
#    def test_1D(self):
#        self.assertRaises(IndexError, lambda: DictTensor(1).unfolded(0))

    def test_oob(self):
        self.assertRaises(IndexError, lambda: DictTensor(3).unfolded(3))


class UnfoldedSetTests(unittest.TestCase):
    def setUp(self):
        self.sets = [
            OrderedSet([2,4,6,8,10]),      # 5 items
            IdentitySet(10),               # 10 items
            OrderedSet(['a','b','c','d']), # 4 items
            ]

    def test_index(self):
        uset = UnfoldedSet(self.sets)
        self.assertEqual(uset.index((2, 0, 'a')), 0)
        self.assertEqual(uset.index((2, 0, 'b')), 1)
        self.assertEqual(uset.index((2, 1, 'a')), 4)
        self.assertEqual(uset.index((4, 0, 'a')), 40)

    def test_label(self):
        uset = UnfoldedSet(self.sets)
        self.assertEqual(uset[0], (2, 0, 'a'))
        self.assertEqual(uset[1], (2, 0, 'b'))
        self.assertEqual(uset[4], (2, 1, 'a'))
        self.assertEqual(uset[40], (4, 0, 'a'))

    def test_len(self):
        uset = UnfoldedSet(self.sets)
        self.assertEqual(len(uset), 5*10*4)

    def test_from_unfolding(self):
        u0 = UnfoldedSet.from_unfolding(0, self.sets)
        self.assertEqual(u0[0], (0, 'a'))
        self.assertEqual(u0[1], (0, 'b'))
        self.assertEqual(u0[4], (1, 'a'))

        u1 = UnfoldedSet.from_unfolding(1, self.sets)
        self.assertEqual(u1[0], (2, 'a'))
        self.assertEqual(u1[1], (2, 'b'))
        self.assertEqual(u1[4], (4, 'a'))

        u2 = UnfoldedSet.from_unfolding(2, self.sets)
        self.assertEqual(u2[0], (2, 0))
        self.assertEqual(u2[1], (2, 1))
        self.assertEqual(u2[10], (4, 0))


class TestMult(unittest.TestCase):
    '''Tests that DenseTensors and DictTensors behave identically for multiplication.'''
    def rand_pair(self):
        leftdim = random.randrange(1,30)
        innerdim = random.randrange(1,30)
        rightdim = random.randrange(1,30)
        dense1 = DenseTensor(numpy.random.random((leftdim, innerdim)))
        dense2 = DenseTensor(numpy.random.random((innerdim, rightdim)))
        return dense1, dense2

    def test_denseprod(self):
        for i in range(5):
            # Create pairs of arrays
            dense1, dense2 = self.rand_pair()
            result = dense1 * dense2
            self.assertEqual(result.shape, (dense1.shape[0], dense2.shape[1]))

    def test_tensordot(self):
        if True: # FIXME XXX: skip this test.
            return
        # Test degenerate case of two 1-d vectors
        t1 = DictTensor(ndim=1)
        t2 = DictTensor(ndim=1)
        t1[0] = 1
        t1[2] = 2
        t2[0] = 3
        t2[1] = 4
        t2[2] = 5
        self.assertEqual(13, t1.tensordot(t2, 0))
        self.assertEqual(13, t1.tensordot(t2.to_dense(), 0))
        self.assertEqual(13, t1.to_dense().tensordot(t2, 0))
        self.assertEqual(13, t1.to_dense().tensordot(t2.to_dense(), 0))

        for i in range(5):
            # Make a random, randomly-shaped 3D tensor
            shape = random.sample(xrange(1,30), 3)
            tensor = DenseTensor(numpy.random.random(shape))

            # Pick a random one of those dimensions
            dim = random.randrange(3)

            # Make a random vector of that length
            vec = DenseTensor(numpy.random.random((shape[dim],)))

            # Try the dense result
            result = tensor.tensordot(vec, dim)

            self.assertEqual(result.shape, tuple(shape[:dim]+shape[dim+1:]))

            # Try it with the tensor being sparse.
            sparset = tensor.to_sparse()
            result_s = sparset.tensordot(vec, dim)
            self.assertEqual(result_s.shape, result.shape)
            for key, val in result.iteritems():
                self.assertAlmostEqual(val, result_s[key])

class NormalizeTest(unittest.TestCase):
    data = [[-5, 0, 0, 0],
            [0, 1, -2, 3],
            [100, 2, 0, 0]]

    def setUp(self):
        self.tensor = DenseTensor(zeros((3, 4)))
        self.tensor.update(nested_list_to_dict(self.data))

        self.normalized = NormalizedView(self.tensor, mode=0)

        self.randomtensor = DenseTensor(numpy.random.normal(size=(5, 8)))
        self.randomnormal_0 = NormalizedView(self.randomtensor, mode=0)
        self.randomnormal_1 = NormalizedView(self.randomtensor, mode=1)

    def test_norms(self):
        self.assertAlmostEqual(self.normalized[0,0], -1.0)
        self.assertAlmostEqual(self.normalized[2,0], 0.99980006)

        for i in range(5):
            row = [self.randomnormal_0[i,j] for j in range(8)]
            self.assertAlmostEqual(numpy.dot(row, row), 1.0)

        for i in range(8):
            col = [self.randomnormal_1[i,j] for i in range(5)]
            self.assertAlmostEqual(numpy.dot(col, col), 1.0)


class NormalizedSVD2DTest(unittest.TestCase):
    def setUp(self):
        self.tensor = DictTensor(2)
        self.tensor.update(nested_list_to_dict(
                numpy.random.random_sample((10, 12))))
        self.normalized_tensor = self.tensor.normalized()
        self.svd = self.normalized_tensor.svd(k=3)
        self.u, self.svals, self.v = self.svd.u, self.svd.svals, self.svd.v

    def test_decomposition(self):
        self.assertEqual(self.u.shape[0], self.tensor.shape[0])
        self.assertEqual(len(self.svals), self.u.shape[1])
        self.assertEqual(len(self.svals), self.v.shape[1])
        self.assertEqual(self.v.shape[0], self.tensor.shape[1])

        # Assert that the singular values are decreasing
        for i in range(1,len(self.svals)):
            self.assert_(self.svals[i] < self.svals[i-1])

    def test_reconstructed(self):
        pass # TODO

    def test_orthonormality(self):
        assertTensorEqual(self.u.T * self.u, numpy.eye(self.u.shape[1]))
        assertTensorEqual(self.v.T * self.v, numpy.eye(self.u.shape[1]))

    def test_variance(self):
        return # TODO
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


normalize_testcase2 = [[1, 0],
                       [3, 4]]

normalize_testcase3 = [[70, 0],
                       [3, 4]]

def tensor_from_nested_list(data, tensor_cls=DictTensor):
    t = tensor_cls(2)
    t.update(nested_list_to_dict(data))
    return t


class NormalizedSVD2DTest2(unittest.TestCase):
    def assertItemsEqual(self, t1, t2):
        for key in t1.iterkeys():
            self.assertAlmostEqual(t1[key], t2[key])


    def test_svd(self):
        '''Ensure SVDs of tensors match up after normalizing.'''
        t1 = tensor_from_nested_list(normalize_testcase2)
        svd1 = t1.normalized().svd(k=1)

        t2 = tensor_from_nested_list(normalize_testcase3)
        svd2 = t2.normalized().svd(k=1)
        self.assertItemsEqual(svd1.u, svd2.u)
        self.assertItemsEqual(svd1.svals, svd2.svals)
        self.assertItemsEqual(svd1.v, svd2.v)

        # Now try transposing and normalizing over the other mode
        t3 = t1.T
        t4 = t2.T
        svd3 = t3.normalized(mode=1).svd(k=1)
        svd4 = t4.normalized(mode=1).svd(k=1)
        self.assertItemsEqual(svd1.u, svd3.v)
        self.assertItemsEqual(svd1.svals, svd3.svals)
        self.assertItemsEqual(svd1.v, svd3.u)

        self.assertItemsEqual(svd3.u, svd4.u)
        self.assertItemsEqual(svd3.svals, svd4.svals)
        self.assertItemsEqual(svd3.v, svd4.v)


if __name__ == '__main__':
    unittest.main()
