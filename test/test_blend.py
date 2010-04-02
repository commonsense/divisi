# if __name__ == '__main__' and __package__ is None:
#     import sys, os
#     sys.path.insert(0, os.path.split(os.dirname(os.abspath(__file__)))[0])
#     __package__ = 'test.test_blend'

import logging
from nose.tools import *
from csc.divisi.blend import Blend
from numpy import allclose
from test_util import ez_matrix, assertSetEquals
from tensor_util import assertTensorEqualCompleteDict

t1 = t2 = None
def setup():
    global t1, t2

    t1 = ez_matrix('aab', '121', [1]*3)
    t2 = ez_matrix('ac', '21', [1]*2)

def test_bare_blend():
    '''
    A matrix blended with nothing else changes nothing.
    '''
    b = Blend([t1], weights=[1])
    assertSetEquals(set(t1.label_list(0)), set(b.label_list(0)))

@raises(TypeError)
def test_nonlist():
    Blend(t1, weights=[1])
    

def test_manual_weights():
    '''
    Specifying weights manually causes the result to be the weighted sum.
    '''
    b = Blend([t1, t2], weights=[.75,.25])
    b.build_tensor()
    expected = dict(zip(zip('aabc', '1211'), (.75, 1.0, .75, .25)))
    assertTensorEqualCompleteDict(b, expected)

def test_fake_tensor():
    '''
    Blends have a fake tensor that does iteritems really fast.
    '''
    # labels will be abc, 12
    t3 = ez_matrix('c', '1', [1])
    b = Blend([t1, t3], weights=[.75,.25])
    fake_tensor = b.fake_tensor()
    items = set(fake_tensor.iteritems())
    expected_items = set((
            ((0,0), .75),
            ((0,1), .75),
            ((1,0), .75),
            ((2,0), .25)))
    assertSetEquals(items, expected_items)

def test_no_overlap():
    '''
    Certain optimizations are possible if there is no overlap.
    '''
    t2 = ez_matrix('dc', '21', [1]*2)
    b = Blend([t1, t2], weights=[.75,.25])
    b.build_tensor()
    expected = dict(zip(zip('aabdc', '12121'), (.75, .75, .75, .25, .25)))
    assertTensorEqualCompleteDict(b, expected)

def test_names():
    '''
    Blends can name their tensors.
    '''
    # Uses repr() by default
    b = Blend([t1, t2])
    eq_(b.names, (repr(t1), repr(t2)))

    # but you can specify it with a dict:
    b = Blend(dict(t1=t1, t2=t2))
    tensors = sorted(list(zip(b.names, b.tensors)))
    eq_(tensors, [('t1', t1), ('t2', t2)])

    # or a sequence of items
    b = Blend([('t1', t1), ('t2', t2)])
    eq_(b.names, ('t1', 't2'))
    eq_(b.tensors, (t1, t2))


def test_sval_linearity():
    '''
    Show that singular values are linear with respect to constant multiplication.
    '''
    t1 = ez_matrix('0013', '0421', [1,2,3,4])
    t2 = t1*3
    assert_almost_equal(t2.svd(k=1).svals[0], t1.svd(k=1).svals[0]*3)

@raises(TypeError)
def test_wrong_dims():
    from csc.divisi.labeled_view import make_sparse_labeled_tensor
    t1 = make_sparse_labeled_tensor(ndim=1)
    t2 = make_sparse_labeled_tensor(ndim=2)
    Blend([t1, t2])

def test_autoblend():
    '''
    If weights are not specified explicitly, Blend computes them automatically
    so as to maximize the amount of interaction between the two matrices.

    This is hard to test in general. The strategy used here is to
    blend two copies of the same matrix (so the singular values are
    the same), but with different labels.
    '''

    t1 = ez_matrix('0013', '0421', [1,2,3,4])
    t2 = ez_matrix('2214', '0421', [3,6,9,12]) # one overlapping label, 3x the values
    b = Blend([t1, t2]) # don't specify weights => autoblend.

    eq_(b.label_overlap[0], 1)
    eq_(b.label_overlap[1], 4)

    # This should result in t2 getting weighted 1/3 the weight of t1:
    logging.info(b.weights)
    assert allclose(b.weights, [.75, .25])

    # Test the resulting tensor
    b.build_tensor()
    # -non-overlapping elements
    assert t1['0', '0'] == 1
    assert_almost_equal(b['0', '0'], .75*1) # remember that the original tensors had non-unity values.
    assert t2['4', '1'] == 4*3
    assert_almost_equal(b['4', '1'], .25*4*3)
    # -overlapping element
    assert t1['1', '2'] == 3
    assert t2['1', '2'] == 3*3
    assert_almost_equal(b['1', '2'],  0.75*3 + 0.25*3*3) # just to be explicit...


class MockTensor(object):
    ndim=2
    def label_list(self, idx):
        return range(5)
    def svd(self, k=50):
        import numpy
        from csc.divisi.tensor import DenseTensor
        return MockSVDResults(None, DenseTensor(numpy.array(self.svals[:k])), None)
    def stack_contains(self, cls):
        from csc.divisi.labeled_view import make_sparse_labeled_tensor
        return make_sparse_labeled_tensor(2).stack_contains(cls)

class MockSVDResults(object):
    def __init__(self, u, svals, v):
        self.u, self.svals, self.v = u, svals, v

def test_rough_blend():
    '''
    The rough weights are 1/svals[0], normalized.
    '''
    t1 = MockTensor()
    t1.svals = range(5, 0, -1)
    t2 = MockTensor()
    t2.svals = range(15, 0, -3)

    b = Blend([t1, t2], k_values=1)
    eq_(b.weights, (.75, .25))

def test_rough_blend2():
    '''
    You can pass in the svals and it doesn't try to compute them.
    '''
    t1 = MockTensor()
    t2 = MockTensor()
    b = Blend([t1, t2], k_values=1, svals=[range(5,0,-1), range(15,0,-3)])
    eq_(b.weights, (.75, .25))

def test_predicted_svals():
    '''
    The predicted_svals function shows the predicted singular values.
    '''
    t1 = MockTensor()
    t1.svals = range(5, 0, -1)
    t2 = MockTensor()
    t2.svals = range(10, 0, -2)

    # Weighting one side heavily should make its svals uniquely show up.
    weight = 0.999999
    b = Blend([t1, t2], weights=[weight, 1-weight], k_values=1)

    # with origin tracking:
    svals = b.predicted_svals(num=5, for_each_tensor=5, track_origin=True)
    for expected, (actual, src) in zip(t1.svals, svals):
        assert_almost_equal(actual/weight, expected)
        eq_(src, 0)

    # without origin tracking
    svals = b.predicted_svals(num=5, for_each_tensor=5)
    for expected, actual in zip(t1.svals, svals):
        assert_almost_equal(actual/weight, expected)

    # Flip it around.
    b.weights = [1-weight, weight]
    # Note: this is an easy way to transpose the "matrix"
    sval, src = zip(*b.predicted_svals(num=5, for_each_tensor=5, track_origin=True))
    for actual, expected in zip(sval, t2.svals):
        assert_almost_equal(actual/weight, expected)
    eq_(src, (1,)*5)

def test_specifying_factor():
    '''
    When a factor is supplied, use it as the weight of the second matrix.
    '''
    eq_(Blend([t1, t2], factor=.25).weights, (0.75, 0.25))

@raises(ValueError)
def test_factor_too_big():
    "Factor is between 0 and 1."
    Blend([t1, t2], factor=1.5)

@raises(TypeError)
def test_factor_wrong_dims():
    "Factor only applies to two matrices."
    Blend([t1, t2, t1], factor=0.5)

@raises(TypeError)
def test_specifying_factor_and_weights():
    "Shouldn't specify both factor and weights at the same time."
    Blend([t1, t2], factor=0.5, weights=[0.5, 0.5])
