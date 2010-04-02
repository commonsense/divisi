''' tensor.extreme_items() is a fast and flexible way to determine the
maximum and/or minimum items in a Tensor (often a vector).
'''

from csc.divisi import make_sparse_labeled_tensor
from nose.tools import eq_, raises

small_vec, large_vec = None, None
def setup():
    global small_vec, large_vec
    small_vec = make_sparse_labeled_tensor(ndim=1)
    for i in range(5):
        small_vec[i] = -i

    large_vec = make_sparse_labeled_tensor(ndim=1)
    for i in range(100):
        large_vec[i] = i-50

def test_large_vec():
    'Extreme items of a vector.'
    largest, smallest = large_vec.extreme_items(n_biggest=10, n_smallest=3)

    eq_(len(largest), 10)
    eq_(len(smallest), 3)

    (k,), v = smallest[0]
    eq_(k, 0)
    eq_(int(v), -50)

    (k,), v = largest[-1]
    eq_(k, 99)
    eq_(int(v), 49)


def test_small_vec():
    'Extreme items of a vector.'
    largest, smallest = small_vec.extreme_items(n_biggest=10, n_smallest=3)

    eq_(len(largest), 5)
    eq_(len(smallest), 3)

    (k,), v = smallest[0]
    eq_(k, 4)
    eq_(int(v), -4)

    (k,), v = largest[-1]
    eq_(k, 0)
    eq_(int(v), 0)


def test_detuple():
    'extreme_items can optionally pull its keys out of a tuple'
    largest, smallest = large_vec.extreme_items(n_biggest=10, n_smallest=3, detuple=True)

    eq_(len(largest), 10)
    eq_(len(smallest), 3)

    k, v = smallest[0]
    eq_(k, 0)
    eq_(int(v), -50)

    k, v = largest[-1]
    eq_(k, 99)
    eq_(int(v), 49)


@raises(IndexError)
def test_detuple_too_big():
    make_sparse_labeled_tensor(ndim=2, initial=(((i, i), i) for i in range(30))).extreme_items(detuple=True)
    
