from csc.divisi.labeled_view import make_sparse_labeled_tensor
from csc.divisi.ordered_set import OrderedSet
from nose.tools import eq_, assert_almost_equal, raises

def test_make_shape():
    labels = OrderedSet(list('abcde'))
    t = make_sparse_labeled_tensor(ndim=1, labels=[labels])
    eq_(t.shape[0], len(labels))
    eq_(t.tensor.shape[0], len(labels))

def test_op_shape():
    labels = OrderedSet(list('abcde'))
    t = make_sparse_labeled_tensor(ndim=1, labels=[labels])
    t = t * 2
    eq_(t.shape[0], len(labels))
    eq_(t.tensor.shape[0], len(labels))
