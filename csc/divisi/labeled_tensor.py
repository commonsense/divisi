'''Some deprecated stuff. Look in :mod:`csc.divisi.labeled_view`.'''

from csc.divisi.tensor import DictTensor, DenseTensor
from csc.divisi.normalized_view import NormalizedView
from csc.divisi.labeled_view import LabeledView
from csc.divisi.ordered_set import OrderedSet

if False: # This will get deprecated sometime...
    import warnings
    warnings.warn('SparseLabeledTensor and DenseTensor have moved to '+
                  'csc.divisi.labeled_view.make_sparse_labeled_tensor and make_dense_labeled_tensor.',
                  DeprecationWarning)

from csc.divisi.labeled_view import make_sparse_labeled_tensor as SparseLabeledTensor, make_dense_labeled_tensor as DenseLabeledTensor

import numpy
class OldDenseLabeledTensor(LabeledView):
    # FIXME: These docs are now WRONG!
    '''A tensor of arbitrary dimension that can be indexed by label.

    Create an example tensor.
    >>> t = DenseLabeledTensor(numpy.array([[4, -2, 0], [2, 0, 1], [-1, 0, 0]]), [OrderedSet(['apple', 'banana', 'pear']), OrderedSet(['red', 'yellow', 'green'])])

    Get the entry for 'apple', 'red'.
    >>> t['apple', 'red']
    4

    Set 'banana', 'yellow' to 6.
    >>> t['banana', 'yellow'] = 6
    >>> t['banana', 'yellow']
    6

    It has some semantics of a numpy array also:
    >>> t.shape
    (3, 3)

    You can slice it along any axis and get another labeled tensor.
    >>> banana = t['banana',:]
    >>> banana['yellow']
    6
    >>> sorted(banana.keys())
    [('green',), ('red',), ('yellow',)]

    >>> yellow = t[:,'yellow']
    >>> yellow['banana']
    6
    >>> sorted(yellow.keys())
    [('apple',), ('banana',), ('pear',)]

    Run an svd. Show that the matrices multiply back out. For a 2D SVD,
    get_factor(0) corresponds to U, get_core() to \Sigma, and get_factor(1)
    to V, in the notation of the literature.
    >>> Ap = t.get_factor(0) * t.get_core() * t.get_factor(1).transpose()
    >>> numpy.allclose(t._data, Ap._data)
    True

    Sometimes one dimension will have no labels, in which case it will be
    indexed by value.
    >>> t2 = DenseLabeledTensor(numpy.array([[4, -2, 0], [2, 0, 1], [-1, 0, 0]]), [['apple', 'banana', 'pear'], None])
    >>> t2['apple', 0]
    4

    It is sometimes useful to slice out certain indices.
    >>> sl = t2['apple', :]
    >>> sl.shape
    (3,)
    >>> sl[0]
    4
    >>> sl2 = t2['apple', 0:2]
    >>> sl2.shape
    (2,)
    >>> sl2[0]
    4
    '''
    # TODO: constructor.

    def __repr__(self):
        res = '<DenseLabeledTensor:'
        res += ' shape: %s' % repr(self.shape)
        res += ' keys like: %s' % repr(self.iterkeys().next())
        res += '>'
        return res



class OldSparseLabeledTensor(LabeledView):
    # FIXME: These docs are WRONG!
    '''A labeled tensor that only stores nonzero entries

    >>> t = SparseLabeledTensor(2)
    >>> t[('apple', 'red')] = 1
    >>> t[('apple', 'red')]
    1

    Internally the labels are mapped to indices:
    >>> t.indices(('apple', 'red'))
    (0, 0)
    >>> t.labels((0, 0))
    ('apple', 'red')

    It behaves like a dictionary in other ways, too:
    >>> t.keys()
    [('apple', 'red')]

    But it behaves like a numpy array:
    >>> t.shape
    (1, 1)

    Fill in some other values:
    >>> for fruit in ['apple', 'banana', 'pear', 'peach']:
    ...     for color in ['red', 'yellow', 'green']:
    ...        t[(fruit, color)] = 0

    We can make a dense version of it:
    >>> tt = t.to_dense()
    >>> tt.shape
    (4, 3)

    Run an svd. Show that the matrices multiply back out. For a 2D SVD,
    get_factor(0) corresponds to U, get_core() to \Sigma, and get_factor(1)
    to V, in the notation of the literature.
    >>> Ap = t.get_factor(0) * t.get_core() * t.get_factor(1).transpose()
    >>> numpy.allclose(tt._data, Ap._data)
    True
    '''

    def __init__(self, *a, **kw):
        if 'ndim' in kw:
            ndim = kw.pop('ndim')
            data = DictTensor(ndim)
            label_lists = [OrderedSet() for i in xrange(ndim)]
            LabeledView.__init__(self, data, label_lists, *a, **kw)
        else:
            LabeledView.__init__(self, *a, **kw)
        self._slice_cache = {}

    def __repr__(self):
        return '<SparseLabeledTensor: %s>' % LabeledView.__repr__(self)

    def dot(self, other):
        """
        For first-order tensors, this is a dot product. For second-order,
        it performs matrix multiplication. Like Numpy matrices (not arrays),
        this is the operation performed by the * operator.
        """
        # FIXME: move this code.
        # FIXME.
        if self.ndim == 1:
            #assert isinstance(other, LabeledTensor)
            # can deal with other cases later, if necessary
            if other.ndim == 2:
                labels = other.label_list(1)
                out = DenseLabeledTensor(numpy.zeros(len(labels)), labels)
                for label in self.vector_keys():
                    out += self[label] * other[label]
                return out
            elif other.ndim == 1:
                out = 0.0

            for label in self.vector_keys():
                out += self[label] * other[label]

            return out
        else:
            raise NotImplementedError # not done yet


    def array_op(self, ufunc, *others):
        # FIXME!
        """
        Apply a NumPy operation (a ufunc) to the data in this tensor,
        preserving labels.
        """
        def take_data(tensor):
            if isinstance(tensor, DenseLabeledTensor): return tensor._data
            elif isinstance(tensor, SparseLabeledTensor):
                return tensor.to_dense()._data
            else: return tensor
        arglist = [self._data] + [take_data(o) for o in others]
        newdata = ufunc(*arglist)
        return DenseLabeledTensor(newdata, self._labels)



    def slice(self, mode, label):
        # FIXME...
        if self._slice_cache.has_key((mode, label)):
            items = [(k, self[k]) for k in self._slice_cache[mode, label]]
            found = True
        else:
            items = self.items()
            found = False
        theslice = SparseLabeledTensor(self.ndim-1)
        theslice._labels = self._labels[:mode] + self._labels[mode+1:]
        for key, val in items:
            if not found:
                majorkey = key[mode]
                self._slice_cache.setdefault((slice, majorkey), set()).add(key)
            if key[mode] == label:
                theslice[key[:mode] + key[mode+1:]] = val
        return theslice

    def dense_slice(self, mode, label):
        # FIXME...
        if self._slice_cache.has_key((mode, label)):
            items = [(k, self[k]) for k in self._slice_cache[mode, label]]
            found = True
        else:
            items = self.items()
            found = False
        theslice = DenseLabeledTensor(numpy.zeros(self.shape[:mode] +
        self.shape[mode+1:]), self._labels[:mode] + self._labels[mode+1:])

        for key, val in items:
            if not found:
                majorkey = key[mode]
                self._slice_cache.setdefault((mode, majorkey), set()).add(key)
            if key[mode] == label:
                theslice[key[:mode] + key[mode+1:]] = val
        return theslice

    ### Loading and saving
    @classmethod
    def load(cls, filebase):
        tensor = DictTensor.load(filebase)
        try:
            tensor = NormalizedView.load(filebase, tensor)
        except IOError:
            pass
        return super(SparseLabeledTensor,cls).load(filebase, tensor)
