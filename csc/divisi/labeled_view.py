from csc.divisi.ordered_set import OrderedSet, IdentitySet, indexable_set
from csc.divisi.tensor import View, TheEmptySlice, DictTensor, DenseTensor
import copy
import heapq
from operator import itemgetter
from itertools import izip

# Some convenience functions

def make_sparse_labeled_tensor(ndim, labels=None,
                               initial=None, accumulate=None,
                               normalize=False):
    '''
    Create a sparse labeled tensor.

    ndim: number of dimensions (usually 2)
    
    labels: if you already have label lists, pass them in here. (A
    None in this list means an unlabeled dimension. If you simply
    don't have labels yet, pass an OrderedSet().)

    initial / accumulate: sequences of (key, value) pairs to add to
    the tensor. ``initial`` is applied first by ``.update``, meaning
    that later values will override earlier ones. ``accumulate`` is
    applied afterwards, and all values add to anything already there.

    normalize:
     an int or tuple of ints: normalize along that dimension
     True: normalize along axis 0
     'tfidf': use tf-idf
     'tfidf.T': use tf-idf, transposed (matrix is documents by terms)
     a class: adds that class as a layer.
    '''
    if labels is None: labels = [OrderedSet() for _ in xrange(ndim)]
    tensor = LabeledView(DictTensor(ndim), labels)
    tensor.tensor._shape[:] = [len(label_list) for label_list in labels]
    if initial is not None:
        tensor.update(initial)
    for k, v in accumulate or []:
        tensor.inc(k, v)

    if normalize:
        return tensor.normalized(normalize)
    else:
        return tensor


def make_dense_labeled_tensor(data, labels):
    return LabeledView(DenseTensor(data), labels)


class LabelingMixin(object):
    def vector_keys(self):
        """
        Like keys(), but returns plain labels instead of singletons of labels
        for a 1st-order tensor.
        """
        if self.tensor.ndim == 1: return self.label_lists()[0]
        else: return self.keys()

    # Index -> label.
    def label(self, mode, index):
        """
        Get the label for a particular place in the matrix -- that is,
        a given index along a given mode.

        The inverse of this function is index().
        """
        return self._labels[mode][index]

    def row_label(self, index): return self.label(0, index)
    def column_label(self, index): return self.label(1, index)
    def labels(self, indices):
        return tuple(label_list[index] for label_list, index in izip(self._labels, indices))
    def label_lists(self):
        """
        A list of all label lists for this tensor, in order by mode.
        """
        return self._labels
    def label_list(self, mode):
        """
        Get the label list for a particular mode.
        """
        return self._labels[mode]
    label_sets = label_lists

    # Label -> index
    def index(self, mode, label, create=False):
        """
        Find the index, along a given mode, that is labeled with a given label.

        The inverse of this function is label().
        """
        if label == TheEmptySlice: return TheEmptySlice
        if create:
            return self._labels[mode].add(label)
        else:
            return self._labels[mode].index(label)

    def row_index(self, index): return self.label(0, index)
    def column_index(self, index): return self.label(1, index)
    def indices(self, labels, create=False):
        '''
        Look up the underlying tensor keys for a tuple of labels.
        '''
        if not create:
            try:
                return tuple(label_list.index(label) for label_list, label in izip(self._labels, labels))
            except TypeError: # happens for slices
                pass
        return tuple(self.index(mode, label, create)
                     for mode, label in enumerate(labels))


class LabeledView(LabelingMixin, View):
    '''
    A LabeledView creates a layer of labels (probably strings) that
    map to the numerical keys of the underlying tensor.
    '''

    # Flag to the layering procedure that this view takes numeric inputs and maps
    # them to non-numeric outputs.
    input_is_numeric = True
    output_is_numeric = False

    def __init__(self, tensor, label_lists=None):
        View.__init__(self, tensor)
        shape = tensor.shape
        if label_lists is None:
            label_lists = [None]*self.tensor.ndim

        if tensor.ndim != len(label_lists):
            raise IndexError('Number of label lists (%d) does not match number of dimensions (%s)' % (len(label_lists), tensor.ndim))

        self._labels = [indexable_set(l, shape[mode])
                        for mode, l in enumerate(label_lists)]

        self.tensor.ensure_index_is_valid(tuple(len(label_list)-1 for label_list in self._labels))


    def __repr__(self):
        return '<LabeledView of %r, keys like: %r>' % (
            self.tensor, self.example_key())


    @property
    def shape(self):
#        assert self._label_dims_correct()
        return self.tensor.shape

    def _label_dims_correct(self):
        shape = self.tensor.shape
        for mode, label in enumerate(self._labels):
            if isinstance(label, IdentitySet):
                continue
            if len(label) != shape[mode]:
                return False
        return True


    def transpose(self):
        if self.tensor.ndim != 2:
            raise TypeError('Only 2D tensors can be transposed.')
        return LabeledView(self.tensor.transpose(), [self._labels[1], self._labels[0]])

    @property
    def T(self):
        return self.transpose()

    def __iter__(self):
        labels = self._labels
        for indices, v in self.tensor.iteritems():
            label = tuple(label_list[index] for label_list, index in izip(labels, indices))
            if None in label: continue
            yield label

    def iteritems(self):
        labels = self._labels
        for indices, v in self.tensor.iteritems():
            result = (tuple(label_list[index] for label_list, index in izip(labels, indices)), v)
            if None in result[0]: continue
            yield result

    def iter_dim_keys(self, dim):
        return iter(self.label_list(dim))

    def dim_keys(self, dim):
        return self.label_list(dim)

    def unfolded_labels(self, mode):
        '''Return the set of labels for the remaining dimensions if this view were to be
        unfolded at the given mode.'''
        if mode >= self.tensor.ndim:
            raise IndexError('Dimension out of range.')
        if self.tensor.ndim == 2:
            if mode == 0: return self._labels[1]
            else: return self.labels[0]
        # FIXME: finish this

    # __contains__ is defined in terms of has_key by MyDictMixin (see View / Tensor)
    def has_key(self, labels):
        # Handle single dimensions
        if not isinstance(labels, (list, tuple)):
            labels = (labels,)

        # Compute indices
        try:
            indices = self.indices(labels)
        except KeyError:
            return False

        return indices in self.tensor

    def __getitem__(self, labels):
        # Handle single dimensions
        if not isinstance(labels, (list, tuple)):
            labels = (labels,)

        # Compute indices
        try:
            indices = self.indices(labels)
        except KeyError:
            if hasattr(self.tensor, 'default_value'):
                return self.tensor.default_value
            else: raise

        # Extract data
        data = self.tensor[indices]

        # Return a plain number if all dimensions are specified.
        if len(getattr(data, 'shape', ())) == 0:
            return data

        # Compute new labels.
        labellists = [self._labels[mode] for mode, idx in enumerate(indices)
                      if idx == TheEmptySlice]
        return LabeledView(data, labellists)

    def __setitem__(self, labels, value):
        if not isinstance(labels, (list, tuple)):
            labels = (labels,)

        d = self.tensor.ndim
        if len(labels) != d: raise KeyError("You need %d indices" % d)

        # Make sure we have ordered numbers for each label
        for mode in xrange(d):
            self._labels[mode].add(labels[mode])

        self.tensor[self.indices(labels)] = value
        # TODO: Add support for slice assignment.


    ### Math operations
    def scalar_iadd(self, other):
        self.tensor += other
        return self

    def icmul(self, other):
        self.tensor *= other
        return self

    def tensor_iadd(self, other):
        if not hasattr(other, 'label_list'):
            raise TypeError('Only labeled views can be added to labeled views.')
        if self.tensor.ndim != other.ndim:
            raise ValueError('Number of dimensions do not match.')

        # Adding the wrapped tensors directly is only valid if the labels
        # match up.
        if all(self.label_list(mode) == other.label_list(mode)
               for mode in xrange(self.ndim)):
            self.tensor += other.tensor
            return self

        # Otherwise, do something simple.
        for k, v in other.iteritems():
            if v:
                self[k] = self.get(k, 0) + v
        return self

    def __neg__(self):
        return self.layer_on(-self.tensor)

    def to_dense(self):
        '''Change the underlying representation to a dense tensor.'''
        return self.layer_on(self.tensor.to_dense())
    
    def to_sparse(self):
        '''Change the underlying representation to a sparse tensor.'''
        return self.layer_on(self.tensor.to_sparse())
    
    def bake(self):
        '''Simplify the representation to a LabeledView of a DictTensor.'''
        return make_sparse_labeled_tensor(ndim=self.ndim, initial=self.iteritems())

    def dot(self, other):
        """
        For first-order tensors, this is a dot product. For second-order,
        it performs matrix multiplication. Like Numpy matrices (not arrays),
        this is the operation performed by the * operator.
        """
        label_lists = [self.label_list(i) for i in xrange(self.ndim-1)]
        if hasattr(other, 'label_list'):
            data = self.tensor.dot(other.tensor)
            label_lists += [other.label_list(i) for i in xrange(1, other.ndim)]
        else:
            data = self.tensor.dot(other)
            assert isinstance(self.label_list(self.tensor.ndim-1), IdentitySet)
            label_lists += [None]*(len(other.shape)-1)

        if len(label_lists) == 0: return float(data)
        return LabeledView(data, label_lists)


    def tensordot(self, other, mode):
        shape = self.shape
        if other.ndim != 1 or shape[mode] != other.shape[0]:
            raise IndexError('Incompatible dimensions for sparse tensordot (%r.tensordot(%r, %d)' % (self, other, mode))

        # The operation will collapse the specified mode. Get the remaining labels.
        label_sets = self._labels[:mode] + self._labels[mode+1:]

        if hasattr(other, 'label_list'):
            if self._labels[mode] != other.label_list(0):
                raise IndexError("Labels for mode %d of %r don't match labels for the vector %r." % (mode, self, other))
            data = self.tensor.tensordot(other.tensor, mode)
        else:
            if not isinstance(self._labels[mode], IdentitySet):
                import warnings
                warnings.warn("Mode %d of %r is labeled, but the vector %r is not." % (mode, self, other), RuntimeWarning)
            data = self.tensor.tensordot(other, mode)

        if len(label_sets) == 0: return float(data)
        return LabeledView(data, label_sets)


    def cmul(self, other):
        return self.layer_on(self.tensor.cmul(other))

    def hat(self):
        return self.layer_on(self.tensor.hat())

    def svd(self, *a, **kw):
        if self.ndim == 2:
            _svd = self.tensor.svd(*a, **kw)
            return LabeledSVD2DResults.layer_on(_svd, self)
        else:
            # The unfoldings will take care of labeling.
            from csc.divisi.svd import svd_sparse
            return svd_sparse(self, *a, **kw)

    def incremental_svd(self, *a, **kw):
        if self.ndim == 2:
            _svd = self.tensor.incremental_svd(*a, **kw)
            return LabeledSVD2DResults.layer_on(_svd, self)
        else:
            raise NotImplementedError("Incremental SVDs can only be run on 2d tensors!")

    ### Views
    def layer_on(self, tensor):
        return LabeledView(tensor, self._labels)

    def unnormalized(self):
        '''
        Remove the topmost layer of normalization from this tensor stack.
        '''
        return self.layer_on(self.tensor.unnormalized())
    
    def array_op(self, op, labels=None, *args, **kwargs):
        """
        Apply an operation to the tensor inside (which must be a
        DenseTensor), and return the result wrapped in the same view.

        This assumes the resulting tensor has the same shape, unless
        you provide a labels= parameter.
        """
        inner = self.tensor.array_op(op, *args, **kwargs)
        if labels is None: labels = self._labels
        return inner.labeled(labels)


    ### Utility extractions
    def vector_to_dict(self):
        '''Like #items for 1D tensors, but uses indices directly instead of
        one-item tuples.'''
        if self.ndim != 1:
            raise ValueError("vector_to_dict only works for 1D tensors. Take a slice first.")
        labels = self.label_list(0)
        return dict((labels[i], float(self.tensor[i])) for i in xrange(self.shape[0]))

    def top_items(self, n=10, largest=True, key=None):
        '''
        For each of the top n items with greatest values, return a pair of
        the item and its value.

        Parameters:
         n: number of items to return
         largest: True (default) to get the largest items, else get the smallest
         key: key to use when sorting. (e.g., abs)

        If the tensor is 1D, gives indices instead of one-item tuples.
        Always expresses the output in floats directly.
        '''
        if key is not None:
            extreme_key = lambda ent: key(ent[1])
        else:
            extreme_key = itemgetter(1)

        find_extreme = heapq.nlargest if largest else heapq.nsmallest
        items = find_extreme(n, self.iteritems(), key=extreme_key)

        if self.ndim == 1:
            return [(k[0], float(v)) for k, v in items]
        else:
            return [(k, float(v)) for k, v in items]

    def concatenate(self, other):
        concat_dense = self.tensor.concatenate(other.tensor)
        newlabels = OrderedSet(list(self.label_list(0)) +
                               list(other.label_list(0)))
        return LabeledView(concat_dense, [newlabels] + self.label_lists()[1:])

    def with_zeros_removed(self):
        '''
        Make a new LabeledView of a DictTensor with all the non-zero
        items in this tensor.
        '''
        return make_sparse_labeled_tensor(ndim=self.ndim, initial=(
                (k, v) for k, v in self.iteritems() if v))

    def k_means(self, k=20, **kw):
        means, clusters = self.tensor.k_means(k, **kw)
        labeled_clusters = [[self.label(0, index) for index in cluster] for cluster in clusters]
        return means, labeled_clusters

    def label_histograms(self):
        '''
        Counts the number and magnitude of items in each row / column.

        Returns (counts, magnitudes).
        '''
        counts, magnitudes = self.tensor.label_histograms()
        return ([LabeledView(count, [labels]) for count, labels
                 in izip(counts, self.label_sets())],
                [LabeledView(mag, [labels]) for mag, labels
                 in izip(magnitudes, self.label_sets())])

    def squish(self, cutoff=5):
        import numpy as np
        dims = range(self.ndim)
        counts = [np.zeros((self.shape[mode],)) for mode in dims]
        for labelindex in self:
            indices = self.indices(labelindex)
            for mode in dims:
                counts[mode][indices[mode]] += 1

        print "done counting."
        newtensor = make_sparse_labeled_tensor(ndim=self.ndim)
        for labelindex, value in self.iteritems():
            keep = True
            indices = self.indices(labelindex)
            for mode in dims:
                if counts[mode][indices[mode]] < cutoff:
                    keep = False
                    break
            if keep:
                newtensor[labelindex] = value
        return newtensor
    
    def means(self):
        '''
        Computes the labeled by-slice means for each dimension.
        '''
        means = self.tensor.means()
        return [make_dense_labeled_tensor(means, [labels])
                for means, labels in zip(means, self.label_lists())]


from csc.divisi.svd import SVD2DResults
class LabeledSVD2DResults(SVD2DResults):
    @classmethod
    def layer_on(cls, results, src):
        if hasattr(src, 'label_lists'):
            src = src.label_lists()
        return cls(
            LabeledView(results.u, [src[0], None]),
            LabeledView(results.v, [src[1], None]),
            results.svals)

    def save_pytables(self, filename, title='SVD results', filters=None, **kw):
        from csc.divisi.pyt_utils import get_pyt_handle
        from tables import ObjectAtom, Filters, Atom
        fileh = get_pyt_handle(filename, title)
        if filters is None and kw:
            filters = Filters(**kw)
        try:
            root = fileh.root

            def store_tensor(name, tensor):
                data = tensor._data
                arr = fileh.createCArray(root, name, Atom.from_dtype(data.dtype), tensor.shape, filters=filters)
                arr[:] = data
            # Labeled stuff
            for name in ('u', 'v', 'weighted_u', 'weighted_v'):
                store_tensor(name, getattr(self, name).tensor)
            # Unlabeled stuff
            for name in ('svals', 'core'):
                store_tensor(name, getattr(self, name))

            # Ordered sets
            def write_labels(name, view):
                arr = fileh.createVLArray(root, name, ObjectAtom(), filters=filters)
                for label in view.label_lists():
                    arr.append(label)
            write_labels('u_labels', self.u)
            write_labels('v_labels', self.v)
        finally:
            fileh.close()

    @classmethod
    def load_pytables(cls, filename, copy=False):
        import numpy as np
        from csc.divisi.pyt_utils import get_pyt_handle
        fileh = get_pyt_handle(filename)

        root = fileh.root
        def wrapped_tensor(name):
            data = getattr(root, name)
            if copy: data = data.read()
            return DenseTensor(data)
        def wrapped_labeled_tensor(name, label_name):
            return wrapped_tensor(name).labeled(list(getattr(root, label_name)))
        u = wrapped_labeled_tensor('u', 'u_labels')
        v = wrapped_labeled_tensor('v', 'v_labels')
        svals = wrapped_tensor('svals')
        svd = cls(u, v, svals)
        svd._weighted_u = wrapped_tensor('weighted_u').labeled(u.label_lists())
        svd._weighted_v = wrapped_tensor('weighted_v').labeled(v.label_lists())
        svd._core = wrapped_tensor('core')
        return svd

    # Pickle also works for loading and saving.
    def __getstate__(self):
        return dict(
            u=self.u,
            v=self.v,
            svals=self.svals)
    def __setstate__(self, state):
        self.u = state['u']
        self.v = state['v']
        self.svals = state['svals']
        self.clear_cache()

    def get_weighted_u(self):
        return self.u.layer_on(DenseTensor(self.u.tensor.unwrap() * self.svals.unwrap()))
    def get_weighted_v(self):
        return self.v.layer_on(DenseTensor(self.v.tensor.unwrap() * self.svals.unwrap()))


    def slice(self, k=None, rows=None, cols=None):
        '''
        Slice out a part of the SVD results, fewer svals, rows, and/or columns.

        k : int - how many svals to keep. None -> keep all.
        rows : [obj] - a list of rows to keep. None -> keep all.
        cols : [obj] - a list of cols to keep. None -> keep all.
        '''
        row_labels = self.u.label_list(0)
        col_labels = self.v.label_list(0)
        if rows is not None:
            rows = map(row_labels.index, rows)
            row_labels = indexable_set(rows, len(rows))
        if cols is not None:
            cols = map(col_labels.index, rows)
            col_labels = indexable_set(cols, len(cols))
        return LabeledSVD2DResults.layer_on(super(LabeledSVD2DResults, self).slice(k, rows, cols), [row_labels, col_labels])

    def unlabeled(self):
        return SVD2DResults(self.u.unwrap(), self.v.unwrap(), self.svals)
    def label_lists(self):
        return self.u.label_list(0), self.v.label_list(0)
        
    def denormalized(self, norms):
        return LabeledSVD2DResults.layer_on(self.unlabeled().denormalized(norms), self.label_lists())

from collections import defaultdict
def reduce_keys(matrix, cutoff=2):
    # only works on 2-D matrices for now

    labels0 = defaultdict(int)
    labels1 = defaultdict(int)
    reduced = make_sparse_labeled_tensor(ndim=2)
    print 'reduce: scanning for useful keys'
    for key, value in matrix.iteritems():
        label0, label1 = key
        labels0[label0] += 1
        labels1[label1] += 1
    print 'reduce: copying values'
    for key, value in matrix.iteritems():
        label0, label1 = key
        if labels0[label0] >= cutoff and labels1[label1] >= cutoff:
            reduced[label0, label1] = value
    return reduced

