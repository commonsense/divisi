"""
A :class:`Tensor`, in short, is an n-dimensional array that we can do math to.

Often, n=2, in which case the tensor would be better known as a *matrix*.
Or sometimes n=1, in which case it's a *vector*. But Divisi can also deal with
n=3 and beyond.

Divisi uses many different kinds of tensors to store data. Fundamentally, there
are :class:`DenseTensors <DenseTensor>`, which are built around NumPy arrays,
and :class:`DictTensors <DictTensor>`, which
store data sparsely in dictionaries; then, there are various kinds of
:class:`views <View>` that wrap around them to let you work with your data.
"""

import numpy, sys
from csc.divisi.dict_mixin import MyDictMixin
import copy, random, math
from csc.divisi.exceptions import DimensionMismatch, InvalidLayeringException
from csc.divisi._svdlib import svd, dictSvd, isvd, dictIsvd
from warnings import warn
import logging
from itertools import izip

__all__ = ['TheEmptySlice',
           'Tensor', 'DictTensor', 'DenseTensor', # Data storage
           'View', 'InvalidLayeringException', # Layering
           'outer_tuple_iterator',  'data', 'wrap_unless_scalar', # Utilities
           ]

def isscalar(obj):
    return len(getattr(obj, 'shape', ())) == 0

TheEmptySlice = slice(None)
EPSILON = 1e-12

# Subclasses of Tensor need at a bare minimum:
#  - ndim
#  - shape
#  - __iter__()
#  - __getitem__()
# and you can probably make a faster implementation of:
#  - __len__(): number of keys
#  - iter_dim_keys()
#  - inc

class Tensor(MyDictMixin):
    """
    Tensors are the main type of object handled by Divisi. This is the base
    class for all Tensors.

    Tensors act like dictionaries whenever possible, so you can use
    dictionary methods such as ``keys``, ``iteritems``, and ``update`` as you
    would for a dictionary. They also act like Numpy arrays in some ways, but
    acting like a dictionary takes priority.

    Many tensor operations refer to *modes*; these are the dimensions by which
    a tensor is indexed. For example, a matrix has two modes, numbered 0 and 1.
    Mode 0 refers to rows, and mode 1 refers to columns.

    *Obscure terminology*: For higher-dimensional tensors, mode 2 has sometimes
    been called "tubes". Modes 3 and higher don't have names.
    """
    def __repr__(self): raise NotImplementedError
    def example_key(self):
        try:
            return self.__iter__().next()
        except StopIteration:
            return None

    @property
    def dims(self):
        """The number of dimensions of data in this tensor. For compatibility
        with Numpy, this is the same as ``self.ndim``."""
        return self.ndim

    def inc(self, key, amt):
        '''
        Increment the value at key by amt.
        '''
        self[key] += amt

    def __len__(self):
        """
        The number of *entries* stored in this tensor.

        This does not count the implied zeros in a :class:`DictTensor`.

        Note that this differs from the len() of a Numpy array, which would
        instead count its number of *rows*.
        """
        # This should be implemented more efficiently by subclasses.
        return len(self.keys())

    def iter_dim_keys(self, mode):
        '''
        Get an iterator over the keys of a specified mode of the tensor.

        Example usage::

            for row in tensor.iter_dim_keys(0):
                do_something_to(row)
        '''
        # Some subclasses may override this with a more efficient method
        assert(mode >= 0 and mode < self.dims)
        seen_keys = {}
        for key in self.iterkeys():
            if key[mode] not in seen_keys:
                seen_keys[key[mode]] = True
                yield key[mode]

    def dim_keys(self, mode):
        return list(self.iter_dim_keys(mode))

    def combine_by_element(self, other, op):
        '''
        Takes two tensors and combines them element by element using `op`.

        For example, given input tensors *a* and *b*, ``result[i] = op(a[i],
        b[i])``, for all indices *i* in *a* and *b*. This operation requires *a*
        and *b* to have the same indices; otherwise, it doesn't make any sense.

        (Note that, for the sake of efficiency, this doesn't run *op* on keys that
        neither *a* nor *b* have.)
        '''
        if other.shape != self.shape:
            raise IndexError('Element-by-element combination requires that both matrices have the same shape (%r.element_multiply(%r)' % (self, other))

        # This assumes that having the same dimensions => having the same keys.
        res = copy.deepcopy(self)
        seen_keys = {}
        for key, value in self.iteritems():
            res[key] = op(value, other[key])
            seen_keys[key] = True

        # Run the operation for all items in the other tensor that we haven't already
        # evaluated.
        for key, value in other.iteritems():
            if key not in seen_keys:
                res[key] = op(self[key], other[key])

        return res

    # Extraction and views
    def slice(self, mode, index):
        """
        Gets a tensor of only the entries indexed by *index* on mode *mode*.

        The resulting tensor will have one mode fewer than the original tensor.
        For example, a slice of a 2-D matrix is a 1-D vector.

        Examples:

        - To get a particular row of a matrix, use ``matrix.slice(0, row)``.
        - To get a particular column, use ``matrix.slice(1, col)``.
        """
        indices = [TheEmptySlice]*self.ndim
        indices[mode] = index
        return self[indices]

    def label(self, mode, index):
        """
        Returns the index as the *label* for that entry, so the label for an
        unlabeled index is simply the index itself.  Added to allow certain
        operations which work on labeled views to work on all tensors.
        """
        return index

    def add_layer(self, cls, *a, **kw):
        """
        Layer a view onto this tensor.

        Tensors can be wrapped by various kinds of
        :class:`View`. This method adds a view in the
        appropriate place.

        In this case, this is a plain Tensor, so it simply
        passes it to the view's constructor, but other Views will override
        this to layer on the new View in the appropriate way.
        """
        return cls(self, *a, **kw)

    def stack_contains(self, cls):
        '''
        Check if the tensor stack below here includes an instance of ``cls``.
        
        You may pass a tuple of classes as well, which will check if
        the stack contains _any_ of those classes.
        '''
        if isinstance(self, cls):
            return True
        if hasattr(self, 'tensor'):
            return self.tensor.stack_contains(cls)
        else:
            return False

    def unfolded(self, mode):
        """
        Get an :class:`UnfoldedView <divisi.unfolded_view.UnfoldedView>` of
        this tensor,
        which represents a tensor of order 3 or higher as an order-2 matrix.
        """
        from csc.divisi.unfolded_view import UnfoldedView
        return self.add_layer(UnfoldedView, mode)

    def labeled(self, labels):
        """
        Get a :class:`LabeledView` of this tensor.

        *labels* should be a list of
        :class:`OrderedSet` s, one for each mode,
        which assign labels to its indices, or ``None`` if that mode should
        remain unlabeled.
        """
        from csc.divisi.labeled_view import LabeledView
        return self.add_layer(LabeledView, labels)

    def normalized(self, mode=0):
        """
        Get a :class:`divisi.normalized_view.NormalizedView`
        of this tensor.
        
        The parameter can be:
         an int or tuple of ints: normalize along that dimension
         True: normalize along axis 0
         'tfidf': use tf-idf
         'tfidf.T': use tf-idf, transposed (matrix is documents by terms)
         a class: adds that class as a layer.
        """
        if mode is True: mode = 0 # careful, 1 == True but, but 1 is not True.
        if isinstance(mode, (int, tuple, list)):
            from csc.divisi.normalized_view import NormalizedView
            return self.add_layer(NormalizedView, mode)
        elif mode == 'tfidf':
            from csc.divisi.normalized_view import TfIdfView
            return self.add_layer(TfIdfView)
        elif mode == 'tfidf.T':
            from csc.divisi.normalized_view import TfIdfView
            return self.add_layer(TfIdfView, transposed=True)
        elif isinstance(mode, type):
            return self.add_layer(mode)

        raise TypeError('Unknown normalization parameter %r' % mode)
            

    def mean_subtracted(self, mode=0):
        from csc.divisi.normalized_view import MeanSubtractedView
        return self.add_layer(MeanSubtractedView, mode)

    # Math
    def __add__(self, other):
        res = copy.deepcopy(self)
        res += other
        return res

    def __iadd__(self, other):
        if isscalar(other):
            return self.scalar_iadd(other)
        else:
            return self.tensor_iadd(other)

    def weighted_add(self, other, weight):
        '''
        this + (weight * other).
        '''
        return self + weight * other

    def weighted_iadd(self, other, weight):
        self += other * weight
        return self

    def __sub__(self, other):
        return self + (-other)

    def __isub__(self, other):
        self += (-other)

    def __mul__(self, other):
        """
        This performs matrix multiplication, not elementwise multiplication.
        See the documentation for :meth:`dot`.

        If *other* is a scalar, this will perform scalar multiplication
        instead.
        """
        if isscalar(other):
            return self.cmul(other)
        else:
            return self.dot(other)

    def __rmul__(self, other):
        '''
        Handles multiplication in cases where the tensor is the second thing,
        e.g., ``2.5*tensor``. Multiplication is commutative, so this is trivial.
        '''
        return self * other

    def __imul__(self, other):
        """
        Handle multiplication in-place (``*=``). See the documentation
        for :meth:`__mul__`.
        """
        if isscalar(other):
            return self.icmul(other)
        else:
            raise NotImplementedError, "Don't know how to *= for non-scalars."

    def icmul(self, other):
        '''
        Multiplies this tensor in-place by a scalar.

        The ``*=`` operator uses this method.
        '''
        raise NotImplementedError, "Don't know how to multiply by a constant in place for %s" % type(self)

    def dot(self, other):
        """
        Get the product of two tensors, using matrix multiplication.

        When two tensors *a* and *b* are multiplied, the entries in the result
        come from the dot products of the last mode of *a* with the first mode
        of *b*. So the product of a *k*-dimensional tensor with an
        *m*-dimensional tensor will have *(k + m - 2)* dimensions.

        The ``*`` operation on two tensors uses this method.
        """
        raise NotImplementedError

    def cmul(self, other):
        """
        Returns the product of this tensor and a scalar.

        The ``*`` operation does this when given a scalar as its second argument.
        """
        raise NotImplementedError, "Don't know how to cmul a %s" % type(self)

    def __div__(self, other):
        """
        Performs scalar division.

        It actually just multiplies by 1/*other*, so if you're clever enough
        you might be able to make it divide by some other kind of object.
        """
        return self * (1.0/other)

    def __idiv__(self, other):
        """
        Performs in-place scalar division.

        It actually just multiplies by 1/*other*, so if you're clever enough
        you might be able to make it divide by some other kind of object.
        """
        self *= (1.0/other)
        return self

    def norm(self):
        '''
        Calculate the Frobenius (or Euclidean) norm of this tensor.

        The Frobenius norm of a tensor is simply the square root of the sum of
        the squares of its elements. For a vector, this is the same as the
        Euclidean norm.

        NOTE: This function was incorrect before 2009-06-01. Check your usage.
        '''
        mag = 0.0
        for v in self.itervalues():
            mag += v*v
        return math.sqrt(mag)

    magnitude = norm

    def hat(self):
        '''
        Return a normalized version of this tensor (generally a vector), by
        dividing by its norm.

        This is not the same as .normalized(), which normalizes all the things
        (generally vectors) that make up a tensor (generally a matrix), using a
        NormalizedView.
        '''
        return self / max(self.norm(), EPSILON)
    normalized_copy = hat

    ## Vector operations
    # Most of the operations below only make sense on vectors.



    def projection_onto(self, other):
        '''
        Return the component of this vector in the direction of another.
        '''
        assert self.ndim == 1
        absVal = other * other
        if absVal < EPSILON:
            return 0.0
        else:
            return (self * other) / absVal * other


    def components_wrt(self, other):
        '''
        Return the (parallel, perpendicular) components of this vector
        with respect to another vector.
        '''
        projection = self.projection_onto(other)
        return projection, (self-projection)

    def cosine_of_angle_to(self, other):
        '''
        Return the cosine of the angle of this vector with another.

        A dot B = |A| |B| cos \theta
        => cos \theta = (A dot B) / (|A| |B|)
        '''
        if self.ndim != 1: raise TypeError('Must be a vector')
        if (self.norm() < EPSILON) or (other.norm() < EPSILON):
            return 0.0
        else:
            return (self*other) / (self.norm() * other.norm())


    ## Summarization operations
    def extreme_items(self, n_biggest=10, n_smallest=10, filter=None, detuple=False):
        '''
        Extract the items at either extreme of the tensor. Returns
        (biggest, smallest).

        In many applications, we're not interested in all the values
        in a particular tensor we calculated, just the ones with the
        highest magnitude. *extremes* returns the *n_biggest* biggest
        and *n_smallest* smallest values in a tensor, along with their
        indices, in the form used by ``dict.items()``.

        The results are ordered by increasing value. For example, if
        you ask for::

            biggest, smallest = tensor.extreme_items(10, 10)

        Then the smallest item will be found as ``smallest[0]`` and the
        biggest as ``biggest[-1]``.

        If this is a vector (as it often is), note that the indices
        are going to be tuples with one thing in them, for
        consistency. If you don't like that, pass detuple=True.

        You can pass a ``filter(key, value)`` that will be called on
        each item; the item will be included only if the filter
        returns a true value. ``key`` is always a tuple, regardless of
        the value of ``detuple``.
        '''
#         TODO:
#         Somewhat like Python's ``sorted`` function, you can specify a
#         ``key``, e.g., abs, which is applied to the value before it is
#         compared. The output uses the original values.
        if detuple and self.ndim != 1:
            raise IndexError('Can only detuple for vectors.')
        
        sentinel = object()
        
        from heapq import heapreplace
        from sys import maxint
        biggest = [(-maxint,sentinel)] * n_biggest
        smallest = [(-maxint,sentinel)] * n_smallest # maintains the negative of the smallest part

        # Filter
        if filter is None:
            items = self.iteritems()
        else:
            items = ((k, v) for k, v in self.iteritems() if filter(k, v))

        # Pop the smallest values out, push larger values in. The heap
        # makes this efficient.
        for k, v in items:
            if n_biggest and v > biggest[0][0]: heapreplace(biggest, (v, k))
            if n_smallest and -v > smallest[0][0]: heapreplace(smallest, (-v, k))

        biggest.sort()
        smallest.sort(reverse=True)
        biggest = [(k, v) for v, k in biggest if k is not sentinel]
        smallest = [(k, -v) for v, k in smallest if k is not sentinel]

        if detuple:
            biggest = [(k, v) for (k,), v in biggest]
            smallest = [(k, v) for (k,), v in smallest]

        return biggest, smallest

    def extremes(self, n=10, top_only=False):
        """
        Extract the interesting parts of this tensor.

        """
        n_biggest = n
        if top_only: n_smallest = 0
        else: n_smallest = n

        biggest, smallest = self.extreme_items(n_biggest, n_smallest)

        if top_only:
            return biggest
        else:
            return smallest + biggest


    def show_extremes(self, n=10, output=sys.stdout):
        """
        Display the `extremes` of this tensor in a nice tabular format.
        """
        extremes = self.extremes(n)
        for key, value in [x for x in extremes if x[1] < 0]:
            if len(key) == 1: key = key[0]
            if isinstance(key, unicode): key = key.encode('utf-8')
            print >> output, "%+5.5f\t%s" % (value, key)
        print >> output
        for key, value in [x for x in extremes if x[1] > 0]:
            if len(key) == 1: key = key[0]
            if isinstance(key, unicode): key = key.encode('utf-8')
            print >> output, "%+5.5f\t%s" % (value, key)

    def unwrap(self):
        """
        Get the object being wrapped by this Tensor, which may be a NumPy
        array or a dictionary.
        """
        return self._data

    def svd(self, k=50, *a, **kw):
        from csc.divisi.svd import svd_sparse
        return svd_sparse(self, k, *a, **kw)

    def _svd(self, k=50, *a, **kw):
        return svd(self, k=k, *a, **kw)

    def _isvd(self, k=50, niter=100, lrate=.001, *a, **kw):
        return isvd(self, k=k, niter=niter, lrate=lrate, *a, **kw)

    def weighted_slice_sum(self, mode, weights,
                           constant_weight=None, ignore_missing=False):
        '''
        Computes the weighted sum of slices along `mode`. Weights are
        specified as (key, weight).

        This is a slight generalization of the concept of an ad hoc category.

        You can also pass in just a list of keys as weights and set
        constant_weight to a value (like 1.0) that you want to weight
        everything by.

        Some items may be missing. Pass ignore_missing=True, and
        you'll get back a tuple: (weighted sum, missing items).
        '''
        # TODO: this can be done more efficiently by avoiding making copies
        # when multiplying.

        if constant_weight is not None:
            if constant_weight is True: constant_weight = 1.0
            weight_iter = ((item, constant_weight) for item in weights)
        else:
            weight_iter = iter(weights)

        res = None
        missing = []

        # Add in other items.
        for key, weight in weight_iter:
            try:
                slice = self.slice(mode, key)
            except KeyError: # key missing
                if not ignore_missing:
                    raise KeyError(key)
                missing.append((key, weight))
            else:
                if res is None: # first successful time
                    res = slice * weight # the multiplication makes it a copy.
                else:
                    res.weighted_iadd(self.slice(mode, key), weight)

        if ignore_missing:
            return res, missing
        else:
            return res

    def means(self):
        '''
        Computes the by-slice means for each dimension.

        Returns numpy arrays, not DenseTensors.
        '''
        import numpy as np
        accumulator_for_mode = [numpy.zeros(n) for n in self.shape]
        for k, v in self.iteritems():
            for accumulator, key in izip(accumulator_for_mode, k):
                accumulator[key] += v
        
        # Divide by the number of items that could have gone into each
        # accumulator slot.
        total_possible_items = np.product(self.shape)
        for mode, accumulator in enumerate(accumulator_for_mode):
            possible_items = total_possible_items / self.shape[mode]
            accumulator /= possible_items
        return accumulator_for_mode
            

    def label_histograms(self):
        '''
        Counts the number and magnitude of items in each row / column.

        Returns (counts, magnitudes).
        '''
        import numpy
        counts = [numpy.zeros(n, dtype=numpy.uint16) for n in self.shape]
        magnitudes = [numpy.zeros(n) for n in self.shape]

        for k, v in self.iteritems():
            vv = v**2
            for count, mag, key in izip(counts, magnitudes, k):
                count[key] += 1
                mag[key] += vv

        return (map(DenseTensor, counts),
                map(DenseTensor, magnitudes))

    # Sparse shape handling (TODO: consistent semantics)
    def ensure_index_is_valid(self, idx):
        '''
        Ensure that an index is valid, possibly updating the shape to reflect that.
        '''
        # Subclasses with adjustable shape should override this.
        shape = self.shape
        for mode in xrange(self.ndim):
            if idx[mode] >= shape[mode]:
                raise ValueError('Index (%d) is out of range (%d) for dimension %d' % (idx[mode], shape[mode], mode))

forward_to_tensor = set([
        'shape', 'ndim', 'has_key', 'default_value', 'density'])

class View(Tensor):
    """
    A *view* is a wrapper around a tensor (always ``self.tensor``)
    that performs some operations (usually ``__getitem__`` and/or
    ``__setitem__``) differently.

    For almost all purposes, it acts like a ``Tensor`` itself. Unknown
    methods are passed through to the underlying ``Tensor``.

    This is just the abstract base class. Some useful ``View``s
    include :class:`csc.divisi.labeled_view.LabeledView` and
    :class:`csc.divisi.labeled_view.NormalizedView`.
    """

    def __init__(self, tensor):
        """
        Create a new View wrapping a Tensor.
        """
        self.tensor = tensor
        self.ndim = tensor.ndim

    def __repr__(self):
        return '<View of %s>' % repr(self.tensor)

    def __getattr__(self, name):
        '''Fall back to the tensor operation.'''
        if name in forward_to_tensor:
            attr = getattr(self.tensor, name)
            #setattr(self, name, attr)
            return attr
        else:
            raise AttributeError(name)

    def __iter__(self):
        return self.tensor.__iter__()

    def __len__(self):
        return len(self.tensor)

    def add_layer(self, cls, *a, **kw):
        """
        Given a class of View, determine where it should go in the stack
        and add it as a layer there.
        """
        try:
            return cls(self, *a, **kw)
        except InvalidLayeringException:
            # Layer it below here.
            return self.layer_on(self.tensor.add_layer(cls, *a, **kw))

    def layer_on(self, tensor):
        # Just make a copy of the same class viewing the new tensor.
        # Subclasses should override this.
        return self.__class__(tensor)

    def unwrap(self):
        """
        Get the object being wrapped by this View. This may be a
        :class:`Tensor`,
        or another :class:`View`.

        Essentially, this removes the outermost View layer, so that you can
        get at what's inside.
        """
        return self.tensor


class TransposedView(View):
    '''
    A simple view that just transposes its input.
    '''
    def __init__(self, tensor):
        if tensor.ndim != 2:
            raise TypeError('Can only transpose matrices.')
        View.__init__(self, tensor)
        r, c = tensor.shape
        self.shape = c, r

    def __getitem__(self, item):
        r, c = item
        return self.tensor[c, r]

    def iteritems(self):
        for (r, c), v in self.tensor.iteritems():
            yield (c, r), v

    def __iter__(self):
        for r, c in self.tensor.iterkeys():
            yield c, r

    def __setitem__(self, key, value):
        r, c = key
        self.tensor[c, r] = value

    def __delitem__(self, key):
        r, c = key
        del self.tensor[c, r]


    
class DictTensor(Tensor):
    '''
    A sparse tensor that stores data in nested dictionaries.

    The first level of dictionaries specifies the rows, the second level
    specifies columns, and so on for higher modes. Therefore, slicing by
    rows is the easiest to do. Despite this, you can slice on any mode,
    possibly returning a :class:`divisi.sliced_view.SlicedView`
    for the sake of efficiency.

    DictTensors can save a lot of memory, can efficiently provide input
    to a Lanczos :ref:`SVD`, and work well with
    :class:`divisi.labeled_view.LabeledView` objects. However, for some
    operations you may need to convert the DictTensor to a
    :class:`DenseTensor`.
    '''

    # Indicates that not all possible keys will necessarily be iterated over.
    # NOTE: not used anywhere yet.
    is_sparse = True

    def __init__(self, ndim, default_value=0.0):
        """
        Create a new, empty DictTensor with *ndim* dimensions.

        Frequently, *ndim* is 2, creating a sparse matrix.

        default_value is the value of all unspecified entries. An SVD will
        only work when *default_value*=0.0.
        """
        self._data = {}
        self.ndim = ndim
        self._shape = [0]*ndim
        self.default_value = default_value

    def __repr__(self):
        return '<DictTensor shape: %s; %d items>' % (repr(self.shape),
                                                     len(self))

    def density(self):
        '''Calculate how dense the tensor is.

        Returns (num specified elements)/(num possible elements).

        Note that some specified elements may be zero.
        '''
        possible_elements = 1
        for dim in self.shape: possible_elements *= dim
        return float(len(self))/possible_elements

    def sparsity(self):
        warn(DeprecationWarning('the proper name for this method is `density`.'))
        return self.density()

    def __getstate__(self):
        return dict(self.__dict__, version=1)

    def __setstate__(self, state):
        version = state.pop('version', None)
        if version is None:
            # Version-less loading
            self.ndim = state['ndim']
            self.default_value = state['default_value']
            self._shape = state['_shape']

            # Try to handle the non-nested format.
            data = state['_data']
            if len(data) > 0 and not isinstance(data.iterkeys().next(), int):
                print 'Attempting to load non-nested DictTensor.'
                self.update(data)
            else:
                # Should be the current format.
                self._data = data
        elif version == 1:
            self.__dict__.update(state)
        else:
            raise TypeError('unsupported version of DictTensor: '+str(version))

    def __len__(self):
        '''
        Compute the number of specified items.

        Note that some specified items may be zero.
        '''
        # If speed is important, store the counter and
        # increment the count every time an item is added.
        def subcount(dct, to_go):
            if to_go == 0:
                return len(dct)
            else:
                return sum(subcount(d, to_go-1) for d in dct.itervalues())
        return subcount(self._data, self.ndim-1)

    def __iter__(self):
        def iter_helper(dct, num_nested, key_base):
            if num_nested == 1:
                for x in dct.__iter__():
                    # TODO: is using append here too slow?
                    yield key_base + (x,)
            else:
                for key, child_dict in dct.iteritems():
                    for x in iter_helper(child_dict, num_nested - 1, key_base + (key,)):
                        yield x
        return iter_helper(self._data, self.ndim, ())

    def itervalues(self):
        def iter_helper(dct, num_nested):
            if num_nested == 1:
                for v in dct.itervalues():
                    yield v
            else:
                for child_dict in dct.itervalues():
                    for x in iter_helper(child_dict, num_nested - 1):
                        yield x
        return iter_helper(self._data, self.ndim)

    def iteritems(self):
        def iter_helper(dct, num_nested, key_base):
            if num_nested == 1:
                for x, v in dct.iteritems():
                    # TODO: is using append here too slow?
                    yield (key_base + (x,)), v
            else:
                for key, child_dict in dct.iteritems():
                    for x in iter_helper(child_dict, num_nested - 1, key_base + (key,)):
                        yield x
        return iter_helper(self._data, self.ndim, ())

    @property
    def shape(self):
        """
        A tuple representing the shape of this tensor.
        """
        return tuple(self._shape)

    # Support negative reference semantics
    # TODO: it would be nice if this method could be eliminated by
    # somehow making _dict_walk directly return a reference to the
    # *value* in the final dictionary instead of the final dictionary.
    def _adjust_index(self, index, dim):
        if index < 0:
            new_index = self.shape[dim] + index
            # Don't allow the tensor to have actual negative indices
            if new_index < 0:
                raise IndexError('List index out of range')
            return new_index
        return index

    def _dict_walk(self, indices, create=False):
        # Walk nested dictionaries to get to the dictionary that
        # contains the value
        cur = self._data
        for i, index in enumerate(indices):
            index = self._adjust_index(index, i)

            try:
                cur = cur[index]
            except KeyError:
                if not create: raise
                cur[index] = {}
                cur = cur[index]
        return cur

    # Note: __contains__ is defined in terms of has_key by DictMixin

    def has_key(self, indices):
        """
        Given a tuple of *indices*, is there a specified value at those indices?
        """
        try:
            self._dict_walk(indices, create=False)
            return True
        except KeyError:
            return False

    def __getitem__(self, indices):
        """Get an item from the dictionary. Return 0.0 if no such entry exists.
        This doesn't bother to check if the element is out of bounds."""
        if not isinstance(indices, (list, tuple)):
            indices = (indices,)

        if self.dims != len(indices):
            raise DimensionMismatch

        try:
            return self._dict_walk(indices)

        except TypeError: # Dictionary raises this when you try and index into it with a slice
            # Two kinds of slices:
            if not isinstance(indices[0], slice)\
                    and all(isinstance(index, slice) for index in indices[1:]):
                # The optimal kind of slice is retrieving an entire
                # first dimension, e.g. tensor[1, :]. This resolves
                # to a dictionary lookup.

                # This is sort of a hack...
                t = DictTensor(self.dims - 1)
                t._data = self._data[indices[0]]
                t._shape = self.shape[1:]
                t.default_value = self.default_value
                return t
            else:
                # The other slice is a SlicedView
                from csc.divisi.sliced_view import SlicedView
                return SlicedView(self, indices)

        except KeyError:
            return self.default_value

    def __setitem__(self, indices, value):
        if not isinstance(indices, (list, tuple)):
            indices = (indices,)

        d = self.ndim
        if len(indices) != d: raise KeyError("You need %d indices" % d)

        s = self._shape
        for mode, idx in enumerate(indices):
            assert isinstance(idx, (int, long))
            if idx >= s[mode]:
                s[mode] = idx + 1

        # Walk nested dictionaries to where the item should
        # go, creating the dictionaries if necessary
        innermost_dict = self._dict_walk(indices[:-1], create=True)
        final_index = self._adjust_index(indices[-1], self.ndim - 1)
        if False:#value == self.default_value:
            if final_index in innermost_dict:
                del innermost_dict[final_index]
        else:
            innermost_dict[final_index] = value

    def __delitem__(self, indices):
        # TODO: update shape correctly.
        ndim = self.ndim
        if len(indices) != ndim: raise DimensionMismatch
        indices = [self._adjust_index(index, i) for i, index in enumerate(indices)]
        cur = self._data
        for i, index in enumerate(indices):
            if i == ndim-1:
                del cur[index]
            else:
                cur = cur[index]

    def purge(self):
        '''
        Removes any values that are specified as the default value.

        Note: this method can't remove any empty rows or columns,
        since that would require changing indices. You may be
        interested in
        :meth:`csc.divisi.labeled_view.LabeledView.with_zeros_removed`.
        '''
        self._recursive_purge(self._data, self.ndim - 1)

    def _recursive_purge(self, data, nesting_level):
        default_value = self.default_value
        if nesting_level == 0:
            # Purge values
            to_purge = [k for k, v in data.iteritems() if v == default_value]
        else:
            to_purge = []
            for k, v in data.iteritems():
                self._recursive_purge(v, nesting_level - 1)
                if len(v) == 0:
                    to_purge.append(k)
        for k in to_purge:
            del data[k]

    def incremental_svd(self, *a, **kw):
        """
        Take the singular value decomposition of this tensor using an
        incremental SVD algorithm.
        """
        from csc.divisi.svd import incremental_svd
        return incremental_svd(self, *a, **kw)

    def to_dense(self):
        """
        Convert this to a :class:`DenseTensor`.
        """
        dense = numpy.zeros(self.shape)
        for idx, value in self.iteritems():
            dense[idx] = value
        return DenseTensor(dense)

    def cmul(self, other):
        '''Multiply by a constant.'''
        res = self.__class__(ndim=self.ndim)
        for i, value in self.iteritems():
            res[i] = value * other
        return res

    # Products:
    # * dot product: vector by vector
    # * matrix product: matrix by {vector, matrix}
    # * tensor products:
    #   - tensordot: tensor by vector (param: mode)
    #   - tensor by matrix: also takes mode param

    def dot(self, other, mode=0, into=None, reverse=False):
        if max(self.ndim, other.ndim) == 1:
            return self._vectordot(other, reverse=reverse)
        if max(self.ndim, other.ndim) == 2:
            return self._2ddot(other, reverse=reverse)
        raise NotImplementedError
#         if into is None:
#             into = DictTensor(2) # FIXME
#         for key, val in self.iteritems():
#             pass#FIXME

    def tensordot(self, other, mode):
        shape = self.shape
        if other.ndim != 1 or shape[mode] != other.shape[0]:
            raise IndexError('Incompatible dimensions for sparse tensordot (%r.tensordot(%r, %d)' % (self, other, mode))
        result_shape = tuple(shape[:mode] + shape[mode+1:])
        result = numpy.zeros(result_shape)
        for key, val in self.iteritems():
            result[key[:mode]+key[mode+1:]] += val * other[key[mode]]
        # Check if the result is a 0-d array, in which case return a scalar
        if(result.shape == ()):
            return result[()]
        return DenseTensor(result)

    def _vectordot(self, other, reverse=False):
        if other.ndim != 1:
            raise IndexError("Incompatible dimensions for sparse vectordot (%r._vectordot(%r)" % (self, other))
        if not isinstance(other, DictTensor):
            other = other.to_sparse()
        if len(other) < len(self):
            other, self = self, other
        return sum(value * other.get(key, 0) for key, value in self.iteritems())

    def _2ddot(self, other, reverse=False):
        if reverse:
            other, self = self, other
        if self.ndim < other.ndim:
            raise IndexError("Incompatible order in sparse 2ddot: (%r._2ddot(%r))" % (self, other))
        if self.shape[1] != other.shape[0]:
            raise IndexError("Dimension Mismatch in sparse 2ddot: (%r._2ddot(%r))" % (self, other))

        result = DictTensor(other.ndim)

        if other.ndim == 1:
            # Simple case:
            for key, value in self.iteritems():
                i, j = key
                result._data[i] = result._data.get(i, 0) + value * other._data.get(j, 0)
        else:
            for key, value in self.iteritems():
                i, j = key
                for k, v2 in other._data.get(j, {}).iteritems():
                    result[(i, k)] = result.get((i, k)) + value * v2

        return result

    def tensor_by_matrix(self, other, mode):
        pass

    def transpose(self):
        '''
        Returns a new DictTensor that is the transpose of this tensor.

        Only works for matrices (i.e., ``tensor.ndim=2``.)
        '''
        if self.ndim != 2:
            raise DimensionMismatch('Can only transpose a 2D tensor')
        tensor = DictTensor(2)
        for key, value in self.iteritems():
            tensor[key[1], key[0]] = value

        return tensor

    @property
    def T(self):
        '''
        Returns a new DictTensor that is the transpose of this tensor.

        Only works for matrices (i.e., ``tensor.ndim=2``.)
        '''
        return self.transpose()

    def scalar_iadd(self, other):
        '''
        Add *other* to every value in this tensor.
        Mutates the value of this tensor.
        '''
        self.default_value += other
        for k in self.iterkeys():
            self[k] += other
        return self

    def tensor_iadd(self, other):
        '''
        Element-by-element tensor addition. For all keys *k* in this
        tensor *t* and the other tensor *o*, set ``t[k] = t[k] + o[k]``.
        '''
        if self.ndim != other.ndim:
            raise DimensionMismatch()
        assert getattr(other, 'default_value', 0) == 0 # Lazy...
        for k, v in other.iteritems():
            if v:
                self[k] += v
        return self

    def __sub__(self, other):
        '''
        Element-by-element tensor subtraction. For all keys *k* in this
        tensor *t* and the other tensor *o*, return a new tensor *r*
        such that ``r[k] = t[k] - o[k]``.
        '''
        res = copy.deepcopy(self)
        res -= other
        return res

    def __isub__(self, other):
        '''
        Element-by-element tensor subtraction. For all keys *k* in this
        tensor *t* and the other tensor *o*, set ``t[k] = t[k] - o[k]``.
        '''
        if self.ndim != other.ndim:
            raise DimensionMismatch()
        assert getattr(other, 'default_value', 0) == 0 # Lazy...
        for k, v in other.iteritems():
            if v:
                self[k] -= v
        return self

    def icmul(self, other):
        def subiter(dct, to_go):
            if to_go == 0:
                for k in dct:
                    dct[k] *= other
            else:
                for d in dct.itervalues():
                    subiter(d, to_go-1)
        subiter(self._data, self.ndim-1)
        return self

    def __neg__(self):
        '''
        Return a new tensor whose values are the negation of this
        tensor's values. That is, return a tensor *r* such that for all keys
        *k*, ``r[k] = -t[k]``, where *t* is this tensor.
        '''
        res = DictTensor(self.ndim, -1*self.default_value)
        for key, value in self.iteritems():
            res[key] = -value
        return res

    def ensure_index_is_valid(self, indices):
        # A bit of code in __setitem__ duplicates this for speed.
        s = self._shape
        for mode, idx in enumerate(indices):
            assert isinstance(idx, (int, long))
            if idx >= s[mode]:
                s[mode] = idx + 1

    def _svd(self, k=50, *a, **kw):
        return dictSvd(self, k=k, *a, **kw)

    def _isvd(self, k=50, niter=100, lrate=.001, *a, **kw):
        return dictIsvd(self, k=k, niter=niter, lrate=lrate, *a, **kw)

try:
    from itertools import product
    def outer_tuple_iterator(shape):
        return product(*map(xrange, shape))
except ImportError:
    def outer_tuple_iterator(shape):
        '''
        Generate all valid indices in a tensor of the given shape.

        >>> list(outer_tuple_iterator((3,)))
        [(0,), (1,), (2,)]
        >>> list(outer_tuple_iterator((2,2)))
        [(0, 0), (0, 1), (1, 0), (1, 1)]
        >>> list(outer_tuple_iterator(()))
        []
        '''
        if len(shape) == 0: return
        idx = [0]*len(shape)
        maxdim = len(shape) - 1
        while True:
            yield tuple(idx)
            idx[maxdim] += 1
            # Wraparound
            cur_dim = maxdim
            while idx[cur_dim] >= shape[cur_dim]:
                idx[cur_dim] = 0
                cur_dim -= 1
                if cur_dim == -1: raise StopIteration
                idx[cur_dim] += 1

def data(tensor):
    """
    Get the chewy (NumPy|dictionary) center of the tensor, if at all possible.
    """
    if hasattr(tensor, '_data'): return tensor._data
    elif hasattr(tensor, 'tensor'): return data(tensor.tensor)
    else: return tensor

def wrap_unless_scalar(result):
    if isscalar(result): return result
    else: return DenseTensor(result)

class DenseTensor(Tensor):
    '''
    A representation of a :class:`Tensor`, based on Numpy arrays.

    DenseTensors can be created from Numpy arrays and converted to
    Numpy arrays. This makes DenseTensors good for performing math
    operations, since it allows you to use Numpy's optimized math
    libraries.
    '''

    def __init__(self, data):
        """Create a DenseTensor from a numpy array."""
        if isinstance(data, DenseTensor):
            self._data = data._data
        else:
            self._data = data
        self.ndim = len(data.shape)


    def __repr__(self):
        return '<DenseTensor shape: %s>' % (repr(self.shape),)


    @property
    def shape(self):
        return self._data.shape

    def __getstate__(self):
        bytes = self._data.tostring()
        shape = self._data.shape
        return dict(bytes=bytes, shape=shape, version=1)

    def __setstate__(self, state):
        version = state.pop('version', None)
        if version is None:
            self.__dict__ = state
        elif version == 1:
            array = numpy.fromstring(state['bytes']).reshape(state['shape'])
            self._data = array
            self.ndim = len(state['shape'])
        else:
            raise ValueError("I don't know how to unpickle this version")

    def __getitem__(self, indices):
        res = self._data[indices]
        if isinstance(indices, (int, long)) or all(isinstance(idx, (int, long)) for idx in indices):
            return res
        else:
            return DenseTensor(res)

    def __setitem__(self, indices, value):
        self._data[indices] = value

    def has_key(self, indices):
        # This is broken in numpy.
        # return indices in self._data
        if not isinstance(indices, (list, tuple)):
            indices = (indices,)
        if len(indices) != self.ndim:
            raise ValueError, 'Wrong number of dimensions'
        for dim, idx in enumerate(indices):
            if not (0 <= idx < self._data.shape[dim]): return False
        return True

    def __iter__(self):
        return outer_tuple_iterator(self.shape)

    def itervalues(self):
        return iter(numpy.reshape(self._data, (-1,)))

    def iteritems(self):
        return izip(iter(self), self.itervalues())

    def iter_dim_keys(self, dim):
        assert(dim >= 0 and dim < self.ndim)
        return xrange(0, self.shape[dim])

    # Math operations
    def dot(self, other):
        if isinstance(other, DictTensor):
            return other.dot(self, reverse=True)
        # FIXME: this will need additional cases for other types of tensors.
        return wrap_unless_scalar(numpy.dot(self._data, data(other)))

    def cmul(self, other):
        return DenseTensor(self._data * other)

    def tensordot(self, other, mode):
        '''This is almost like numpy's tensordot function, but at the moment
        only supports summing over a single mode (axis), specified by the
        integer parameter *mode*.'''
        return wrap_unless_scalar(numpy.tensordot(self._data, data(other), [mode, 0]))

    def __add__(self, other):
        return DenseTensor(self._data + data(other))

    def scalar_iadd(self, other):
        self._data += data(other)
        return self

    def tensor_iadd(self, other):
        self._data += data(other)
        return self

    def icmul(self, other):
        self._data *= other
        return self

    def __neg__(self):
        return DenseTensor(-self._data)

    def transpose(self):
        if self.ndim != 2:
            raise DimensionMismatch('Can only transpose a 2D tensor')
        return DenseTensor(self._data.T)

    @property
    def T(self):
        return self.transpose()

    def to_dense(self):
        return self

    def to_sparse(self, default_value=0.0):
        """
        Convert this to a :class:`DictTensor`.
        """
        result = DictTensor(self.ndim)
        for key, val in self.iteritems():
            if val == default_value:
                continue
            result[key] = val
        return result

    def to_array(self):
        return self._data

    def array_op(self, op, *args, **kwargs):
        """Apply a Numpy operation to this tensor, returning a new DenseTensor."""

        def extract_data(t):
            if isinstance(t, DenseTensor): return t.to_array()
            else: return t

        newargs = [self.to_array()] + [extract_data(a) for a in args]
        result = op(*newargs, **kwargs)
        return DenseTensor(result)

    def concatenate(self, other):
        """
        Make a new DenseTensor containing all the rows in this DenseTensor,
        followed by all the rows in another.

        The Tensors to be concatenated need to have compatible shapes.
        """
        assert self.shape[1:] == other.shape[1:]
        newdata = numpy.concatenate((self._data, other._data))
        return DenseTensor(newdata)

    def k_means_inner(self, k, tolerance, iterlim):
        m, n = self.shape

        # Normalize the data so that we're working with directions
        directions = self._data[:]
        for row in xrange(m):
            norm = math.sqrt(numpy.dot(directions[row], directions[row]))
            if norm == 0: continue
            directions[row] /= norm

        # Initialize to random points chosen from the data
        unique = []
        for i in xrange(m):
            found = False
            for u in unique:
                if numpy.all(directions[i] == u):
                    found = True
                    break
            if not found:
                unique.append(directions[i])

        whichmeans = random.sample(unique, k)
        means = numpy.zeros((k, n))
        for row in xrange(k):
            means[row,:] = whichmeans[row]

        error = 1000.0
        dist = 1000.0
        clusters = numpy.zeros((m,), 'i')
        iteration = 0
        while error > numpy.linalg.norm(means)*tolerance:
            # Find the closest mean for each row
            dots = numpy.dot(directions, means.T)
            dist = 0.0
            for row in xrange(m):
                clusters[row] = numpy.argmax(dots[row])
                themax = numpy.max(dots[row])
                if themax > 1.0: themax = 1.0
                dist += math.acos(themax)
                assert str(dist) != 'nan'

            newmeans = numpy.zeros((k, n))
            # Update the means
            for row in xrange(m):
                newmeans[clusters[row], :] += directions[row, :]

            for row in xrange(k):
                norm = math.sqrt(numpy.dot(newmeans[row], newmeans[row]))
                if norm == 0:
                    newmeans[row, :] = means[row, :]
                else:
                    newmeans[row] /= norm

            error = numpy.linalg.norm(newmeans - means)
            means = newmeans

            iteration += 1
            if iteration > iterlim:
                logging.warn('k_means iteration limit exceeded')
                break

        clusterlists = [[] for i in xrange(k)]
        for row in xrange(m):
            cluster = clusters[row]
            clusterlists[cluster].append(row)

        return DenseTensor(means), clusterlists, dist

    def _svd(self, k=50, *a, **kw):
        from csc.divisi.svd import SVD2DResults
        # perform a NumPy SVD on the data
        u, sigma, vt = numpy.linalg.svd(data(self), full_matrices=False)
        return u.T, sigma, vt

    def k_means(self, k=20, iter=10, tolerance=0.001, iterlim=1000):
        """
        Divide the data into clusters using k-means clustering.

        Returns the pair of lists `(means, clusters)`. For each index *i*,
        `means[i]` is the mean vector of a cluster, and `clusters[i]` is
        the list of items in that cluster.
        """
        best_results = (None, None)
        best_dist = 1e300
        for iteration in xrange(iter):
            means, clusters, dist = self.k_means_inner(k, tolerance, iterlim)
            print dist
            if dist < best_dist:
                best_dist = dist
                best_results = (means, clusters)
        return best_results
    
    # FIXME: be consistent about whether these functions return copies.

