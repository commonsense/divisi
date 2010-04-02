from csc.divisi.tensor import View
from csc.divisi.exceptions import InvalidLayeringException

EPSILON = 0.000000001   # for floating point comparisons

from numpy import zeros
import numpy as np
from math import sqrt, log

class BaseNormalizedView(View):
    '''
    The base class for normalized views -- views that scale the values in a
    tensor in some way.
    '''
    # Specify that this view maps numeric labels to (the same) numeric labels.
    input_is_numeric = True
    output_is_numeric = True

    # Default to normalizing on mode 0.
    default_modes = 0

    def __init__(self, tensor, mode=None):
        '''
        * tensor: tensor to view.
        * mode: which mode to normalize over (0 = rows, 1 = columns, etc.)
        '''
        if len(tensor) == 0:
            raise ValueError('Tensor ' + repr(tensor) + ' is empty.')

        # If this isn't a MeanSubtractedView, the underlying tensor can't be a normalized view.
        if not isinstance(self, MeanSubtractedView) and tensor.stack_contains(BaseNormalizedView):
            raise TypeError('Already normalized; .unnormalized() first.')

        # The underlying tensor must be numeric.
        for part in tensor.example_key():
            if not isinstance(part, (int, long)):
                raise InvalidLayeringException

        View.__init__(self, tensor)
        self.base = 0.0
        self._normalize_mode = mode

        self.refresh_norms()
        # FIXME: register change notification.

    def __repr__(self):
        return '<%s of %r, mode=%r, base=%s>' % (
            self.__class__.__name__, self.tensor, self._normalize_mode, self.base)

    def __setitem__(self, unused, _):
        raise TypeError('Normalized views are read-only.')

    @property
    def normalize_modes(self):
        """
        Which modes of the tensor is normalized (0 for rows, 1 for columns,
        etc.)
        """
        modes = self._normalize_mode
        if modes is None: modes = self.default_modes
        if not isinstance(modes, (list, tuple)):
            modes = [modes]
        return modes

    @property
    def norms(self):
        """
        A list of the normalization factors that were computed for each
        row/column/whatever.
        """
        # FIXME: this method should be in a subclass only.
        return self._norms

    def item_changed(self, indices, oldvalue, newvalue):
        # FIXME: nobody ever even complained that this doesn't
        # work. Maybe change notification just isn't worth it, and
        # normalized views are just immutable.
        self.update_norm(indices, oldvalue, newvalue)

    def update_norm(self, indices, prev, current):
        raise NotImplementedError

    def refresh_norms():
        """
        Compute all of the normalization factors from scratch.
        """
        import inspect
        caller = inspect.getouterframes(inspect.currentframe())[1][3]
        raise NotImplementedError(caller + ' must be implemented in subclass')

    def unnormalized(self):
        return self.tensor


class NormalizedView(BaseNormalizedView):
    '''A normalized view of a tensor.

    The norm is a Euclidean norm (sqrt of sum of squares) over one or
    more dimensions.

    To demonstrate its use, let's create a demonstration ordinary tensor:
    >>> a = SparseLabeledTensor(ndim=2)
    >>> for fruit in ['apple', 'banana', 'pear', 'peach']:
    ...     for color in ['red', 'yellow', 'green']:
    ...        a[fruit, color] = 1

    Then we can make the normalized view.
    >>> an = NormalizedView(a, mode=0)

    Getting at item returns the normalized value. The dimension
    along which it normalizes is specified in the constructor;
    it defaults to 0 (the first dimension).

    >>> an['apple', 'red'] * sqrt(3)
    1.0

    You can also specify a base norm that is added to every vector.
    It is as if every vector had an extra element of base**2. It
    defaults to 0.
    >>> an.base
    0.0
    >>> an.base = 1
    >>> an['apple', 'red'] * sqrt(4)
    1.0

    You can also supply a list of modes to normalize over.
    '''

    def refresh_norms(self):
        tensor = self.tensor
        modes = self.normalize_modes
        norms = [zeros((tensor.shape[mode],)) for mode in modes]
        for key, value in self.tensor.iteritems():
            for mode in modes:
                # FIXME: I want to change this to enumerate(modes),
                # but the `.index` isn't covered by a test and I'd
                # mess it up. -kca
                norms[modes.index(mode)][key[mode]] += value*value
        self._norms = norms

    def update_norm(self, indices, prev, current):
        modes = self.normalize_modes
        # Handle growing the norms array.
        for i_mode, mode in enumerate(modes):
            index = indices[mode]
            while index >= self._norms[i_mode].shape[0]:
                self._norms[i_mode].resize((2*self._norms[i_mode].shape[0],))
            self._norms[i_mode][index] += (current*current - prev*prev)

    def __getitem__(self, indices):
        # FIXME: this doesn't work well for slices.
        if not isinstance(indices, (list, tuple)):
            indices = (indices,)
        modes = self.normalize_modes
        norm = self.base ** 2 + sum(self._norms[i_mode][indices[mode]]
                                    for i_mode, mode in enumerate(modes))
        data = self.tensor[indices]
        #if abs(data) < EPSILON and abs(norm) < EPSILON: return 0.0
        return data / sqrt(norm)

    def iteritems(self):
        modes = self.normalize_modes
        base_squared = self.base ** 2
        for indices, value in self.tensor.iteritems():
            norm = base_squared + sum(self._norms[i_mode][indices[mode]]
                                      for i_mode, mode in enumerate(modes))
            yield indices, value / sqrt(norm)

    ### Extraction and views
    def layer_on(self, tensor):
        # FIXME: This looks shady.
        return NormalizedView(self.tensor, self.modes)

    def _svd(self, **kw):
        if list(self.normalize_modes) == [0]:
            return self.tensor._svd(row_factors=self.norms[0], **kw)
        return super(NormalizedView, self)._svd(**kw)

    def _isvd(self, **kw):
        if list(self.normalize_modes) == [0]:
            return self.tensor._isvd(row_factors=self.norms[0], **kw)
        return super(NormalizedView, self)._isvd(**kw)

    # All other operations fall back to the tensor.


class MeanSubtractedView(BaseNormalizedView):
    '''Subtracts out the mean of each row and/or column.

    Uses offset parameters to the SVD to compute this efficiently.
    '''
    default_modes = 0
    def refresh_norms(self):
        self.means = self.tensor.means()

    def __getitem__(self, indices):
        # FIXME: slices.
        data = self.tensor[indices]
        for mode in self.normalize_modes:
            # not in-place, in case of slicing.
            data = data - self.means[mode][indices[mode]]
        return data

    # FIXME: iteritems is incorrect if the tensor is not dense, since
    # unspecified items will probably be non-zero.

    def _svd(self, **kw):
        modes = self.normalize_modes
        if not all(0 <= m <= 1 for m in modes):
            raise NotImplementedError
        if 0 in modes:
            kw['offset_for_row'] = -self.means[0]
        if 1 in modes:
            kw['offset_for_col'] = -self.means[1]
        print kw
        return self.tensor._svd(**kw)

    def _isvd(self, **kw):
        raise NotImplementedError


class OldAvgNormalizedView(BaseNormalizedView):
    '''A NormalizedView that makes the average to be zero, instead of using a Euclidean norm.'''
    def refresh_norms(self):
        tensor = self.tensor
        mode = self._normalize_mode
        if not isinstance(mode, list): modes = [mode]
        else: modes = mode
        norms = [zeros((tensor.shape[mode],2)) for mode in modes]
        for key, value in self.tensor.iteritems():
            for mode in modes:
            # sum, count
                norms[modes.index(mode)][key[mode]][0] += value
                norms[modes.index(mode)][key[mode]][1] += 1
        self._norms = norms

    def __getitem__(self, indices):
        mode = self._normalize_mode
        if not isinstance(mode, list): modes = [mode]
        else: modes = mode
        norm = self.base
        for i, i_mode in [(indices[mode], modes.index(mode)) for mode in modes]:
            norm += self._norms[i_mode][i][0] / self._norms[i_mode][i][1]
        data = self.tensor[indices]
        if abs(data) < EPSILON and abs(norm) < EPSILON: return 0.0
        return data - norm


class TfIdfView(BaseNormalizedView):
    ''' Normalizes by document length (the "term frequency"/tf part)
    and by term importance, measured by what fraction of the documents
    the term appears in (the "inverse document frequency"/idf part).

    In the resulting matrix, the entry ``(term, document)`` is given
    by: ``tf(term, document) * idf(term)``, where ``tf(term, document)
    = occurrances(term, document) / occurrances(*, document)`` and
    ``idf(term) = log(num_documents /
    num_docs_that_contain_term(term)``.
    '''
    def __init__(self, tensor, transposed=False):
        '''
        transposed: if True, then the matrix is document by term instead.
        '''
        self.transposed = transposed # before the super call because
                                     # super calls refresh_norms.
        super(TfIdfView, self).__init__(tensor) # Modes don't really make sense.

    def __repr__(self):
        return 'TfIdfView(%r, transposed=%r)' % (self.tensor, self.transposed)
    
    def refresh_norms(self):
        tensor, transposed = self.tensor, self.transposed
        if transposed:
            documents, terms = self.tensor.shape
        else:
            terms, documents = self.tensor.shape
        self.num_documents = documents

        # Compute aggregate counts.
        self.counts_for_document = counts_for_document = zeros((documents,))
        self.num_docs_that_contain_term = num_docs_that_contain_term = zeros((terms,), dtype=np.uint32)
        for (term, document), value in self.tensor.iteritems():
            if transposed:
                document, term = term, document
            counts_for_document[document] += value
            num_docs_that_contain_term[term] += 1

    def tf(self, term, document):
        if self.transposed:
            term_count = self.tensor[document, term]
        else:
            term_count = self.tensor[term, document]
        #if (term_count < EPSILON or self.counts_for_document[document] < EPSILON):
            #return 0.0
        return term_count / self.counts_for_document[document]

    def idf(self, term):
        #if (self.num_docs_that_contain_term[term] < EPSILON): return float('-inf')
        return log(self.num_documents / self.num_docs_that_contain_term[term])

    def __getitem__(self, indices):
        if self.transposed:
            document, term = indices
        else:
            term, document = indices
        return self.tf(term, document) * self.idf(term)

