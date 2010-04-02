from csc.divisi.tensor import View, outer_tuple_iterator
from math import ceil
import numpy

class SlicedView(View):
    '''A view representing a slice of a tensor.'''

    # Specify that this view maps numeric labels to (the same) numeric labels.
    input_is_numeric = True
    output_is_numeric = True

    def __init__(self, tensor, slice_indices):
        View.__init__(self, tensor)
        self._slice = slice_indices
        self._tensor_ndim = len(slice_indices)

        # Compute the slice's dimensionality and shape.
        ndim = 0
        shape = []
        offset = numpy.zeros(self._tensor_ndim)
        interval = numpy.zeros(self._tensor_ndim)
        dim_map = {}
        base_indices = numpy.zeros(self._tensor_ndim)
        for i in xrange(0, self._tensor_ndim):
            index = slice_indices[i]
            if hasattr(index, 'indices'):
                start, end, step = index.indices(tensor.shape[i])
                shape.append(int(ceil(float(end - start)/step)))
                offset[ndim] = start
                interval[ndim] = step
                dim_map[ndim] = i
                ndim += 1
            else:
                base_indices[i] = index

        self.ndim = ndim
        self.shape = tuple(shape)
        # These vars help perform the mapping from slice indices -> tensor indices
        self._offset = offset
        self._interval = interval
        self._dim_map = dim_map
        self._base_indices = base_indices

    def __repr__(self):
        return '<SlicedView of %r, slice=%r, shape=%r>' % (self.tensor, self._slice, self.shape)

    def _iter_all(self):
        # Iterates over all possible indices, instead of just the defined
        # indices
        return outer_tuple_iterator(self.shape)

    def __len__(self):
        count = 0
        for x in self.__iter__():
            count += 1
        return count

    def __iter__(self):
        for indices in self._iter_all():
            if self.tensor.has_key(self._translate_indices(indices)):
                yield indices

    def has_key(self, indices):
        return self.tensor.has_key(self._translate_indices(indices))

    def _translate_indices(self, indices):
        # TODO: handle slices
        # Note: doesn't bother checking bounds, so theoretically references
        # to indices off the end of the slice could work
        tensor_indices = self._base_indices.copy()
        for i in xrange(0, self.ndim):
            tensor_indices[self._dim_map[i]] = self._offset[i] + self._interval[i]*indices[i]
        return tuple(tensor_indices)

    def __getitem__(self, indices):
        if not isinstance(indices, (list, tuple)):
            indices = (indices,)

        return self.tensor[self._translate_indices(indices)]

    def __setitem__(self, indices, value):
        self.tensor[self._translate_indices(indices)] = value
