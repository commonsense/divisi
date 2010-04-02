from csc.divisi.tensor import View, DictTensor
from csc.divisi.labeled_view import LabeledView
from csc.divisi.ordered_set import IdentitySet, OrderedSet
from csc.divisi.unfolded_set import UnfoldedSet

class UnfoldedView(View):
    def __init__(self, tensor, dim):
        View.__init__(self, tensor)
        if dim >= tensor.ndim:
            raise IndexError('Unfolding index (%s) out of bounds (%s dims)' %
                             (dim, tensor.ndim))
        self.dim = dim
        self.otherdims = [x for x in range(self.tensor.ndim) if x != dim]

    def __repr__(self):
        return u'<UnfoldedView of %r on dimension %s>' % (self.tensor, self.dim)

    def __iter__(self):
        return (self.label_for(idx) for idx in iter(self.tensor))

    @property
    def shape(self):
        s = self.tensor.shape
        prod = 1
        for dim in self.otherdims:
            prod *= s[dim]
        return (s[self.dim], prod)

    def label_for(self, idx):
        dim = self.dim
        return (idx[self.dim],
                tuple([idx[d] for d in range(self.ndim) if d != dim]))

    def index_for(self, label):
        mainIdx, others = label
        # Splice the main index into "others"
        dim = self.dim
        return others[0:dim] + (mainIdx,) + others[dim:]

    def __getitem__(self, label):
        return self.tensor[self.index_for(label)]

    def __setitem__(self, label, val):
        self.tensor[self.index_for(label)] = val


    def compact_to(self, dest):
        '''Put a 2D compaction into the other tensor. The first dimension is the
        main index; the second dimension is computed from the remaining indices
        by considering the indices as a number with a "digit" for each index,
        and place values equal to the size of the corresponding dimension.

        Dimension sizes are computed as a cumulative product, big endian.

        This function is probably only useful internally, but who knows...'''
        mode = self.dim
        ndim = self.ndim
        shape = self.tensor.shape

        mytensor = self.tensor
        if hasattr(mytensor, 'label_lists'):
            mytensor = mytensor.tensor

        for idx, val in mytensor.iteritems():
            mainIdx = idx[mode]
            cum = 0
            for dim in range(ndim):
                if dim == mode: continue
                cum *= shape[dim]
                cum += idx[dim]
            dest[mainIdx, cum] = val

        return dest


    def svd(self, k=50, normalized=True):
        '''Run an SVD on this unfolding. Compacts, runs, and returns an
        SVD2DResults.'''
        # Set up a LabeledView to map column indices from unfolded products
        # to unique indices.
        col_indices = OrderedSet()
        compact = LabeledView(DictTensor(2),
                              [IdentitySet(0), col_indices])
        self.compact_to(compact)

        if normalized:
            compact = compact.normalized(mode=0)

        svd = compact.svd(k)

        # Wrap the output so that the labeling all works out.
        if hasattr(self.tensor, '_labels'):
            # Case for labeled view beneath
            # FIXME: try not to rely on private vars.
            # TODO: it would be nice to factor this in such a way that we
            #  didn't have to worry about the labeling case here.
            u = LabeledView(svd.u, [self.tensor._labels[self.dim], None])
            v = LabeledView(svd.v, [UnfoldedSet.from_unfolding(self.dim, self.tensor.label_sets()), None])
        else:
            u = svd.u
            v = LabeledView(svd.v,
                            [UnfoldedSet.from_unfolding(self.dim,
                                                        [IdentitySet(dim)
                                                         for dim in self.tensor.shape]),
                             None])

        from csc.divisi.svd import SVD2DResults
        return SVD2DResults(u, v, svd.svals)
