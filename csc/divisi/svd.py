from csc.divisi.tensor import Tensor, DenseTensor, outer_tuple_iterator, TheEmptySlice
#from csc.divisi.labeled_tensor import DenseLabeledTensor
from itertools import imap
import numpy
from math import sqrt
from warnings import warn
import logging
from csc.divisi.exceptions import DimensionMismatch
import random
import sys
EPSILON = 0.000000001   # for floating point comparisons


class ReconstructedTensor(Tensor):
    '''
    A ReconstructedTensor is a wrapper around a
    :class:`SVDResults` object. The
    ReconstructedTensor behaves like the tensor constructed by
    multiplying together the U, Sigma and V matrices (reconstructing
    the original matrix from its SVD).

    ReconstructedTensors require much less storage space than the
    actual reconstructed tensor would require. However, access times
    for a ReconstructedTensor are slower, since accessing an entry
    requires computing a dot product. To mitigate this problem,
    ReconstructedTensors cache the values of recently-accessed entries.
    '''
    def __init__(self, svd_results):
        '''
        The constructor creates a ReconstructedTensor from
        a :class:`SVDResults` object.
        '''
        self.svd = svd_results
        self.a = svd_results.orig
        self.r = svd_results.r
        self.clear_cache()

    def __repr__(self):
        return u'<ReconstructedTensor from %r>' % (self.svd,)

    def clear_cache(self):
        '''
        Clear the cache of recently computed entries. (The cache can
        become very large, so this may free up a significant amount of
        memory.)
        '''
        self._cache = {}

    def __getitem__(self, key):
        # Ken says: I did this in a very wrong way that happens to work
        # in one case.
        if key not in self._cache:
            a = self.a
            nskipped = 0
            to_smooth = []
            for i, k in enumerate(key):
                if isinstance(k, slice):
                    nskipped += 1
                    to_smooth.append(i)
                    continue
                smoothing = self.svd.smoothing_slice(i, k)
                a = a.tensordot(smoothing, nskipped)
            for i, dim in enumerate(to_smooth):
                smoothing = self.svd.smoothing_mat(dim)
                a = a.tensordot(smoothing, i)
            self._cache[key] = a
        return self._cache[key]


class SVDResults(object):
    """
    Stores the results of an arbitrary higher-order SVD. If you are simply
    taking the SVD of a matrix, you are probably looking for the SVD2DResults
    class.

    self.r is a list of SVD2DResults, one for each unfolding of the tensor.
    """
    def __init__(self, orig, sub_results):
        self.orig = orig
        self.r = sub_results

    @property
    def reconstructed(self):
        return ReconstructedTensor(self)

    @property
    def core(self):
        raise NotImplementedError

    def smoothing_slice(self, n, idx):
        u = self.r[n].u
        ui = u[idx, :]
        return u*ui

    def smoothing_mat(self, n):
        u = self.r[n].u
        return u*u.transpose()


class SVD2DResults(object):
    '''
    This class wraps the *U*, Sigma (*S*) and *V* matrices that result from
    computing an SVD. An SVD decomposes the matrix *A* such that ``A =
    U*S*(V^T)``. Both *U* and *V* are orthonormal matrices, and *S* is a
    diagonal matrix.

    This class also provides utility methods to make common
    SVD-related math easy.
    '''

    def __init__(self, u, v, svals):
        '''
        The constructor makes an SVD2DResults object from the matrices created
        by an SVD. Note that svals is a 1-D vector of sigma values from
        the SVD.
        '''
        self.u = u
        self.v = v
        self.svals = svals
        self.clear_cache()

    def __repr__(self):
        return u'<SVD2DResults: %d by %d, rank=%d>' % (self.u.shape[0], self.v.shape[0], len(self.svals))

    @property
    def reconstructed(self):
        return Reconstructed2DTensor(self)

    def clear_cache(self):
        '''
        Delete all cached tensors from this object.
        '''
        self._core, self._weighted_u, self._weighted_v = None, None, None

    @property
    def core(self):
        '''
        Get the core tensor. The core tensor is the diagonal *S*
        matrix from an SVD.
        '''
        if self._core is None:
            self._core = DenseTensor(numpy.diag(self.svals.to_dense()._data))
        return self._core

    @property
    def weighted_u(self):
        '''
        Return ``U * S``, the result of weighting each column of the *U*
        matrix by its corresponding sigma value.
        '''
        if self._weighted_u is None:
            self._weighted_u = self.get_weighted_u()
        return self._weighted_u

    def get_weighted_u(self):
        return DenseTensor(self.u.unwrap() * self.svals.unwrap())

    @property
    def weighted_v(self):
        '''
        Return `V * S`, the result of weighting each column of the *V*
        matrix by its corresponding sigma value.
        '''
        if self._weighted_v is None:
            self._weighted_v = self.get_weighted_v()
        return self._weighted_v

    def get_weighted_v(self):
        return DenseTensor(self.v.unwrap() * self.svals.unwrap())

    def u_dotproducts_with(self, vec):
        '''
        Return the dot product of *vec* with every weighted *u*
        vector.
        '''
        return self.weighted_u * vec
    
    def u_angles_to(self, vec):
        '''
        Return the cosine of the angle between *vec* and every weighted
        *u* vector. This value can be used as a similarity measure that
        ranges from -1 to 1.
        '''
        u = self.weighted_u
        angles = self.u_dotproducts_with(vec)
        vec_magnitude = vec.norm()
        if (vec_magnitude < EPSILON):
            for key, value in angles.iteritems():
                angles[key] = 0.0
            return angles
        # Normalize distances to get the cos(angles)
        for key, value in angles.iteritems():
            u_vector_magnitude = self.weighted_u_vec(key[0]).norm()
            if (u_vector_magnitude > EPSILON):
                angles[key] = float(value)/(u_vector_magnitude*vec_magnitude)
            else:
                angles[key] = 0.0

        return angles

    def weighted_u_vec(self, idx, default_zeros=False):
        '''
        Get the weighted *u* vector with key *idx*.

        If *idx* is not present and default_zeros is true, returns a
        vector of zeros.
        '''
        try:
            return self.weighted_u[idx, :]
        except KeyError:
            if not default_zeros: raise
            return DenseTensor(numpy.zeros((len(self.svals),))).labeled([None])

    def v_dotproducts_with(self, vec):
        '''
        Compute the dot product of *vec* with every weighted *v*
        vector.
        '''
        return self.weighted_v * vec

    def v_angles_to(self, vec):
        '''
        Return the cosine of the angle between *vec* and every weighted
        *v* vector. This value can be used as a similarity measure that
        ranges from -1 to 1.
        '''
        v = self.weighted_v
        angles = self.v_dotproducts_with(vec)
        vec_magnitude = sqrt(vec*vec)
        if (vec_magnitude < EPSILON):
            for key, value in angles.iteritems():
                angles[key] = 0.0
            return angles
        # Normalize distances to get the cos(angles)
        for key, value in angles.iteritems():
            v_vector_magnitude = sqrt(v[key[0], :]*v[key[0], :])
            if (v_vector_magnitude > EPSILON):
                angles[key] = float(value)/(v_vector_magnitude*vec_magnitude)
            else:
                angles[key] = 0.0
        return angles

    def weighted_v_vec(self, idx, default_zeros=False):
        '''
        Get the weighted *v* vector with key *idx*.

        If *idx* is not present and default_zeros is true, returns a
        vector of zeros.
        '''
        try:
            return self.weighted_v[idx, :]
        except KeyError:
            if not default_zeros: raise
            return DenseTensor(numpy.zeros((len(self.svals),))).labeled([None])

    def get_ahat(self, indices):
        '''
        Get the value of one entry of the reconstructed matrix (``A^hat
        = U * S * V^T``).
        '''
        i0, i1 = indices
        try:
            return self.u[i0, :]*self.core*self.v[i1, :]
        except KeyError:
            return 0

    def u_similarity(self, idx1, idx2):
        '''
        Get the similarity of two rows of the weighted *U*
        matrix. Similarity is measured by the dot product of the two
        *u* vectors.
        '''
        return self.u[idx1, :]*self.core*self.u[idx2, :]

    def v_similarity(self, idx1, idx2):
        '''
        Get the similarity of two rows of the weighted *V*
        matrix. Similarity is measured by the dot product of the two
        *v* vectors.
        '''
        return self.v[idx1, :]*self.core*self.v[idx2, :]

    def summarize_axis(self, axis, k=None, u_only=False, screen_width=79, output=sys.stdout):
        if k is None: k = len(self.svals)
        else: k = min(k, len(self.svals))

        if isinstance(axis, int):
            print "\nAxis %d (sigma=%5.5f)" % (axis, self.svals[axis])
            if u_only:
                self.u[:,axis].show_extremes(10)
                return

            u_slice, v_slice = self.u[:, axis], self.v[:, axis]

        else:
            print 'Ad-hoc axis (magnitude=%5.5f)' % axis.norm()
            assert not u_only # not implemented yet.
            # TODO: this uses sigma-weighting. Should it?
            u_slice = self.u_dotproducts_with(axis)
            v_slice = self.v_dotproducts_with(axis)

        u_extremes, v_extremes = u_slice.extremes(), v_slice.extremes()

        # Format in two columns
        if screen_width is not None:
            max_width = (screen_width - 2) / 2
        else:
            max_width = None

        def fit_to_width(s, max_width=max_width):
            if max_width is None: return s
            if len(s) > max_width:
                return s[:max_width-3]+'...'
            return s.ljust(max_width)

        def utf8(s):
            if isinstance(s, unicode): return s.encode('utf-8')
            else: return str(s)

        def fix_key(key):
            if not isinstance(key, tuple): return key
            if len(key) == 1: key = key[0]
            if not isinstance(key, tuple): return key
            # Give special formatting to known features.
            if len(key) == 3:
                # FIXME: Features should be objects with this method.
                typ, rel, concept = key
                if typ == 'left':
                    key = u'%s\\%s' % (concept, rel)
                elif typ == 'right':
                    key = u'%s/%s' % (rel, concept)
            return key

        for idx in xrange(20):
            if idx == 10: # halfway through
                print >> output

            u_key, u_value = u_extremes[idx]
            v_key, v_value = v_extremes[idx]

            left = "%+9.5f %s" % (u_value, utf8(fix_key(u_key)))
            right = "%+9.5f %s" % (v_value, utf8(fix_key(v_key)))
            print >> output, '%s  %s' % (fit_to_width(left),
                                         fit_to_width(right))


    def summarize(self, k=None, u_only=False, screen_width=80, output=sys.stdout):
        '''
        For each axis (up to the *k*-th axis), print the items of the
        *u* and *v* matrices with the largest components on the axis.
        '''
        if k is None: k = len(self.svals)
        else: k = min(int(k), len(self.svals))
        for axis in range(k):
            self.summarize_axis(axis, k, u_only, screen_width, output)

    def safe_svals(self):
        '''
        Get the sigma values as an array of floats, no matter what they were before.
        '''
        return numpy.array([float(self.svals[i]) for i in range(len(self.svals))])

    def export_svdview(self, outfn, packed=True, **kw):
        ''' Output a data file suitable for use with svdview. The data
        is saved to the file named _outfn_. See the documentation in
        :mod:`export_svdview <csc.divisi.export_svdview>` for more
        information on the ``denormalize`` and ``packed`` options.
        '''
        import export_svdview
        if packed:
            export_svdview.write_packed(self.u, outfn, **kw)
        else:
            export_svdview.write_tsv(self.u, outfn, **kw)

    def slice(self, k=None, rows=None, cols=None):
        '''
        Slice out a part of the SVD results, fewer svals, rows, and/or columns.

        k : int - how many svals to keep. None -> keep all.
        rows : [int] - a list of rows to keep. None -> keep all.
        cols : [int] - a list of cols to keep. None -> keep all.
        '''
        u, v, svals = self.u.unwrap(), self.v.unwrap(), self.svals.unwrap()
        if svals is not None:
            svals = svals[:k]
            u = u[:,:k]
            v = v[:,:k]
        if rows is not None:
            u = u[rows,:]
        if cols is not None:
            v = v[cols,:]
        return SVD2DResults(DenseTensor(u), DenseTensor(v), DenseTensor(svals))

    def denormalized(self, norms):
        row_norms, col_norms = norms
        u, v, svals = self.u.unwrap(), self.v.unwrap(), self.svals.unwrap()

        import numpy as np
        if row_norms is not None:
            u = self.u.unwrap() * row_norms[:, np.newaxis]
        if col_norms is not None:
            v = self.v.unwrap() * col_norms[:, np.newaxis]
        return SVD2DResults(DenseTensor(u), DenseTensor(v), DenseTensor(svals))
        

    # TODO: u_distances_to and v_distances_to had these notes:

    # Once the old use of this method is removed, we can replace
    # it with something that actually behaves according to its
    # name, i.e., returns _||vec - ((vec * u) * u^hat)||_, that is
    # it returns the magnitude of the component of _vec_
    # orthogonal to _u_.

    # Once the old use of this method is removed, we can replace
    # it with something that actually behaves according to its
    # name, i.e., returns _||vec - ((vec * u) * u^hat)||_, that is
    # it returns the magnitude of the component of _vec_
    # orthogonal to _u_.


class Reconstructed2DTensor(Tensor):
    '''
    This class wraps the results of a 2-D SVD (represented by
    an :class:`SVD2DResults` object) and
    behaves like the :class:`Tensor` created
    by multiplying together the *U*, *S* and *V* matrices of the
    SVD result.

    Reconstructed2DTensors require much less storage space than the
    actual reconstructed tensor. However, access times
    for a Reconstructed2DTensor are slower, since accessing an entry
    requires computing a dot product.
    '''
    ndim = 2
    def __init__(self, svd):
        '''
        The constructor creates a Reconstructed2DTensor from
        a :class:`SVD2DResults` object.
        '''
        self.svd = svd
        self.shape = svd.u.shape[0], svd.v.shape[0]

    def __repr__(self):
        return u'<%r shape=%r svd=%r>' % (self.__class__, self.shape, self.svd)

    def __len__(self):
        return self.shape[0] * self.shape[1]

    def __iter__(self):
        return imap(lambda x: (self.svd.u.label(0, x[0]), self.svd.v.label(0, x[1])), outer_tuple_iterator(self.shape))

    def __getitem__(self, idx):
        if len(idx) != 2:
            raise DimensionMismatch
        i0, i1 = idx

        if i0 == TheEmptySlice and i1 == TheEmptySlice:
            return self
        elif i0 == TheEmptySlice:
            return self.svd.weighted_u*self.svd.v[i1, :]
        elif i1 == TheEmptySlice:
            return self.svd.v*self.svd.weighted_u[i0, :]
        elif isinstance(i0, slice) or isinstance(i1, slice):
            raise ValueError('Only the `:` slice supported.')
        else:
            return self.svd.get_ahat(idx)

    def has_key(self, idx):
        if len(idx) != 2: return False
        return idx[0] in self.svd.u.dim_keys(0) and idx[1] in self.svd.v.dim_keys(0)

    def iter_dim_keys(self, dim):
        if dim == 0:
            return self.svd.u.iter_dim_keys(0)
        elif dim == 1:
            return self.svd.v.iter_dim_keys(0)
        else:
            raise DimensionMismatch

    def sample(self, source, power=2):
        def sample_dist(vec):
            """
            Samples from a labeled vector.
            """
            data = vec._data
            if random.randint(0, 1): data = -data
            no_neg = numpy.maximum(vec._data, 0)
            cumulprob = numpy.cumsum(no_neg) / numpy.sum(no_neg)
            index = numpy.searchsorted(cumulprob, random.random())
            return vec.label(0, index), no_neg[index]

        while True:
            which_sval, prob1 = sample_dist(self.svd.svals)
            which_u, prob2 = sample_dist(self.svd.u[:, which_sval])
            which_v, prob3 = sample_dist(self.svd.v[:, which_sval])

            # don't sample things we already knew
            if source[which_u, which_v] > 0: continue
            if (isinstance(which_v, basestring) and
                (which_v.startswith('InheritsFrom') or
                 which_v.endswith('InheritsFrom'))):
                continue
            if (isinstance(which_v, tuple) and
                (which_v[1] == 'InheritsFrom')): continue

            u_modified = numpy.maximum(self.svd.u[which_u, :]._data, 0)
            v_modified = numpy.maximum(self.svd.v[which_v, :]._data, 0)
            u_modified2 = numpy.minimum(self.svd.u[which_u, :]._data, 0)
            v_modified2 = numpy.minimum(self.svd.v[which_v, :]._data, 0)
            sample_prob1 = numpy.dot(u_modified, v_modified)\
                         * self.svd.svals[which_sval]
            sample_prob2 = numpy.dot(u_modified2, v_modified2)\
                         * self.svd.svals[which_sval]
            sample_prob = sample_prob1 + sample_prob2
            u_row = self.svd.u[which_u, :]._data
            v_row = self.svd.v[which_v, :]._data
            actual_prob = max(0, numpy.dot(u_row, v_row)
                                 * self.svd.svals[which_sval])

            acceptance_prob = actual_prob / sample_prob
            assert acceptance_prob <= 1.000001
            if random.random() < acceptance_prob:
                if power == 1 or random.random() < actual_prob:
                    return (which_u, which_v, actual_prob)

class DenormalizedReconstructed2DTensor(Reconstructed2DTensor):
    def __init__(self, svd, norms, normdim=0):
        Reconstructed2DTensor.__init__(self, svd)
        self.norms = norms
        if normdim != 0:
            raise NotImplementedError('normdim != 0 not yet implemented.')

    def __getitem__(self, idx):
        if len(idx) != 2:
            raise DimensionMismatch
        i0, i1 = idx

        if i0 == TheEmptySlice and i1 == TheEmptySlice:
            return self
        elif i0 == TheEmptySlice:
            return self.svd.weighted_u*self.svd.v[i1, :] * self.norms
        elif i1 == TheEmptySlice:
            return self.svd.v * self.svd.weighted_u[i0, :] * self.norms[i0]
        elif isinstance(i0, slice) or isinstance(i1, slice):
            raise ValueError('Only the `:` slice supported.')
        else:
            return self.svd.u[i0, :]*self.svd.core*self.svd.v[i1, :] * self.norms[i0]

def svd_sparse(tensor, k=50, **kw):
    '''
    Compute the SVD of sparse tensor *tensor* using Lanczos' algorithm.
    The optional parameter *k* is the number of sigma values to retain
    (default value is 50).
    '''
    if tensor.ndim < 2:
        raise NotImplementedError
    elif tensor.ndim == 2:
        return svd_sparse_order2(tensor, k, **kw)
    else:
        return svd_sparse_orderN(tensor, k, **kw)

def svd_sparse_order2(tensor, k=50, **kw):
    '''
    Compute the SVD of sparse 2-D tensor *tensor*. The optional
    parameter *k* is the number of sigma values to retain (default
    value is 50).
    '''

    ut, s, vt = tensor._svd(k=k, **kw)
    #cnorms = dMatNorms(vt)


    logging.info('ut(%s), s(%s), vt(%s)\n' % (ut.shape, s.shape, vt.shape))

    # Truncate ut and vt to the number of singular values we actually got.
    n_svals = s.shape[0]
    ut = ut[:n_svals, :]
    vt = vt[:n_svals, :]
    return SVD2DResults(DenseTensor(ut.T), DenseTensor(vt.T), DenseTensor(s))

def svd_sparse_orderN(tensor, k=50):
    '''
    Compute the higher-order SVD of *tensor*. The optional
    parameter *k* is the number of sigma values to retain (default
    value is 50).
    '''

    '''Negative k values mean not to normalize that unfolding.'''
    ndim = tensor.ndim

    # Assume that the given k applies for all unfoldings
    if not isinstance(k, (list, tuple)):
        k=[k]*ndim

    def sub_svd(dim):
        print 'Unfolding along dimension %s' % dim
        u = tensor.unfolded(dim)
        cur_k = k[dim]
        normalized = (cur_k > 0)
        cur_k = abs(cur_k)
        return u.svd(k=cur_k, normalized=normalized)

    return SVDResults(tensor, [sub_svd(dim) for dim in xrange(ndim)])

def incremental_svd(tensor, k=50, niter=100, lrate=.1, logging_interval=None):
    '''
    Compute SVD of 2-D tensor *tensor* using an incremental SVD algorithm.
    This algorithm ignores unknown entries in the sparse tensor instead
    of treating them as zeros (like Lanczos' algorithm does). This means that
    the incremental SVD of a sparse matrix will not necessarily equal the
    Lanczos' SVD of the same matrix.

    The incremental SVD algorithm performs gradient descent to find the
    *U*, *V* and *S* matrices that minimize the
    Root Mean Squared error of the reconstructed matrix.

    The optional parameter *k* is the number of sigma values to retain
    (default 50). The *lrate* parameter controls the "learning rate," a
    factor that changes the rate of gradient descent (default .1). A larger value
    means the gradient descent will take larger steps toward its goal.
    Setting *lrate* to a "large" value may cause the algorithm to diverge,
    and will cause a mysterious overflow error.
    The *niter* parameter controls the number of steps performed by the
    gradient descent process (default 100).
    '''
    assert(tensor.ndim == 2)
    if not isinstance(tensor, Tensor):
        raise TypeError("This function can only be run on a Tensor. Use the "
                        ".incremental_svd() method for other objects.")

    u, v, sigma = tensor._isvd(k=k, niter=niter, lrate=lrate)
    return SVD2DResults(u, v, sigma)

