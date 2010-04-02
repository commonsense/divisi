# ccipca.py -- A Candid Covariance-free Incremental Principal Component Analysis implementation
# Jason B. Alonso <jalonso@media.mit.edu>
# Copyright 2009 Massachusetts Institute of Technology
# MIT Media Laboratory, Personal Robots Group

import logging
logger = logging.getLogger(__name__)
from csc.divisi.labeled_view import make_sparse_labeled_tensor

# The algorithm is based on:
#   Book Series - Lecture Notes in Computer Science
#   Book Title  - Intelligent Data Engineering and Automated Learning
#   Chapter Title  - A Fast Algorithm for Incremental Principal Component Analysis
#   First Page  - 876
#   Last Page  - 881
#   Copyright  - 2003
#   Author  - Juyang Weng
#   Author  - Yilu Zhang
#   Author  - Wey-Shiuan Hwang
#   DOI  -
#   Link  - http://www.springerlink.com/content/cd8br967h808bw7h

# NOTE: The algorithm has been modified to include a "remembrance" parameter, which determines
#   the largest iteration number to be used in computations.  Set sufficiently high, this
#   parameter has no effect on the outcome of the computation.  If it does come into effect,
#   though, the eigenvectors will be able to be attracted to a moving set of eigenvectors.

# NOTE: Eigenvector 0 is a ``pseudo-eigenvector.''  That is, it is the mean of all input vectors.

def zerovec():
    return make_sparse_labeled_tensor(1)

class CCIPCA(object):
    """A Candid Covariance-free Incremental Principal Component Analysis implementation"""

    def __init__(self, k, ev=None, i=0, bootstrap=20, amnesia=3.0, remembrance=100000.0, auto_baseline=True):
        """Construct a CCIPCA computation with k initial eigenvectors ev at iteration i, using simple averaging until the iteration given by bootstrap, afterward using CCIPCA given amnesic parameter amnesia, rememberance parameter remembrance, and a weight vector and subspace criteria for simultaneous vector presentation"""
        if ev is not None: self._v = ev
        else: self._v = [zerovec() for iter in xrange(k)]

        self._k = k
        self._amnesia = amnesia
        self._iteration = i
        self._bootstrap = bootstrap
        self._remembrance = remembrance
        self._auto_baseline = auto_baseline

    def compute_attractor(self, k, u, v_hat=None):
        """Compute the attractor vector for eigenvector k with vector u"""

        if k == 0: return u
        if v_hat is None: v_hat = self._v[k].hat()
        partial = u * (u * v_hat)
        return partial

    def update_eigenvector(self, k, u, learn=False):
        """Update eigenvector k with vector u, returning pair
        containing magnitude of eigenvector component and residue vector"""
        if learn:
            # Handle elementary cases
            if self._iteration < k:
                return 0.0, zerovec()

            if self._iteration == k:
                self._v[k] = make_sparse_labeled_tensor(1, initial=u)
                mag = self._v[k].norm()
                return mag, zerovec()

            # Compute weighting factors
            n = min(self._iteration, self._remembrance)
            if n < self._bootstrap:
                w_old = float(n - 1) / n
                w_new = 1.0 / n
            else:
                l = self._amnesia
                w_old = float(n - l) / n
                w_new = float(l) / n

            # Compute the attractor
            attractor = self.compute_attractor(k, u)

            # Approach attractor
            self._v[k] *= w_old
            self._v[k] += attractor * w_new

        # Calculate component magnitudes
        v_hat = self._v[k].hat()
        if k == 0:
            if self._auto_baseline: base_mag = self._v[k].norm()
            else: base_mag = 0.0
        else: base_mag = u * v_hat

        u_residue = u - (v_hat * base_mag)
        if k == 0:
            logger.debug('e0: %s' % str(self._v[k]))
            logger.debug('u: %s' % str(u))
            if learn:
                logger.debug('attractor: %s' % str(attractor))
            logger.debug('residue: %s' % str(u_residue))
            logger.debug('')
        return base_mag, u_residue

    def iteration(self, u, learn=False):
        """Train the eigenvector table with new vector u"""
        print "iteration = ", self._iteration
        mags = []
        u_copy = make_sparse_labeled_tensor(1, initial=u)   # copy
        for k in xrange(min(self._k, self._iteration+1)):
            mag, new_u = self.update_eigenvector(k, u_copy, learn)
            u_copy = new_u
            mags.append(mag)
        if learn:
            self._iteration += 1
            self._v[1:] = sorted(self._v[1:], reverse=True, key=lambda v: v.norm())
        return mags

    def reconstruct(self, weights):
        # Create a linear combination of the eigenvectors
        sum = zerovec()
        for index, w in enumerate(weights):
            sum += self._v[index].hat() * w
        return sum

    def smooth(self, u, k_max=None, learn=False):
        mags = self.iteration(u, learn)
        if k_max is not None:
            mags = mags[:k_max]
        vec = self.reconstruct(mags)
        if not learn:
            logger.debug("decomp: %s" % str(mags))
        return vec

# vim:tw=160:
