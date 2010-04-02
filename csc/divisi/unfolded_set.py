class UnfoldedSet(object):
    '''An UnfoldedSet is like an OrderedSet in that it maps bidirectionally
    between items and their indices. But it's designed for unfoldings like
    T(a,b,c,d,e) => U(a, (b,c,d,e))
    where the tuple (b,c,d,e) is treated as in index.

    An UnfoldedSet doesn't store those tuples; instead, it references the original
    OrderedSets.
    '''
    index_is_efficient = True

    def __init__(self, labels):
        self.labels = labels
        self.lens = [len(s) for s in labels]
        self.ndim = len(labels)

        factors = [None]*len(labels)
        factors[-1] = 1
        for dim in xrange(len(factors)-2, -1, -1):
            factors[dim] = factors[dim+1]*self.lens[dim+1]
        self.factors = factors

    def __repr__(self):
        return u'<UnfoldedSet of %r>' % (self.labels,)

    def index(self, label):
        idx = 0
        lens = self.lens
        labels = self.labels
        for i in xrange(self.ndim):
            idx *= lens[i]
            idx += labels[i].index(label[i])
        return idx

    def __getitem__(self, idx):
        ndim = self.ndim
        label = [None]*ndim
        factors = self.factors
        labels = self.labels
        for i in xrange(ndim):
            quo, rem = divmod(idx, factors[i])
            idx = rem
            label[i] = labels[i][quo]
        return tuple(label)

    def __len__(self):
        return self.factors[0]*self.lens[0]


    @classmethod
    def from_unfolding(cls, mode, labels):
        return cls(labels[0:mode]+labels[mode+1:])
