class MyDictMixin(object):
    '''Emulates a dictionary interface, more efficiently than DictMixin.

    Mixin defining all dictionary methods for classes that already have
    a minimum dictionary interface including getitem, setitem, delitem,
    and __iter__. Without knowledge of the subclass constructor, the mixin
    does not define __init__() or copy().  In addition to the four base
    methods, progressively more efficiency comes with defining
    __contains__() and iteritems().

    Based on UserDict in python 2.4.
    '''
    __slots__ = []
    # second level definitions support higher levels
#     def has_key(self, key):
#         """
#         Does this object have a value for the given key?
#         """
#         try:
#             _ = self[key]
#         except (KeyError, IndexError):
#             return False
#         return True
    ## Note: the above definition of has_key is incorrect for dicts
    ## with default values.
    def __contains__(self, key):
        return self.has_key(key)

    # third level takes advantage of second level definitions
    def iteritems(self):
        """
        Iterate over a list of (key, value) tuples.
        """
        for k in self:
            yield (k, self[k])
    def iterkeys(self):
        """
        An iterator over the keys of this object.
        """
        return self.__iter__()
    def keys(self):
        """
        List the keys of this object.
        """
        return list(self.__iter__())

    # fourth level uses definitions from lower levels
    def itervalues(self):
        """
        An iterator over the values of this object.
        """
        for _, v in self.iteritems():
            yield v
    def values(self):
        """
        List the values of this object.
        """
        return [v for _, v in self.iteritems()]
    def items(self):
        """
        Express this object as a list of (key, value) tuples.
        """
        return list(self.iteritems())
    def clear(self):
        """
        Remove all elements from this object.
        """
        for key in self.keys():
            del self[key]
    def setdefault(self, key, default=None):
        """`D.setdefault(k,d)` -> `D.get(k,d)`, and also sets `D[k]=d` if
        `k not in D`"""
        try:
            return self[key]
        except KeyError:
            self[key] = default
        return default
    def pop(self, key, *args):
        """
        Get a value from this object (disregarding its key) and delete it.
        """
        if len(args) > 1:
            raise TypeError, "pop expected at most 2 arguments, got "\
                              + repr(1 + len(args))
        try:
            value = self[key]
        except KeyError:
            if args:
                return args[0]
            raise
        del self[key]
        return value
    def popitem(self):
        """
        Get a (key, value) tuple from this object, and delete that key.
        """
        try:
            k, v = self.iteritems().next()
        except StopIteration:
            raise KeyError, 'container is empty'
        del self[k]
        return (k, v)
    def update(self, other=None, **kwargs):
        """
        Add all the items from `other` into `self`, overwriting values when
        that key already exists.
        """
        # Make progressively weaker assumptions about "other"
        if other is None:
            pass
        elif hasattr(other, 'iteritems'):  # iteritems saves memory and lookups
            for k, v in other.iteritems():
                self[k] = v
        elif hasattr(other, 'keys'):
            for k in other.keys():
                self[k] = other[k]
        else:
            for k, v in other:
                self[k] = v
        if kwargs:
            self.update(kwargs)
    def get(self, key, default=None):
        """
        Get the value associated with `key`, or `default` if the key does not
        exist.
        """
        try:
            return self[key]
        except KeyError:
            return default
    def __repr__(self):
        return repr(dict(self.iteritems()))
    def __cmp__(self, other):
        if other is None:
            return 1
        if isinstance(other, MyDictMixin):
            other = dict(other.iteritems())
        return cmp(dict(self.iteritems()), other)
    def __len__(self):
        return len(self.keys())
