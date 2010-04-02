from csc.divisi.tensor import Tensor
import tables
from itertools import izip, count
from csc.divisi.pyt_utils import get_pyt_handle

# I noticed the following in the user manual: "Note that, from
# PyTables 1.1 on, you can nest several iterators over the same
# table."  This looks worrisome; we may want to avoid returning
# generators.

class PTTensor(Tensor):
    is_sparse = True

    @classmethod
    def open(cls, filename, pt_path='/', pt_name='tensor'):
        '''
        Open an existing PyTables tensor.

        pt_path and pt_name are the "path" and "filename" within the
        PyTables file.

        Raises a tables.NoSuchNodeError if the table doesn't exist.

        (FIXME: it will still create a new, empty file.)
        '''
        fileh = get_pyt_handle(filename)
        table = fileh.getNode(pt_path, pt_name)
        return cls(table)


    @classmethod
    def create(cls, filename, ndim, pt_path='/', pt_name='tensor', filters=None):
        '''
        Create a new PyTables tensor.

        pt_path and pt_name are the "path" and "filename" within the
        PyTables file.

        Raises tables.NodeError if the table already exists.
        '''
        fileh = get_pyt_handle(filename)
        table = fileh.createTable(pt_path, pt_name, cls.descriptor(ndim), filters=filters)
        return cls(table)


    def __init__(self, table):
        '''
        This tensor stores its data in a densely-packed PyTables table.

        Generally, you will call the :meth:`create` or :meth:`open`
        class methods rather than calling the constructor
        directly.

        Let's create a new tensor:

        >>> from tempfile import mktemp # for uniqueness, not security
        >>> filename = mktemp()
        >>> t = PTTensor.create(filename=filename, ndim=2)

        It obeys (most of) the same dict-like interface as other tensors:
        >>> t[0,0] = 1.5
        >>> t[0,0]
        1.5

        But the fastest way to get data in is by using :meth:`update`
        (just like a Python dict):

        >>> t.update([((0,0), 0.), ((0,1), 0.01), ((1,0), 0.10), ((1,1), 0.11)])
        >>> t[0,0]
        0.0
        >>> t[1,1]
        0.11

        If you're adding to a value that you think already exists, use
        the :meth:`inc` method:

        >>> t.inc((0,0), 1)
        >>> t[0,0]
        1.0

        Containment is supported, though it currently requires a
        linear scan, so it's a bit slow:

        >>> (0,1) in t
        True

        It also supports other tensor functions, including the very
        useful :meth:`iteritems`:

        >>> (0,1) in t.keys()
        True
        >>> sorted(t.iteritems())[0]
        ((0L, 0L), 1.0)

        Like other tensors, you can query its shape:

        >>> t.shape
        (2L, 2L)

        You can poke around at the PyTables table underneath this if
        you want, but be warned that the PyTables API is kinda
        obfuscated because they want to do "natural naming" for
        data. Variables are prefixed by ``_v_``, file functions by
        ``_f_``, etc. To show that the stuff above actually stored
        data on disk, let's close the file and re-open the
        tensor. Don't try this at home, folks:

        >>> t.table._v_file.close()
        >>> t = PTTensor.open(filename)
        >>> t.shape
        (2L, 2L)


        Internally, data is stored in a densely packed table of ndim+1
        columns, one entry per row. ndim UInt32 columns store the
        "key" (the indices for each mode), and the final Float64
        column stores the value. The _last_ occurrence of a key is
        used as the current value; this allows updates to be
        constant-time. This property means that the table may gather
        "junk" over time; TODO(kcarnold) implement a garbage collector
        to delete the unused rows.

        The unordered disk structure means that a sequential scan is
        necessary to find any arbitrary key (though iteration through
        all items is very fast). If you want fast key access, sort the
        table and use the :mod:`bisect` module.
        '''

        self.table = table

        table.flush()
        self.ndim = len([x for x in table.colnames if x.startswith('i')])
        self._col_names = map(self._column_name, xrange(self.ndim))

        # Precompute the query for getting a single item.
        self._getitem_key = ' & '.join('(%s==%%d)' % name for name in self._col_names)

        # Compute what keys are present for each index.
        dim_entries = [set() for _ in xrange(self.ndim)]
        for key in self:
            for dim_ent, idx in izip(dim_entries, key):
                dim_ent.add(idx)
        self._dim_entries = dim_entries

        # Compute the shape
        if all(dim_entries):
            self._shape = [max(dim_ent) + 1 for dim_ent in dim_entries]
        else:
            # No contents.
            self._shape = [0L for _ in xrange(self.ndim)]


    def __repr__(self):
        return '<PTTensor shape: %r; %d items>' % (self.shape, len(self))

    def update(self, entries):
        row = self.table.row
        dim_entries = self._dim_entries
        col_names = self._col_names
        shape = self._shape
        for key, val in entries:
            for idx, col_name, ent, dim_ent in izip(count(), col_names, key, dim_entries):
                row[col_name] = ent
                dim_ent.add(ent)
                shape[idx] = max(shape[idx], ent + 1L)
            row['v'] = val
            row.append()
        self.table.flush()

    def _idx_for_key(self, key):
        indices = self.table.getWhereList(self._getitem_key % key)
        if len(indices) == 0:
            return None
        return indices[-1] # only the last one is valid.

    def update_existing_key(self, key, val):
        self.table.cols.v[self._idx_for_key(key)] = val

    #
    # Tensor API
    #
    def inc(self, key, amt):
        '''
        Add amt to key.
        '''
        idx = self._idx_for_key(key)
        if idx is None:
            self[key] = amt
        else:
            self.table.cols.v[idx] += amt


    def __getitem__(self, key):
        idx = self._idx_for_key(key)
        if idx is None:
            raise KeyError(key)
        return self.table.cols.v[idx]

    def __setitem__(self, key, val):
        self.update([(key, val)])

    def __delitem__(self, key):
        raise NotImplementedError('not yet implemented.')


    def __iter__(self):
        '''Iterate over keys.

        Note: if the table is not garbage-collected, the same index may be iterated over multiple times.
        '''
        col_names = self._col_names
        return (tuple(r[name] for name in col_names) for r in self.table)

    def iteritems(self):
        '''Iterate over key-value pairs.

        Note: if the table is not garbage-collected, the same index
        may be iterated over multiple times. The last value returned
        is the correct one.
        '''
        col_names = self._col_names
        return ((tuple(r[name] for name in col_names), r['v']) for r in self.table)

    def has_key(self, key):
        if any(kent not in dim_ent for kent, dim_ent in izip(key, self._dim_entries)):
            # Optimization: we don't have that key
            return False
        return len(self.table.getWhereList(self._getitem_key % key)) == 1

    @property
    def shape(self):
        return tuple(self._shape)

    def __len__(self):
        '''
        Return the number of specified items in the tensor.

        Note: if the table is not garbage-collected, this count may be
        overestimated.
        '''
        return self.table.shape[0]



    @classmethod
    def descriptor(cls, ndim):
        '''
        Construct the "descriptor" that PyTables uses to define the kind of table to create.
        '''
        desc = dict((cls._column_name(idx), tables.UInt32Col())
                    for idx in range(ndim))
        desc['v'] = tables.Float64Col()
        return type('PyT', (tables.IsDescription,), desc)

    @classmethod
    def _column_name(cls, idx):
        return 'i%d' % (idx,)
