from csc.divisi.labeled_view import make_sparse_labeled_tensor

def ez_matrix(rows, cols, vals):
    '''
    Simple way to make a matrix (2D sparse tensor).
    '''
    return make_sparse_labeled_tensor(ndim=2, initial=zip(zip(rows, cols), vals))

def test_ez_matrix():
    t = ez_matrix('abc', '123', [1]*3)
    assert ('a', '1') in t.keys()
    assert ('c', '3') in t.keys()
    assert ('a', '3') not in t.keys()

class TestFailError(StandardError): pass

def assertSetEquals(set1, set2, msg=None):
    # Modified from Python 2.7 source, I think.
    """A set-specific equality assertion.

    Args:
        set1: The first set to compare.
        set2: The second set to compare.
        msg: Optional message to use on failure instead of a list of
                differences.

    For more general containership equality, assertSameElements will work
    with things other than sets.    This uses ducktyping to support
    different types of sets, and is optimized for sets specifically
    (parameters must support a difference method).
    """
    try:
        difference1 = set1.difference(set2)
    except TypeError, e:
        raise TestFailError('invalid type when attempting set difference: %s' % e)
    except AttributeError, e:
        raise TestFailError('first argument does not support set difference: %s' % e)

    try:
        difference2 = set2.difference(set1)
    except TypeError, e:
        raise TestFailError('invalid type when attempting set difference: %s' % e)
    except AttributeError, e:
        raise TestFailError('second argument does not support set difference: %s' % e)

    if not (difference1 or difference2):
        return

    if msg is not None:
        raise TestFailError(msg)

    lines = []
    if difference1:
        lines.append('Items in the first set but not the second:')
        for item in difference1:
            lines.append(repr(item))
    if difference2:
        lines.append('Items in the second set but not the first:')
        for item in difference2:
            lines.append(repr(item))
    raise TestFailError('\n'.join(lines))
