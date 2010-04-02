from nose.tools import eq_, assert_almost_equal
from csc.divisi.util import nested_list_to_dict

## Utilities
def get_shape_for_dict(expected):
    if len(expected.keys()) == 0:
        return ()
    ndim = len(expected.keys()[0])
    return tuple(len(set(k[dim] for k in expected.iterkeys()))
                 for dim in xrange(ndim))

def nones_removed(d):
    return dict((k, v) for k, v in d.iteritems() if v is not None)
def zeros_removed(d):
    return dict((k, v) for k, v in d.iteritems() if v)

def dict_fill_in_missing_dims(d):
    items = d.items()
    ndim = len(items[0])
    dim_keys = [set() for i in xrange(ndim)]

    for cur_dim in xrange(ndim):
        for key, value in items:
            dim_keys[cur_dim].add(key[cur_dim])

    def enumerate_keys(dim_keys):
        if len(dim_keys) == 0:
            yield ()
        else:
            for key_end in enumerate_keys(dim_keys[1:]):
                for key_part in dim_keys[0]:
                    yield (key_part,) + key_end

    new_dict = {}
    for key in enumerate_keys(dim_keys):
        new_dict[key] = d.get(key, None)

    return new_dict



## Assertions

def assert_dims_consistent(tensor):
    eq_(tensor.ndim, len(tensor.shape), 'dims inconsistent')

def assertShapeEqual(actual, expected):
    expected_shape = get_shape_for_dict(expected)
    eq_(expected_shape, actual.shape, 'shape mismatch')
    eq_(len(expected_shape), actual.ndim, 'shape mismatch')
    eq_(actual.dims, actual.ndim, 'ndim != dims')

def assertValuesEqual(actual, expected, absolute):
    '''
    Test which values are specified in the tensor.

    Args:
     actual: the Tensor result
     expected: a dictionary of expected results
    '''
    for index, expected_value in expected.iteritems():
        if expected_value is not None:
            assert index in actual
            assert actual.has_key(index)
            if absolute:
                assert_almost_equal(abs(actual[index]), abs(expected_value))
            else:
                assert_almost_equal(actual[index], expected_value)
        else:
            assert index not in actual
            assert not actual.has_key(index)
            if absolute:
                assert_almost_equal(abs(actual[index]), abs(actual.default_value))
            else:
                assert_almost_equal(actual[index], actual.default_value)

    # Test some the shape boundary condition
    assert get_shape_for_dict(expected) not in actual

def assertItemsCorrect(actual, expected, absolute):
    # Filter just what's specified.
    specified = nones_removed(expected)

    # Test __iter__
    specified_keys = specified.keys()
    specified_keys.sort()
    indices = list(iter(actual))

    eq_(sorted(indices), specified_keys)
    eq_(sorted(actual.keys()), specified_keys)

    # Same test for iteritems()
    items_iter = sorted(list(actual.iteritems()))
    items_noniter = sorted(actual.items())
    specified_items = sorted(specified.items())
    # Values are subject to floating-point error, so an
    # AlmostEqual test is required.
    for (actual_key, actual_val), (expected_key, expected_val) \
            in zip(items_iter, specified_items):
        eq_(actual_key, expected_key)
        if absolute:
            assert_almost_equal(abs(actual_val), abs(expected_val))
        else:
            assert_almost_equal(actual_val, expected_val)
    for (actual_key, actual_val), (expected_key, expected_val) \
            in zip(items_noniter, specified_items):
        eq_(actual_key, expected_key)
        if absolute:
            assert_almost_equal(abs(actual_val), abs(expected_val))
        else:
            assert_almost_equal(actual_val, expected_val)

    # Test the values() method
    specified_values = specified.values()
    if not absolute:
        for a, e in zip(sorted(actual.values()), sorted(specified_values)):
            assert_almost_equal(a, e)

    # Test that the length of the tensor is equal to the number of specified items
    eq_(len(actual), len(specified_keys))

    # Test the __contains__ method
    assert specified_keys[0] in actual
    # can't easily test the contrapositive, since we can't easily make up an index that's not in there.

def assertIterDimKeysCorrect(actual, expected):
    specified_indices = [k for k, v in expected.iteritems()
                         if v is not None]
    for dim in xrange(len(specified_indices[0])):
        expected_keys = set(key[dim] for key in specified_indices)
        dim_keys = list(actual.iter_dim_keys(dim))
        eq_(sorted(dim_keys), sorted(expected_keys))

def assertTensorEqual(actual, expected, abs=False):
    '''Takes an n-dimensional array 'expected' and runs several
    tests on the tensor 'actual'. For the test description, see
    assertTensorEqualCompleteDict.
    '''
    expected_dict = nested_list_to_dict(expected)
    assertTensorEqualCompleteDict(actual, expected_dict, abs)

def assertTensorEqualDict(actual, expected_dict, abs=False):
    expected_complete_dict = dict_fill_in_missing_dims(expected_dict)
    assertTensorEqualCompleteDict(actual, expected_complete_dict, abs)

def assertTensorEqualCompleteDict(actual, expected_dict, abs=False):
    '''
    Takes a tensor and a dictionary of key => value pairs, e.g.
    {(0,0) : 1, (0, 1) : 2}
    and performs the following tests:

    1. actual's dimension and shape match that of expected
    2. Every index of actual has the same value as expected.
    3. Every index of expected is contained (i.e. using python's "in")
       in actual (Note: values in expected can be None, in which case
       actual is assumed to not contain the index and
       the value of actual at that index is the tensor's default
       value)
    4. Iteration (__iter__ and iteritems) correctly iterate over
       only the specified indices of actual
    5. iter_dim_keys iterates over specified keys of the specified dimension
    6. keys() and values() methods return only the specified indices
       of actual
    7. len(actual) returns the number of specified indices in expected
    8. has_key returns True for specified indices and False for unspecified indices
    '''
    assertShapeEqual(actual, expected_dict)
    assertValuesEqual(actual, expected_dict, abs)
    assertItemsCorrect(actual, expected_dict, abs)
    assertIterDimKeysCorrect(actual, expected_dict)
