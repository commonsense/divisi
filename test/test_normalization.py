from csc.divisi.tensor import DictTensor
from csc.divisi.normalized_view import NormalizedView
from nose.tools import raises, assert_almost_equal
from tensor_util import assertTensorEqual, nones_removed, nested_list_to_dict

normalize_testcase = [[1, None],
                      [3, 4]]

normalize_expected_result = [[1, None],
                             [3/5., 4/5.]]

raw = DictTensor(2)
raw.update(nones_removed(nested_list_to_dict(normalize_testcase)))
tensor = NormalizedView(raw, 0)

def test_result():
    assertTensorEqual(tensor, normalize_expected_result)

def test_contains():
    assert (0,0) in tensor
    assert tensor.has_key((0,0))
    assert (0,1) not in tensor
    assert not tensor.has_key((0,1))

def test_unnormalize():
    assert_almost_equal(tensor[1,0], 3/5.)
    assert_almost_equal(tensor.unnormalized()[1,0], 3)

def test_labeled_unnormalize():
    labeled = tensor.labeled([['a','b'],['A','B']])
    assert_almost_equal(labeled['b','A'], 3/5.)
    labeled_unnormalized = labeled.unnormalized()
    assert_almost_equal(labeled_unnormalized['b','A'], 3)

def test_stack_contains():
    labeled = tensor.labeled([['a','b'],['A','B']])
    assert labeled.stack_contains(NormalizedView)
    assert labeled.stack_contains(DictTensor)
    assert labeled.stack_contains((NormalizedView, DictTensor))
    assert not labeled.stack_contains(int)

@raises(TypeError)
def test_two_normalize_layers():
    tensor.normalized()
    
