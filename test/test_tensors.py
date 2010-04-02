from csc.divisi.tensor import DenseTensor, DictTensor, data
from csc.divisi.labeled_view import LabeledView
from nose.tools import eq_, assert_raises, assert_almost_equal
from numpy import zeros, ndarray

def test_iter_dim_keys():
    raw = DenseTensor(zeros((2, 3)))
    labels = [['a', 'b'], ['c', 'd', 'e']]
    tensor = LabeledView(raw, labels)

    i = 0
    for key in tensor.iter_dim_keys(0):
        eq_(key, labels[0][i])
        i += 1
    eq_(i, 2)

    i = 0
    for key in tensor.iter_dim_keys(1):
        eq_(key, labels[1][i])
        i += 1
    eq_(i, 3)

def test_combine_by_element():
    t1 = LabeledView(DenseTensor(zeros((2,2))), [['a', 'b'], ['c', 'd']])
    t2 = LabeledView(DenseTensor(zeros((2,2))), [['a', 'b'], ['c', 'd']])
    t1['a', 'c'] = 1
    t1['b', 'c'] = 2
    t2['a', 'c'] = 4
    t2['a', 'd'] = 5

    t3 = t1.combine_by_element(t2, lambda x, y: x + (2*y))
    eq_(t3['a', 'c'], 9)
    eq_(t3['b', 'c'], 2)
    eq_(t3['a', 'd'], 10)
    eq_(t3['b', 'd'], 0)

    t4 = DenseTensor(zeros((3, 2)))
    assert_raises(IndexError, lambda: t1.combine_by_element(t4, lambda x, y: x + y))
    t4 = DenseTensor(zeros((2, 2, 1)))
    assert_raises(IndexError, lambda: t1.combine_by_element(t4, lambda x, y: x + y))

def test_DictMatrixVectorDot():
    # Numbers computed using numpy separately (and checked by hand)...
    A = DictTensor(2)
    b = DictTensor(1)
    A.update({(0, 0): 0.18850744743616121,
              (0, 1): 0.64380371397047509,
              (1, 0): 0.40673500155569442,
              (1, 1): 0.77961381386745443,
              (2, 0): 0.38745898104117782,
              (2, 1): 0.39479530812173591})
    b.update({0: 0.95308634444417639, 1: 0.41483520394218798})

    test_result = {(0,): 0.44673631896111365,
                   (1,): 0.71106483126206554,
                   (2,): 0.53305685602270081}

    result = A * b

    for k, value in result.iteritems():
        assert_almost_equal(value, test_result[k])

def test_DictMatrixMatrixDot():
    # Numbers computed using numpy separately (and checked by hand)...
    A = DictTensor(2)
    B = DictTensor(2)

    A.update({(0, 0): 0.97878770132160475,
             (0, 1): 0.38968165255179188,
             (0, 2): 0.62726841877492023,
             (1, 0): 0.077757604769237876,
             (1, 1): 0.081345677776447523,
             (1, 2): 0.64136810022648949})

    B.update({(0, 0): 0.062059208836173663,
              (0, 1): 0.67286767409459525,
              (0, 2): 0.55410453533854442,
              (0, 3): 0.74671274663041698,
              (1, 0): 0.11565332983247767,
              (1, 1): 0.48262692547766795,
              (1, 2): 0.76280138705455269,
              (1, 3): 0.50230554417370143,
              (2, 0): 0.67149114912362429,
              (2, 1): 0.7656884479264322,
              (2, 2): 0.69286881606948747,
              (2, 3): 0.82598232206483091})

    test_result = {(0, 0): 0.52701596238696313,
                   (0, 1): 1.3269576439118278,
                   (0, 2): 1.2742151361864653,
                   (0, 3): 1.4447251324591062,
                   (1, 0): 0.444906476567622,
                   (1, 1): 0.58266833824233299,
                   (1, 2): 0.54952039356712779,
                   (1, 3): 0.62868169229370208}

    result = A * B

    for key, value in result.iteritems():
        assert_almost_equal(value, test_result[key])

def test_dense_data():
    t1 = LabeledView(DenseTensor(zeros((2,2))), [['a', 'b'], ['c', 'd']])
    assert isinstance(data(t1), ndarray)
