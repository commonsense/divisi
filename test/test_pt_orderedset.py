from csc.divisi.pt_ordered_set import PTOrderedSet
from nose.tools import eq_
import tempfile, os.path
def test_pt_ordered_set():
    tmpdir = tempfile.mkdtemp()
    tfile = os.path.join(tmpdir, 'pttensor.h5')

    ordered_set = PTOrderedSet.create(tfile, '/', 'labels_0')
    eq_(ordered_set.add('apple'), 0)
    eq_(ordered_set.add('banana'), 1)
    eq_(ordered_set[1], 'banana')
    eq_(ordered_set.index('apple'), 0)

    del ordered_set

    ordered_set = PTOrderedSet.open(tfile, '/', 'labels_0')
    eq_(ordered_set.add('apple'), 0)
    eq_(ordered_set.add('banana'), 1)
    eq_(ordered_set[1], 'banana')
    eq_(ordered_set.index('apple'), 0)
