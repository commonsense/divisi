from csc.divisi.ordered_set import OrderedSet

from csc.divisi.tensor import Tensor
import tables
from itertools import izip, count
from csc.divisi.pyt_utils import get_pyt_handle

class PTOrderedSet(OrderedSet):
    @classmethod
    def create(cls, filename, pt_path, pt_name, filters=None):
        fileh = get_pyt_handle(filename)
        array = fileh.createVLArray(pt_path, pt_name, tables.ObjectAtom(), filters=filters)
        return cls(array)

    @classmethod
    def open(cls, filename, pt_path, pt_name):
        fileh = get_pyt_handle(filename)
        array = fileh.getNode(pt_path, pt_name)
        return cls(array)


    def __init__(self, array):
        self.items = array
        self.indices = dict((item, idx) for idx, item in enumerate(array))
        self._setup_quick_lookup_methods()

    def __setitem__(self, n, newkey):
        raise TypeError('Existing items in a PTOrderedSet cannot be changed.')
    def __delitem__(self, n):
        raise TypeError('Existing items in a PTOrderedSet cannot be changed.')

    #def __del__(self):
     #   self.array._
