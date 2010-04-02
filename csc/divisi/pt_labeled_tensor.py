from csc.divisi.pt_ordered_set import PTOrderedSet
from csc.divisi.pt_tensor import PTTensor
from csc.divisi.labeled_view import LabeledView

class PTLabeledTensor(LabeledView):
    @classmethod
    def create(cls, filename, ndim, pt_path='/', filters=None):
        tensor = PTTensor.create(filename, ndim, pt_path, pt_name='tensor', filters=filters)
        label_lists = [PTOrderedSet.create(filename, pt_path, pt_name='labels_%d' % mode, filters=filters)
                       for mode in range(ndim)]
        return cls(tensor, label_lists)

    @classmethod
    def open(cls, filename, pt_path='/'):
        tensor = PTTensor.open(filename, pt_path, pt_name='tensor')
        ndim = tensor.ndim
        label_lists = [PTOrderedSet.open(filename, pt_path, pt_name='labels_%d' % mode)
                       for mode in range(ndim)]
        return cls(tensor, label_lists)

    def __repr__(self):
        return '<PTLabeledTensor: %s>' % LabeledView.__repr__(self)
