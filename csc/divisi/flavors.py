from csc.divisi.tensor import DictTensor
from csc.divisi.ordered_set import OrderedSet
from csc.divisi.labeled_view import LabeledView


def add_triple_to_matrix(matrix, triple, value=1.0):
    '''
    Adds a triple (left, relation, right) to the matrix in the 2D unfolded format.

    This is the new add_assertion_tuple.
    '''
    left, relation, right = triple

    lfeature = ('left',  relation, left)
    rfeature = ('right', relation, right)
    matrix.inc((left, rfeature), value)
    matrix.inc((right, lfeature), value)

def set_triple_in_matrix(matrix, triple, value=1.0):
    ''' Sets a triple (left, relation, right) in the matrix in the 2D
    unfolded format to the specified value.
    '''
    left, relation, right = triple

    lfeature = ('left',  relation, left)
    rfeature = ('right', relation, right)
    matrix[left, rfeature] = value
    matrix[right, lfeature] = value

###
### Assertion Tensors
###
class AssertionTensor(LabeledView):
    '''
    All AssertionTensors have the following functions:
     .add_triple(triple, value)
     .set_triple(triple, value)
     .add_identity(text, value=1.0, relation='Identity')
    where triple is (concept1, relation, concept2).

    They also have the convenience classmethod from_triples.
    '''
    def add_identity(self, text, value=1.0, relation='Identity'):
        self.add_triple((text, relation, text), value)

    def bake(self):
        '''
        Simplify the representation.
        '''
        return LabeledView(self.tensor, self._labels)

    def add_triples(self, triples, accumulate=True, constant_weight=None):
        if accumulate: add = self.add_triple
        else: add = self.set_triple

        if constant_weight:
            for triple in triples:
                add(triple, constant_weight)
        else:
            for triple, weight in triples:
                add(triple, weight)

    @classmethod
    def from_triples(cls, triples, accumulate=True, constant_weight=None):
        mat = cls()
        mat.add_triples(triples, accumulate, constant_weight)
        return mat

    def add_identities(self, value=1.0, relation='Identity'):
        if not value: return # 0 or False means not to actually add identities.
        for concept in self.concepts():
            self.add_triple((concept, relation, concept), value)

class ConceptByFeatureMatrix(AssertionTensor):
    '''
    This is the typical AnalogySpace matrix. It stores each assertion
    twice: once as (c1, ('right', rel, c2)) and once as (c2, ('left',
    rel, c1)).

    This class is a convenience for building matrices in this
    format. Once you've add_triple'sed everything, you can call
    .bake() to convert it back to a plain old LabeledView of a
    DictTensor, just like make_sparse_labeled_tensor does.
    '''
    def __init__(self):
        super(ConceptByFeatureMatrix, self).__init__(
            DictTensor(2), [OrderedSet() for _ in '01'])

    add_triple = add_triple_to_matrix
    set_triple = set_triple_in_matrix
    def concepts(self): return self.label_list(0)


class FeatureByConceptMatrix(AssertionTensor):
    '''
    A transposed ConceptByFeatureMatrix; see it for documentation.
    '''
    def __init__(self):
        super(FeatureByConceptMatrix, self).__init__(
            DictTensor(2), [OrderedSet() for _ in '01'])

    def add_triple(self, triple, value=1.0):
        left, relation, right = triple

        lfeature = ('left',  relation, left)
        rfeature = ('right', relation, right)
        self.inc((rfeature, left), value)
        self.inc((lfeature, right), value)

    def set_triple(self, triple, value=1.0):
        left, relation, right = triple

        lfeature = ('left',  relation, left)
        rfeature = ('right', relation, right)
        self[rfeature, left] = value
        self[lfeature, right] = value

    def concepts(self): return self.label_list(1)

            
class ConceptRelationConceptTensor(AssertionTensor):
    '''
    This is a straightforward encoding of concepts as a 3D tensor.
    '''
    def __init__(self):
        # FIXME: yes this saves space, but it might make a row or column be zero.
        concepts, relations = OrderedSet(), OrderedSet()
        super(ConceptRelationConceptTensor, self).__init__(
            DictTensor(3), [concepts, relations, concepts])

    def concepts(self): return self.label_list(0)

    def add_triple(self, triple, value=1.0):
        left, relation, right = triple
        self.inc((left, relation, right), value)

    def set_triple(self, triple, value=1.0):
        left, relation, right = triple
        self[left, relation, right] = value


class MirroringCRCTensor(ConceptRelationConceptTensor):
    '''
    Every assertion (c1, r, c2) in this tensor has an inverse,
    (c2, r', c1).

    This is analogous to how the 2D tensor makes left and right features.

    Inverse relations are constructed from ordinary relations by
    prefixing a '-'.
    '''
    def add_triple(self, triple, value=1.0):
        left, relation, right = triple
        self.inc((left, relation, right), value) # normal
        self.inc((right, '-'+relation, left), value) # inverse

    def set_triple(self, triple, value=1.0):
        left, relation, right = triple
        self[left, relation, right] = value
        self[left, '-'+relation, right] = value
        
