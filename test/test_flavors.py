from nose.tools import eq_

def test_conceptfeature():
    from csc.divisi.flavors import ConceptByFeatureMatrix
    mat = ConceptByFeatureMatrix()
    mat.add_triple(('dog', 'IsA', 'pet'), 1.5)
    eq_(mat['dog', ('right', 'IsA', 'pet')], 1.5)
    eq_(mat['pet', ('left', 'IsA', 'dog')], 1.5)
    eq_(len(mat), 2)

def test_conceptfeature_fromtriple():
    from csc.divisi.flavors import ConceptByFeatureMatrix
    mat = ConceptByFeatureMatrix.from_triples(
        [('dog', 'IsA', 'pet'),
         ('dog', 'IsA', 'pet'),
         ('dog', 'IsA', 'animal')],
        accumulate=True,
        constant_weight=1.5)
    eq_(mat['dog', ('right', 'IsA', 'pet')], 3.0)
    eq_(mat['pet', ('left', 'IsA', 'dog')], 3.0)
    eq_(mat['animal', ('left', 'IsA', 'dog')], 1.5)
    eq_(len(mat), 4)

def test_featureconcept():
    from csc.divisi.flavors import FeatureByConceptMatrix
    mat = FeatureByConceptMatrix()
    mat.add_triple(('dog', 'IsA', 'pet'), 1.5)
    eq_(mat[('right', 'IsA', 'pet'), 'dog'], 1.5)
    eq_(mat[('left', 'IsA', 'dog'), 'pet'], 1.5)
    eq_(len(mat), 2)

def test_featureconcept_fromtriple():
    from csc.divisi.flavors import FeatureByConceptMatrix
    mat = FeatureByConceptMatrix.from_triples(
        [('dog', 'IsA', 'pet'),
         ('dog', 'IsA', 'pet'),
         ('dog', 'IsA', 'animal')],
        accumulate=True,
        constant_weight=1.5)
    eq_(mat[('right', 'IsA', 'pet'), 'dog'], 3.0)
    eq_(mat[('left', 'IsA', 'dog'), 'pet'], 3.0)
    eq_(mat[('left', 'IsA', 'dog'), 'animal'], 1.5)
    eq_(len(mat), 4)

