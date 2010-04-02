from csc.divisi.labeled_view import LabeledView
from csc.divisi.tensor import DictTensor

def weight_feature_vector(vec, weight_dct, default_weight=0.0):
    '''
    Weights a feature vector by relation.

    vec: a feature vector (e.g., a slice of a reconstructed tensor)

    weight_dct: a mapping from (side, relation) tuples to weights,
    where side is 'left' or 'right'.

    default_weight: the weight to give entries that are not specified.

    Example:
    >>> from csc.conceptnet4.analogyspace import conceptnet_2d_from_db
    >>> t = conceptnet_2d_from_db('en')
    >>> svd = t.svd()
    >>> baseball = svd.reconstructed['baseball',:]
    >>> weights = {}
    >>> weights['right', 'IsA'] = 1.0
    >>> weights['right', 'AtLocation'] = 0.8
    >>> weight_feature_vector(baseball, weights).top_items()
    '''
    if vec.ndim != 1:
        raise TypeError('Feature vectors can only have one dimension')

    res = LabeledView(DictTensor(ndim=1), label_lists=vec.label_lists())
    for k, v in vec.iteritems():
        res[k] = v*weight_dct.get(k[0][:2], default_weight)
    return res


def concepts_for_feature_vector(feature_vector_items):
    for k, v in feature_vector_items:
        yield k[2], v


def concept_vector(svd, feature_vector_items):
    from operator import add
    return reduce(add, (svd.weighted_u_vec(k)*v for k, v
                        in concepts_for_feature_vector(feature_vector_items)))


def related_concepts(svd, concept, weight_dct, n=10):
    feature_vector = weight_feature_vector(svd.reconstructed[concept, :], weight_dct).top_items(n)
    vec = concept_vector(svd, feature_vector)
    return svd.u_dotproducts_with(vec).top_items()
