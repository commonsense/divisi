from nose.tools import *
from nose.plugins.attrib import attr

@attr('slow')
def test_build_conceptnet():
    from csc.conceptnet4.analogyspace import conceptnet_2d_from_db

    tensor = conceptnet_2d_from_db('en')
    svd = tensor.normalized(mode=0).svd()
    svd.summarize(2)


@attr('slow')
def test_conceptnet_selfblend():
    from csc.conceptnet4.analogyspace import conceptnet_selfblend
    #from csc.divisi.blend import Blend

    blend = conceptnet_selfblend('en')
    svd = blend.svd()
    svd.summarize(2)
