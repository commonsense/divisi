from nose.plugins.attrib import attr

@attr('slow')
def test_tensor_download():
    import urllib
    import gzip
    import cPickle as pickle

    print 'Retrieving tensor...'
    fname, headers = urllib.urlretrieve('http://commons.media.mit.edu/en/tensor.gz')
    print 'loading pickle...'
    t=pickle.load(gzip.open(fname,'rb'))

    assert t.shape[0]*t.shape[1] > 10000000, 'Downloaded ConceptNet tensor is kinda small.'
