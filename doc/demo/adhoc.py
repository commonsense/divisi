from csc.divisi.cnet import *
import cPickle as pickle

tensor=pickle.load(open('conceptnet_tensor','rb'))
tensor
svd = tensor.svd()
svd
concept_similarity(svd, 'guitar').top_items(3)

predict_features(svd, 'guitar').top_items(3)


cat = make_category(svd, ['car','train','bus'])
concepts, features = category_similarity(svd, cat)

concepts.top_items(5)

features.top_items(3)
