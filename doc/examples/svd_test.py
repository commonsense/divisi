from csc.divisi.cnet import *
print 'Loading from db...'
ct = conceptnet_2d_from_db('en', identities=3.0)
print 'Normalizing...'
n = ct.normalized()
print 'Running SVD...'
svd = ct.svd(k=50)

print 'Similar concepts for "teach":'
print concept_similarity(svd, 'teach').top_items(20)
print 'Similar concepts for "dog":'
print concept_similarity(svd, 'dog').top_items(20)
print 'Similar concepts for "trumpet":'
print concept_similarity(svd, 'trumpet').top_items(20)
print 'Similar concepts for "tv watch":'
print concept_similarity(svd, 'tv watch').top_items(20)

print 'Predicting features for "trumpet":'
print predict_features(svd, 'trumpet').top_items(20)

print 'Predicting features for "dog":'
print predict_features(svd, 'dog').top_items(20)

print 'Is a dog a pet?'
# Please use keyword arguments here.
print eval_assertion(svd, relationtype='IsA',
                     ltext='dog', rtext='pet')

print 'Is an elephant a pet?'
# stem is 'eleph'
print eval_assertion(svd, relationtype='IsA',
                     ltext='eleph', rtext='pet')
