from csc.divisi.cnet3 import conceptnet_3d_from_db
print 'Loading from db...'
a=conceptnet_3d_from_db(lang='en')

print 'Unfolding and running SVDs...'
r = a.svd([100, 17, 100])
