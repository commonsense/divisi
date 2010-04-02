# Put libraries such as Divisi in the PYTHONPATH.
import sys, pickle, os
sys.path = ['/stuff/openmind'] + sys.path
from csc.divisi.cnet import *
from csc.divisi.graphics import output_svg
from vendor_db import iter_info
from csamoa.corpus.models import *
from csamoa.conceptnet.models import *

# Load the OMCS language model
en = Language.get('en')
en_nl=get_nl('en')

# Load OMCS stopwords
sw = open('stopwords.txt', 'r')
swords = [x.strip() for x in sw.readlines()]

# Parameters
factor = 1
wsize = 2

def check_concept(concept):
    try:
        Concept.get(concept, 'en')
        return True
    except:
        return False

def english_window(text):
    windows = []
    words = [x for x in text.lower().replace('&', 'and').split() if x not in swords]
    for x in range(len(words)-wsize+1):
        pair = " ".join(words[x:x+wsize])
        if check_concept(pair): windows.append(pair)
        if check_concept(words[x]): windows.append(words[x])
    for c in range(wsize-1):
        if check_concept(words[c]): windows.append(words[c])
    return windows

if 'vendor_only.pickle' in os.listdir('.'):
    print "Loading saved matrix."
    matrix = pickle.load(open("vendor_only.pickle"))
else:
    print "Creating New Tensor"
    matrix = SparseLabeledTensor(ndim=2)
    print "Adding Vendors"
    for co, englist in iter_info('CFB_Cities'):
        print co
        for phrase in englist:
            parts = english_window(phrase)
            print parts
            for part in parts:
                matrix[co, ('sells', part)] += factor
                matrix[part, ('sells_inv', co)] += factor
    pickle.dump(matrix, open("vendor_only.pickle", 'w'))

print "Normalizing."
matrix = matrix.normalized()

print "Matrix constructed. Running SVD."
svd = matrix.svd(k=10)
svd.summarize()

output_svg(svd.u, "vendorplain.svg", xscale=3000, yscale=3000, min=0.03)
