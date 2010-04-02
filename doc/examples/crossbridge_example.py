
'''
From a use standpoint, there are three major steps to run crossbridge:
1. Convert ConceptNet from a tensor into a semantic network
2. Initialize Crossbridge on the semantic network (A lot happens under the
hood here).
3. Find analogies using the crossbridge object created in step 2.

I've written a paragraph or so about each step, intermixed with the
code to run that step.

(This is mainly documentation, but it's also runnable code. The whole
script runs in a few minutes -- you have to be somewhat patient.)
'''

from csc.divisi.crossbridge import CrossBridge, graph_from_triples
from csc.divisi.semantic_network import SemanticNetwork
from csc.conceptnet4.analogyspace import conceptnet_triples
import logging

'''
Crossbridge uses the logging module to print out status messages,
which makes waiting for long computations much more tolerable. If you
do the same initialization, you'll get helpful messages too.
'''

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

'''
Step 1: Create a semantic network from ConceptNet

You can convert any sequence of triples into a semantic network using
graph_from_triples (in crossbridge.py for the moment).

You can use the min_weight parameter to limit the number of assertions
included in the final semantic network. *This is very important and
has a BIG impact on the running time of the second step*. I suggest
using min_weight=1 to run quick tests, though the setting really
depends on how long you're willing to wait. (Note that the default
min_weight is 0, which means running crossbridge will take *forever*.)

The omit_relations parameter removes some edge types from the
resulting network. In this example, no 'IsA' edges will be included. I
typically remove 'IsA' edges, which in my experience have produced
nonintuitive analogies.

(With min_weight = 1, this takes about 10 minutes to
run. With min_weight = 1.5, it takes 2 minutes, but
gives pretty bad analogies.)
'''

cnet = graph_from_triples(conceptnet_triples('en', cutoff=2),
                          min_weight=1,
                          omit_relations=["IsA"])


'''
# Eliminate self-loops
for vertex in cnet.vertices():
    for reltype in copy.copy(cnet.get_connecting_edge_types(vertex, vertex)):
        cnet.del_edge((vertex, vertex, reltype))
'''

logging.debug("ConceptNet has %d vertices and %d edges...", len(cnet.vertices()), len(cnet.edges()))


'''
Step 2: Instantiate a CrossBridge object

This is step is easy, but there are a lot of potential parameters to
tweak. The default parameter settings are sane for ConceptNet (and are
also what I'm using here, just to be clear about what they are), but
here's some exposition:

num_nodes sets the number of nodes in each indexed subgraph
(you should probably use 3 -- fewer than that and crossbridge makes no
sense, and more takes forever.) Every set of concepts
in the crossbridge matrix will contain exactly num_nodes concepts.

min_graph_edges sets the minimum number of edges in each indexed
subgraph. 

(num_nodes and min_graph_edges both affect how many subgraphs are
included in the CrossBridge SVD. Increasing num_nodes or decreasing
min_graph_edges will increase the number of subgraphs in the matrix.)

min_feature_edges / max_feature_edges control the size of the graph
features. Again, decreasing min_feature_edges or increasing
max_feature_edges will increase the size of the matrix. 

Under the hood, this function searches around the graph for
"promising" subgraphs, then puts them all into a tensor and takes
their SVD. The searching process is quite expensive (O(V^(num_nodes))
time and space), and therefore can take a very long time / a lot of
memory with large semantic networks. This is why it's important to use
the min_weight parameter in step 1. (Details of this
processing are in my thesis)

(I highly recommend caching this object as a pickle if you plan on
doing multiple experiments with it)
'''

crossbridge = CrossBridge(cnet, num_nodes=3, 
                          min_graph_edges=3, 
                          min_feature_edges=2, 
                          max_feature_edges=3,
                          svd_dims=100)
                                 

'''
Step 3: Find analogies

Supply a set of concepts to the crossbridge.analogy method, and you'll
get back a list of analogies for the concepts. For large sets of
concepts, this merges together many small analogies using a beam
search. If you find that this method takes a long time, you can set
the beam_width parameter to reduce the number of candidates maintained
during the search, with the caveat that you may find worse
analogies. 

The num_candidates parameter is the number of analogies
retrieved from the SVD and fed into the merging process. A higher
number means potentially better analogies, but a slower search process.

Again, default parameters are sane.

(also: the relations returned are pretty garbage. I don't trust them.)
'''

analogies, relations = crossbridge.analogy(set(['bird','wing','fly', 'sky']), num_candidates=300, beam_width=10000)

print "Analogy / Score"
for analogy, score in analogies:
    for source, target in analogy:
        print "%s -> %s," % (source, target),
    print "(", score, ")"
