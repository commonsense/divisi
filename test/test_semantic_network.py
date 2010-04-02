from csc.divisi.semantic_network import SemanticNetwork
def test_semantic_network():
    g = SemanticNetwork()
    g2 = SemanticNetwork()

    g.add_vertex("v1")
    g.add_edge("v1", "v2", 4)
    g.add_edge("v2", "v3", 5)

    g2.add_vertex("v1")
    g2.add_edge("v1", "v2", 4)
    g2.add_edge("v2", "v3", 5)

    print list(g2.topological_sort("v1"))

    #f = g.fingerprint()
    #h = g.fingerprint()
    #assert (f == h)
    #assert (g.fingerprint() == g2.fingerprint())
    #assert (hash(f) == hash(h))

    reachable = g2.get_reachable_vertices("v1")
    assert(reachable == frozenset(["v2", "v1", "v3"]))
    reachable = g2.get_reachable_vertices("v1", max_depth=1)
    assert(reachable == frozenset(["v2", "v1"]))

    assert(g.vertices() == set(["v1", "v2", "v3"]))
    assert(g.edges() == set([("v1", "v2", 4), ("v2", "v3", 5)]))

    g = SemanticNetwork()
    g.add_edge('a', 'b', 1)
    g.add_edge('a', 'b', 2)
    g.add_edge('a', 'b', 3)
    g.add_edge('b', 'c', 1)
    g.add_edge('b', 'c', 2)
    g.add_edge('a', 'c', 1)
    g.add_vertex('d')

    for h in g.enumerate_without_repeated_edges():
        print h
