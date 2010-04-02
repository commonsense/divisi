from collections import defaultdict
import copy

class SemanticNetwork(object):
    '''
    This class represents a directed graph with typed edges. Vertices
    are represented by hashable objects, and edges are (source vertex,
    target vertex, <type>) tuples. There can be many different edges
    with the same type, but there cannot be multiple vertices with the
    same value.

    SemanticNetworks are mutable.
    '''
    def __init__(self):
        self._vertices = set()
        self._edges = set()

        self._forward_edges = defaultdict(set)
        self._reverse_edges = defaultdict(set)
        self._connecting_edges = defaultdict(set)

        self._edge_types = set()

    def vertices(self):
        return frozenset(self._vertices)

    def edges(self):
        '''
        Returns all edges in this graph as a set of (source, target, value) tuples
        '''
        return frozenset(self._edges)

    def edge_types(self):
        return frozenset(self._edge_types)

    def add_vertex(self, v):
        self._vertices.add(v)

    def add_edge(self, src, dst, value):
        edge = (src, dst, value)
        self.add_vertex(src)
        self.add_vertex(dst)
        self._edges.add(edge)
        self._edge_types.add(value)

        self._forward_edges[src].add(edge)
        self._reverse_edges[dst].add(edge)
        self._connecting_edges[(src, dst)].add(value)

    def del_edge(self, edge):
        if edge in self._edges:
            src, dst, val = edge
            self._edges.remove(edge)
            self._forward_edges[src].remove(edge)
            self._reverse_edges[dst].remove(edge)
            self._connecting_edges[(src, dst)].remove(val)

    def get_out_edges(self, src, reltype=None):
        '''
        Returns the set of outbound edges as (src, <target>, <value>) tuples
        '''
        if reltype is None:
            return self._forward_edges.get(src, set())
        else:
            return set([edge for edge in self._forward_edges.get(src, set()) if edge[2] == reltype])

    def get_in_edges(self, dst, reltype=None):
        '''
        Returns the set of inbound edges as (<source>, dst, <value>) tuples
        '''
        if reltype is None:
            return self._reverse_edges.get(dst, set())
        else:
            return set([edge for edge in self._reverse_edges.get(dst, set()) if edge[2] == reltype])

    def get_neighbors(self, vertex):
        out_neighbors = frozenset([t for s, t, v in self.get_out_edges(vertex)])
        in_neighbors = frozenset([s for s, t, v in self.get_in_edges(vertex)])

        return out_neighbors.union(in_neighbors)

    def get_reachable_vertices(self, src, max_depth=None):
        '''
        Returns the set of all vertices that can be reached by following
        directed edges out of src.
        '''
        found_vertices = set()

        def bfs(graph, v, depth):
            found_vertices.add(v)
            if max_depth is not None and depth == max_depth:
                return
            else:
                for _, next_v, _ in graph.get_out_edges(v):
                    if next_v not in found_vertices:
                        bfs(graph, next_v, depth + 1)
        bfs(self, src, 0)

        return frozenset(found_vertices)

    def topological_sort(self, start):
        done_order = []

        def dfs_traverse(graph, v):
            for _, next_v, _ in graph.get_out_edges(v):
                dfs_traverse(graph, next_v)
            done_order.append(v)

        dfs_traverse(self, start)
        return reversed(done_order)

    def get_connecting_edge_types(self, src, dst):
        return self._connecting_edges[(src, dst)]

    def subgraph_from_vertices(self, vertices):
        g = SemanticNetwork()

        for v in vertices:
            g.add_vertex(v)
            for source, target, value in self.get_out_edges(v):
                if target in vertices:
                    g.add_edge(source, target, value)

        return g

    def enumerate_without_repeated_edges(self):
        '''
        Enumerates graphs h containing a subset of the edges of this
        graph g s.t. (u, v, r) in g => ((u, v, x) or (v, u, x)) in h
        (that is, all vertices connected by one or more relationships
        in g are connected by exactly one of those relationships in h)
        '''
        if len(self.edges()) == 0:
            yield copy.deepcopy(self)
        else:
            g = copy.deepcopy(self)
            for vertices, reltypes in self._connecting_edges.iteritems():
                src,dst = vertices

                backward_reltypes = set()
                if (dst, src) in self._connecting_edges:
                    backward_reltypes = self._connecting_edges[(dst, src)]

                # This means there actually aren't any edges between
                # the two vertices
                if len(reltypes) + len(backward_reltypes) == 0:
                    continue

                for rel in reltypes:
                    g.del_edge((src, dst, rel))
                for rel in backward_reltypes:
                    g.del_edge((dst, src, rel))

                for graph_portion in g.enumerate_without_repeated_edges():
                    yield graph_portion

                    for rel in reltypes:
                        h = copy.deepcopy(graph_portion)
                        h.add_edge(src, dst, rel)
                        yield h

                    for rel in backward_reltypes:
                        h = copy.deepcopy(graph_portion)
                        h.add_edge(dst, src, rel)
                        yield h
                break

    def __repr__(self):
        return "[SemanticNetwork:\nVertices: %r \nEdges: %r\n]" % (self.vertices(), self.edges())

