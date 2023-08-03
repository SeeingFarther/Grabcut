import igraph
import numpy as np


# Graph for the min cut problem
class Graph:
    # Init graph
    def __init__(self, source, target, vertices_num, h, w):
        self.source = source
        self.target = target
        self.vertices_map = np.arange(1, h * w + 1, dtype=int).reshape(h, w)
        self.vertices_num = vertices_num
        self.graph = None
        self.hard_t_edges = []
        self.soft_t_edges = []
        self.n_edges = []
        self.n_edge_capacities = []
        self.hard_t_edge_capacities = []
        self.soft_t_edge_capacities = []
        self.K = 0

    # Get and set functions
    def set_K(self, num):
        self.K = num

    def get_K(self):
        return self.K

    def get_vertex_name(self, i, j):
        return self.vertices_map[i][j]

    def get_vertices_map(self):
        return self.vertices_map

    def get_target(self):
        return self.target

    def get_source(self):
        return self.source

    def set_n_edges_cap(self, capacities):
        self.n_edge_capacities = capacities

    def set_hard_t_edges_cap(self, capacities):
        self.hard_t_edge_capacities = capacities

    def set_soft_t_edges_cap(self, capacities):
        self.soft_t_edge_capacities = capacities

    def set_n_edges(self, edges):
        self.n_edges = edges

    def set_soft_t_edges(self, edges):
        self.soft_t_edges = edges

    def set_hard_t_edges(self, edges):
        self.hard_t_edges = edges

    # Calculate min cut for the graph
    def mincut(self):
        # Merge N-edges with T-edges
        edges = self.n_edges + self.hard_t_edges + self.soft_t_edges
        edge_capacities = self.n_edge_capacities + self.hard_t_edge_capacities + self.soft_t_edge_capacities

        # Build graph
        self.graph = igraph.Graph()
        self.graph.add_vertices(self.vertices_num)
        self.graph.add_edges(edges)

        # Return min cut
        return self.graph.st_mincut(self.source, self.target, edge_capacities)
