from collections import deque
from cospomdp.utils.graph import Node, Graph, Edge
from cospomdp.utils.math import euclidean_dist

class TopoNode(Node):
    """TopoNode is a node on the grid map."""

    def __init__(self, node_id, grid_pos):
        super().__init__(node_id, grid_pos)
        self._coords = grid_pos

    @property
    def pos(self):
        return self.data


class TopoMap(Graph):

    """To create a TopoMap,
    construct a mapping called e.g. `edges` that maps from edge id to Edge,
    and do TopoMap(edges)."""

    def closest_node(self, x, y):
        """Given a point at (x,y) find the node that is closest to this point.
        """
        return min(self.nodes,
                   key=lambda nid: euclidean_dist(self.nodes[nid].pose[:2], (x,y)))

    def navigable(self, nid1, nid2):
        # DFS find path from nid1 to nid2
        stack = deque()
        stack.append(nid1)
        visited = set()
        while len(stack) > 0:
            nid = stack.pop()
            if nid == nid2:
                return True
            for neighbor_nid in self.neighbors(nid):
                if neighbor_nid not in visited:
                    stack.append(neighbor_nid)
                    visited.add(neighbor_nid)
        return False
