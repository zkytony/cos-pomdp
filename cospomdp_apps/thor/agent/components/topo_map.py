import cv2
import numpy as np
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


class TopoEdge(Edge):
    def __init__(self, id, node1, node2, grid_path):
        super().__init__(id, node1, node2, data=grid_path)

    @property
    def grid_path(self):
        return self.data

    @property
    def grid_dist(self):
        if self.grid_path is None:
            return float('inf')
        else:
            return len(self.grid_path)


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


#------ Visualization -----#
# In all fucntions, r means resolution, in pygmae visualziation
def draw_edge(img, pos1, pos2, r, thickness=2, color=(0, 0, 0)):
    x1, y1 = pos1
    x2, y2 = pos2
    cv2.line(img, (y1*r+r//2, x1*r+r//2), (y2*r+r//2, x2*r+r//2),
             color, thickness=thickness)
    return img

def draw_topo(img, topo_map, r, draw_grid_path=False,
              path_color=(52, 235, 222), edge_color=(0, 0, 0), edge_thickness=2):
    for eid in topo_map.edges:
        edge = topo_map.edges[eid]

        if draw_grid_path and edge.grid_path is not None:
            for x, y in edge.grid_path:
                cv2.rectangle(img,
                              (y*r, x*r),
                              (y*r+r, x*r+r),
                              path_color, -1)

        node1, node2 = edge.nodes
        pos1 = node1.pos
        pos2 = node2.pos
        img = draw_edge(img, pos1, pos2, r, edge_thickness, color=edge_color)

    for nid in topo_map.nodes:
        pos = topo_map.nodes[nid].pos
        img = mark_cell(img, pos, int(nid), r)
    return img

def mark_cell(img, pos, nid, r, linewidth=1, unmark=False):
    if unmark:
        color = (255, 255, 255, 255)
    else:
        color = (242, 227, 15, 255)
    x, y = pos
    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                  color, -1)
    # Draw boundary
    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                  (0, 0, 0), linewidth)

    if not unmark:
        font                   = cv2.FONT_HERSHEY_COMPLEX_SMALL
        fontScale              = 0.72
        fontColor              = (43, 13, 4)
        lineType               = 1
        imgtxt = np.full((r, r, 4), color, dtype=np.uint8)
        text_loc = (int(round(r/4)), int(round(r/1.5)))
        cv2.putText(imgtxt, str(nid), text_loc, #(y*r+r//4, x*r+r//2),
                    font, fontScale, fontColor, lineType)
        imgtxt = cv2.rotate(imgtxt, cv2.ROTATE_90_CLOCKWISE)
        img[x*r:x*r+r, y*r:y*r+r] = imgtxt
    return img
