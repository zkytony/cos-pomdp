from collections import deque
import json

from .graph import Node, Edge, Graph
from ..utils import thor_camera_pose
from ..actions import ThorAction


class PoseNode(Node):
    """
    PoseNode represents one pose (position, rotation) the agent can reach.
    Note that this should be the camera pose.
    """
    def __init__(self, id, position, rotation):
        """
        Args:
           id (int): ID of the node
           position (tuple): tuple (x, y, z)
           rotation (tuple): tuple (x, y, z); pitch, yaw, roll.
              Not doing quaternion because in ai2thor the mobile robot
              can only do two of the rotation axes so there's no problem using
              Euclidean.  Will restrict the angles to be between 0 to 360.
              Units in degrees (to be consistent with ai2thor).

              yaw refers to rotation of the agent's body.
              pitch refers to rotation of the camera up and down.
        """
        self.position = position
        self.rotation = rotation
        super().__init__(id)


class ActionEdge(Edge):
    def __init__(self, id, node1, node2, action):
        self.action = action
        super().__init__(self, node1, node2, action)


class SceneGraph(Graph):
    def __init__(self, scene, edges):
        """
        Args:
            scene (str): e.g. "FloorPlan2"
            edges (set or dict): ActionEdge objects.
        """
        self.scene = scene
        super().__init__(edges)

    def save(self, outputfile, **dump_params):
        nodes = []
        for nid in self.nodes:
            node = self.nodes[nid]
            nodes.append({"node_id": nid,
                          "position": list(node.position),
                          "rotation": list(node.rotation)})
        edges = []
        for eid in self.edges:
            edge = self.edges[eid]
            node1, node2 = edge.nodes
            edges.append({"edge_id": eid,
                          "node_id1": node1.id,
                          "node_id2": node2.id,
                          "action": edge.action.to_json()})
        json.dump({"nodes": nodes,
                   "edges": edges,
                   "scene": self.scene}, outputfile, **dump_params)


    @classmethod
    def load(self, inputfile):
        obj = json.load(inputfile)
        nodes = {}
        for nobj in obj["nodes"]:
            node = PoseNode(nobj["node_id"],
                            tuple(nobj["position"]),
                            tuple(nobj["rotation"]))
            nodes[node.id] = node
        edges = {}
        for eobj in obj["edges"]:
            edge = ActionEdge(eobj["edge_id"],
                              nodes[eobj["node_id1"]],
                              nodes[eobj["node_id2"]],
                              ThorAction.from_json(eobj["action"]))

            edges[edge.id] = edge
        return SceneGraph(obj["scene"], edges)


def build_scene_graph(controller, actions, outputfile, **dump_params):
    """
    Args:
        controller (ai2thor.Controller)
        actions (list): List of (movement) Action objects
        outputfile (fileobject): file to save this graph
    """
    pose_to_node = {}
    poses_to_edge = {}
    nodes = {}
    edges = {}
    worklist = deque([])

    # Add the first node
    position, rotation = thor_camera_pose(controller, get_tuples=True)
    nodes[0] = PoseNode(0, position, rotation)
    worklist.append((position, rotation))
    pose_to_node[(position, rotation)] = 0

    while len(worklist) > 0:
        pose = worklist.pop()
        node = nodes[pose_to_node[pose]]
        position, rotation = pose
        x, y, z = position
        pitch, yaw, roll = rotation
        for action in actions:
            controller.step(action="Teleport",
                            position=dict(x=x, y=y, z=z),
                            rotation=dict(x=0, y=yaw, z=0),
                            horizon=pitch,
                            standing=True)
            event = controller.step(action=action.name,
                                    **action.params)
            new_pose = thor_camera_pose(event, get_tuples=True)
            if new_pose not in pose_to_node:
                worklist.append(new_pose)
                # Add node
                new_nid = len(nodes)
                pose_to_node[new_pose] = new_nid
                new_position, new_rotation = new_pose
                nodes[new_nid] = PoseNode(new_nid, new_position, new_rotation)

            # Add edge
            if (pose, new_pose) not in poses_to_edge\
               and (new_pose, pose) not in poses_to_edge:
                eid = len(edges)
                new_node = nodes[new_nid]
                edge = ActionEdge(eid, node, new_node, action)
                edges[eid] = edge
                poses_to_edge[(pose, new_pose)] = eid
        # Printing progress information
        print("|V|: {} ; |E|: {}".format(len(nodes), len(edges)), end="\r", flush=True)

    graph = SceneGraph(controller.scene.split("_")[0], edges)
    graph.save(outputfile, **dump_params)
    return graph
