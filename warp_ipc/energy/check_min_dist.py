from typing import TYPE_CHECKING
import torch
import warp as wp
from icecream import ic
import warp_ipc.collision_detection as collision_detection
from warp_ipc.contact.edge_edge_distance import edge_edge_distance
from warp_ipc.contact.point_triangle_distance import point_triangle_distance
from warp_ipc.energy.barrier_energy_gradient import *
from warp_ipc.energy.barrier_energy_hessian import *
from warp_ipc.utils.constants import *
if TYPE_CHECKING:
    from warp_ipc.sim_model import ASRModel

@wp.kernel
def dist_IPC_hs(energy_x: wp.array(dtype=wp.float64), hs_node: wp.array(dtype=wp.int32), hs_ground: wp.array(dtype=wp.int32), x: wp.array(dtype=wp.vec3d), surf_vi: wp.array(dtype=wp.int32), node_xi: wp.array(dtype=wp.float64), half_space_n: wp.array(dtype=wp.vec3d), half_space_o: wp.array(dtype=wp.vec3d)):
    pass

@wp.kernel
def dist_IPC_collisions(energy_x: wp.array(dtype=wp.float64), nodeI: wp.array(dtype=wp.int32), nodeJ: wp.array(dtype=wp.int32), collision_type: wp.array(dtype=wp.int32), x: wp.array(dtype=wp.vec3d), surf_vi: wp.array(dtype=wp.int32), node_xi: wp.array(dtype=wp.float64), edge_xi: wp.array(dtype=wp.float64), face_xi: wp.array(dtype=wp.float64), edge: wp.array(dtype=wp.vec2i), face: wp.array(dtype=wp.vec3i)):
    pass

def min_dist(sim: 'ASRModel', cdw: collision_detection.CollisionData, x: wp.array):
    num_connectivity = int(cdw.num_collisions.numpy()[0])
    num_hs_pair = int(cdw.num_hs_pair.numpy()[0])
    hs_dist = wp.array(shape=num_hs_pair, dtype=wp.float64)
    c_dist = wp.array(shape=num_connectivity, dtype=wp.float64)
    wp.launch(dist_IPC_hs, num_hs_pair, inputs=[hs_dist, cdw.hs_node, cdw.hs_ground, x, sim.surf_vi, sim.node_xi, sim.half_space_n, sim.half_space_o])
    wp.launch(dist_IPC_collisions, num_connectivity, inputs=[c_dist, cdw.nodeI, cdw.nodeJ, cdw.collision_type, x, sim.surf_vi, sim.node_xi, sim.edge_xi, sim.face_xi, sim.edge, sim.face])
    if len(hs_dist) == 0:
        hs_dist = torch.ones(1, dtype=torch.float64).to(sim.device) * sim.dhat
    else:
        hs_dist = wp.to_torch(hs_dist)
    if len(c_dist) == 0:
        c_dist = torch.ones(1, dtype=torch.float64).to(sim.device) * sim.dhat
    else:
        c_dist = wp.to_torch(c_dist)
    dist = torch.cat([hs_dist, c_dist]) / sim.dhat
    return dist.min().item()