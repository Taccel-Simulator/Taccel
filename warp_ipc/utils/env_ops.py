from typing import TYPE_CHECKING
import torch
import warp as wp
if TYPE_CHECKING:
    from warp_ipc.sim_model import ASRModel

def reduce_env_energy_soft_tet(soft_tet_energy: wp.array, env_energy: wp.array, sim: 'ASRModel'):
    if sim.soft_vol_tets_num > 0:
        wp.to_torch(env_energy).scatter_add_(0, wp.to_torch(sim.tet_envs).to(torch.int64), wp.to_torch(soft_tet_energy))

def reduce_env_energy_soft_shell(soft_shell_energy: wp.array, env_energy: wp.array, sim: 'ASRModel'):
    if sim.soft_shell_tris_num > 0:
        wp.to_torch(env_energy).scatter_add_(0, wp.to_torch(sim.face2env[sim.affine_tris_num:sim.affine_tris_num + sim.soft_shell_tris_num]).to(torch.int64), wp.to_torch(soft_shell_energy))

def reduce_env_energy_soft_vert(soft_vert_energy: wp.array, env_energy: wp.array, sim: 'ASRModel'):
    if sim.soft_verts_num > 0:
        wp.to_torch(env_energy).scatter_add_(0, wp.to_torch(sim.node2env[sim.affine_verts_num:]).to(torch.int64), wp.to_torch(soft_vert_energy))

def reduce_env_energy_affine_body(affine_body_energy: wp.array, env_energy: wp.array, sim: 'ASRModel'):
    if sim.affine_body_num > 0:
        wp.to_torch(env_energy).scatter_add_(0, wp.to_torch(sim.body_env_id)[:sim.affine_body_num].to(torch.int64), wp.to_torch(affine_body_energy))

def reduce_env_energy_edge(edge_energy: wp.array, env_energy: wp.array, sim: 'ASRModel'):
    if sim.num_edge > 0:
        wp.to_torch(env_energy).scatter_add_(0, wp.to_torch(sim.edge2env).to(torch.int64), wp.to_torch(edge_energy))

def reduce_env_energy_vert(vert_energy: wp.array, env_energy: wp.array, sim: 'ASRModel'):
    if sim.num_x > 0:
        wp.to_torch(env_energy).scatter_add_(0, wp.to_torch(sim.node2env).to(torch.int64), wp.to_torch(vert_energy))