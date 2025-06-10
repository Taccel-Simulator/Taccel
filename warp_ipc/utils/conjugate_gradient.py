from typing import TYPE_CHECKING
import torch
import warp as wp
import warp.sparse as wps
from warp.sparse import BsrMatrix
import warp_ipc.utils.log as log
if TYPE_CHECKING:
    from warp_ipc.sim_model import ASRModel

@wp.kernel
def array_inv(x: wp.array(dtype=wp.mat33d)):
    print('[ERROR] Unexpected Recompilation: array_inv')

@wp.kernel
def array_matmul(A: wp.array(dtype=wp.mat33d), y: wp.array(dtype=wp.vec3d), z: wp.array(dtype=wp.vec3d)):
    print('[ERROR] Unexpected Recompilation: array_matmul')

@wp.kernel
def axpy(y: wp.array(dtype=wp.vec3d), x: wp.array(dtype=wp.vec3d), env_beta: wp.array(dtype=wp.float64), ret: wp.array(dtype=wp.vec3d), affine_dof_env_id: wp.array(dtype=wp.int64), node2env: wp.array(dtype=wp.int32), n_affine_dofs: wp.int32, num_constraints: wp.int32, affine_verts_num: wp.int32):
    print('[ERROR] Unexpected Recompilation: axpy')

@wp.kernel
def cg_one_iter(x: wp.array(dtype=wp.vec3d), r: wp.array(dtype=wp.vec3d), q: wp.array(dtype=wp.vec3d), env_alpha: wp.array(dtype=wp.float64), p: wp.array(dtype=wp.vec3d), Ap: wp.array(dtype=wp.vec3d), diag_inv: wp.array(dtype=wp.mat33d), affine_dof_env_id: wp.array(dtype=wp.int64), node2env: wp.array(dtype=wp.int32), n_affine_dofs: wp.int32, num_constraints: wp.int32, affine_verts_num: wp.int32):
    print('[ERROR] Unexpected Recompilation: cg_one_iter')

def conjugate_gradient(x: wp.array, A: BsrMatrix, b: wp.array, sim: 'ASRModel', tol=0.001, max_iters: None | int=500):
    if max_iters is None:
        max_iters = b.shape[0]
    x.zero_()
    if sim.linear_constraint_helper.num_constraints > 0:
        wp.copy(x[:sim.linear_constraint_helper.num_constraints // 3], b[:sim.linear_constraint_helper.num_constraints // 3])
    num_block = b.shape[0]

    def _array_inner_per_env(a: wp.array, b: wp.array, out: wp.array):
        torch.sum(wp.to_torch(a) * wp.to_torch(b), dim=1, out=wp.to_torch(sim.sim_cache.cg_inner_tmp))
        out.fill_(0.0)
        if sim.affine_body_num > 0:
            wp.to_torch(out).scatter_add_(0, wp.to_torch(sim.linear_constraint_helper.affine_dof_env_id), wp.to_torch(sim.sim_cache.cg_inner_tmp[sim.linear_constraint_helper.num_constraints // 3:sim.n_affine_dofs // 3]))
        if sim.soft_verts_num > 0:
            wp.to_torch(out).scatter_add_(0, wp.to_torch(sim.node2env[sim.affine_verts_num:]).to(torch.int64), wp.to_torch(sim.sim_cache.cg_inner_tmp[sim.n_affine_dofs // 3:]))
    sim.sim_cache.cg_diag_inv.dtype = A.values.dtype
    wps.bsr_get_diag(A, sim.sim_cache.cg_diag_inv)
    wp.launch(kernel=array_inv, dim=num_block, inputs=[sim.sim_cache.cg_diag_inv], device=sim.device)
    wp.copy(sim.sim_cache.cg_r, b)
    wp.launch(kernel=array_matmul, dim=num_block, inputs=[sim.sim_cache.cg_diag_inv, sim.sim_cache.cg_r, sim.sim_cache.cg_q], device=b.device)
    wp.copy(sim.sim_cache.cg_p, sim.sim_cache.cg_q)
    _array_inner_per_env(sim.sim_cache.cg_r, sim.sim_cache.cg_q, sim.sim_cache.cg_zTrk)
    residual = wp.to_torch(sim.sim_cache.cg_residual)
    tolerance = wp.to_torch(sim.sim_cache.cg_tol)
    residual[:] = wp.to_torch(sim.sim_cache.cg_zTrk) ** 0.5
    if (residual < 1e-16).all():
        return x
    tolerance[:] = torch.minimum(tol * residual, torch.tensor(1.0, dtype=torch.float64, device=sim.device))
    sim.sim_cache.cg_solved.zero_()
    solved = wp.to_torch(sim.sim_cache.cg_solved)
    for iter in range(max_iters):
        residual[:] = wp.to_torch(sim.sim_cache.cg_zTrk) ** 0.5
        if (residual < tolerance).all():
            break
        solved |= residual < tolerance
        wps.bsr_mv(A, sim.sim_cache.cg_p, sim.sim_cache.cg_Ap)
        _array_inner_per_env(sim.sim_cache.cg_Ap, sim.sim_cache.cg_p, sim.sim_cache.cg_inner_ApTp)
        wp.to_torch(sim.sim_cache.cg_alpha)[:] = wp.to_torch(sim.sim_cache.cg_zTrk) / (wp.to_torch(sim.sim_cache.cg_inner_ApTp) + 1e-30)
        wp.to_torch(sim.sim_cache.cg_alpha)[solved] = 0.0
        wp.launch(kernel=cg_one_iter, dim=num_block, inputs=[x, sim.sim_cache.cg_r, sim.sim_cache.cg_q, sim.sim_cache.cg_alpha, sim.sim_cache.cg_p, sim.sim_cache.cg_Ap, sim.sim_cache.cg_diag_inv, sim.linear_constraint_helper.affine_dof_env_id, sim.node2env, sim.n_affine_dofs, sim.linear_constraint_helper.num_constraints, sim.affine_verts_num], device=sim.device)
        wp.copy(sim.sim_cache.cg_zTrk_last, sim.sim_cache.cg_zTrk)
        _array_inner_per_env(sim.sim_cache.cg_q, sim.sim_cache.cg_r, sim.sim_cache.cg_zTrk)
        wp.to_torch(sim.sim_cache.cg_beta)[:] = wp.to_torch(sim.sim_cache.cg_zTrk) / (wp.to_torch(sim.sim_cache.cg_zTrk_last) + 1e-30)
        wp.to_torch(sim.sim_cache.cg_beta)[solved] = 0
        wp.launch(kernel=axpy, dim=num_block, inputs=[sim.sim_cache.cg_q, sim.sim_cache.cg_p, sim.sim_cache.cg_beta, sim.sim_cache.cg_p, sim.linear_constraint_helper.affine_dof_env_id, sim.node2env, sim.n_affine_dofs, sim.linear_constraint_helper.num_constraints, sim.affine_verts_num], device=sim.device)
    log.debug(f'[Conjugate Gradient] converge iter={iter}, residual={residual.cpu()}')