from typing import TYPE_CHECKING
import warp as wp
import warp_ipc.utils.matrix as matrix
from warp_ipc.utils.matrix import COOMatrix3x3
if TYPE_CHECKING:
    from warp_ipc.sim_model import ASRModel
from warp_ipc.utils.wp_types import mat66d

@wp.kernel
def construct_stitch_map(stitch_pair: wp.array(dtype=wp.vec2i), stitch_map: wp.array2d(dtype=wp.int32), num_stitch_per_x: wp.array(dtype=wp.int32)):
    pass

@wp.kernel
def stitch_energy_kernel(energy_buffer: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.vec3d), xhat: wp.array(dtype=wp.vec3d), node_area: wp.array(dtype=wp.float64), stitches: wp.array(dtype=wp.vec2i), stitch_stiffness: wp.float64, stitch_damping: wp.float64, dt: wp.float64, scale: wp.float64):
    pass

@wp.kernel
def stitch_gradient_kernel(soft_gradient_x: wp.array(dtype=wp.vec3d), x: wp.array(dtype=wp.vec3d), xhat: wp.array(dtype=wp.vec3d), node_area: wp.array(dtype=wp.float64), stitches: wp.array(dtype=wp.vec2i), stitch_stiffness: wp.float64, stitch_damping: wp.float64, dt: wp.float64, scale: wp.float64, affine_verts_num: wp.int32):
    pass

@wp.kernel
def stitch_hessian_kernel(hess_soft_diag: COOMatrix3x3, hess_soft_stitch: COOMatrix3x3, node_area: wp.array(dtype=wp.float64), stitches: wp.array(dtype=wp.vec2i), stitch_stiffness: wp.float64, stitch_damping: wp.float64, scale: wp.float64, affine_verts_num: wp.int32, affine_dofs: wp.int32):
    pass

def val(x: wp.array, sim: 'ASRModel', scale: float, dt: float) -> float:
    energy = wp.zeros((sim.stitch.shape[0],), dtype=wp.float64, device=sim.device)
    wp.launch(stitch_energy_kernel, dim=sim.stitch.shape[0], inputs=[energy, x, sim.hat_x, sim.node_area, sim.stitch, sim.stitch_stiffness, sim.stitch_damping, dt, scale])
    return wp.utils.array_sum(energy)

def grad(x: wp.array, sim: 'ASRModel', scale: float, soft_gradient_x: wp.array, dt: float):
    wp.launch(stitch_gradient_kernel, dim=sim.stitch.shape[0], inputs=[soft_gradient_x, x, sim.hat_x, sim.node_area, sim.stitch, sim.stitch_stiffness, sim.stitch_damping, dt, scale, sim.affine_verts_num])

def grad_adjoint(x: wp.array, sim: 'ASRModel', scale: float, soft_gradient_x: wp.array, dt: float):
    wp.launch(stitch_gradient_kernel, dim=sim.stitch.shape[0], inputs=[soft_gradient_x, x, sim.hat_x, sim.node_area, sim.stitch, sim.stitch_stiffness, sim.stitch_damping, dt, scale, sim.affine_verts_num], adjoint=True, adj_inputs=[None, None, None, None, None, 0.0, 0.0, 0.0, 0.0, 0])

def hess(sim: 'ASRModel', scale: float):
    wp.launch(stitch_hessian_kernel, dim=sim.stitch.shape[0], inputs=[sim.hess_soft_diag, sim.hess_stitch, sim.node_area, sim.stitch, sim.stitch_stiffness, sim.stitch_damping, scale, sim.affine_verts_num, sim.n_affine_dofs])