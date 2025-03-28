from typing import TYPE_CHECKING
import warp as wp
import warp_ipc.utils.matrix as matrix
from warp_ipc.utils.matrix import COOMatrix3x3
if TYPE_CHECKING:
    from warp_ipc.sim_model import ASRModel
from warp_ipc.utils.make_pd import make_pd_12x12
from warp_ipc.utils.wp_types import mat12x12d

@wp.func
def dihedral_angle(v0: wp.vec3d, v1: wp.vec3d, v2: wp.vec3d, v3: wp.vec3d):
    pass

@wp.func
def dihedral_angle_gradient(v2: wp.vec3d, v0: wp.vec3d, v1: wp.vec3d, v3: wp.vec3d):
    pass

@wp.func
def compute_mHat(xp: wp.vec3d, xe0: wp.vec3d, xe1: wp.vec3d):
    pass

@wp.func
def dihedral_angle_hessian(v2: wp.vec3d, v0: wp.vec3d, v1: wp.vec3d, v3: wp.vec3d):
    pass

@wp.kernel
def dihedral_bending_energy_kernel(energy_buffer: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.vec3d), bendings: wp.array(dtype=wp.vec4i), bending_rest_angle: wp.array(dtype=wp.float64), bending_e: wp.array(dtype=wp.float64), bending_h: wp.array(dtype=wp.float64), bending_stiff: wp.array(dtype=wp.float64), scale: wp.float64):
    pass

@wp.kernel
def dihedral_bending_gradient_kernel(soft_gradient_x: wp.array(dtype=wp.vec3d), x: wp.array(dtype=wp.vec3d), bendings: wp.array(dtype=wp.vec4i), bending_rest_angle: wp.array(dtype=wp.float64), bending_e: wp.array(dtype=wp.float64), bending_h: wp.array(dtype=wp.float64), bending_stiff: wp.array(dtype=wp.float64), scale: wp.float64, affine_verts_num: wp.int32):
    pass

@wp.kernel
def dihedral_bending_hessian_kernel(hess_soft_diag: COOMatrix3x3, hess_soft_bending: COOMatrix3x3, x: wp.array(dtype=wp.vec3d), bendings: wp.array(dtype=wp.vec4i), bending_rest_angle: wp.array(dtype=wp.float64), bending_e: wp.array(dtype=wp.float64), bending_h: wp.array(dtype=wp.float64), bending_stiff: wp.array(dtype=wp.float64), scale: wp.float64, project_pd: wp.bool, affine_verts_num: wp.int32, affine_dofs: wp.int32):
    pass

def val(x: wp.array, sim: 'ASRModel', scale: float) -> float:
    energy = wp.zeros((sim.bending.shape[0],), dtype=wp.float64, device=sim.device)
    wp.launch(dihedral_bending_energy_kernel, dim=sim.bending.shape[0], inputs=[energy, x, sim.bending, sim.bending_rest_angle, sim.bending_e, sim.bending_h, sim.bending_stiffness, scale])
    return wp.utils.array_sum(energy)

def grad(x: wp.array, sim: 'ASRModel', scale: float, soft_gradient_x: wp.array):
    wp.launch(dihedral_bending_gradient_kernel, dim=sim.bending.shape[0], inputs=[soft_gradient_x, x, sim.bending, sim.bending_rest_angle, sim.bending_e, sim.bending_h, sim.bending_stiffness, scale, sim.affine_verts_num])

def grad_adjoint(x: wp.array, sim: 'ASRModel', scale: float, soft_gradient_x: wp.array):
    wp.launch(dihedral_bending_gradient_kernel, dim=sim.bending.shape[0], inputs=[soft_gradient_x, x, sim.bending, sim.bending_rest_angle, sim.bending_e, sim.bending_h, sim.bending_stiffness, scale, sim.affine_verts_num], adjoint=True, adj_inputs=[None, None, None, None, None, None, None, 0.0, 0])

def hess(x: wp.array, sim: 'ASRModel', scale: float, project_pd: bool=True):
    wp.launch(dihedral_bending_hessian_kernel, dim=sim.bending.shape[0], inputs=[sim.hess_soft_diag, sim.hess_soft_bending_elastic, x, sim.bending, sim.bending_rest_angle, sim.bending_e, sim.bending_h, sim.bending_stiffness, scale, project_pd, sim.affine_verts_num, sim.n_affine_dofs])