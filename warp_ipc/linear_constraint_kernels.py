import warp as wp
from warp_ipc.utils.constants import _0, _1

@wp.kernel
def linear_constraint_project_system_kernel(offsets: wp.array(dtype=wp.int32), columns: wp.array(dtype=wp.int32), values: wp.array(dtype=wp.mat33d), gradient: wp.array(dtype=wp.vec3d), n_reduced_dofs: wp.int32):
    print('[ERROR] Unexpected Recompilation: linear_constraint_project_system_kernel')