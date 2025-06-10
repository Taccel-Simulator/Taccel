import warp as wp
vec12d = wp.vec(12, dtype=wp.float64)
vec12i = wp.vec(12, dtype=wp.int32)
mat32d = wp.mat((3, 2), dtype=wp.float64)
mat33d = wp.mat((3, 3), dtype=wp.float64)
vec6d = wp.vec(6, dtype=wp.float64)
vec9d = wp.vec(9, dtype=wp.float64)
mat46d = wp.mat((4, 6), dtype=wp.float64)
mat96d = wp.mat((9, 6), dtype=wp.float64)
mat66d = wp.mat((6, 6), dtype=wp.float64)
mat99d = wp.mat((9, 9), dtype=wp.float64)
mat12x9d = wp.mat((12, 9), dtype=wp.float64)
mat12x12d = wp.mat((12, 12), dtype=wp.float64)
vec8i = wp.vec(8, dtype=wp.int32)

@wp.kernel
def vec3d_to_vec3(vec_in: wp.array(dtype=wp.vec3), vec_out: wp.array(dtype=wp.vec3)):
    print('[ERROR] Unexpected Recompilation: vec3d_to_vec3')