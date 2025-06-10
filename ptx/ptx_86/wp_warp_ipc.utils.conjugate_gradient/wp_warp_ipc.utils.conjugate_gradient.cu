
#define WP_TILE_BLOCK_DIM 256
#define WP_NO_CRT
#include "builtin.h"

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

#define builtin_tid1d() wp::tid(_idx, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, _idx, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, _idx, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, _idx, dim)

extern "C" {
}


extern "C" __global__ void array_matmul_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::mat_t<3,3,wp::float64>> var_A,
    wp::array_t<wp::vec_t<3,wp::float64>> var_y,
    wp::array_t<wp::vec_t<3,wp::float64>> var_z)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        wp::mat_t<3,3,wp::float64>* var_1;
        wp::vec_t<3,wp::float64>* var_2;
        wp::vec_t<3,wp::float64> var_3;
        wp::mat_t<3,3,wp::float64> var_4;
        wp::vec_t<3,wp::float64> var_5;
        //---------
        // forward
        // def array_matmul(A: wp.array(dtype=wp.mat33d), y: wp.array(dtype=wp.vec3d), z: wp.array(dtype=wp.vec3d)):       <L 21>
        // tid = wp.tid()                                                                         <L 22>
        var_0 = builtin_tid1d();
        // z[tid] = A[tid] @ y[tid]                                                               <L 23>
        var_1 = wp::address(var_A, var_0);
        var_2 = wp::address(var_y, var_0);
        var_4 = wp::load(var_1);
        var_5 = wp::load(var_2);
        var_3 = wp::mul(var_4, var_5);
        wp::array_store(var_z, var_0, var_3);
    }
}



extern "C" __global__ void array_matmul_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::mat_t<3,3,wp::float64>> var_A,
    wp::array_t<wp::vec_t<3,wp::float64>> var_y,
    wp::array_t<wp::vec_t<3,wp::float64>> var_z,
    wp::array_t<wp::mat_t<3,3,wp::float64>> adj_A,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_y,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_z)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        wp::mat_t<3,3,wp::float64>* var_1;
        wp::vec_t<3,wp::float64>* var_2;
        wp::vec_t<3,wp::float64> var_3;
        wp::mat_t<3,3,wp::float64> var_4;
        wp::vec_t<3,wp::float64> var_5;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::mat_t<3,3,wp::float64> adj_1 = {};
        wp::vec_t<3,wp::float64> adj_2 = {};
        wp::vec_t<3,wp::float64> adj_3 = {};
        wp::mat_t<3,3,wp::float64> adj_4 = {};
        wp::vec_t<3,wp::float64> adj_5 = {};
        //---------
        // forward
        // def array_matmul(A: wp.array(dtype=wp.mat33d), y: wp.array(dtype=wp.vec3d), z: wp.array(dtype=wp.vec3d)):       <L 21>
        // tid = wp.tid()                                                                         <L 22>
        var_0 = builtin_tid1d();
        // z[tid] = A[tid] @ y[tid]                                                               <L 23>
        var_1 = wp::address(var_A, var_0);
        var_2 = wp::address(var_y, var_0);
        var_4 = wp::load(var_1);
        var_5 = wp::load(var_2);
        var_3 = wp::mul(var_4, var_5);
        // wp::array_store(var_z, var_0, var_3);
        //---------
        // reverse
        wp::adj_array_store(var_z, var_0, var_3, adj_z, adj_0, adj_3);
        wp::adj_mul(var_4, var_5, adj_1, adj_2, adj_3);
        wp::adj_load(var_2, adj_2, adj_5);
        wp::adj_load(var_1, adj_1, adj_4);
        wp::adj_address(var_y, var_0, adj_y, adj_0, adj_2);
        wp::adj_address(var_A, var_0, adj_A, adj_0, adj_1);
        // adj: z[tid] = A[tid] @ y[tid]                                                          <L 23>
        // adj: tid = wp.tid()                                                                    <L 22>
        // adj: def array_matmul(A: wp.array(dtype=wp.mat33d), y: wp.array(dtype=wp.vec3d), z: wp.array(dtype=wp.vec3d)):  <L 21>
        continue;
    }
}



extern "C" __global__ void axpy_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float64>> var_y,
    wp::array_t<wp::vec_t<3,wp::float64>> var_x,
    wp::array_t<wp::float64> var_env_beta,
    wp::array_t<wp::vec_t<3,wp::float64>> var_ret,
    wp::array_t<wp::int64> var_affine_dof_env_id,
    wp::array_t<wp::int32> var_node2env,
    wp::int32 var_n_affine_dofs,
    wp::int32 var_num_constraints,
    wp::int32 var_affine_verts_num)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::float32 var_1 = 0.0;
        wp::float64 var_2;
        const wp::int32 var_3 = 3;
        wp::int32 var_4;
        bool var_5;
        const wp::int32 var_6 = 3;
        wp::int32 var_7;
        bool var_8;
        const wp::int32 var_9 = 3;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int64* var_12;
        wp::float64* var_13;
        wp::int64 var_14;
        wp::float64 var_15;
        wp::float64 var_16;
        wp::float64 var_17;
        const wp::int32 var_18 = 3;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::int32* var_22;
        wp::float64* var_23;
        wp::int32 var_24;
        wp::float64 var_25;
        wp::float64 var_26;
        wp::float64 var_27;
        wp::float64 var_28;
        wp::vec_t<3,wp::float64>* var_29;
        wp::vec_t<3,wp::float64>* var_30;
        wp::vec_t<3,wp::float64> var_31;
        wp::vec_t<3,wp::float64> var_32;
        wp::vec_t<3,wp::float64> var_33;
        wp::vec_t<3,wp::float64> var_34;
        //---------
        // forward
        // def axpy(                                                                              <L 27>
        // tid = wp.tid()                                                                         <L 38>
        var_0 = builtin_tid1d();
        // beta = wp.float64(0.0)                                                                 <L 39>
        var_2 = wp::float64(var_1);
        // if tid < num_constraints // 3:                                                         <L 40>
        var_4 = wp::floordiv(var_num_constraints, var_3);
        var_5 = (var_0 < var_4);
        if (var_5) {
            // return                                                                             <L 41>
            continue;
        }
        if (!var_5) {
            // elif tid < n_affine_dofs // 3:                                                     <L 42>
            var_7 = wp::floordiv(var_n_affine_dofs, var_6);
            var_8 = (var_0 < var_7);
            if (var_8) {
                // beta = env_beta[affine_dof_env_id[tid - num_constraints // 3]]                 <L 43>
                var_10 = wp::floordiv(var_num_constraints, var_9);
                var_11 = wp::sub(var_0, var_10);
                var_12 = wp::address(var_affine_dof_env_id, var_11);
                var_14 = wp::load(var_12);
                var_13 = wp::address(var_env_beta, var_14);
                var_16 = wp::load(var_13);
                var_15 = wp::copy(var_16);
            }
            var_17 = wp::where(var_8, var_15, var_2);
            if (!var_8) {
                // beta = env_beta[node2env[tid - n_affine_dofs // 3 + affine_verts_num]]         <L 45>
                var_19 = wp::floordiv(var_n_affine_dofs, var_18);
                var_20 = wp::sub(var_0, var_19);
                var_21 = wp::add(var_20, var_affine_verts_num);
                var_22 = wp::address(var_node2env, var_21);
                var_24 = wp::load(var_22);
                var_23 = wp::address(var_env_beta, var_24);
                var_26 = wp::load(var_23);
                var_25 = wp::copy(var_26);
            }
            var_27 = wp::where(var_8, var_17, var_25);
        }
        var_28 = wp::where(var_5, var_2, var_27);
        // ret[tid] = y[tid] + beta * x[tid]                                                      <L 46>
        var_29 = wp::address(var_y, var_0);
        var_30 = wp::address(var_x, var_0);
        var_32 = wp::load(var_30);
        var_31 = wp::mul(var_28, var_32);
        var_34 = wp::load(var_29);
        var_33 = wp::add(var_34, var_31);
        wp::array_store(var_ret, var_0, var_33);
    }
}



extern "C" __global__ void axpy_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float64>> var_y,
    wp::array_t<wp::vec_t<3,wp::float64>> var_x,
    wp::array_t<wp::float64> var_env_beta,
    wp::array_t<wp::vec_t<3,wp::float64>> var_ret,
    wp::array_t<wp::int64> var_affine_dof_env_id,
    wp::array_t<wp::int32> var_node2env,
    wp::int32 var_n_affine_dofs,
    wp::int32 var_num_constraints,
    wp::int32 var_affine_verts_num,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_y,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_x,
    wp::array_t<wp::float64> adj_env_beta,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_ret,
    wp::array_t<wp::int64> adj_affine_dof_env_id,
    wp::array_t<wp::int32> adj_node2env,
    wp::int32 adj_n_affine_dofs,
    wp::int32 adj_num_constraints,
    wp::int32 adj_affine_verts_num)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::float32 var_1 = 0.0;
        wp::float64 var_2;
        const wp::int32 var_3 = 3;
        wp::int32 var_4;
        bool var_5;
        const wp::int32 var_6 = 3;
        wp::int32 var_7;
        bool var_8;
        const wp::int32 var_9 = 3;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int64* var_12;
        wp::float64* var_13;
        wp::int64 var_14;
        wp::float64 var_15;
        wp::float64 var_16;
        wp::float64 var_17;
        const wp::int32 var_18 = 3;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::int32* var_22;
        wp::float64* var_23;
        wp::int32 var_24;
        wp::float64 var_25;
        wp::float64 var_26;
        wp::float64 var_27;
        wp::float64 var_28;
        wp::vec_t<3,wp::float64>* var_29;
        wp::vec_t<3,wp::float64>* var_30;
        wp::vec_t<3,wp::float64> var_31;
        wp::vec_t<3,wp::float64> var_32;
        wp::vec_t<3,wp::float64> var_33;
        wp::vec_t<3,wp::float64> var_34;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::float32 adj_1 = {};
        wp::float64 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        bool adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        bool adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int64 adj_12 = {};
        wp::float64 adj_13 = {};
        wp::int64 adj_14 = {};
        wp::float64 adj_15 = {};
        wp::float64 adj_16 = {};
        wp::float64 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::float64 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::float64 adj_25 = {};
        wp::float64 adj_26 = {};
        wp::float64 adj_27 = {};
        wp::float64 adj_28 = {};
        wp::vec_t<3,wp::float64> adj_29 = {};
        wp::vec_t<3,wp::float64> adj_30 = {};
        wp::vec_t<3,wp::float64> adj_31 = {};
        wp::vec_t<3,wp::float64> adj_32 = {};
        wp::vec_t<3,wp::float64> adj_33 = {};
        wp::vec_t<3,wp::float64> adj_34 = {};
        //---------
        // forward
        // def axpy(                                                                              <L 27>
        // tid = wp.tid()                                                                         <L 38>
        var_0 = builtin_tid1d();
        // beta = wp.float64(0.0)                                                                 <L 39>
        var_2 = wp::float64(var_1);
        // if tid < num_constraints // 3:                                                         <L 40>
        var_4 = wp::floordiv(var_num_constraints, var_3);
        var_5 = (var_0 < var_4);
        if (var_5) {
            // return                                                                             <L 41>
            goto label0;
        }
        if (!var_5) {
            // elif tid < n_affine_dofs // 3:                                                     <L 42>
            var_7 = wp::floordiv(var_n_affine_dofs, var_6);
            var_8 = (var_0 < var_7);
            if (var_8) {
                // beta = env_beta[affine_dof_env_id[tid - num_constraints // 3]]                 <L 43>
                var_10 = wp::floordiv(var_num_constraints, var_9);
                var_11 = wp::sub(var_0, var_10);
                var_12 = wp::address(var_affine_dof_env_id, var_11);
                var_14 = wp::load(var_12);
                var_13 = wp::address(var_env_beta, var_14);
                var_16 = wp::load(var_13);
                var_15 = wp::copy(var_16);
            }
            var_17 = wp::where(var_8, var_15, var_2);
            if (!var_8) {
                // beta = env_beta[node2env[tid - n_affine_dofs // 3 + affine_verts_num]]         <L 45>
                var_19 = wp::floordiv(var_n_affine_dofs, var_18);
                var_20 = wp::sub(var_0, var_19);
                var_21 = wp::add(var_20, var_affine_verts_num);
                var_22 = wp::address(var_node2env, var_21);
                var_24 = wp::load(var_22);
                var_23 = wp::address(var_env_beta, var_24);
                var_26 = wp::load(var_23);
                var_25 = wp::copy(var_26);
            }
            var_27 = wp::where(var_8, var_17, var_25);
        }
        var_28 = wp::where(var_5, var_2, var_27);
        // ret[tid] = y[tid] + beta * x[tid]                                                      <L 46>
        var_29 = wp::address(var_y, var_0);
        var_30 = wp::address(var_x, var_0);
        var_32 = wp::load(var_30);
        var_31 = wp::mul(var_28, var_32);
        var_34 = wp::load(var_29);
        var_33 = wp::add(var_34, var_31);
        // wp::array_store(var_ret, var_0, var_33);
        //---------
        // reverse
        wp::adj_array_store(var_ret, var_0, var_33, adj_ret, adj_0, adj_33);
        wp::adj_add(var_34, var_31, adj_29, adj_31, adj_33);
        wp::adj_load(var_29, adj_29, adj_34);
        wp::adj_mul(var_28, var_32, adj_28, adj_30, adj_31);
        wp::adj_load(var_30, adj_30, adj_32);
        wp::adj_address(var_x, var_0, adj_x, adj_0, adj_30);
        wp::adj_address(var_y, var_0, adj_y, adj_0, adj_29);
        // adj: ret[tid] = y[tid] + beta * x[tid]                                                 <L 46>
        wp::adj_where(var_5, var_2, var_27, adj_5, adj_2, adj_27, adj_28);
        if (!var_5) {
            wp::adj_where(var_8, var_17, var_25, adj_8, adj_17, adj_25, adj_27);
            if (!var_8) {
                wp::adj_copy(var_26, adj_23, adj_25);
                wp::adj_load(var_23, adj_23, adj_26);
                wp::adj_address(var_env_beta, var_24, adj_env_beta, adj_22, adj_23);
                wp::adj_load(var_22, adj_22, adj_24);
                wp::adj_address(var_node2env, var_21, adj_node2env, adj_21, adj_22);
                wp::adj_add(var_20, var_affine_verts_num, adj_20, adj_affine_verts_num, adj_21);
                wp::adj_sub(var_0, var_19, adj_0, adj_19, adj_20);
                wp::adj_floordiv(var_n_affine_dofs, var_18, adj_n_affine_dofs, adj_18, adj_19);
                // adj: beta = env_beta[node2env[tid - n_affine_dofs // 3 + affine_verts_num]]    <L 45>
            }
            wp::adj_where(var_8, var_15, var_2, adj_8, adj_15, adj_2, adj_17);
            if (var_8) {
                wp::adj_copy(var_16, adj_13, adj_15);
                wp::adj_load(var_13, adj_13, adj_16);
                wp::adj_address(var_env_beta, var_14, adj_env_beta, adj_12, adj_13);
                wp::adj_load(var_12, adj_12, adj_14);
                wp::adj_address(var_affine_dof_env_id, var_11, adj_affine_dof_env_id, adj_11, adj_12);
                wp::adj_sub(var_0, var_10, adj_0, adj_10, adj_11);
                wp::adj_floordiv(var_num_constraints, var_9, adj_num_constraints, adj_9, adj_10);
                // adj: beta = env_beta[affine_dof_env_id[tid - num_constraints // 3]]            <L 43>
            }
            wp::adj_floordiv(var_n_affine_dofs, var_6, adj_n_affine_dofs, adj_6, adj_7);
            // adj: elif tid < n_affine_dofs // 3:                                                <L 42>
        }
        if (var_5) {
            label0:;
            // adj: return                                                                        <L 41>
        }
        wp::adj_floordiv(var_num_constraints, var_3, adj_num_constraints, adj_3, adj_4);
        // adj: if tid < num_constraints // 3:                                                    <L 40>
        wp::adj_float64(var_1, adj_1, adj_2);
        // adj: beta = wp.float64(0.0)                                                            <L 39>
        // adj: tid = wp.tid()                                                                    <L 38>
        // adj: def axpy(                                                                         <L 27>
        continue;
    }
}



extern "C" __global__ void array_inv_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::mat_t<3,3,wp::float64>> var_x)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        wp::mat_t<3,3,wp::float64>* var_1;
        wp::mat_t<3,3,wp::float64> var_2;
        wp::mat_t<3,3,wp::float64> var_3;
        //---------
        // forward
        // def array_inv(x: wp.array(dtype=wp.mat33d)):                                           <L 15>
        // tid = wp.tid()                                                                         <L 16>
        var_0 = builtin_tid1d();
        // x[tid] = wp.inverse(x[tid])                                                            <L 17>
        var_1 = wp::address(var_x, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::inverse(var_3);
        wp::array_store(var_x, var_0, var_2);
    }
}



extern "C" __global__ void array_inv_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::mat_t<3,3,wp::float64>> var_x,
    wp::array_t<wp::mat_t<3,3,wp::float64>> adj_x)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        wp::mat_t<3,3,wp::float64>* var_1;
        wp::mat_t<3,3,wp::float64> var_2;
        wp::mat_t<3,3,wp::float64> var_3;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::mat_t<3,3,wp::float64> adj_1 = {};
        wp::mat_t<3,3,wp::float64> adj_2 = {};
        wp::mat_t<3,3,wp::float64> adj_3 = {};
        //---------
        // forward
        // def array_inv(x: wp.array(dtype=wp.mat33d)):                                           <L 15>
        // tid = wp.tid()                                                                         <L 16>
        var_0 = builtin_tid1d();
        // x[tid] = wp.inverse(x[tid])                                                            <L 17>
        var_1 = wp::address(var_x, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::inverse(var_3);
        // wp::array_store(var_x, var_0, var_2);
        //---------
        // reverse
        wp::adj_array_store(var_x, var_0, var_2, adj_x, adj_0, adj_2);
        wp::adj_inverse(var_3, var_2, adj_1, adj_2);
        wp::adj_load(var_1, adj_1, adj_3);
        wp::adj_address(var_x, var_0, adj_x, adj_0, adj_1);
        // adj: x[tid] = wp.inverse(x[tid])                                                       <L 17>
        // adj: tid = wp.tid()                                                                    <L 16>
        // adj: def array_inv(x: wp.array(dtype=wp.mat33d)):                                      <L 15>
        continue;
    }
}



extern "C" __global__ void cg_one_iter_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float64>> var_x,
    wp::array_t<wp::vec_t<3,wp::float64>> var_r,
    wp::array_t<wp::vec_t<3,wp::float64>> var_q,
    wp::array_t<wp::float64> var_env_alpha,
    wp::array_t<wp::vec_t<3,wp::float64>> var_p,
    wp::array_t<wp::vec_t<3,wp::float64>> var_Ap,
    wp::array_t<wp::mat_t<3,3,wp::float64>> var_diag_inv,
    wp::array_t<wp::int64> var_affine_dof_env_id,
    wp::array_t<wp::int32> var_node2env,
    wp::int32 var_n_affine_dofs,
    wp::int32 var_num_constraints,
    wp::int32 var_affine_verts_num)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::float32 var_1 = 0.0;
        wp::float64 var_2;
        const wp::int32 var_3 = 3;
        wp::int32 var_4;
        bool var_5;
        const wp::int32 var_6 = 3;
        wp::int32 var_7;
        bool var_8;
        const wp::int32 var_9 = 3;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int64* var_12;
        wp::float64* var_13;
        wp::int64 var_14;
        wp::float64 var_15;
        wp::float64 var_16;
        wp::float64 var_17;
        const wp::int32 var_18 = 3;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::int32* var_22;
        wp::float64* var_23;
        wp::int32 var_24;
        wp::float64 var_25;
        wp::float64 var_26;
        wp::float64 var_27;
        wp::float64 var_28;
        wp::vec_t<3,wp::float64>* var_29;
        wp::vec_t<3,wp::float64> var_30;
        wp::vec_t<3,wp::float64> var_31;
        wp::vec_t<3,wp::float64> var_32;
        wp::vec_t<3,wp::float64>* var_33;
        wp::vec_t<3,wp::float64> var_34;
        wp::vec_t<3,wp::float64> var_35;
        wp::vec_t<3,wp::float64> var_36;
        wp::mat_t<3,3,wp::float64>* var_37;
        wp::vec_t<3,wp::float64>* var_38;
        wp::vec_t<3,wp::float64> var_39;
        wp::mat_t<3,3,wp::float64> var_40;
        wp::vec_t<3,wp::float64> var_41;
        //---------
        // forward
        // def cg_one_iter(                                                                       <L 50>
        // tid = wp.tid()                                                                         <L 64>
        var_0 = builtin_tid1d();
        // alpha = wp.float64(0.0)                                                                <L 65>
        var_2 = wp::float64(var_1);
        // if tid < num_constraints // 3:                                                         <L 66>
        var_4 = wp::floordiv(var_num_constraints, var_3);
        var_5 = (var_0 < var_4);
        if (var_5) {
            // return                                                                             <L 67>
            continue;
        }
        if (!var_5) {
            // elif tid < n_affine_dofs // 3:                                                     <L 68>
            var_7 = wp::floordiv(var_n_affine_dofs, var_6);
            var_8 = (var_0 < var_7);
            if (var_8) {
                // alpha = env_alpha[affine_dof_env_id[tid - num_constraints // 3]]               <L 69>
                var_10 = wp::floordiv(var_num_constraints, var_9);
                var_11 = wp::sub(var_0, var_10);
                var_12 = wp::address(var_affine_dof_env_id, var_11);
                var_14 = wp::load(var_12);
                var_13 = wp::address(var_env_alpha, var_14);
                var_16 = wp::load(var_13);
                var_15 = wp::copy(var_16);
            }
            var_17 = wp::where(var_8, var_15, var_2);
            if (!var_8) {
                // alpha = env_alpha[node2env[tid - n_affine_dofs // 3 + affine_verts_num]]       <L 71>
                var_19 = wp::floordiv(var_n_affine_dofs, var_18);
                var_20 = wp::sub(var_0, var_19);
                var_21 = wp::add(var_20, var_affine_verts_num);
                var_22 = wp::address(var_node2env, var_21);
                var_24 = wp::load(var_22);
                var_23 = wp::address(var_env_alpha, var_24);
                var_26 = wp::load(var_23);
                var_25 = wp::copy(var_26);
            }
            var_27 = wp::where(var_8, var_17, var_25);
        }
        var_28 = wp::where(var_5, var_2, var_27);
        // x[tid] += alpha * p[tid]                                                               <L 72>
        var_29 = wp::address(var_p, var_0);
        var_31 = wp::load(var_29);
        var_30 = wp::mul(var_28, var_31);
        var_32 = wp::atomic_add(var_x, var_0, var_30);
        // r[tid] -= alpha * Ap[tid]                                                              <L 73>
        var_33 = wp::address(var_Ap, var_0);
        var_35 = wp::load(var_33);
        var_34 = wp::mul(var_28, var_35);
        var_36 = wp::atomic_sub(var_r, var_0, var_34);
        // q[tid] = diag_inv[tid] @ r[tid]                                                        <L 74>
        var_37 = wp::address(var_diag_inv, var_0);
        var_38 = wp::address(var_r, var_0);
        var_40 = wp::load(var_37);
        var_41 = wp::load(var_38);
        var_39 = wp::mul(var_40, var_41);
        wp::array_store(var_q, var_0, var_39);
    }
}



extern "C" __global__ void cg_one_iter_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float64>> var_x,
    wp::array_t<wp::vec_t<3,wp::float64>> var_r,
    wp::array_t<wp::vec_t<3,wp::float64>> var_q,
    wp::array_t<wp::float64> var_env_alpha,
    wp::array_t<wp::vec_t<3,wp::float64>> var_p,
    wp::array_t<wp::vec_t<3,wp::float64>> var_Ap,
    wp::array_t<wp::mat_t<3,3,wp::float64>> var_diag_inv,
    wp::array_t<wp::int64> var_affine_dof_env_id,
    wp::array_t<wp::int32> var_node2env,
    wp::int32 var_n_affine_dofs,
    wp::int32 var_num_constraints,
    wp::int32 var_affine_verts_num,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_x,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_r,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_q,
    wp::array_t<wp::float64> adj_env_alpha,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_p,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_Ap,
    wp::array_t<wp::mat_t<3,3,wp::float64>> adj_diag_inv,
    wp::array_t<wp::int64> adj_affine_dof_env_id,
    wp::array_t<wp::int32> adj_node2env,
    wp::int32 adj_n_affine_dofs,
    wp::int32 adj_num_constraints,
    wp::int32 adj_affine_verts_num)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        // reset shared memory allocator
        wp::tile_alloc_shared(0, true);

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::float32 var_1 = 0.0;
        wp::float64 var_2;
        const wp::int32 var_3 = 3;
        wp::int32 var_4;
        bool var_5;
        const wp::int32 var_6 = 3;
        wp::int32 var_7;
        bool var_8;
        const wp::int32 var_9 = 3;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int64* var_12;
        wp::float64* var_13;
        wp::int64 var_14;
        wp::float64 var_15;
        wp::float64 var_16;
        wp::float64 var_17;
        const wp::int32 var_18 = 3;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::int32* var_22;
        wp::float64* var_23;
        wp::int32 var_24;
        wp::float64 var_25;
        wp::float64 var_26;
        wp::float64 var_27;
        wp::float64 var_28;
        wp::vec_t<3,wp::float64>* var_29;
        wp::vec_t<3,wp::float64> var_30;
        wp::vec_t<3,wp::float64> var_31;
        wp::vec_t<3,wp::float64> var_32;
        wp::vec_t<3,wp::float64>* var_33;
        wp::vec_t<3,wp::float64> var_34;
        wp::vec_t<3,wp::float64> var_35;
        wp::vec_t<3,wp::float64> var_36;
        wp::mat_t<3,3,wp::float64>* var_37;
        wp::vec_t<3,wp::float64>* var_38;
        wp::vec_t<3,wp::float64> var_39;
        wp::mat_t<3,3,wp::float64> var_40;
        wp::vec_t<3,wp::float64> var_41;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::float32 adj_1 = {};
        wp::float64 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        bool adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        bool adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int64 adj_12 = {};
        wp::float64 adj_13 = {};
        wp::int64 adj_14 = {};
        wp::float64 adj_15 = {};
        wp::float64 adj_16 = {};
        wp::float64 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::float64 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::float64 adj_25 = {};
        wp::float64 adj_26 = {};
        wp::float64 adj_27 = {};
        wp::float64 adj_28 = {};
        wp::vec_t<3,wp::float64> adj_29 = {};
        wp::vec_t<3,wp::float64> adj_30 = {};
        wp::vec_t<3,wp::float64> adj_31 = {};
        wp::vec_t<3,wp::float64> adj_32 = {};
        wp::vec_t<3,wp::float64> adj_33 = {};
        wp::vec_t<3,wp::float64> adj_34 = {};
        wp::vec_t<3,wp::float64> adj_35 = {};
        wp::vec_t<3,wp::float64> adj_36 = {};
        wp::mat_t<3,3,wp::float64> adj_37 = {};
        wp::vec_t<3,wp::float64> adj_38 = {};
        wp::vec_t<3,wp::float64> adj_39 = {};
        wp::mat_t<3,3,wp::float64> adj_40 = {};
        wp::vec_t<3,wp::float64> adj_41 = {};
        //---------
        // forward
        // def cg_one_iter(                                                                       <L 50>
        // tid = wp.tid()                                                                         <L 64>
        var_0 = builtin_tid1d();
        // alpha = wp.float64(0.0)                                                                <L 65>
        var_2 = wp::float64(var_1);
        // if tid < num_constraints // 3:                                                         <L 66>
        var_4 = wp::floordiv(var_num_constraints, var_3);
        var_5 = (var_0 < var_4);
        if (var_5) {
            // return                                                                             <L 67>
            goto label0;
        }
        if (!var_5) {
            // elif tid < n_affine_dofs // 3:                                                     <L 68>
            var_7 = wp::floordiv(var_n_affine_dofs, var_6);
            var_8 = (var_0 < var_7);
            if (var_8) {
                // alpha = env_alpha[affine_dof_env_id[tid - num_constraints // 3]]               <L 69>
                var_10 = wp::floordiv(var_num_constraints, var_9);
                var_11 = wp::sub(var_0, var_10);
                var_12 = wp::address(var_affine_dof_env_id, var_11);
                var_14 = wp::load(var_12);
                var_13 = wp::address(var_env_alpha, var_14);
                var_16 = wp::load(var_13);
                var_15 = wp::copy(var_16);
            }
            var_17 = wp::where(var_8, var_15, var_2);
            if (!var_8) {
                // alpha = env_alpha[node2env[tid - n_affine_dofs // 3 + affine_verts_num]]       <L 71>
                var_19 = wp::floordiv(var_n_affine_dofs, var_18);
                var_20 = wp::sub(var_0, var_19);
                var_21 = wp::add(var_20, var_affine_verts_num);
                var_22 = wp::address(var_node2env, var_21);
                var_24 = wp::load(var_22);
                var_23 = wp::address(var_env_alpha, var_24);
                var_26 = wp::load(var_23);
                var_25 = wp::copy(var_26);
            }
            var_27 = wp::where(var_8, var_17, var_25);
        }
        var_28 = wp::where(var_5, var_2, var_27);
        // x[tid] += alpha * p[tid]                                                               <L 72>
        var_29 = wp::address(var_p, var_0);
        var_31 = wp::load(var_29);
        var_30 = wp::mul(var_28, var_31);
        // var_32 = wp::atomic_add(var_x, var_0, var_30);
        // r[tid] -= alpha * Ap[tid]                                                              <L 73>
        var_33 = wp::address(var_Ap, var_0);
        var_35 = wp::load(var_33);
        var_34 = wp::mul(var_28, var_35);
        // var_36 = wp::atomic_sub(var_r, var_0, var_34);
        // q[tid] = diag_inv[tid] @ r[tid]                                                        <L 74>
        var_37 = wp::address(var_diag_inv, var_0);
        var_38 = wp::address(var_r, var_0);
        var_40 = wp::load(var_37);
        var_41 = wp::load(var_38);
        var_39 = wp::mul(var_40, var_41);
        // wp::array_store(var_q, var_0, var_39);
        //---------
        // reverse
        wp::adj_array_store(var_q, var_0, var_39, adj_q, adj_0, adj_39);
        wp::adj_mul(var_40, var_41, adj_37, adj_38, adj_39);
        wp::adj_load(var_38, adj_38, adj_41);
        wp::adj_load(var_37, adj_37, adj_40);
        wp::adj_address(var_r, var_0, adj_r, adj_0, adj_38);
        wp::adj_address(var_diag_inv, var_0, adj_diag_inv, adj_0, adj_37);
        // adj: q[tid] = diag_inv[tid] @ r[tid]                                                   <L 74>
        wp::adj_atomic_sub(var_r, var_0, var_34, adj_r, adj_0, adj_34, adj_36);
        wp::adj_mul(var_28, var_35, adj_28, adj_33, adj_34);
        wp::adj_load(var_33, adj_33, adj_35);
        wp::adj_address(var_Ap, var_0, adj_Ap, adj_0, adj_33);
        // adj: r[tid] -= alpha * Ap[tid]                                                         <L 73>
        wp::adj_atomic_add(var_x, var_0, var_30, adj_x, adj_0, adj_30, adj_32);
        wp::adj_mul(var_28, var_31, adj_28, adj_29, adj_30);
        wp::adj_load(var_29, adj_29, adj_31);
        wp::adj_address(var_p, var_0, adj_p, adj_0, adj_29);
        // adj: x[tid] += alpha * p[tid]                                                          <L 72>
        wp::adj_where(var_5, var_2, var_27, adj_5, adj_2, adj_27, adj_28);
        if (!var_5) {
            wp::adj_where(var_8, var_17, var_25, adj_8, adj_17, adj_25, adj_27);
            if (!var_8) {
                wp::adj_copy(var_26, adj_23, adj_25);
                wp::adj_load(var_23, adj_23, adj_26);
                wp::adj_address(var_env_alpha, var_24, adj_env_alpha, adj_22, adj_23);
                wp::adj_load(var_22, adj_22, adj_24);
                wp::adj_address(var_node2env, var_21, adj_node2env, adj_21, adj_22);
                wp::adj_add(var_20, var_affine_verts_num, adj_20, adj_affine_verts_num, adj_21);
                wp::adj_sub(var_0, var_19, adj_0, adj_19, adj_20);
                wp::adj_floordiv(var_n_affine_dofs, var_18, adj_n_affine_dofs, adj_18, adj_19);
                // adj: alpha = env_alpha[node2env[tid - n_affine_dofs // 3 + affine_verts_num]]  <L 71>
            }
            wp::adj_where(var_8, var_15, var_2, adj_8, adj_15, adj_2, adj_17);
            if (var_8) {
                wp::adj_copy(var_16, adj_13, adj_15);
                wp::adj_load(var_13, adj_13, adj_16);
                wp::adj_address(var_env_alpha, var_14, adj_env_alpha, adj_12, adj_13);
                wp::adj_load(var_12, adj_12, adj_14);
                wp::adj_address(var_affine_dof_env_id, var_11, adj_affine_dof_env_id, adj_11, adj_12);
                wp::adj_sub(var_0, var_10, adj_0, adj_10, adj_11);
                wp::adj_floordiv(var_num_constraints, var_9, adj_num_constraints, adj_9, adj_10);
                // adj: alpha = env_alpha[affine_dof_env_id[tid - num_constraints // 3]]          <L 69>
            }
            wp::adj_floordiv(var_n_affine_dofs, var_6, adj_n_affine_dofs, adj_6, adj_7);
            // adj: elif tid < n_affine_dofs // 3:                                                <L 68>
        }
        if (var_5) {
            label0:;
            // adj: return                                                                        <L 67>
        }
        wp::adj_floordiv(var_num_constraints, var_3, adj_num_constraints, adj_3, adj_4);
        // adj: if tid < num_constraints // 3:                                                    <L 66>
        wp::adj_float64(var_1, adj_1, adj_2);
        // adj: alpha = wp.float64(0.0)                                                           <L 65>
        // adj: tid = wp.tid()                                                                    <L 64>
        // adj: def cg_one_iter(                                                                  <L 50>
        continue;
    }
}

