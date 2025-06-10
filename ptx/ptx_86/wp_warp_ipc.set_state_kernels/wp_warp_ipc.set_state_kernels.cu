
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


extern "C" __global__ void set_sim_soft_state_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_soft_update_mask,
    wp::array_t<wp::vec_t<3,wp::float64>> var_soft_verts_positions,
    wp::array_t<wp::vec_t<3,wp::float64>> var_x,
    wp::array_t<wp::vec_t<3,wp::float64>> var_x_hat,
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
        wp::int32* var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        const wp::int32 var_4 = 0;
        bool var_5;
        wp::vec_t<3,wp::float64>* var_6;
        wp::vec_t<3,wp::float64> var_7;
        wp::vec_t<3,wp::float64> var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        //---------
        // forward
        // def set_sim_soft_state(                                                                <L 83>
        // tid = wp.tid()                                                                         <L 90>
        var_0 = builtin_tid1d();
        // mask = soft_update_mask[tid]                                                           <L 91>
        var_1 = wp::address(var_soft_update_mask, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // if mask == 0:                                                                          <L 92>
        var_5 = (var_2 == var_4);
        if (var_5) {
            // return                                                                             <L 93>
            continue;
        }
        // pos = soft_verts_positions[tid]                                                        <L 94>
        var_6 = wp::address(var_soft_verts_positions, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // x[tid + affine_verts_num] = pos                                                        <L 95>
        var_9 = wp::add(var_0, var_affine_verts_num);
        wp::array_store(var_x, var_9, var_7);
        // x_hat[tid + affine_verts_num] = pos                                                    <L 96>
        var_10 = wp::add(var_0, var_affine_verts_num);
        wp::array_store(var_x_hat, var_10, var_7);
    }
}



extern "C" __global__ void set_sim_soft_state_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_soft_update_mask,
    wp::array_t<wp::vec_t<3,wp::float64>> var_soft_verts_positions,
    wp::array_t<wp::vec_t<3,wp::float64>> var_x,
    wp::array_t<wp::vec_t<3,wp::float64>> var_x_hat,
    wp::int32 var_affine_verts_num,
    wp::array_t<wp::int32> adj_soft_update_mask,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_soft_verts_positions,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_x,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_x_hat,
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
        wp::int32* var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        const wp::int32 var_4 = 0;
        bool var_5;
        wp::vec_t<3,wp::float64>* var_6;
        wp::vec_t<3,wp::float64> var_7;
        wp::vec_t<3,wp::float64> var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        bool adj_5 = {};
        wp::vec_t<3,wp::float64> adj_6 = {};
        wp::vec_t<3,wp::float64> adj_7 = {};
        wp::vec_t<3,wp::float64> adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        //---------
        // forward
        // def set_sim_soft_state(                                                                <L 83>
        // tid = wp.tid()                                                                         <L 90>
        var_0 = builtin_tid1d();
        // mask = soft_update_mask[tid]                                                           <L 91>
        var_1 = wp::address(var_soft_update_mask, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // if mask == 0:                                                                          <L 92>
        var_5 = (var_2 == var_4);
        if (var_5) {
            // return                                                                             <L 93>
            goto label0;
        }
        // pos = soft_verts_positions[tid]                                                        <L 94>
        var_6 = wp::address(var_soft_verts_positions, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // x[tid + affine_verts_num] = pos                                                        <L 95>
        var_9 = wp::add(var_0, var_affine_verts_num);
        // wp::array_store(var_x, var_9, var_7);
        // x_hat[tid + affine_verts_num] = pos                                                    <L 96>
        var_10 = wp::add(var_0, var_affine_verts_num);
        // wp::array_store(var_x_hat, var_10, var_7);
        //---------
        // reverse
        wp::adj_array_store(var_x_hat, var_10, var_7, adj_x_hat, adj_10, adj_7);
        wp::adj_add(var_0, var_affine_verts_num, adj_0, adj_affine_verts_num, adj_10);
        // adj: x_hat[tid + affine_verts_num] = pos                                               <L 96>
        wp::adj_array_store(var_x, var_9, var_7, adj_x, adj_9, adj_7);
        wp::adj_add(var_0, var_affine_verts_num, adj_0, adj_affine_verts_num, adj_9);
        // adj: x[tid + affine_verts_num] = pos                                                   <L 95>
        wp::adj_copy(var_8, adj_6, adj_7);
        wp::adj_load(var_6, adj_6, adj_8);
        wp::adj_address(var_soft_verts_positions, var_0, adj_soft_verts_positions, adj_0, adj_6);
        // adj: pos = soft_verts_positions[tid]                                                   <L 94>
        if (var_5) {
            label0:;
            // adj: return                                                                        <L 93>
        }
        // adj: if mask == 0:                                                                     <L 92>
        wp::adj_copy(var_3, adj_1, adj_2);
        wp::adj_load(var_1, adj_1, adj_3);
        wp::adj_address(var_soft_update_mask, var_0, adj_soft_update_mask, adj_0, adj_1);
        // adj: mask = soft_update_mask[tid]                                                      <L 91>
        // adj: tid = wp.tid()                                                                    <L 90>
        // adj: def set_sim_soft_state(                                                           <L 83>
        continue;
    }
}



extern "C" __global__ void set_sim_affine_state_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_affine_update_mask,
    wp::array_t<wp::mat_t<3,3,wp::float64>> var_affine_rotations,
    wp::array_t<wp::vec_t<3,wp::float64>> var_affine_translations,
    wp::array_t<wp::vec_t<12,wp::float64>> var_y,
    wp::array_t<wp::vec_t<12,wp::float64>> var_y_hat,
    wp::array_t<wp::vec_t<3,wp::float64>> var_virtual_object_centers,
    wp::array_t<wp::vec_t<3,wp::float64>> var_ABD_centers)
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
        wp::int32* var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        const wp::int32 var_4 = 0;
        bool var_5;
        wp::vec_t<3,wp::float64>* var_6;
        wp::vec_t<3,wp::float64> var_7;
        wp::vec_t<3,wp::float64> var_8;
        wp::mat_t<3,3,wp::float64>* var_9;
        wp::mat_t<3,3,wp::float64> var_10;
        wp::mat_t<3,3,wp::float64> var_11;
        wp::vec_t<3,wp::float64>* var_12;
        wp::vec_t<3,wp::float64> var_13;
        wp::vec_t<3,wp::float64> var_14;
        wp::vec_t<3,wp::float64>* var_15;
        wp::vec_t<3,wp::float64> var_16;
        wp::vec_t<3,wp::float64> var_17;
        const wp::float64 var_18 = 1.0;
        const wp::float64 var_19 = 0.0;
        wp::vec_t<3,wp::float64> var_20;
        wp::vec_t<3,wp::float64> var_21;
        wp::vec_t<3,wp::float64> var_22;
        wp::vec_t<3,wp::float64> var_23;
        wp::vec_t<3,wp::float64> var_24;
        wp::vec_t<3,wp::float64> var_25;
        wp::vec_t<3,wp::float64> var_26;
        wp::vec_t<3,wp::float64> var_27;
        wp::vec_t<3,wp::float64> var_28;
        wp::vec_t<3,wp::float64> var_29;
        wp::vec_t<3,wp::float64> var_30;
        wp::vec_t<3,wp::float64> var_31;
        wp::vec_t<3,wp::float64> var_32;
        wp::vec_t<3,wp::float64> var_33;
        wp::vec_t<3,wp::float64> var_34;
        wp::vec_t<3,wp::float64> var_35;
        wp::vec_t<3,wp::float64> var_36;
        wp::vec_t<3,wp::float64> var_37;
        wp::vec_t<3,wp::float64> var_38;
        wp::vec_t<3,wp::float64> var_39;
        wp::vec_t<3,wp::float64> var_40;
        wp::vec_t<3,wp::float64> var_41;
        const wp::int32 var_42 = 0;
        wp::float64 var_43;
        const wp::int32 var_44 = 1;
        wp::float64 var_45;
        const wp::int32 var_46 = 2;
        wp::float64 var_47;
        const wp::int32 var_48 = 0;
        wp::float64 var_49;
        const wp::int32 var_50 = 1;
        wp::float64 var_51;
        const wp::int32 var_52 = 2;
        wp::float64 var_53;
        const wp::int32 var_54 = 0;
        wp::float64 var_55;
        const wp::int32 var_56 = 1;
        wp::float64 var_57;
        const wp::int32 var_58 = 2;
        wp::float64 var_59;
        const wp::int32 var_60 = 0;
        wp::float64 var_61;
        const wp::int32 var_62 = 1;
        wp::float64 var_63;
        const wp::int32 var_64 = 2;
        wp::float64 var_65;
        wp::vec_t<12,wp::float64> var_66;
        //---------
        // forward
        // def set_sim_affine_state(                                                              <L 8>
        // tid = wp.tid()                                                                         <L 17>
        var_0 = builtin_tid1d();
        // mask = affine_update_mask[tid]                                                         <L 18>
        var_1 = wp::address(var_affine_update_mask, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // if mask == 0:                                                                          <L 19>
        var_5 = (var_2 == var_4);
        if (var_5) {
            // return                                                                             <L 20>
            continue;
        }
        // virtual_center = virtual_object_centers[tid]                                           <L 21>
        var_6 = wp::address(var_virtual_object_centers, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // rot = affine_rotations[tid]                                                            <L 22>
        var_9 = wp::address(var_affine_rotations, var_0);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // trans = affine_translations[tid]                                                       <L 23>
        var_12 = wp::address(var_affine_translations, var_0);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // rest_O = ABD_centers[tid]                                                              <L 24>
        var_15 = wp::address(var_ABD_centers, var_0);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // rest_A = rest_O + wp.vec3d(_1, _0, _0)                                                 <L 25>
        var_20 = wp::vec_t<3,wp::float64>(var_18, var_19, var_19);
        var_21 = wp::add(var_16, var_20);
        // rest_B = rest_O + wp.vec3d(_0, _1, _0)                                                 <L 26>
        var_22 = wp::vec_t<3,wp::float64>(var_19, var_18, var_19);
        var_23 = wp::add(var_16, var_22);
        // rest_C = rest_O + wp.vec3d(_0, _0, _1)                                                 <L 27>
        var_24 = wp::vec_t<3,wp::float64>(var_19, var_19, var_18);
        var_25 = wp::add(var_16, var_24);
        // O = rot @ (rest_O - virtual_center) + trans + virtual_center                           <L 28>
        var_26 = wp::sub(var_16, var_7);
        var_27 = wp::mul(var_10, var_26);
        var_28 = wp::add(var_27, var_13);
        var_29 = wp::add(var_28, var_7);
        // A = rot @ (rest_A - virtual_center) + trans + virtual_center                           <L 29>
        var_30 = wp::sub(var_21, var_7);
        var_31 = wp::mul(var_10, var_30);
        var_32 = wp::add(var_31, var_13);
        var_33 = wp::add(var_32, var_7);
        // B = rot @ (rest_B - virtual_center) + trans + virtual_center                           <L 30>
        var_34 = wp::sub(var_23, var_7);
        var_35 = wp::mul(var_10, var_34);
        var_36 = wp::add(var_35, var_13);
        var_37 = wp::add(var_36, var_7);
        // C = rot @ (rest_C - virtual_center) + trans + virtual_center                           <L 31>
        var_38 = wp::sub(var_25, var_7);
        var_39 = wp::mul(var_10, var_38);
        var_40 = wp::add(var_39, var_13);
        var_41 = wp::add(var_40, var_7);
        // new_y = vec12d(O[0], O[1], O[2], A[0], A[1], A[2], B[0], B[1], B[2], C[0], C[1], C[2])       <L 32>
        var_43 = wp::extract(var_29, var_42);
        var_45 = wp::extract(var_29, var_44);
        var_47 = wp::extract(var_29, var_46);
        var_49 = wp::extract(var_33, var_48);
        var_51 = wp::extract(var_33, var_50);
        var_53 = wp::extract(var_33, var_52);
        var_55 = wp::extract(var_37, var_54);
        var_57 = wp::extract(var_37, var_56);
        var_59 = wp::extract(var_37, var_58);
        var_61 = wp::extract(var_41, var_60);
        var_63 = wp::extract(var_41, var_62);
        var_65 = wp::extract(var_41, var_64);
        var_66 = wp::vec_t<12,wp::float64>({var_43, var_45, var_47, var_49, var_51, var_53, var_55, var_57, var_59, var_61, var_63, var_65});
        // y[tid] = new_y                                                                         <L 33>
        wp::array_store(var_y, var_0, var_66);
        // y_hat[tid] = new_y                                                                     <L 34>
        wp::array_store(var_y_hat, var_0, var_66);
    }
}



extern "C" __global__ void set_sim_affine_state_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_affine_update_mask,
    wp::array_t<wp::mat_t<3,3,wp::float64>> var_affine_rotations,
    wp::array_t<wp::vec_t<3,wp::float64>> var_affine_translations,
    wp::array_t<wp::vec_t<12,wp::float64>> var_y,
    wp::array_t<wp::vec_t<12,wp::float64>> var_y_hat,
    wp::array_t<wp::vec_t<3,wp::float64>> var_virtual_object_centers,
    wp::array_t<wp::vec_t<3,wp::float64>> var_ABD_centers,
    wp::array_t<wp::int32> adj_affine_update_mask,
    wp::array_t<wp::mat_t<3,3,wp::float64>> adj_affine_rotations,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_affine_translations,
    wp::array_t<wp::vec_t<12,wp::float64>> adj_y,
    wp::array_t<wp::vec_t<12,wp::float64>> adj_y_hat,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_virtual_object_centers,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_ABD_centers)
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
        wp::int32* var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        const wp::int32 var_4 = 0;
        bool var_5;
        wp::vec_t<3,wp::float64>* var_6;
        wp::vec_t<3,wp::float64> var_7;
        wp::vec_t<3,wp::float64> var_8;
        wp::mat_t<3,3,wp::float64>* var_9;
        wp::mat_t<3,3,wp::float64> var_10;
        wp::mat_t<3,3,wp::float64> var_11;
        wp::vec_t<3,wp::float64>* var_12;
        wp::vec_t<3,wp::float64> var_13;
        wp::vec_t<3,wp::float64> var_14;
        wp::vec_t<3,wp::float64>* var_15;
        wp::vec_t<3,wp::float64> var_16;
        wp::vec_t<3,wp::float64> var_17;
        const wp::float64 var_18 = 1.0;
        const wp::float64 var_19 = 0.0;
        wp::vec_t<3,wp::float64> var_20;
        wp::vec_t<3,wp::float64> var_21;
        wp::vec_t<3,wp::float64> var_22;
        wp::vec_t<3,wp::float64> var_23;
        wp::vec_t<3,wp::float64> var_24;
        wp::vec_t<3,wp::float64> var_25;
        wp::vec_t<3,wp::float64> var_26;
        wp::vec_t<3,wp::float64> var_27;
        wp::vec_t<3,wp::float64> var_28;
        wp::vec_t<3,wp::float64> var_29;
        wp::vec_t<3,wp::float64> var_30;
        wp::vec_t<3,wp::float64> var_31;
        wp::vec_t<3,wp::float64> var_32;
        wp::vec_t<3,wp::float64> var_33;
        wp::vec_t<3,wp::float64> var_34;
        wp::vec_t<3,wp::float64> var_35;
        wp::vec_t<3,wp::float64> var_36;
        wp::vec_t<3,wp::float64> var_37;
        wp::vec_t<3,wp::float64> var_38;
        wp::vec_t<3,wp::float64> var_39;
        wp::vec_t<3,wp::float64> var_40;
        wp::vec_t<3,wp::float64> var_41;
        const wp::int32 var_42 = 0;
        wp::float64 var_43;
        const wp::int32 var_44 = 1;
        wp::float64 var_45;
        const wp::int32 var_46 = 2;
        wp::float64 var_47;
        const wp::int32 var_48 = 0;
        wp::float64 var_49;
        const wp::int32 var_50 = 1;
        wp::float64 var_51;
        const wp::int32 var_52 = 2;
        wp::float64 var_53;
        const wp::int32 var_54 = 0;
        wp::float64 var_55;
        const wp::int32 var_56 = 1;
        wp::float64 var_57;
        const wp::int32 var_58 = 2;
        wp::float64 var_59;
        const wp::int32 var_60 = 0;
        wp::float64 var_61;
        const wp::int32 var_62 = 1;
        wp::float64 var_63;
        const wp::int32 var_64 = 2;
        wp::float64 var_65;
        wp::vec_t<12,wp::float64> var_66;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        bool adj_5 = {};
        wp::vec_t<3,wp::float64> adj_6 = {};
        wp::vec_t<3,wp::float64> adj_7 = {};
        wp::vec_t<3,wp::float64> adj_8 = {};
        wp::mat_t<3,3,wp::float64> adj_9 = {};
        wp::mat_t<3,3,wp::float64> adj_10 = {};
        wp::mat_t<3,3,wp::float64> adj_11 = {};
        wp::vec_t<3,wp::float64> adj_12 = {};
        wp::vec_t<3,wp::float64> adj_13 = {};
        wp::vec_t<3,wp::float64> adj_14 = {};
        wp::vec_t<3,wp::float64> adj_15 = {};
        wp::vec_t<3,wp::float64> adj_16 = {};
        wp::vec_t<3,wp::float64> adj_17 = {};
        wp::float64 adj_18 = {};
        wp::float64 adj_19 = {};
        wp::vec_t<3,wp::float64> adj_20 = {};
        wp::vec_t<3,wp::float64> adj_21 = {};
        wp::vec_t<3,wp::float64> adj_22 = {};
        wp::vec_t<3,wp::float64> adj_23 = {};
        wp::vec_t<3,wp::float64> adj_24 = {};
        wp::vec_t<3,wp::float64> adj_25 = {};
        wp::vec_t<3,wp::float64> adj_26 = {};
        wp::vec_t<3,wp::float64> adj_27 = {};
        wp::vec_t<3,wp::float64> adj_28 = {};
        wp::vec_t<3,wp::float64> adj_29 = {};
        wp::vec_t<3,wp::float64> adj_30 = {};
        wp::vec_t<3,wp::float64> adj_31 = {};
        wp::vec_t<3,wp::float64> adj_32 = {};
        wp::vec_t<3,wp::float64> adj_33 = {};
        wp::vec_t<3,wp::float64> adj_34 = {};
        wp::vec_t<3,wp::float64> adj_35 = {};
        wp::vec_t<3,wp::float64> adj_36 = {};
        wp::vec_t<3,wp::float64> adj_37 = {};
        wp::vec_t<3,wp::float64> adj_38 = {};
        wp::vec_t<3,wp::float64> adj_39 = {};
        wp::vec_t<3,wp::float64> adj_40 = {};
        wp::vec_t<3,wp::float64> adj_41 = {};
        wp::int32 adj_42 = {};
        wp::float64 adj_43 = {};
        wp::int32 adj_44 = {};
        wp::float64 adj_45 = {};
        wp::int32 adj_46 = {};
        wp::float64 adj_47 = {};
        wp::int32 adj_48 = {};
        wp::float64 adj_49 = {};
        wp::int32 adj_50 = {};
        wp::float64 adj_51 = {};
        wp::int32 adj_52 = {};
        wp::float64 adj_53 = {};
        wp::int32 adj_54 = {};
        wp::float64 adj_55 = {};
        wp::int32 adj_56 = {};
        wp::float64 adj_57 = {};
        wp::int32 adj_58 = {};
        wp::float64 adj_59 = {};
        wp::int32 adj_60 = {};
        wp::float64 adj_61 = {};
        wp::int32 adj_62 = {};
        wp::float64 adj_63 = {};
        wp::int32 adj_64 = {};
        wp::float64 adj_65 = {};
        wp::vec_t<12,wp::float64> adj_66 = {};
        //---------
        // forward
        // def set_sim_affine_state(                                                              <L 8>
        // tid = wp.tid()                                                                         <L 17>
        var_0 = builtin_tid1d();
        // mask = affine_update_mask[tid]                                                         <L 18>
        var_1 = wp::address(var_affine_update_mask, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // if mask == 0:                                                                          <L 19>
        var_5 = (var_2 == var_4);
        if (var_5) {
            // return                                                                             <L 20>
            goto label0;
        }
        // virtual_center = virtual_object_centers[tid]                                           <L 21>
        var_6 = wp::address(var_virtual_object_centers, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // rot = affine_rotations[tid]                                                            <L 22>
        var_9 = wp::address(var_affine_rotations, var_0);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // trans = affine_translations[tid]                                                       <L 23>
        var_12 = wp::address(var_affine_translations, var_0);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // rest_O = ABD_centers[tid]                                                              <L 24>
        var_15 = wp::address(var_ABD_centers, var_0);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // rest_A = rest_O + wp.vec3d(_1, _0, _0)                                                 <L 25>
        var_20 = wp::vec_t<3,wp::float64>(var_18, var_19, var_19);
        var_21 = wp::add(var_16, var_20);
        // rest_B = rest_O + wp.vec3d(_0, _1, _0)                                                 <L 26>
        var_22 = wp::vec_t<3,wp::float64>(var_19, var_18, var_19);
        var_23 = wp::add(var_16, var_22);
        // rest_C = rest_O + wp.vec3d(_0, _0, _1)                                                 <L 27>
        var_24 = wp::vec_t<3,wp::float64>(var_19, var_19, var_18);
        var_25 = wp::add(var_16, var_24);
        // O = rot @ (rest_O - virtual_center) + trans + virtual_center                           <L 28>
        var_26 = wp::sub(var_16, var_7);
        var_27 = wp::mul(var_10, var_26);
        var_28 = wp::add(var_27, var_13);
        var_29 = wp::add(var_28, var_7);
        // A = rot @ (rest_A - virtual_center) + trans + virtual_center                           <L 29>
        var_30 = wp::sub(var_21, var_7);
        var_31 = wp::mul(var_10, var_30);
        var_32 = wp::add(var_31, var_13);
        var_33 = wp::add(var_32, var_7);
        // B = rot @ (rest_B - virtual_center) + trans + virtual_center                           <L 30>
        var_34 = wp::sub(var_23, var_7);
        var_35 = wp::mul(var_10, var_34);
        var_36 = wp::add(var_35, var_13);
        var_37 = wp::add(var_36, var_7);
        // C = rot @ (rest_C - virtual_center) + trans + virtual_center                           <L 31>
        var_38 = wp::sub(var_25, var_7);
        var_39 = wp::mul(var_10, var_38);
        var_40 = wp::add(var_39, var_13);
        var_41 = wp::add(var_40, var_7);
        // new_y = vec12d(O[0], O[1], O[2], A[0], A[1], A[2], B[0], B[1], B[2], C[0], C[1], C[2])       <L 32>
        var_43 = wp::extract(var_29, var_42);
        var_45 = wp::extract(var_29, var_44);
        var_47 = wp::extract(var_29, var_46);
        var_49 = wp::extract(var_33, var_48);
        var_51 = wp::extract(var_33, var_50);
        var_53 = wp::extract(var_33, var_52);
        var_55 = wp::extract(var_37, var_54);
        var_57 = wp::extract(var_37, var_56);
        var_59 = wp::extract(var_37, var_58);
        var_61 = wp::extract(var_41, var_60);
        var_63 = wp::extract(var_41, var_62);
        var_65 = wp::extract(var_41, var_64);
        var_66 = wp::vec_t<12,wp::float64>({var_43, var_45, var_47, var_49, var_51, var_53, var_55, var_57, var_59, var_61, var_63, var_65});
        // y[tid] = new_y                                                                         <L 33>
        // wp::array_store(var_y, var_0, var_66);
        // y_hat[tid] = new_y                                                                     <L 34>
        // wp::array_store(var_y_hat, var_0, var_66);
        //---------
        // reverse
        wp::adj_array_store(var_y_hat, var_0, var_66, adj_y_hat, adj_0, adj_66);
        // adj: y_hat[tid] = new_y                                                                <L 34>
        wp::adj_array_store(var_y, var_0, var_66, adj_y, adj_0, adj_66);
        // adj: y[tid] = new_y                                                                    <L 33>
        wp::adj_vec_t({var_43, var_45, var_47, var_49, var_51, var_53, var_55, var_57, var_59, var_61, var_63, var_65}, {&adj_43, &adj_45, &adj_47, &adj_49, &adj_51, &adj_53, &adj_55, &adj_57, &adj_59, &adj_61, &adj_63, &adj_65}, adj_66);
        wp::adj_extract(var_41, var_64, adj_41, adj_64, adj_65);
        wp::adj_extract(var_41, var_62, adj_41, adj_62, adj_63);
        wp::adj_extract(var_41, var_60, adj_41, adj_60, adj_61);
        wp::adj_extract(var_37, var_58, adj_37, adj_58, adj_59);
        wp::adj_extract(var_37, var_56, adj_37, adj_56, adj_57);
        wp::adj_extract(var_37, var_54, adj_37, adj_54, adj_55);
        wp::adj_extract(var_33, var_52, adj_33, adj_52, adj_53);
        wp::adj_extract(var_33, var_50, adj_33, adj_50, adj_51);
        wp::adj_extract(var_33, var_48, adj_33, adj_48, adj_49);
        wp::adj_extract(var_29, var_46, adj_29, adj_46, adj_47);
        wp::adj_extract(var_29, var_44, adj_29, adj_44, adj_45);
        wp::adj_extract(var_29, var_42, adj_29, adj_42, adj_43);
        // adj: new_y = vec12d(O[0], O[1], O[2], A[0], A[1], A[2], B[0], B[1], B[2], C[0], C[1], C[2])  <L 32>
        wp::adj_add(var_40, var_7, adj_40, adj_7, adj_41);
        wp::adj_add(var_39, var_13, adj_39, adj_13, adj_40);
        wp::adj_mul(var_10, var_38, adj_10, adj_38, adj_39);
        wp::adj_sub(var_25, var_7, adj_25, adj_7, adj_38);
        // adj: C = rot @ (rest_C - virtual_center) + trans + virtual_center                      <L 31>
        wp::adj_add(var_36, var_7, adj_36, adj_7, adj_37);
        wp::adj_add(var_35, var_13, adj_35, adj_13, adj_36);
        wp::adj_mul(var_10, var_34, adj_10, adj_34, adj_35);
        wp::adj_sub(var_23, var_7, adj_23, adj_7, adj_34);
        // adj: B = rot @ (rest_B - virtual_center) + trans + virtual_center                      <L 30>
        wp::adj_add(var_32, var_7, adj_32, adj_7, adj_33);
        wp::adj_add(var_31, var_13, adj_31, adj_13, adj_32);
        wp::adj_mul(var_10, var_30, adj_10, adj_30, adj_31);
        wp::adj_sub(var_21, var_7, adj_21, adj_7, adj_30);
        // adj: A = rot @ (rest_A - virtual_center) + trans + virtual_center                      <L 29>
        wp::adj_add(var_28, var_7, adj_28, adj_7, adj_29);
        wp::adj_add(var_27, var_13, adj_27, adj_13, adj_28);
        wp::adj_mul(var_10, var_26, adj_10, adj_26, adj_27);
        wp::adj_sub(var_16, var_7, adj_16, adj_7, adj_26);
        // adj: O = rot @ (rest_O - virtual_center) + trans + virtual_center                      <L 28>
        wp::adj_add(var_16, var_24, adj_16, adj_24, adj_25);
        wp::adj_vec_t(var_19, var_19, var_18, adj_19, adj_19, adj_18, adj_24);
        // adj: rest_C = rest_O + wp.vec3d(_0, _0, _1)                                            <L 27>
        wp::adj_add(var_16, var_22, adj_16, adj_22, adj_23);
        wp::adj_vec_t(var_19, var_18, var_19, adj_19, adj_18, adj_19, adj_22);
        // adj: rest_B = rest_O + wp.vec3d(_0, _1, _0)                                            <L 26>
        wp::adj_add(var_16, var_20, adj_16, adj_20, adj_21);
        wp::adj_vec_t(var_18, var_19, var_19, adj_18, adj_19, adj_19, adj_20);
        // adj: rest_A = rest_O + wp.vec3d(_1, _0, _0)                                            <L 25>
        wp::adj_copy(var_17, adj_15, adj_16);
        wp::adj_load(var_15, adj_15, adj_17);
        wp::adj_address(var_ABD_centers, var_0, adj_ABD_centers, adj_0, adj_15);
        // adj: rest_O = ABD_centers[tid]                                                         <L 24>
        wp::adj_copy(var_14, adj_12, adj_13);
        wp::adj_load(var_12, adj_12, adj_14);
        wp::adj_address(var_affine_translations, var_0, adj_affine_translations, adj_0, adj_12);
        // adj: trans = affine_translations[tid]                                                  <L 23>
        wp::adj_copy(var_11, adj_9, adj_10);
        wp::adj_load(var_9, adj_9, adj_11);
        wp::adj_address(var_affine_rotations, var_0, adj_affine_rotations, adj_0, adj_9);
        // adj: rot = affine_rotations[tid]                                                       <L 22>
        wp::adj_copy(var_8, adj_6, adj_7);
        wp::adj_load(var_6, adj_6, adj_8);
        wp::adj_address(var_virtual_object_centers, var_0, adj_virtual_object_centers, adj_0, adj_6);
        // adj: virtual_center = virtual_object_centers[tid]                                      <L 21>
        if (var_5) {
            label0:;
            // adj: return                                                                        <L 20>
        }
        // adj: if mask == 0:                                                                     <L 19>
        wp::adj_copy(var_3, adj_1, adj_2);
        wp::adj_load(var_1, adj_1, adj_3);
        wp::adj_address(var_affine_update_mask, var_0, adj_affine_update_mask, adj_0, adj_1);
        // adj: mask = affine_update_mask[tid]                                                    <L 18>
        // adj: tid = wp.tid()                                                                    <L 17>
        // adj: def set_sim_affine_state(                                                         <L 8>
        continue;
    }
}



extern "C" __global__ void set_sim_soft_vel_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_soft_update_mask,
    wp::array_t<wp::vec_t<3,wp::float64>> var_soft_verts_vel,
    wp::array_t<wp::vec_t<3,wp::float64>> var_v_x,
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
        wp::int32* var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        const wp::int32 var_4 = 0;
        bool var_5;
        wp::vec_t<3,wp::float64>* var_6;
        wp::vec_t<3,wp::float64> var_7;
        wp::vec_t<3,wp::float64> var_8;
        wp::int32 var_9;
        //---------
        // forward
        // def set_sim_soft_vel(                                                                  <L 100>
        // tid = wp.tid()                                                                         <L 106>
        var_0 = builtin_tid1d();
        // mask = soft_update_mask[tid]                                                           <L 107>
        var_1 = wp::address(var_soft_update_mask, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // if mask == 0:                                                                          <L 108>
        var_5 = (var_2 == var_4);
        if (var_5) {
            // return                                                                             <L 109>
            continue;
        }
        // vel = soft_verts_vel[tid]                                                              <L 110>
        var_6 = wp::address(var_soft_verts_vel, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // v_x[tid + affine_verts_num] = vel                                                      <L 111>
        var_9 = wp::add(var_0, var_affine_verts_num);
        wp::array_store(var_v_x, var_9, var_7);
    }
}



extern "C" __global__ void set_sim_soft_vel_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_soft_update_mask,
    wp::array_t<wp::vec_t<3,wp::float64>> var_soft_verts_vel,
    wp::array_t<wp::vec_t<3,wp::float64>> var_v_x,
    wp::int32 var_affine_verts_num,
    wp::array_t<wp::int32> adj_soft_update_mask,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_soft_verts_vel,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_v_x,
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
        wp::int32* var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        const wp::int32 var_4 = 0;
        bool var_5;
        wp::vec_t<3,wp::float64>* var_6;
        wp::vec_t<3,wp::float64> var_7;
        wp::vec_t<3,wp::float64> var_8;
        wp::int32 var_9;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        bool adj_5 = {};
        wp::vec_t<3,wp::float64> adj_6 = {};
        wp::vec_t<3,wp::float64> adj_7 = {};
        wp::vec_t<3,wp::float64> adj_8 = {};
        wp::int32 adj_9 = {};
        //---------
        // forward
        // def set_sim_soft_vel(                                                                  <L 100>
        // tid = wp.tid()                                                                         <L 106>
        var_0 = builtin_tid1d();
        // mask = soft_update_mask[tid]                                                           <L 107>
        var_1 = wp::address(var_soft_update_mask, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // if mask == 0:                                                                          <L 108>
        var_5 = (var_2 == var_4);
        if (var_5) {
            // return                                                                             <L 109>
            goto label0;
        }
        // vel = soft_verts_vel[tid]                                                              <L 110>
        var_6 = wp::address(var_soft_verts_vel, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // v_x[tid + affine_verts_num] = vel                                                      <L 111>
        var_9 = wp::add(var_0, var_affine_verts_num);
        // wp::array_store(var_v_x, var_9, var_7);
        //---------
        // reverse
        wp::adj_array_store(var_v_x, var_9, var_7, adj_v_x, adj_9, adj_7);
        wp::adj_add(var_0, var_affine_verts_num, adj_0, adj_affine_verts_num, adj_9);
        // adj: v_x[tid + affine_verts_num] = vel                                                 <L 111>
        wp::adj_copy(var_8, adj_6, adj_7);
        wp::adj_load(var_6, adj_6, adj_8);
        wp::adj_address(var_soft_verts_vel, var_0, adj_soft_verts_vel, adj_0, adj_6);
        // adj: vel = soft_verts_vel[tid]                                                         <L 110>
        if (var_5) {
            label0:;
            // adj: return                                                                        <L 109>
        }
        // adj: if mask == 0:                                                                     <L 108>
        wp::adj_copy(var_3, adj_1, adj_2);
        wp::adj_load(var_1, adj_1, adj_3);
        wp::adj_address(var_soft_update_mask, var_0, adj_soft_update_mask, adj_0, adj_1);
        // adj: mask = soft_update_mask[tid]                                                      <L 107>
        // adj: tid = wp.tid()                                                                    <L 106>
        // adj: def set_sim_soft_vel(                                                             <L 100>
        continue;
    }
}



extern "C" __global__ void set_sim_affine_vel_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_affine_update_mask,
    wp::array_t<wp::vec_t<3,wp::float64>> var_affine_linear_vel,
    wp::array_t<wp::vec_t<3,wp::float64>> var_affine_angular_vel,
    wp::array_t<wp::vec_t<12,wp::float64>> var_y,
    wp::array_t<wp::vec_t<12,wp::float64>> var_v_y,
    wp::array_t<wp::vec_t<3,wp::float64>> var_virtual_object_centers,
    wp::array_t<wp::vec_t<3,wp::float64>> var_ABD_centers)
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
        wp::int32* var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        const wp::int32 var_4 = 0;
        bool var_5;
        wp::vec_t<3,wp::float64>* var_6;
        wp::vec_t<3,wp::float64> var_7;
        wp::vec_t<3,wp::float64> var_8;
        wp::vec_t<3,wp::float64>* var_9;
        wp::vec_t<3,wp::float64> var_10;
        wp::vec_t<3,wp::float64> var_11;
        wp::vec_t<3,wp::float64>* var_12;
        wp::vec_t<3,wp::float64> var_13;
        wp::vec_t<3,wp::float64> var_14;
        wp::vec_t<3,wp::float64>* var_15;
        wp::vec_t<3,wp::float64> var_16;
        wp::vec_t<3,wp::float64> var_17;
        wp::vec_t<12,wp::float64>* var_18;
        wp::vec_t<12,wp::float64> var_19;
        wp::vec_t<12,wp::float64> var_20;
        wp::vec_t<3,wp::float64> var_21;
        const wp::int32 var_22 = 0;
        wp::float64 var_23;
        const wp::int32 var_24 = 1;
        wp::float64 var_25;
        const wp::int32 var_26 = 2;
        wp::float64 var_27;
        wp::vec_t<3,wp::float64> var_28;
        const wp::int32 var_29 = 3;
        wp::float64 var_30;
        const wp::int32 var_31 = 4;
        wp::float64 var_32;
        const wp::int32 var_33 = 5;
        wp::float64 var_34;
        wp::vec_t<3,wp::float64> var_35;
        const wp::int32 var_36 = 6;
        wp::float64 var_37;
        const wp::int32 var_38 = 7;
        wp::float64 var_39;
        const wp::int32 var_40 = 8;
        wp::float64 var_41;
        wp::vec_t<3,wp::float64> var_42;
        const wp::int32 var_43 = 9;
        wp::float64 var_44;
        const wp::int32 var_45 = 10;
        wp::float64 var_46;
        const wp::int32 var_47 = 11;
        wp::float64 var_48;
        wp::vec_t<3,wp::float64> var_49;
        const wp::float64 var_50 = 1.0;
        const wp::int32 var_51 = 0;
        wp::float64 var_52;
        wp::float64 var_53;
        const wp::int32 var_54 = 1;
        wp::float64 var_55;
        wp::float64 var_56;
        const wp::int32 var_57 = 2;
        wp::float64 var_58;
        wp::float64 var_59;
        wp::vec_t<3,wp::float64> var_60;
        const wp::int32 var_61 = 0;
        wp::float64 var_62;
        wp::vec_t<3,wp::float64> var_63;
        wp::vec_t<3,wp::float64> var_64;
        const wp::int32 var_65 = 1;
        wp::float64 var_66;
        wp::vec_t<3,wp::float64> var_67;
        wp::vec_t<3,wp::float64> var_68;
        const wp::int32 var_69 = 2;
        wp::float64 var_70;
        wp::vec_t<3,wp::float64> var_71;
        wp::vec_t<3,wp::float64> var_72;
        wp::vec_t<3,wp::float64> var_73;
        wp::vec_t<3,wp::float64> var_74;
        wp::vec_t<3,wp::float64> var_75;
        wp::vec_t<3,wp::float64> var_76;
        wp::vec_t<3,wp::float64> var_77;
        wp::vec_t<3,wp::float64> var_78;
        wp::vec_t<3,wp::float64> var_79;
        wp::vec_t<3,wp::float64> var_80;
        wp::vec_t<3,wp::float64> var_81;
        wp::vec_t<3,wp::float64> var_82;
        wp::vec_t<3,wp::float64> var_83;
        wp::vec_t<3,wp::float64> var_84;
        const wp::int32 var_85 = 0;
        wp::float64 var_86;
        const wp::int32 var_87 = 1;
        wp::float64 var_88;
        const wp::int32 var_89 = 2;
        wp::float64 var_90;
        const wp::int32 var_91 = 0;
        wp::float64 var_92;
        const wp::int32 var_93 = 1;
        wp::float64 var_94;
        const wp::int32 var_95 = 2;
        wp::float64 var_96;
        const wp::int32 var_97 = 0;
        wp::float64 var_98;
        const wp::int32 var_99 = 1;
        wp::float64 var_100;
        const wp::int32 var_101 = 2;
        wp::float64 var_102;
        const wp::int32 var_103 = 0;
        wp::float64 var_104;
        const wp::int32 var_105 = 1;
        wp::float64 var_106;
        const wp::int32 var_107 = 2;
        wp::float64 var_108;
        wp::vec_t<12,wp::float64> var_109;
        //---------
        // forward
        // def set_sim_affine_vel(                                                                <L 38>
        // tid = wp.tid()                                                                         <L 47>
        var_0 = builtin_tid1d();
        // mask = affine_update_mask[tid]                                                         <L 48>
        var_1 = wp::address(var_affine_update_mask, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // if mask == 0:                                                                          <L 49>
        var_5 = (var_2 == var_4);
        if (var_5) {
            // return                                                                             <L 50>
            continue;
        }
        // linear_vel = affine_linear_vel[tid]                                                    <L 51>
        var_6 = wp::address(var_affine_linear_vel, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // angular_vel = affine_angular_vel[tid]                                                  <L 52>
        var_9 = wp::address(var_affine_angular_vel, var_0);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // rest_center = virtual_object_centers[tid]                                              <L 53>
        var_12 = wp::address(var_virtual_object_centers, var_0);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // rest_O = ABD_centers[tid]                                                              <L 54>
        var_15 = wp::address(var_ABD_centers, var_0);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // current_y = y[tid]                                                                     <L 55>
        var_18 = wp::address(var_y, var_0);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // w = rest_center - rest_O                                                               <L 56>
        var_21 = wp::sub(var_13, var_16);
        // O = wp.vec3d(current_y[0], current_y[1], current_y[2])                                 <L 57>
        var_23 = wp::extract(var_19, var_22);
        var_25 = wp::extract(var_19, var_24);
        var_27 = wp::extract(var_19, var_26);
        var_28 = wp::vec_t<3,wp::float64>(var_23, var_25, var_27);
        // A = wp.vec3d(current_y[3], current_y[4], current_y[5])                                 <L 58>
        var_30 = wp::extract(var_19, var_29);
        var_32 = wp::extract(var_19, var_31);
        var_34 = wp::extract(var_19, var_33);
        var_35 = wp::vec_t<3,wp::float64>(var_30, var_32, var_34);
        // B = wp.vec3d(current_y[6], current_y[7], current_y[8])                                 <L 59>
        var_37 = wp::extract(var_19, var_36);
        var_39 = wp::extract(var_19, var_38);
        var_41 = wp::extract(var_19, var_40);
        var_42 = wp::vec_t<3,wp::float64>(var_37, var_39, var_41);
        // C = wp.vec3d(current_y[9], current_y[10], current_y[11])                               <L 60>
        var_44 = wp::extract(var_19, var_43);
        var_46 = wp::extract(var_19, var_45);
        var_48 = wp::extract(var_19, var_47);
        var_49 = wp::vec_t<3,wp::float64>(var_44, var_46, var_48);
        // center = (_1 - w[0] - w[1] - w[2]) * O + w[0] * A + w[1] * B + w[2] * C                <L 61>
        var_52 = wp::extract(var_21, var_51);
        var_53 = wp::sub(var_50, var_52);
        var_55 = wp::extract(var_21, var_54);
        var_56 = wp::sub(var_53, var_55);
        var_58 = wp::extract(var_21, var_57);
        var_59 = wp::sub(var_56, var_58);
        var_60 = wp::mul(var_59, var_28);
        var_62 = wp::extract(var_21, var_61);
        var_63 = wp::mul(var_62, var_35);
        var_64 = wp::add(var_60, var_63);
        var_66 = wp::extract(var_21, var_65);
        var_67 = wp::mul(var_66, var_42);
        var_68 = wp::add(var_64, var_67);
        var_70 = wp::extract(var_21, var_69);
        var_71 = wp::mul(var_70, var_49);
        var_72 = wp::add(var_68, var_71);
        // vel_O = wp.cross(angular_vel, O - center) + linear_vel                                 <L 62>
        var_73 = wp::sub(var_28, var_72);
        var_74 = wp::cross(var_10, var_73);
        var_75 = wp::add(var_74, var_7);
        // vel_A = wp.cross(angular_vel, A - center) + linear_vel                                 <L 63>
        var_76 = wp::sub(var_35, var_72);
        var_77 = wp::cross(var_10, var_76);
        var_78 = wp::add(var_77, var_7);
        // vel_B = wp.cross(angular_vel, B - center) + linear_vel                                 <L 64>
        var_79 = wp::sub(var_42, var_72);
        var_80 = wp::cross(var_10, var_79);
        var_81 = wp::add(var_80, var_7);
        // vel_C = wp.cross(angular_vel, C - center) + linear_vel                                 <L 65>
        var_82 = wp::sub(var_49, var_72);
        var_83 = wp::cross(var_10, var_82);
        var_84 = wp::add(var_83, var_7);
        // v_y[tid] = vec12d(                                                                     <L 66>
        // vel_O[0],                                                                              <L 67>
        var_86 = wp::extract(var_75, var_85);
        // vel_O[1],                                                                              <L 68>
        var_88 = wp::extract(var_75, var_87);
        // vel_O[2],                                                                              <L 69>
        var_90 = wp::extract(var_75, var_89);
        // vel_A[0],                                                                              <L 70>
        var_92 = wp::extract(var_78, var_91);
        // vel_A[1],                                                                              <L 71>
        var_94 = wp::extract(var_78, var_93);
        // vel_A[2],                                                                              <L 72>
        var_96 = wp::extract(var_78, var_95);
        // vel_B[0],                                                                              <L 73>
        var_98 = wp::extract(var_81, var_97);
        // vel_B[1],                                                                              <L 74>
        var_100 = wp::extract(var_81, var_99);
        // vel_B[2],                                                                              <L 75>
        var_102 = wp::extract(var_81, var_101);
        // vel_C[0],                                                                              <L 76>
        var_104 = wp::extract(var_84, var_103);
        // vel_C[1],                                                                              <L 77>
        var_106 = wp::extract(var_84, var_105);
        // vel_C[2],                                                                              <L 78>
        var_108 = wp::extract(var_84, var_107);
        var_109 = wp::vec_t<12,wp::float64>({var_86, var_88, var_90, var_92, var_94, var_96, var_98, var_100, var_102, var_104, var_106, var_108});
        // v_y[tid] = vec12d(                                                                     <L 66>
        wp::array_store(var_v_y, var_0, var_109);
    }
}



extern "C" __global__ void set_sim_affine_vel_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_affine_update_mask,
    wp::array_t<wp::vec_t<3,wp::float64>> var_affine_linear_vel,
    wp::array_t<wp::vec_t<3,wp::float64>> var_affine_angular_vel,
    wp::array_t<wp::vec_t<12,wp::float64>> var_y,
    wp::array_t<wp::vec_t<12,wp::float64>> var_v_y,
    wp::array_t<wp::vec_t<3,wp::float64>> var_virtual_object_centers,
    wp::array_t<wp::vec_t<3,wp::float64>> var_ABD_centers,
    wp::array_t<wp::int32> adj_affine_update_mask,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_affine_linear_vel,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_affine_angular_vel,
    wp::array_t<wp::vec_t<12,wp::float64>> adj_y,
    wp::array_t<wp::vec_t<12,wp::float64>> adj_v_y,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_virtual_object_centers,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_ABD_centers)
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
        wp::int32* var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        const wp::int32 var_4 = 0;
        bool var_5;
        wp::vec_t<3,wp::float64>* var_6;
        wp::vec_t<3,wp::float64> var_7;
        wp::vec_t<3,wp::float64> var_8;
        wp::vec_t<3,wp::float64>* var_9;
        wp::vec_t<3,wp::float64> var_10;
        wp::vec_t<3,wp::float64> var_11;
        wp::vec_t<3,wp::float64>* var_12;
        wp::vec_t<3,wp::float64> var_13;
        wp::vec_t<3,wp::float64> var_14;
        wp::vec_t<3,wp::float64>* var_15;
        wp::vec_t<3,wp::float64> var_16;
        wp::vec_t<3,wp::float64> var_17;
        wp::vec_t<12,wp::float64>* var_18;
        wp::vec_t<12,wp::float64> var_19;
        wp::vec_t<12,wp::float64> var_20;
        wp::vec_t<3,wp::float64> var_21;
        const wp::int32 var_22 = 0;
        wp::float64 var_23;
        const wp::int32 var_24 = 1;
        wp::float64 var_25;
        const wp::int32 var_26 = 2;
        wp::float64 var_27;
        wp::vec_t<3,wp::float64> var_28;
        const wp::int32 var_29 = 3;
        wp::float64 var_30;
        const wp::int32 var_31 = 4;
        wp::float64 var_32;
        const wp::int32 var_33 = 5;
        wp::float64 var_34;
        wp::vec_t<3,wp::float64> var_35;
        const wp::int32 var_36 = 6;
        wp::float64 var_37;
        const wp::int32 var_38 = 7;
        wp::float64 var_39;
        const wp::int32 var_40 = 8;
        wp::float64 var_41;
        wp::vec_t<3,wp::float64> var_42;
        const wp::int32 var_43 = 9;
        wp::float64 var_44;
        const wp::int32 var_45 = 10;
        wp::float64 var_46;
        const wp::int32 var_47 = 11;
        wp::float64 var_48;
        wp::vec_t<3,wp::float64> var_49;
        const wp::float64 var_50 = 1.0;
        const wp::int32 var_51 = 0;
        wp::float64 var_52;
        wp::float64 var_53;
        const wp::int32 var_54 = 1;
        wp::float64 var_55;
        wp::float64 var_56;
        const wp::int32 var_57 = 2;
        wp::float64 var_58;
        wp::float64 var_59;
        wp::vec_t<3,wp::float64> var_60;
        const wp::int32 var_61 = 0;
        wp::float64 var_62;
        wp::vec_t<3,wp::float64> var_63;
        wp::vec_t<3,wp::float64> var_64;
        const wp::int32 var_65 = 1;
        wp::float64 var_66;
        wp::vec_t<3,wp::float64> var_67;
        wp::vec_t<3,wp::float64> var_68;
        const wp::int32 var_69 = 2;
        wp::float64 var_70;
        wp::vec_t<3,wp::float64> var_71;
        wp::vec_t<3,wp::float64> var_72;
        wp::vec_t<3,wp::float64> var_73;
        wp::vec_t<3,wp::float64> var_74;
        wp::vec_t<3,wp::float64> var_75;
        wp::vec_t<3,wp::float64> var_76;
        wp::vec_t<3,wp::float64> var_77;
        wp::vec_t<3,wp::float64> var_78;
        wp::vec_t<3,wp::float64> var_79;
        wp::vec_t<3,wp::float64> var_80;
        wp::vec_t<3,wp::float64> var_81;
        wp::vec_t<3,wp::float64> var_82;
        wp::vec_t<3,wp::float64> var_83;
        wp::vec_t<3,wp::float64> var_84;
        const wp::int32 var_85 = 0;
        wp::float64 var_86;
        const wp::int32 var_87 = 1;
        wp::float64 var_88;
        const wp::int32 var_89 = 2;
        wp::float64 var_90;
        const wp::int32 var_91 = 0;
        wp::float64 var_92;
        const wp::int32 var_93 = 1;
        wp::float64 var_94;
        const wp::int32 var_95 = 2;
        wp::float64 var_96;
        const wp::int32 var_97 = 0;
        wp::float64 var_98;
        const wp::int32 var_99 = 1;
        wp::float64 var_100;
        const wp::int32 var_101 = 2;
        wp::float64 var_102;
        const wp::int32 var_103 = 0;
        wp::float64 var_104;
        const wp::int32 var_105 = 1;
        wp::float64 var_106;
        const wp::int32 var_107 = 2;
        wp::float64 var_108;
        wp::vec_t<12,wp::float64> var_109;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        bool adj_5 = {};
        wp::vec_t<3,wp::float64> adj_6 = {};
        wp::vec_t<3,wp::float64> adj_7 = {};
        wp::vec_t<3,wp::float64> adj_8 = {};
        wp::vec_t<3,wp::float64> adj_9 = {};
        wp::vec_t<3,wp::float64> adj_10 = {};
        wp::vec_t<3,wp::float64> adj_11 = {};
        wp::vec_t<3,wp::float64> adj_12 = {};
        wp::vec_t<3,wp::float64> adj_13 = {};
        wp::vec_t<3,wp::float64> adj_14 = {};
        wp::vec_t<3,wp::float64> adj_15 = {};
        wp::vec_t<3,wp::float64> adj_16 = {};
        wp::vec_t<3,wp::float64> adj_17 = {};
        wp::vec_t<12,wp::float64> adj_18 = {};
        wp::vec_t<12,wp::float64> adj_19 = {};
        wp::vec_t<12,wp::float64> adj_20 = {};
        wp::vec_t<3,wp::float64> adj_21 = {};
        wp::int32 adj_22 = {};
        wp::float64 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::float64 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::float64 adj_27 = {};
        wp::vec_t<3,wp::float64> adj_28 = {};
        wp::int32 adj_29 = {};
        wp::float64 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::float64 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::float64 adj_34 = {};
        wp::vec_t<3,wp::float64> adj_35 = {};
        wp::int32 adj_36 = {};
        wp::float64 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::float64 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::float64 adj_41 = {};
        wp::vec_t<3,wp::float64> adj_42 = {};
        wp::int32 adj_43 = {};
        wp::float64 adj_44 = {};
        wp::int32 adj_45 = {};
        wp::float64 adj_46 = {};
        wp::int32 adj_47 = {};
        wp::float64 adj_48 = {};
        wp::vec_t<3,wp::float64> adj_49 = {};
        wp::float64 adj_50 = {};
        wp::int32 adj_51 = {};
        wp::float64 adj_52 = {};
        wp::float64 adj_53 = {};
        wp::int32 adj_54 = {};
        wp::float64 adj_55 = {};
        wp::float64 adj_56 = {};
        wp::int32 adj_57 = {};
        wp::float64 adj_58 = {};
        wp::float64 adj_59 = {};
        wp::vec_t<3,wp::float64> adj_60 = {};
        wp::int32 adj_61 = {};
        wp::float64 adj_62 = {};
        wp::vec_t<3,wp::float64> adj_63 = {};
        wp::vec_t<3,wp::float64> adj_64 = {};
        wp::int32 adj_65 = {};
        wp::float64 adj_66 = {};
        wp::vec_t<3,wp::float64> adj_67 = {};
        wp::vec_t<3,wp::float64> adj_68 = {};
        wp::int32 adj_69 = {};
        wp::float64 adj_70 = {};
        wp::vec_t<3,wp::float64> adj_71 = {};
        wp::vec_t<3,wp::float64> adj_72 = {};
        wp::vec_t<3,wp::float64> adj_73 = {};
        wp::vec_t<3,wp::float64> adj_74 = {};
        wp::vec_t<3,wp::float64> adj_75 = {};
        wp::vec_t<3,wp::float64> adj_76 = {};
        wp::vec_t<3,wp::float64> adj_77 = {};
        wp::vec_t<3,wp::float64> adj_78 = {};
        wp::vec_t<3,wp::float64> adj_79 = {};
        wp::vec_t<3,wp::float64> adj_80 = {};
        wp::vec_t<3,wp::float64> adj_81 = {};
        wp::vec_t<3,wp::float64> adj_82 = {};
        wp::vec_t<3,wp::float64> adj_83 = {};
        wp::vec_t<3,wp::float64> adj_84 = {};
        wp::int32 adj_85 = {};
        wp::float64 adj_86 = {};
        wp::int32 adj_87 = {};
        wp::float64 adj_88 = {};
        wp::int32 adj_89 = {};
        wp::float64 adj_90 = {};
        wp::int32 adj_91 = {};
        wp::float64 adj_92 = {};
        wp::int32 adj_93 = {};
        wp::float64 adj_94 = {};
        wp::int32 adj_95 = {};
        wp::float64 adj_96 = {};
        wp::int32 adj_97 = {};
        wp::float64 adj_98 = {};
        wp::int32 adj_99 = {};
        wp::float64 adj_100 = {};
        wp::int32 adj_101 = {};
        wp::float64 adj_102 = {};
        wp::int32 adj_103 = {};
        wp::float64 adj_104 = {};
        wp::int32 adj_105 = {};
        wp::float64 adj_106 = {};
        wp::int32 adj_107 = {};
        wp::float64 adj_108 = {};
        wp::vec_t<12,wp::float64> adj_109 = {};
        //---------
        // forward
        // def set_sim_affine_vel(                                                                <L 38>
        // tid = wp.tid()                                                                         <L 47>
        var_0 = builtin_tid1d();
        // mask = affine_update_mask[tid]                                                         <L 48>
        var_1 = wp::address(var_affine_update_mask, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // if mask == 0:                                                                          <L 49>
        var_5 = (var_2 == var_4);
        if (var_5) {
            // return                                                                             <L 50>
            goto label0;
        }
        // linear_vel = affine_linear_vel[tid]                                                    <L 51>
        var_6 = wp::address(var_affine_linear_vel, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // angular_vel = affine_angular_vel[tid]                                                  <L 52>
        var_9 = wp::address(var_affine_angular_vel, var_0);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // rest_center = virtual_object_centers[tid]                                              <L 53>
        var_12 = wp::address(var_virtual_object_centers, var_0);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // rest_O = ABD_centers[tid]                                                              <L 54>
        var_15 = wp::address(var_ABD_centers, var_0);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // current_y = y[tid]                                                                     <L 55>
        var_18 = wp::address(var_y, var_0);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // w = rest_center - rest_O                                                               <L 56>
        var_21 = wp::sub(var_13, var_16);
        // O = wp.vec3d(current_y[0], current_y[1], current_y[2])                                 <L 57>
        var_23 = wp::extract(var_19, var_22);
        var_25 = wp::extract(var_19, var_24);
        var_27 = wp::extract(var_19, var_26);
        var_28 = wp::vec_t<3,wp::float64>(var_23, var_25, var_27);
        // A = wp.vec3d(current_y[3], current_y[4], current_y[5])                                 <L 58>
        var_30 = wp::extract(var_19, var_29);
        var_32 = wp::extract(var_19, var_31);
        var_34 = wp::extract(var_19, var_33);
        var_35 = wp::vec_t<3,wp::float64>(var_30, var_32, var_34);
        // B = wp.vec3d(current_y[6], current_y[7], current_y[8])                                 <L 59>
        var_37 = wp::extract(var_19, var_36);
        var_39 = wp::extract(var_19, var_38);
        var_41 = wp::extract(var_19, var_40);
        var_42 = wp::vec_t<3,wp::float64>(var_37, var_39, var_41);
        // C = wp.vec3d(current_y[9], current_y[10], current_y[11])                               <L 60>
        var_44 = wp::extract(var_19, var_43);
        var_46 = wp::extract(var_19, var_45);
        var_48 = wp::extract(var_19, var_47);
        var_49 = wp::vec_t<3,wp::float64>(var_44, var_46, var_48);
        // center = (_1 - w[0] - w[1] - w[2]) * O + w[0] * A + w[1] * B + w[2] * C                <L 61>
        var_52 = wp::extract(var_21, var_51);
        var_53 = wp::sub(var_50, var_52);
        var_55 = wp::extract(var_21, var_54);
        var_56 = wp::sub(var_53, var_55);
        var_58 = wp::extract(var_21, var_57);
        var_59 = wp::sub(var_56, var_58);
        var_60 = wp::mul(var_59, var_28);
        var_62 = wp::extract(var_21, var_61);
        var_63 = wp::mul(var_62, var_35);
        var_64 = wp::add(var_60, var_63);
        var_66 = wp::extract(var_21, var_65);
        var_67 = wp::mul(var_66, var_42);
        var_68 = wp::add(var_64, var_67);
        var_70 = wp::extract(var_21, var_69);
        var_71 = wp::mul(var_70, var_49);
        var_72 = wp::add(var_68, var_71);
        // vel_O = wp.cross(angular_vel, O - center) + linear_vel                                 <L 62>
        var_73 = wp::sub(var_28, var_72);
        var_74 = wp::cross(var_10, var_73);
        var_75 = wp::add(var_74, var_7);
        // vel_A = wp.cross(angular_vel, A - center) + linear_vel                                 <L 63>
        var_76 = wp::sub(var_35, var_72);
        var_77 = wp::cross(var_10, var_76);
        var_78 = wp::add(var_77, var_7);
        // vel_B = wp.cross(angular_vel, B - center) + linear_vel                                 <L 64>
        var_79 = wp::sub(var_42, var_72);
        var_80 = wp::cross(var_10, var_79);
        var_81 = wp::add(var_80, var_7);
        // vel_C = wp.cross(angular_vel, C - center) + linear_vel                                 <L 65>
        var_82 = wp::sub(var_49, var_72);
        var_83 = wp::cross(var_10, var_82);
        var_84 = wp::add(var_83, var_7);
        // v_y[tid] = vec12d(                                                                     <L 66>
        // vel_O[0],                                                                              <L 67>
        var_86 = wp::extract(var_75, var_85);
        // vel_O[1],                                                                              <L 68>
        var_88 = wp::extract(var_75, var_87);
        // vel_O[2],                                                                              <L 69>
        var_90 = wp::extract(var_75, var_89);
        // vel_A[0],                                                                              <L 70>
        var_92 = wp::extract(var_78, var_91);
        // vel_A[1],                                                                              <L 71>
        var_94 = wp::extract(var_78, var_93);
        // vel_A[2],                                                                              <L 72>
        var_96 = wp::extract(var_78, var_95);
        // vel_B[0],                                                                              <L 73>
        var_98 = wp::extract(var_81, var_97);
        // vel_B[1],                                                                              <L 74>
        var_100 = wp::extract(var_81, var_99);
        // vel_B[2],                                                                              <L 75>
        var_102 = wp::extract(var_81, var_101);
        // vel_C[0],                                                                              <L 76>
        var_104 = wp::extract(var_84, var_103);
        // vel_C[1],                                                                              <L 77>
        var_106 = wp::extract(var_84, var_105);
        // vel_C[2],                                                                              <L 78>
        var_108 = wp::extract(var_84, var_107);
        var_109 = wp::vec_t<12,wp::float64>({var_86, var_88, var_90, var_92, var_94, var_96, var_98, var_100, var_102, var_104, var_106, var_108});
        // v_y[tid] = vec12d(                                                                     <L 66>
        // wp::array_store(var_v_y, var_0, var_109);
        //---------
        // reverse
        wp::adj_array_store(var_v_y, var_0, var_109, adj_v_y, adj_0, adj_109);
        // adj: v_y[tid] = vec12d(                                                                <L 66>
        wp::adj_vec_t({var_86, var_88, var_90, var_92, var_94, var_96, var_98, var_100, var_102, var_104, var_106, var_108}, {&adj_86, &adj_88, &adj_90, &adj_92, &adj_94, &adj_96, &adj_98, &adj_100, &adj_102, &adj_104, &adj_106, &adj_108}, adj_109);
        wp::adj_extract(var_84, var_107, adj_84, adj_107, adj_108);
        // adj: vel_C[2],                                                                         <L 78>
        wp::adj_extract(var_84, var_105, adj_84, adj_105, adj_106);
        // adj: vel_C[1],                                                                         <L 77>
        wp::adj_extract(var_84, var_103, adj_84, adj_103, adj_104);
        // adj: vel_C[0],                                                                         <L 76>
        wp::adj_extract(var_81, var_101, adj_81, adj_101, adj_102);
        // adj: vel_B[2],                                                                         <L 75>
        wp::adj_extract(var_81, var_99, adj_81, adj_99, adj_100);
        // adj: vel_B[1],                                                                         <L 74>
        wp::adj_extract(var_81, var_97, adj_81, adj_97, adj_98);
        // adj: vel_B[0],                                                                         <L 73>
        wp::adj_extract(var_78, var_95, adj_78, adj_95, adj_96);
        // adj: vel_A[2],                                                                         <L 72>
        wp::adj_extract(var_78, var_93, adj_78, adj_93, adj_94);
        // adj: vel_A[1],                                                                         <L 71>
        wp::adj_extract(var_78, var_91, adj_78, adj_91, adj_92);
        // adj: vel_A[0],                                                                         <L 70>
        wp::adj_extract(var_75, var_89, adj_75, adj_89, adj_90);
        // adj: vel_O[2],                                                                         <L 69>
        wp::adj_extract(var_75, var_87, adj_75, adj_87, adj_88);
        // adj: vel_O[1],                                                                         <L 68>
        wp::adj_extract(var_75, var_85, adj_75, adj_85, adj_86);
        // adj: vel_O[0],                                                                         <L 67>
        // adj: v_y[tid] = vec12d(                                                                <L 66>
        wp::adj_add(var_83, var_7, adj_83, adj_7, adj_84);
        wp::adj_cross(var_10, var_82, adj_10, adj_82, adj_83);
        wp::adj_sub(var_49, var_72, adj_49, adj_72, adj_82);
        // adj: vel_C = wp.cross(angular_vel, C - center) + linear_vel                            <L 65>
        wp::adj_add(var_80, var_7, adj_80, adj_7, adj_81);
        wp::adj_cross(var_10, var_79, adj_10, adj_79, adj_80);
        wp::adj_sub(var_42, var_72, adj_42, adj_72, adj_79);
        // adj: vel_B = wp.cross(angular_vel, B - center) + linear_vel                            <L 64>
        wp::adj_add(var_77, var_7, adj_77, adj_7, adj_78);
        wp::adj_cross(var_10, var_76, adj_10, adj_76, adj_77);
        wp::adj_sub(var_35, var_72, adj_35, adj_72, adj_76);
        // adj: vel_A = wp.cross(angular_vel, A - center) + linear_vel                            <L 63>
        wp::adj_add(var_74, var_7, adj_74, adj_7, adj_75);
        wp::adj_cross(var_10, var_73, adj_10, adj_73, adj_74);
        wp::adj_sub(var_28, var_72, adj_28, adj_72, adj_73);
        // adj: vel_O = wp.cross(angular_vel, O - center) + linear_vel                            <L 62>
        wp::adj_add(var_68, var_71, adj_68, adj_71, adj_72);
        wp::adj_mul(var_70, var_49, adj_70, adj_49, adj_71);
        wp::adj_extract(var_21, var_69, adj_21, adj_69, adj_70);
        wp::adj_add(var_64, var_67, adj_64, adj_67, adj_68);
        wp::adj_mul(var_66, var_42, adj_66, adj_42, adj_67);
        wp::adj_extract(var_21, var_65, adj_21, adj_65, adj_66);
        wp::adj_add(var_60, var_63, adj_60, adj_63, adj_64);
        wp::adj_mul(var_62, var_35, adj_62, adj_35, adj_63);
        wp::adj_extract(var_21, var_61, adj_21, adj_61, adj_62);
        wp::adj_mul(var_59, var_28, adj_59, adj_28, adj_60);
        wp::adj_sub(var_56, var_58, adj_56, adj_58, adj_59);
        wp::adj_extract(var_21, var_57, adj_21, adj_57, adj_58);
        wp::adj_sub(var_53, var_55, adj_53, adj_55, adj_56);
        wp::adj_extract(var_21, var_54, adj_21, adj_54, adj_55);
        wp::adj_sub(var_50, var_52, adj_50, adj_52, adj_53);
        wp::adj_extract(var_21, var_51, adj_21, adj_51, adj_52);
        // adj: center = (_1 - w[0] - w[1] - w[2]) * O + w[0] * A + w[1] * B + w[2] * C           <L 61>
        wp::adj_vec_t(var_44, var_46, var_48, adj_44, adj_46, adj_48, adj_49);
        wp::adj_extract(var_19, var_47, adj_19, adj_47, adj_48);
        wp::adj_extract(var_19, var_45, adj_19, adj_45, adj_46);
        wp::adj_extract(var_19, var_43, adj_19, adj_43, adj_44);
        // adj: C = wp.vec3d(current_y[9], current_y[10], current_y[11])                          <L 60>
        wp::adj_vec_t(var_37, var_39, var_41, adj_37, adj_39, adj_41, adj_42);
        wp::adj_extract(var_19, var_40, adj_19, adj_40, adj_41);
        wp::adj_extract(var_19, var_38, adj_19, adj_38, adj_39);
        wp::adj_extract(var_19, var_36, adj_19, adj_36, adj_37);
        // adj: B = wp.vec3d(current_y[6], current_y[7], current_y[8])                            <L 59>
        wp::adj_vec_t(var_30, var_32, var_34, adj_30, adj_32, adj_34, adj_35);
        wp::adj_extract(var_19, var_33, adj_19, adj_33, adj_34);
        wp::adj_extract(var_19, var_31, adj_19, adj_31, adj_32);
        wp::adj_extract(var_19, var_29, adj_19, adj_29, adj_30);
        // adj: A = wp.vec3d(current_y[3], current_y[4], current_y[5])                            <L 58>
        wp::adj_vec_t(var_23, var_25, var_27, adj_23, adj_25, adj_27, adj_28);
        wp::adj_extract(var_19, var_26, adj_19, adj_26, adj_27);
        wp::adj_extract(var_19, var_24, adj_19, adj_24, adj_25);
        wp::adj_extract(var_19, var_22, adj_19, adj_22, adj_23);
        // adj: O = wp.vec3d(current_y[0], current_y[1], current_y[2])                            <L 57>
        wp::adj_sub(var_13, var_16, adj_13, adj_16, adj_21);
        // adj: w = rest_center - rest_O                                                          <L 56>
        wp::adj_copy(var_20, adj_18, adj_19);
        wp::adj_load(var_18, adj_18, adj_20);
        wp::adj_address(var_y, var_0, adj_y, adj_0, adj_18);
        // adj: current_y = y[tid]                                                                <L 55>
        wp::adj_copy(var_17, adj_15, adj_16);
        wp::adj_load(var_15, adj_15, adj_17);
        wp::adj_address(var_ABD_centers, var_0, adj_ABD_centers, adj_0, adj_15);
        // adj: rest_O = ABD_centers[tid]                                                         <L 54>
        wp::adj_copy(var_14, adj_12, adj_13);
        wp::adj_load(var_12, adj_12, adj_14);
        wp::adj_address(var_virtual_object_centers, var_0, adj_virtual_object_centers, adj_0, adj_12);
        // adj: rest_center = virtual_object_centers[tid]                                         <L 53>
        wp::adj_copy(var_11, adj_9, adj_10);
        wp::adj_load(var_9, adj_9, adj_11);
        wp::adj_address(var_affine_angular_vel, var_0, adj_affine_angular_vel, adj_0, adj_9);
        // adj: angular_vel = affine_angular_vel[tid]                                             <L 52>
        wp::adj_copy(var_8, adj_6, adj_7);
        wp::adj_load(var_6, adj_6, adj_8);
        wp::adj_address(var_affine_linear_vel, var_0, adj_affine_linear_vel, adj_0, adj_6);
        // adj: linear_vel = affine_linear_vel[tid]                                               <L 51>
        if (var_5) {
            label0:;
            // adj: return                                                                        <L 50>
        }
        // adj: if mask == 0:                                                                     <L 49>
        wp::adj_copy(var_3, adj_1, adj_2);
        wp::adj_load(var_1, adj_1, adj_3);
        wp::adj_address(var_affine_update_mask, var_0, adj_affine_update_mask, adj_0, adj_1);
        // adj: mask = affine_update_mask[tid]                                                    <L 48>
        // adj: tid = wp.tid()                                                                    <L 47>
        // adj: def set_sim_affine_vel(                                                           <L 38>
        continue;
    }
}

