
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


extern "C" __global__ void check_is_satisfied_kernel_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<bool> var_y_target_reached,
    wp::array_t<bool> var_y_has_constraint,
    wp::array_t<bool> var_x_target_reached,
    wp::array_t<bool> var_x_has_constraint,
    wp::array_t<bool> var_satisfied_val,
    wp::int32 var_affine_verts_num,
    wp::array_t<wp::int32> var_body_env_id,
    wp::array_t<wp::int32> var_node2env,
    wp::array_t<wp::int32> var_env_states)
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
        const wp::int32 var_1 = 0;
        wp::shape_t* var_2;
        wp::int32 var_3;
        wp::shape_t var_4;
        bool var_5;
        bool* var_6;
        bool var_7;
        bool var_8;
        wp::int32* var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        const wp::int32 var_12 = 1;
        bool var_13;
        wp::int32 var_14;
        wp::int32* var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        const wp::int32 var_18 = 2;
        bool var_19;
        wp::int32 var_20;
        bool var_21;
        bool var_22;
        bool* var_23;
        bool var_24;
        bool var_25;
        const bool var_26 = false;
        wp::int32* var_27;
        wp::int32 var_28;
        const wp::int32 var_29 = 0;
        wp::shape_t* var_30;
        wp::int32 var_31;
        wp::shape_t var_32;
        wp::int32 var_33;
        bool* var_34;
        bool var_35;
        bool var_36;
        wp::int32 var_37;
        wp::int32* var_38;
        wp::int32* var_39;
        wp::int32 var_40;
        bool var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        wp::int32* var_44;
        wp::int32* var_45;
        wp::int32 var_46;
        bool var_47;
        wp::int32 var_48;
        bool var_49;
        bool var_50;
        bool* var_51;
        bool var_52;
        bool var_53;
        const bool var_54 = false;
        wp::int32 var_55;
        wp::int32* var_56;
        wp::int32 var_57;
        wp::int32 var_58;
        //---------
        // forward
        // def check_is_satisfied_kernel(                                                         <L 67>
        // tid = wp.tid()                                                                         <L 78>
        var_0 = builtin_tid1d();
        // if tid < y_target_reached.shape[0]:                                                    <L 79>
        var_2 = &(var_y_target_reached.shape);
        var_4 = wp::load(var_2);
        var_3 = wp::extract(var_4, var_1);
        var_5 = (var_0 < var_3);
        if (var_5) {
            // if not y_has_constraint[tid] or (                                                  <L 80>
            var_6 = wp::address(var_y_has_constraint, var_0);
            var_8 = wp::load(var_6);
            var_7 = wp::unot(var_8);
            // (env_states[body_env_id[tid]] == ENV_STATE_INVALID) or (env_states[body_env_id[tid]] == ENV_STATE_NEWTON_SOLVED)       <L 81>
            var_9 = wp::address(var_body_env_id, var_0);
            var_11 = wp::load(var_9);
            var_10 = wp::address(var_env_states, var_11);
            var_14 = wp::load(var_10);
            var_13 = (var_14 == var_12);
            var_15 = wp::address(var_body_env_id, var_0);
            var_17 = wp::load(var_15);
            var_16 = wp::address(var_env_states, var_17);
            var_20 = wp::load(var_16);
            var_19 = (var_20 == var_18);
            var_21 = var_13 || var_19;
            var_22 = var_7 || var_21;
            if (var_22) {
                // return                                                                         <L 83>
                continue;
            }
            // if not y_target_reached[tid]:                                                      <L 84>
            var_23 = wp::address(var_y_target_reached, var_0);
            var_25 = wp::load(var_23);
            var_24 = wp::unot(var_25);
            if (var_24) {
                // satisfied_val[body_env_id[tid]] = False                                        <L 85>
                var_27 = wp::address(var_body_env_id, var_0);
                var_28 = wp::load(var_27);
                wp::array_store(var_satisfied_val, var_28, var_26);
            }
        }
        if (!var_5) {
            // tid -= y_target_reached.shape[0]                                                   <L 87>
            var_30 = &(var_y_target_reached.shape);
            var_32 = wp::load(var_30);
            var_31 = wp::extract(var_32, var_29);
            var_33 = wp::sub(var_0, var_31);
            // if not x_has_constraint[tid] or (                                                  <L 88>
            var_34 = wp::address(var_x_has_constraint, var_33);
            var_36 = wp::load(var_34);
            var_35 = wp::unot(var_36);
            // (env_states[node2env[tid + affine_verts_num]] == ENV_STATE_INVALID)                <L 89>
            var_37 = wp::add(var_33, var_affine_verts_num);
            var_38 = wp::address(var_node2env, var_37);
            var_40 = wp::load(var_38);
            var_39 = wp::address(var_env_states, var_40);
            var_42 = wp::load(var_39);
            var_41 = (var_42 == var_12);
            // or (env_states[node2env[tid + affine_verts_num]] == ENV_STATE_NEWTON_SOLVED)       <L 90>
            var_43 = wp::add(var_33, var_affine_verts_num);
            var_44 = wp::address(var_node2env, var_43);
            var_46 = wp::load(var_44);
            var_45 = wp::address(var_env_states, var_46);
            var_48 = wp::load(var_45);
            var_47 = (var_48 == var_18);
            var_49 = var_41 || var_47;
            var_50 = var_35 || var_49;
            if (var_50) {
                // return                                                                         <L 92>
                continue;
            }
            // if not x_target_reached[tid]:                                                      <L 93>
            var_51 = wp::address(var_x_target_reached, var_33);
            var_53 = wp::load(var_51);
            var_52 = wp::unot(var_53);
            if (var_52) {
                // satisfied_val[node2env[tid + affine_verts_num]] = False                        <L 94>
                var_55 = wp::add(var_33, var_affine_verts_num);
                var_56 = wp::address(var_node2env, var_55);
                var_57 = wp::load(var_56);
                wp::array_store(var_satisfied_val, var_57, var_54);
            }
        }
        var_58 = wp::where(var_5, var_0, var_33);
    }
}



extern "C" __global__ void check_is_satisfied_kernel_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<bool> var_y_target_reached,
    wp::array_t<bool> var_y_has_constraint,
    wp::array_t<bool> var_x_target_reached,
    wp::array_t<bool> var_x_has_constraint,
    wp::array_t<bool> var_satisfied_val,
    wp::int32 var_affine_verts_num,
    wp::array_t<wp::int32> var_body_env_id,
    wp::array_t<wp::int32> var_node2env,
    wp::array_t<wp::int32> var_env_states,
    wp::array_t<bool> adj_y_target_reached,
    wp::array_t<bool> adj_y_has_constraint,
    wp::array_t<bool> adj_x_target_reached,
    wp::array_t<bool> adj_x_has_constraint,
    wp::array_t<bool> adj_satisfied_val,
    wp::int32 adj_affine_verts_num,
    wp::array_t<wp::int32> adj_body_env_id,
    wp::array_t<wp::int32> adj_node2env,
    wp::array_t<wp::int32> adj_env_states)
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
        const wp::int32 var_1 = 0;
        wp::shape_t* var_2;
        wp::int32 var_3;
        wp::shape_t var_4;
        bool var_5;
        bool* var_6;
        bool var_7;
        bool var_8;
        wp::int32* var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        const wp::int32 var_12 = 1;
        bool var_13;
        wp::int32 var_14;
        wp::int32* var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        const wp::int32 var_18 = 2;
        bool var_19;
        wp::int32 var_20;
        bool var_21;
        bool var_22;
        bool* var_23;
        bool var_24;
        bool var_25;
        const bool var_26 = false;
        wp::int32* var_27;
        wp::int32 var_28;
        const wp::int32 var_29 = 0;
        wp::shape_t* var_30;
        wp::int32 var_31;
        wp::shape_t var_32;
        wp::int32 var_33;
        bool* var_34;
        bool var_35;
        bool var_36;
        wp::int32 var_37;
        wp::int32* var_38;
        wp::int32* var_39;
        wp::int32 var_40;
        bool var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        wp::int32* var_44;
        wp::int32* var_45;
        wp::int32 var_46;
        bool var_47;
        wp::int32 var_48;
        bool var_49;
        bool var_50;
        bool* var_51;
        bool var_52;
        bool var_53;
        const bool var_54 = false;
        wp::int32 var_55;
        wp::int32* var_56;
        wp::int32 var_57;
        wp::int32 var_58;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::shape_t adj_2 = {};
        wp::int32 adj_3 = {};
        wp::shape_t adj_4 = {};
        bool adj_5 = {};
        bool adj_6 = {};
        bool adj_7 = {};
        bool adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        bool adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        bool adj_19 = {};
        wp::int32 adj_20 = {};
        bool adj_21 = {};
        bool adj_22 = {};
        bool adj_23 = {};
        bool adj_24 = {};
        bool adj_25 = {};
        bool adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::shape_t adj_30 = {};
        wp::int32 adj_31 = {};
        wp::shape_t adj_32 = {};
        wp::int32 adj_33 = {};
        bool adj_34 = {};
        bool adj_35 = {};
        bool adj_36 = {};
        wp::int32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::int32 adj_39 = {};
        wp::int32 adj_40 = {};
        bool adj_41 = {};
        wp::int32 adj_42 = {};
        wp::int32 adj_43 = {};
        wp::int32 adj_44 = {};
        wp::int32 adj_45 = {};
        wp::int32 adj_46 = {};
        bool adj_47 = {};
        wp::int32 adj_48 = {};
        bool adj_49 = {};
        bool adj_50 = {};
        bool adj_51 = {};
        bool adj_52 = {};
        bool adj_53 = {};
        bool adj_54 = {};
        wp::int32 adj_55 = {};
        wp::int32 adj_56 = {};
        wp::int32 adj_57 = {};
        wp::int32 adj_58 = {};
        //---------
        // forward
        // def check_is_satisfied_kernel(                                                         <L 67>
        // tid = wp.tid()                                                                         <L 78>
        var_0 = builtin_tid1d();
        // if tid < y_target_reached.shape[0]:                                                    <L 79>
        var_2 = &(var_y_target_reached.shape);
        var_4 = wp::load(var_2);
        var_3 = wp::extract(var_4, var_1);
        var_5 = (var_0 < var_3);
        if (var_5) {
            // if not y_has_constraint[tid] or (                                                  <L 80>
            var_6 = wp::address(var_y_has_constraint, var_0);
            var_8 = wp::load(var_6);
            var_7 = wp::unot(var_8);
            // (env_states[body_env_id[tid]] == ENV_STATE_INVALID) or (env_states[body_env_id[tid]] == ENV_STATE_NEWTON_SOLVED)       <L 81>
            var_9 = wp::address(var_body_env_id, var_0);
            var_11 = wp::load(var_9);
            var_10 = wp::address(var_env_states, var_11);
            var_14 = wp::load(var_10);
            var_13 = (var_14 == var_12);
            var_15 = wp::address(var_body_env_id, var_0);
            var_17 = wp::load(var_15);
            var_16 = wp::address(var_env_states, var_17);
            var_20 = wp::load(var_16);
            var_19 = (var_20 == var_18);
            var_21 = var_13 || var_19;
            var_22 = var_7 || var_21;
            if (var_22) {
                // return                                                                         <L 83>
                goto label0;
            }
            // if not y_target_reached[tid]:                                                      <L 84>
            var_23 = wp::address(var_y_target_reached, var_0);
            var_25 = wp::load(var_23);
            var_24 = wp::unot(var_25);
            if (var_24) {
                // satisfied_val[body_env_id[tid]] = False                                        <L 85>
                var_27 = wp::address(var_body_env_id, var_0);
                var_28 = wp::load(var_27);
                // wp::array_store(var_satisfied_val, var_28, var_26);
            }
        }
        if (!var_5) {
            // tid -= y_target_reached.shape[0]                                                   <L 87>
            var_30 = &(var_y_target_reached.shape);
            var_32 = wp::load(var_30);
            var_31 = wp::extract(var_32, var_29);
            var_33 = wp::sub(var_0, var_31);
            // if not x_has_constraint[tid] or (                                                  <L 88>
            var_34 = wp::address(var_x_has_constraint, var_33);
            var_36 = wp::load(var_34);
            var_35 = wp::unot(var_36);
            // (env_states[node2env[tid + affine_verts_num]] == ENV_STATE_INVALID)                <L 89>
            var_37 = wp::add(var_33, var_affine_verts_num);
            var_38 = wp::address(var_node2env, var_37);
            var_40 = wp::load(var_38);
            var_39 = wp::address(var_env_states, var_40);
            var_42 = wp::load(var_39);
            var_41 = (var_42 == var_12);
            // or (env_states[node2env[tid + affine_verts_num]] == ENV_STATE_NEWTON_SOLVED)       <L 90>
            var_43 = wp::add(var_33, var_affine_verts_num);
            var_44 = wp::address(var_node2env, var_43);
            var_46 = wp::load(var_44);
            var_45 = wp::address(var_env_states, var_46);
            var_48 = wp::load(var_45);
            var_47 = (var_48 == var_18);
            var_49 = var_41 || var_47;
            var_50 = var_35 || var_49;
            if (var_50) {
                // return                                                                         <L 92>
                goto label1;
            }
            // if not x_target_reached[tid]:                                                      <L 93>
            var_51 = wp::address(var_x_target_reached, var_33);
            var_53 = wp::load(var_51);
            var_52 = wp::unot(var_53);
            if (var_52) {
                // satisfied_val[node2env[tid + affine_verts_num]] = False                        <L 94>
                var_55 = wp::add(var_33, var_affine_verts_num);
                var_56 = wp::address(var_node2env, var_55);
                var_57 = wp::load(var_56);
                // wp::array_store(var_satisfied_val, var_57, var_54);
            }
        }
        var_58 = wp::where(var_5, var_0, var_33);
        //---------
        // reverse
        wp::adj_where(var_5, var_0, var_33, adj_5, adj_0, adj_33, adj_58);
        if (!var_5) {
            if (var_52) {
                wp::adj_array_store(var_satisfied_val, var_57, var_54, adj_satisfied_val, adj_56, adj_54);
                wp::adj_load(var_56, adj_56, adj_57);
                wp::adj_address(var_node2env, var_55, adj_node2env, adj_55, adj_56);
                wp::adj_add(var_33, var_affine_verts_num, adj_33, adj_affine_verts_num, adj_55);
                // adj: satisfied_val[node2env[tid + affine_verts_num]] = False                   <L 94>
            }
            wp::adj_unot(var_53, adj_51, adj_52);
            wp::adj_load(var_51, adj_51, adj_53);
            wp::adj_address(var_x_target_reached, var_33, adj_x_target_reached, adj_33, adj_51);
            // adj: if not x_target_reached[tid]:                                                 <L 93>
            if (var_50) {
                label1:;
                // adj: return                                                                    <L 92>
            }
            wp::adj_load(var_45, adj_45, adj_48);
            wp::adj_address(var_env_states, var_46, adj_env_states, adj_44, adj_45);
            wp::adj_load(var_44, adj_44, adj_46);
            wp::adj_address(var_node2env, var_43, adj_node2env, adj_43, adj_44);
            wp::adj_add(var_33, var_affine_verts_num, adj_33, adj_affine_verts_num, adj_43);
            // adj: or (env_states[node2env[tid + affine_verts_num]] == ENV_STATE_NEWTON_SOLVED)  <L 90>
            wp::adj_load(var_39, adj_39, adj_42);
            wp::adj_address(var_env_states, var_40, adj_env_states, adj_38, adj_39);
            wp::adj_load(var_38, adj_38, adj_40);
            wp::adj_address(var_node2env, var_37, adj_node2env, adj_37, adj_38);
            wp::adj_add(var_33, var_affine_verts_num, adj_33, adj_affine_verts_num, adj_37);
            // adj: (env_states[node2env[tid + affine_verts_num]] == ENV_STATE_INVALID)           <L 89>
            wp::adj_unot(var_36, adj_34, adj_35);
            wp::adj_load(var_34, adj_34, adj_36);
            wp::adj_address(var_x_has_constraint, var_33, adj_x_has_constraint, adj_33, adj_34);
            // adj: if not x_has_constraint[tid] or (                                             <L 88>
            wp::adj_sub(var_0, var_31, adj_0, adj_31, adj_33);
            wp::adj_extract(var_32, var_29, adj_30, adj_29, adj_31);
            wp::adj_load(var_30, adj_30, adj_32);
            adj_y_target_reached.shape = adj_30;
            // adj: tid -= y_target_reached.shape[0]                                              <L 87>
        }
        if (var_5) {
            if (var_24) {
                wp::adj_array_store(var_satisfied_val, var_28, var_26, adj_satisfied_val, adj_27, adj_26);
                wp::adj_load(var_27, adj_27, adj_28);
                wp::adj_address(var_body_env_id, var_0, adj_body_env_id, adj_0, adj_27);
                // adj: satisfied_val[body_env_id[tid]] = False                                   <L 85>
            }
            wp::adj_unot(var_25, adj_23, adj_24);
            wp::adj_load(var_23, adj_23, adj_25);
            wp::adj_address(var_y_target_reached, var_0, adj_y_target_reached, adj_0, adj_23);
            // adj: if not y_target_reached[tid]:                                                 <L 84>
            if (var_22) {
                label0:;
                // adj: return                                                                    <L 83>
            }
            wp::adj_load(var_16, adj_16, adj_20);
            wp::adj_address(var_env_states, var_17, adj_env_states, adj_15, adj_16);
            wp::adj_load(var_15, adj_15, adj_17);
            wp::adj_address(var_body_env_id, var_0, adj_body_env_id, adj_0, adj_15);
            wp::adj_load(var_10, adj_10, adj_14);
            wp::adj_address(var_env_states, var_11, adj_env_states, adj_9, adj_10);
            wp::adj_load(var_9, adj_9, adj_11);
            wp::adj_address(var_body_env_id, var_0, adj_body_env_id, adj_0, adj_9);
            // adj: (env_states[body_env_id[tid]] == ENV_STATE_INVALID) or (env_states[body_env_id[tid]] == ENV_STATE_NEWTON_SOLVED)  <L 81>
            wp::adj_unot(var_8, adj_6, adj_7);
            wp::adj_load(var_6, adj_6, adj_8);
            wp::adj_address(var_y_has_constraint, var_0, adj_y_has_constraint, adj_0, adj_6);
            // adj: if not y_has_constraint[tid] or (                                             <L 80>
        }
        wp::adj_extract(var_4, var_1, adj_2, adj_1, adj_3);
        wp::adj_load(var_2, adj_2, adj_4);
        adj_y_target_reached.shape = adj_2;
        // adj: if tid < y_target_reached.shape[0]:                                               <L 79>
        // adj: tid = wp.tid()                                                                    <L 78>
        // adj: def check_is_satisfied_kernel(                                                    <L 67>
        continue;
    }
}



extern "C" __global__ void update_dof_satisfied_kernel_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<bool> var_x_has_constraint,
    wp::array_t<wp::vec_t<3,wp::float64>> var_x,
    wp::array_t<wp::vec_t<3,wp::float64>> var_x_target,
    wp::array_t<bool> var_x_target_reached,
    wp::array_t<bool> var_y_has_constraint,
    wp::array_t<wp::vec_t<12,wp::float64>> var_y,
    wp::array_t<wp::vec_t<12,wp::float64>> var_y_target,
    wp::array_t<bool> var_y_target_reached,
    wp::float64 var_dt,
    wp::float64 var_tol)
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
        const wp::int32 var_1 = 0;
        wp::shape_t* var_2;
        wp::int32 var_3;
        wp::shape_t var_4;
        bool var_5;
        bool* var_6;
        bool* var_7;
        const bool var_8 = false;
        bool var_9;
        bool var_10;
        bool var_11;
        bool var_12;
        wp::vec_t<3,wp::float64>* var_13;
        wp::vec_t<3,wp::float64>* var_14;
        wp::vec_t<3,wp::float64> var_15;
        wp::vec_t<3,wp::float64> var_16;
        wp::vec_t<3,wp::float64> var_17;
        wp::float64 var_18;
        wp::float64 var_19;
        bool var_20;
        const wp::int32 var_21 = 1;
        bool var_22;
        const wp::int32 var_23 = 0;
        wp::shape_t* var_24;
        wp::int32 var_25;
        wp::shape_t var_26;
        wp::int32 var_27;
        bool* var_28;
        bool* var_29;
        const bool var_30 = false;
        bool var_31;
        bool var_32;
        bool var_33;
        bool var_34;
        wp::vec_t<12,wp::float64>* var_35;
        wp::vec_t<12,wp::float64>* var_36;
        wp::vec_t<12,wp::float64> var_37;
        wp::vec_t<12,wp::float64> var_38;
        wp::vec_t<12,wp::float64> var_39;
        wp::float64 var_40;
        wp::float64 var_41;
        bool var_42;
        const wp::int32 var_43 = 1;
        bool var_44;
        wp::int32 var_45;
        //---------
        // forward
        // def update_dof_satisfied_kernel(                                                       <L 8>
        // tid = wp.tid()                                                                         <L 20>
        var_0 = builtin_tid1d();
        // if tid < x_has_constraint.shape[0]:                                                    <L 21>
        var_2 = &(var_x_has_constraint.shape);
        var_4 = wp::load(var_2);
        var_3 = wp::extract(var_4, var_1);
        var_5 = (var_0 < var_3);
        if (var_5) {
            // if x_has_constraint[tid] and x_target_reached[tid] == False:                       <L 22>
            var_6 = wp::address(var_x_has_constraint, var_0);
            var_7 = wp::address(var_x_target_reached, var_0);
            var_10 = wp::load(var_7);
            var_9 = (var_10 == var_8);
            var_11 = wp::load(var_6);
            var_12 = var_11 && var_9;
            if (var_12) {
                // if wp.length(x[tid] - x_target[tid]) < tol * dt:                               <L 23>
                var_13 = wp::address(var_x, var_0);
                var_14 = wp::address(var_x_target, var_0);
                var_16 = wp::load(var_13);
                var_17 = wp::load(var_14);
                var_15 = wp::sub(var_16, var_17);
                var_18 = wp::length(var_15);
                var_19 = wp::mul(var_tol, var_dt);
                var_20 = (var_18 < var_19);
                if (var_20) {
                    // x_target_reached[tid] = wp.bool(1)                                         <L 24>
                    var_22 = bool(var_21);
                    wp::array_store(var_x_target_reached, var_0, var_22);
                }
            }
        }
        if (!var_5) {
            // tid -= x_has_constraint.shape[0]                                                   <L 26>
            var_24 = &(var_x_has_constraint.shape);
            var_26 = wp::load(var_24);
            var_25 = wp::extract(var_26, var_23);
            var_27 = wp::sub(var_0, var_25);
            // if y_has_constraint[tid] and y_target_reached[tid] == False:                       <L 27>
            var_28 = wp::address(var_y_has_constraint, var_27);
            var_29 = wp::address(var_y_target_reached, var_27);
            var_32 = wp::load(var_29);
            var_31 = (var_32 == var_30);
            var_33 = wp::load(var_28);
            var_34 = var_33 && var_31;
            if (var_34) {
                // if wp.length(y[tid] - y_target[tid]) < tol * dt:                               <L 28>
                var_35 = wp::address(var_y, var_27);
                var_36 = wp::address(var_y_target, var_27);
                var_38 = wp::load(var_35);
                var_39 = wp::load(var_36);
                var_37 = wp::sub(var_38, var_39);
                var_40 = wp::length(var_37);
                var_41 = wp::mul(var_tol, var_dt);
                var_42 = (var_40 < var_41);
                if (var_42) {
                    // y_target_reached[tid] = wp.bool(1)                                         <L 29>
                    var_44 = bool(var_43);
                    wp::array_store(var_y_target_reached, var_27, var_44);
                }
            }
        }
        var_45 = wp::where(var_5, var_0, var_27);
    }
}



extern "C" __global__ void update_dof_satisfied_kernel_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<bool> var_x_has_constraint,
    wp::array_t<wp::vec_t<3,wp::float64>> var_x,
    wp::array_t<wp::vec_t<3,wp::float64>> var_x_target,
    wp::array_t<bool> var_x_target_reached,
    wp::array_t<bool> var_y_has_constraint,
    wp::array_t<wp::vec_t<12,wp::float64>> var_y,
    wp::array_t<wp::vec_t<12,wp::float64>> var_y_target,
    wp::array_t<bool> var_y_target_reached,
    wp::float64 var_dt,
    wp::float64 var_tol,
    wp::array_t<bool> adj_x_has_constraint,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_x,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_x_target,
    wp::array_t<bool> adj_x_target_reached,
    wp::array_t<bool> adj_y_has_constraint,
    wp::array_t<wp::vec_t<12,wp::float64>> adj_y,
    wp::array_t<wp::vec_t<12,wp::float64>> adj_y_target,
    wp::array_t<bool> adj_y_target_reached,
    wp::float64 adj_dt,
    wp::float64 adj_tol)
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
        const wp::int32 var_1 = 0;
        wp::shape_t* var_2;
        wp::int32 var_3;
        wp::shape_t var_4;
        bool var_5;
        bool* var_6;
        bool* var_7;
        const bool var_8 = false;
        bool var_9;
        bool var_10;
        bool var_11;
        bool var_12;
        wp::vec_t<3,wp::float64>* var_13;
        wp::vec_t<3,wp::float64>* var_14;
        wp::vec_t<3,wp::float64> var_15;
        wp::vec_t<3,wp::float64> var_16;
        wp::vec_t<3,wp::float64> var_17;
        wp::float64 var_18;
        wp::float64 var_19;
        bool var_20;
        const wp::int32 var_21 = 1;
        bool var_22;
        const wp::int32 var_23 = 0;
        wp::shape_t* var_24;
        wp::int32 var_25;
        wp::shape_t var_26;
        wp::int32 var_27;
        bool* var_28;
        bool* var_29;
        const bool var_30 = false;
        bool var_31;
        bool var_32;
        bool var_33;
        bool var_34;
        wp::vec_t<12,wp::float64>* var_35;
        wp::vec_t<12,wp::float64>* var_36;
        wp::vec_t<12,wp::float64> var_37;
        wp::vec_t<12,wp::float64> var_38;
        wp::vec_t<12,wp::float64> var_39;
        wp::float64 var_40;
        wp::float64 var_41;
        bool var_42;
        const wp::int32 var_43 = 1;
        bool var_44;
        wp::int32 var_45;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::shape_t adj_2 = {};
        wp::int32 adj_3 = {};
        wp::shape_t adj_4 = {};
        bool adj_5 = {};
        bool adj_6 = {};
        bool adj_7 = {};
        bool adj_8 = {};
        bool adj_9 = {};
        bool adj_10 = {};
        bool adj_11 = {};
        bool adj_12 = {};
        wp::vec_t<3,wp::float64> adj_13 = {};
        wp::vec_t<3,wp::float64> adj_14 = {};
        wp::vec_t<3,wp::float64> adj_15 = {};
        wp::vec_t<3,wp::float64> adj_16 = {};
        wp::vec_t<3,wp::float64> adj_17 = {};
        wp::float64 adj_18 = {};
        wp::float64 adj_19 = {};
        bool adj_20 = {};
        wp::int32 adj_21 = {};
        bool adj_22 = {};
        wp::int32 adj_23 = {};
        wp::shape_t adj_24 = {};
        wp::int32 adj_25 = {};
        wp::shape_t adj_26 = {};
        wp::int32 adj_27 = {};
        bool adj_28 = {};
        bool adj_29 = {};
        bool adj_30 = {};
        bool adj_31 = {};
        bool adj_32 = {};
        bool adj_33 = {};
        bool adj_34 = {};
        wp::vec_t<12,wp::float64> adj_35 = {};
        wp::vec_t<12,wp::float64> adj_36 = {};
        wp::vec_t<12,wp::float64> adj_37 = {};
        wp::vec_t<12,wp::float64> adj_38 = {};
        wp::vec_t<12,wp::float64> adj_39 = {};
        wp::float64 adj_40 = {};
        wp::float64 adj_41 = {};
        bool adj_42 = {};
        wp::int32 adj_43 = {};
        bool adj_44 = {};
        wp::int32 adj_45 = {};
        //---------
        // forward
        // def update_dof_satisfied_kernel(                                                       <L 8>
        // tid = wp.tid()                                                                         <L 20>
        var_0 = builtin_tid1d();
        // if tid < x_has_constraint.shape[0]:                                                    <L 21>
        var_2 = &(var_x_has_constraint.shape);
        var_4 = wp::load(var_2);
        var_3 = wp::extract(var_4, var_1);
        var_5 = (var_0 < var_3);
        if (var_5) {
            // if x_has_constraint[tid] and x_target_reached[tid] == False:                       <L 22>
            var_6 = wp::address(var_x_has_constraint, var_0);
            var_7 = wp::address(var_x_target_reached, var_0);
            var_10 = wp::load(var_7);
            var_9 = (var_10 == var_8);
            var_11 = wp::load(var_6);
            var_12 = var_11 && var_9;
            if (var_12) {
                // if wp.length(x[tid] - x_target[tid]) < tol * dt:                               <L 23>
                var_13 = wp::address(var_x, var_0);
                var_14 = wp::address(var_x_target, var_0);
                var_16 = wp::load(var_13);
                var_17 = wp::load(var_14);
                var_15 = wp::sub(var_16, var_17);
                var_18 = wp::length(var_15);
                var_19 = wp::mul(var_tol, var_dt);
                var_20 = (var_18 < var_19);
                if (var_20) {
                    // x_target_reached[tid] = wp.bool(1)                                         <L 24>
                    var_22 = bool(var_21);
                    // wp::array_store(var_x_target_reached, var_0, var_22);
                }
            }
        }
        if (!var_5) {
            // tid -= x_has_constraint.shape[0]                                                   <L 26>
            var_24 = &(var_x_has_constraint.shape);
            var_26 = wp::load(var_24);
            var_25 = wp::extract(var_26, var_23);
            var_27 = wp::sub(var_0, var_25);
            // if y_has_constraint[tid] and y_target_reached[tid] == False:                       <L 27>
            var_28 = wp::address(var_y_has_constraint, var_27);
            var_29 = wp::address(var_y_target_reached, var_27);
            var_32 = wp::load(var_29);
            var_31 = (var_32 == var_30);
            var_33 = wp::load(var_28);
            var_34 = var_33 && var_31;
            if (var_34) {
                // if wp.length(y[tid] - y_target[tid]) < tol * dt:                               <L 28>
                var_35 = wp::address(var_y, var_27);
                var_36 = wp::address(var_y_target, var_27);
                var_38 = wp::load(var_35);
                var_39 = wp::load(var_36);
                var_37 = wp::sub(var_38, var_39);
                var_40 = wp::length(var_37);
                var_41 = wp::mul(var_tol, var_dt);
                var_42 = (var_40 < var_41);
                if (var_42) {
                    // y_target_reached[tid] = wp.bool(1)                                         <L 29>
                    var_44 = bool(var_43);
                    // wp::array_store(var_y_target_reached, var_27, var_44);
                }
            }
        }
        var_45 = wp::where(var_5, var_0, var_27);
        //---------
        // reverse
        wp::adj_where(var_5, var_0, var_27, adj_5, adj_0, adj_27, adj_45);
        if (!var_5) {
            if (var_34) {
                if (var_42) {
                    wp::adj_array_store(var_y_target_reached, var_27, var_44, adj_y_target_reached, adj_27, adj_44);
                    adj_bool(var_43, adj_43, adj_44);
                    // adj: y_target_reached[tid] = wp.bool(1)                                    <L 29>
                }
                wp::adj_mul(var_tol, var_dt, adj_tol, adj_dt, adj_41);
                wp::adj_length(var_37, var_40, adj_37, adj_40);
                wp::adj_sub(var_38, var_39, adj_35, adj_36, adj_37);
                wp::adj_load(var_36, adj_36, adj_39);
                wp::adj_load(var_35, adj_35, adj_38);
                wp::adj_address(var_y_target, var_27, adj_y_target, adj_27, adj_36);
                wp::adj_address(var_y, var_27, adj_y, adj_27, adj_35);
                // adj: if wp.length(y[tid] - y_target[tid]) < tol * dt:                          <L 28>
            }
            wp::adj_load(var_28, adj_28, adj_33);
            wp::adj_load(var_29, adj_29, adj_32);
            wp::adj_address(var_y_target_reached, var_27, adj_y_target_reached, adj_27, adj_29);
            wp::adj_address(var_y_has_constraint, var_27, adj_y_has_constraint, adj_27, adj_28);
            // adj: if y_has_constraint[tid] and y_target_reached[tid] == False:                  <L 27>
            wp::adj_sub(var_0, var_25, adj_0, adj_25, adj_27);
            wp::adj_extract(var_26, var_23, adj_24, adj_23, adj_25);
            wp::adj_load(var_24, adj_24, adj_26);
            adj_x_has_constraint.shape = adj_24;
            // adj: tid -= x_has_constraint.shape[0]                                              <L 26>
        }
        if (var_5) {
            if (var_12) {
                if (var_20) {
                    wp::adj_array_store(var_x_target_reached, var_0, var_22, adj_x_target_reached, adj_0, adj_22);
                    adj_bool(var_21, adj_21, adj_22);
                    // adj: x_target_reached[tid] = wp.bool(1)                                    <L 24>
                }
                wp::adj_mul(var_tol, var_dt, adj_tol, adj_dt, adj_19);
                wp::adj_length(var_15, var_18, adj_15, adj_18);
                wp::adj_sub(var_16, var_17, adj_13, adj_14, adj_15);
                wp::adj_load(var_14, adj_14, adj_17);
                wp::adj_load(var_13, adj_13, adj_16);
                wp::adj_address(var_x_target, var_0, adj_x_target, adj_0, adj_14);
                wp::adj_address(var_x, var_0, adj_x, adj_0, adj_13);
                // adj: if wp.length(x[tid] - x_target[tid]) < tol * dt:                          <L 23>
            }
            wp::adj_load(var_6, adj_6, adj_11);
            wp::adj_load(var_7, adj_7, adj_10);
            wp::adj_address(var_x_target_reached, var_0, adj_x_target_reached, adj_0, adj_7);
            wp::adj_address(var_x_has_constraint, var_0, adj_x_has_constraint, adj_0, adj_6);
            // adj: if x_has_constraint[tid] and x_target_reached[tid] == False:                  <L 22>
        }
        wp::adj_extract(var_4, var_1, adj_2, adj_1, adj_3);
        wp::adj_load(var_2, adj_2, adj_4);
        adj_x_has_constraint.shape = adj_2;
        // adj: if tid < x_has_constraint.shape[0]:                                               <L 21>
        // adj: tid = wp.tid()                                                                    <L 20>
        // adj: def update_dof_satisfied_kernel(                                                  <L 8>
        continue;
    }
}



extern "C" __global__ void project_system_kernel_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_offsets,
    wp::array_t<wp::int32> var_columns,
    wp::array_t<wp::mat_t<3,3,wp::float64>> var_values,
    wp::array_t<wp::vec_t<3,wp::float64>> var_gradient,
    wp::array_t<bool> var_y_target_reached,
    wp::array_t<bool> var_x_target_reached)
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
        const wp::int32 var_1 = 0;
        bool var_2;
        const wp::int32 var_3 = 0;
        wp::shape_t* var_4;
        wp::int32 var_5;
        wp::shape_t var_6;
        const wp::int32 var_7 = 4;
        wp::int32 var_8;
        bool var_9;
        const wp::int32 var_10 = 4;
        wp::int32 var_11;
        wp::int32 var_12;
        bool* var_13;
        bool var_14;
        bool var_15;
        bool var_16;
        const wp::int32 var_17 = 0;
        wp::shape_t* var_18;
        wp::int32 var_19;
        wp::shape_t var_20;
        const wp::int32 var_21 = 4;
        wp::int32 var_22;
        wp::int32 var_23;
        bool* var_24;
        bool var_25;
        bool var_26;
        bool var_27;
        const wp::int32 var_28 = 0;
        wp::float64 var_29;
        wp::vec_t<3,wp::float64> var_30;
        wp::int32* var_31;
        const wp::int32 var_32 = 1;
        wp::int32 var_33;
        wp::int32* var_34;
        wp::range_t var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        wp::int32* var_39;
        wp::int32 var_40;
        wp::int32 var_41;
        const wp::int32 var_42 = 0;
        bool var_43;
        const wp::int32 var_44 = 0;
        wp::shape_t* var_45;
        wp::int32 var_46;
        wp::shape_t var_47;
        const wp::int32 var_48 = 4;
        wp::int32 var_49;
        bool var_50;
        const wp::int32 var_51 = 4;
        wp::int32 var_52;
        wp::int32 var_53;
        bool* var_54;
        bool var_55;
        bool var_56;
        wp::int32 var_57;
        bool var_58;
        const wp::int32 var_59 = 0;
        wp::shape_t* var_60;
        wp::int32 var_61;
        wp::shape_t var_62;
        const wp::int32 var_63 = 4;
        wp::int32 var_64;
        wp::int32 var_65;
        bool* var_66;
        bool var_67;
        bool var_68;
        wp::int32 var_69;
        bool var_70;
        bool var_71;
        const wp::int32 var_72 = 0;
        wp::float64 var_73;
        wp::mat_t<3,3,wp::float64> var_74;
        bool var_75;
        const wp::int32 var_76 = 3;
        wp::mat_t<3,3,wp::float64> var_77;
        //---------
        // forward
        // def project_system_kernel(                                                             <L 33>
        // row_idx = wp.tid()                                                                     <L 41>
        var_0 = builtin_tid1d();
        // project_row = wp.bool(0)                                                               <L 42>
        var_2 = bool(var_1);
        // if row_idx < y_target_reached.shape[0] * 4:                                            <L 43>
        var_4 = &(var_y_target_reached.shape);
        var_6 = wp::load(var_4);
        var_5 = wp::extract(var_6, var_3);
        var_8 = wp::mul(var_5, var_7);
        var_9 = (var_0 < var_8);
        if (var_9) {
            // body_id = wp.int32(row_idx // 4)                                                   <L 44>
            var_11 = wp::floordiv(var_0, var_10);
            var_12 = wp::int32(var_11);
            // project_row = y_target_reached[body_id]                                            <L 45>
            var_13 = wp::address(var_y_target_reached, var_12);
            var_15 = wp::load(var_13);
            var_14 = wp::copy(var_15);
        }
        var_16 = wp::where(var_9, var_14, var_2);
        if (!var_9) {
            // soft_dof = row_idx - y_target_reached.shape[0] * 4                                 <L 47>
            var_18 = &(var_y_target_reached.shape);
            var_20 = wp::load(var_18);
            var_19 = wp::extract(var_20, var_17);
            var_22 = wp::mul(var_19, var_21);
            var_23 = wp::sub(var_0, var_22);
            // project_row = x_target_reached[soft_dof]                                           <L 48>
            var_24 = wp::address(var_x_target_reached, var_23);
            var_26 = wp::load(var_24);
            var_25 = wp::copy(var_26);
        }
        var_27 = wp::where(var_9, var_16, var_25);
        // if project_row:                                                                        <L 49>
        if (var_27) {
            // gradient[row_idx] = wp.vec3d(wp.float64(0))                                        <L 50>
            var_29 = wp::float64(var_28);
            var_30 = wp::vec_t<3,wp::float64>(var_29);
            wp::array_store(var_gradient, var_0, var_30);
        }
        // for i in range(offsets[row_idx], offsets[row_idx + 1]):                                <L 51>
        var_31 = wp::address(var_offsets, var_0);
        var_33 = wp::add(var_0, var_32);
        var_34 = wp::address(var_offsets, var_33);
        var_36 = wp::load(var_31);
        var_37 = wp::load(var_34);
        var_35 = wp::range(var_36, var_37);
        start_for_0:;
            if (iter_cmp(var_35) == 0) goto end_for_0;
            var_38 = wp::iter_next(var_35);
            // col_idx = columns[i]                                                               <L 52>
            var_39 = wp::address(var_columns, var_38);
            var_41 = wp::load(var_39);
            var_40 = wp::copy(var_41);
            // project_col = wp.bool(0)                                                           <L 53>
            var_43 = bool(var_42);
            // if col_idx < y_target_reached.shape[0] * 4:                                        <L 54>
            var_45 = &(var_y_target_reached.shape);
            var_47 = wp::load(var_45);
            var_46 = wp::extract(var_47, var_44);
            var_49 = wp::mul(var_46, var_48);
            var_50 = (var_40 < var_49);
            if (var_50) {
                // body_id = wp.int32(col_idx // 4)                                               <L 55>
                var_52 = wp::floordiv(var_40, var_51);
                var_53 = wp::int32(var_52);
                // project_col = y_target_reached[body_id]                                        <L 56>
                var_54 = wp::address(var_y_target_reached, var_53);
                var_56 = wp::load(var_54);
                var_55 = wp::copy(var_56);
            }
            var_57 = wp::where(var_50, var_53, var_12);
            var_58 = wp::where(var_50, var_55, var_43);
            if (!var_50) {
                // soft_dof = col_idx - y_target_reached.shape[0] * 4                             <L 58>
                var_60 = &(var_y_target_reached.shape);
                var_62 = wp::load(var_60);
                var_61 = wp::extract(var_62, var_59);
                var_64 = wp::mul(var_61, var_63);
                var_65 = wp::sub(var_40, var_64);
                // project_col = x_target_reached[soft_dof]                                       <L 59>
                var_66 = wp::address(var_x_target_reached, var_65);
                var_68 = wp::load(var_66);
                var_67 = wp::copy(var_68);
            }
            var_69 = wp::where(var_50, var_23, var_65);
            var_70 = wp::where(var_50, var_58, var_67);
            // if project_row or project_col:                                                     <L 60>
            var_71 = var_27 || var_70;
            if (var_71) {
                // values[i] = wp.mat33d(wp.float64(0))                                           <L 61>
                var_73 = wp::float64(var_72);
                var_74 = wp::mat_t<3,3,wp::float64>(var_73);
                wp::array_store(var_values, var_38, var_74);
                // if row_idx == col_idx:                                                         <L 62>
                var_75 = (var_0 == var_40);
                if (var_75) {
                    // values[i] = wp.identity(n=3, dtype=wp.float64)                             <L 63>
                    var_77 = wp::identity<3,wp::float64>();
                    wp::array_store(var_values, var_38, var_77);
                }
            }
            wp::assign(var_12, var_57);
            wp::assign(var_23, var_69);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void project_system_kernel_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_offsets,
    wp::array_t<wp::int32> var_columns,
    wp::array_t<wp::mat_t<3,3,wp::float64>> var_values,
    wp::array_t<wp::vec_t<3,wp::float64>> var_gradient,
    wp::array_t<bool> var_y_target_reached,
    wp::array_t<bool> var_x_target_reached,
    wp::array_t<wp::int32> adj_offsets,
    wp::array_t<wp::int32> adj_columns,
    wp::array_t<wp::mat_t<3,3,wp::float64>> adj_values,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_gradient,
    wp::array_t<bool> adj_y_target_reached,
    wp::array_t<bool> adj_x_target_reached)
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
        const wp::int32 var_1 = 0;
        bool var_2;
        const wp::int32 var_3 = 0;
        wp::shape_t* var_4;
        wp::int32 var_5;
        wp::shape_t var_6;
        const wp::int32 var_7 = 4;
        wp::int32 var_8;
        bool var_9;
        const wp::int32 var_10 = 4;
        wp::int32 var_11;
        wp::int32 var_12;
        bool* var_13;
        bool var_14;
        bool var_15;
        bool var_16;
        const wp::int32 var_17 = 0;
        wp::shape_t* var_18;
        wp::int32 var_19;
        wp::shape_t var_20;
        const wp::int32 var_21 = 4;
        wp::int32 var_22;
        wp::int32 var_23;
        bool* var_24;
        bool var_25;
        bool var_26;
        bool var_27;
        const wp::int32 var_28 = 0;
        wp::float64 var_29;
        wp::vec_t<3,wp::float64> var_30;
        wp::int32* var_31;
        const wp::int32 var_32 = 1;
        wp::int32 var_33;
        wp::int32* var_34;
        wp::range_t var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        wp::int32* var_39;
        wp::int32 var_40;
        wp::int32 var_41;
        const wp::int32 var_42 = 0;
        bool var_43;
        const wp::int32 var_44 = 0;
        wp::shape_t* var_45;
        wp::int32 var_46;
        wp::shape_t var_47;
        const wp::int32 var_48 = 4;
        wp::int32 var_49;
        bool var_50;
        const wp::int32 var_51 = 4;
        wp::int32 var_52;
        wp::int32 var_53;
        bool* var_54;
        bool var_55;
        bool var_56;
        wp::int32 var_57;
        bool var_58;
        const wp::int32 var_59 = 0;
        wp::shape_t* var_60;
        wp::int32 var_61;
        wp::shape_t var_62;
        const wp::int32 var_63 = 4;
        wp::int32 var_64;
        wp::int32 var_65;
        bool* var_66;
        bool var_67;
        bool var_68;
        wp::int32 var_69;
        bool var_70;
        bool var_71;
        const wp::int32 var_72 = 0;
        wp::float64 var_73;
        wp::mat_t<3,3,wp::float64> var_74;
        bool var_75;
        const wp::int32 var_76 = 3;
        wp::mat_t<3,3,wp::float64> var_77;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        wp::int32 adj_3 = {};
        wp::shape_t adj_4 = {};
        wp::int32 adj_5 = {};
        wp::shape_t adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        bool adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        bool adj_13 = {};
        bool adj_14 = {};
        bool adj_15 = {};
        bool adj_16 = {};
        wp::int32 adj_17 = {};
        wp::shape_t adj_18 = {};
        wp::int32 adj_19 = {};
        wp::shape_t adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        bool adj_24 = {};
        bool adj_25 = {};
        bool adj_26 = {};
        bool adj_27 = {};
        wp::int32 adj_28 = {};
        wp::float64 adj_29 = {};
        wp::vec_t<3,wp::float64> adj_30 = {};
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::range_t adj_35 = {};
        wp::int32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::int32 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::int32 adj_42 = {};
        bool adj_43 = {};
        wp::int32 adj_44 = {};
        wp::shape_t adj_45 = {};
        wp::int32 adj_46 = {};
        wp::shape_t adj_47 = {};
        wp::int32 adj_48 = {};
        wp::int32 adj_49 = {};
        bool adj_50 = {};
        wp::int32 adj_51 = {};
        wp::int32 adj_52 = {};
        wp::int32 adj_53 = {};
        bool adj_54 = {};
        bool adj_55 = {};
        bool adj_56 = {};
        wp::int32 adj_57 = {};
        bool adj_58 = {};
        wp::int32 adj_59 = {};
        wp::shape_t adj_60 = {};
        wp::int32 adj_61 = {};
        wp::shape_t adj_62 = {};
        wp::int32 adj_63 = {};
        wp::int32 adj_64 = {};
        wp::int32 adj_65 = {};
        bool adj_66 = {};
        bool adj_67 = {};
        bool adj_68 = {};
        wp::int32 adj_69 = {};
        bool adj_70 = {};
        bool adj_71 = {};
        wp::int32 adj_72 = {};
        wp::float64 adj_73 = {};
        wp::mat_t<3,3,wp::float64> adj_74 = {};
        bool adj_75 = {};
        wp::int32 adj_76 = {};
        wp::mat_t<3,3,wp::float64> adj_77 = {};
        //---------
        // forward
        // def project_system_kernel(                                                             <L 33>
        // row_idx = wp.tid()                                                                     <L 41>
        var_0 = builtin_tid1d();
        // project_row = wp.bool(0)                                                               <L 42>
        var_2 = bool(var_1);
        // if row_idx < y_target_reached.shape[0] * 4:                                            <L 43>
        var_4 = &(var_y_target_reached.shape);
        var_6 = wp::load(var_4);
        var_5 = wp::extract(var_6, var_3);
        var_8 = wp::mul(var_5, var_7);
        var_9 = (var_0 < var_8);
        if (var_9) {
            // body_id = wp.int32(row_idx // 4)                                                   <L 44>
            var_11 = wp::floordiv(var_0, var_10);
            var_12 = wp::int32(var_11);
            // project_row = y_target_reached[body_id]                                            <L 45>
            var_13 = wp::address(var_y_target_reached, var_12);
            var_15 = wp::load(var_13);
            var_14 = wp::copy(var_15);
        }
        var_16 = wp::where(var_9, var_14, var_2);
        if (!var_9) {
            // soft_dof = row_idx - y_target_reached.shape[0] * 4                                 <L 47>
            var_18 = &(var_y_target_reached.shape);
            var_20 = wp::load(var_18);
            var_19 = wp::extract(var_20, var_17);
            var_22 = wp::mul(var_19, var_21);
            var_23 = wp::sub(var_0, var_22);
            // project_row = x_target_reached[soft_dof]                                           <L 48>
            var_24 = wp::address(var_x_target_reached, var_23);
            var_26 = wp::load(var_24);
            var_25 = wp::copy(var_26);
        }
        var_27 = wp::where(var_9, var_16, var_25);
        // if project_row:                                                                        <L 49>
        if (var_27) {
            // gradient[row_idx] = wp.vec3d(wp.float64(0))                                        <L 50>
            var_29 = wp::float64(var_28);
            var_30 = wp::vec_t<3,wp::float64>(var_29);
            // wp::array_store(var_gradient, var_0, var_30);
        }
        // for i in range(offsets[row_idx], offsets[row_idx + 1]):                                <L 51>
        var_31 = wp::address(var_offsets, var_0);
        var_33 = wp::add(var_0, var_32);
        var_34 = wp::address(var_offsets, var_33);
        var_36 = wp::load(var_31);
        var_37 = wp::load(var_34);
        var_35 = wp::range(var_36, var_37);
        //---------
        // reverse
        var_35 = wp::iter_reverse(var_35);
        start_for_0:;
            if (iter_cmp(var_35) == 0) goto end_for_0;
            var_38 = wp::iter_next(var_35);
        	adj_39 = {};
        	adj_40 = {};
        	adj_41 = {};
        	adj_42 = {};
        	adj_43 = {};
        	adj_44 = {};
        	adj_45 = {};
        	adj_46 = {};
        	adj_47 = {};
        	adj_48 = {};
        	adj_49 = {};
        	adj_50 = {};
        	adj_51 = {};
        	adj_52 = {};
        	adj_53 = {};
        	adj_54 = {};
        	adj_55 = {};
        	adj_56 = {};
        	adj_57 = {};
        	adj_58 = {};
        	adj_59 = {};
        	adj_60 = {};
        	adj_61 = {};
        	adj_62 = {};
        	adj_63 = {};
        	adj_64 = {};
        	adj_65 = {};
        	adj_66 = {};
        	adj_67 = {};
        	adj_68 = {};
        	adj_69 = {};
        	adj_70 = {};
        	adj_71 = {};
        	adj_72 = {};
        	adj_73 = {};
        	adj_74 = {};
        	adj_75 = {};
        	adj_76 = {};
        	adj_77 = {};
            // col_idx = columns[i]                                                               <L 52>
            var_39 = wp::address(var_columns, var_38);
            var_41 = wp::load(var_39);
            var_40 = wp::copy(var_41);
            // project_col = wp.bool(0)                                                           <L 53>
            var_43 = bool(var_42);
            // if col_idx < y_target_reached.shape[0] * 4:                                        <L 54>
            var_45 = &(var_y_target_reached.shape);
            var_47 = wp::load(var_45);
            var_46 = wp::extract(var_47, var_44);
            var_49 = wp::mul(var_46, var_48);
            var_50 = (var_40 < var_49);
            if (var_50) {
                // body_id = wp.int32(col_idx // 4)                                               <L 55>
                var_52 = wp::floordiv(var_40, var_51);
                var_53 = wp::int32(var_52);
                // project_col = y_target_reached[body_id]                                        <L 56>
                var_54 = wp::address(var_y_target_reached, var_53);
                var_56 = wp::load(var_54);
                var_55 = wp::copy(var_56);
            }
            var_57 = wp::where(var_50, var_53, var_12);
            var_58 = wp::where(var_50, var_55, var_43);
            if (!var_50) {
                // soft_dof = col_idx - y_target_reached.shape[0] * 4                             <L 58>
                var_60 = &(var_y_target_reached.shape);
                var_62 = wp::load(var_60);
                var_61 = wp::extract(var_62, var_59);
                var_64 = wp::mul(var_61, var_63);
                var_65 = wp::sub(var_40, var_64);
                // project_col = x_target_reached[soft_dof]                                       <L 59>
                var_66 = wp::address(var_x_target_reached, var_65);
                var_68 = wp::load(var_66);
                var_67 = wp::copy(var_68);
            }
            var_69 = wp::where(var_50, var_23, var_65);
            var_70 = wp::where(var_50, var_58, var_67);
            // if project_row or project_col:                                                     <L 60>
            var_71 = var_27 || var_70;
            if (var_71) {
                // values[i] = wp.mat33d(wp.float64(0))                                           <L 61>
                var_73 = wp::float64(var_72);
                var_74 = wp::mat_t<3,3,wp::float64>(var_73);
                // wp::array_store(var_values, var_38, var_74);
                // if row_idx == col_idx:                                                         <L 62>
                var_75 = (var_0 == var_40);
                if (var_75) {
                    // values[i] = wp.identity(n=3, dtype=wp.float64)                             <L 63>
                    var_77 = wp::identity<3,wp::float64>();
                    // wp::array_store(var_values, var_38, var_77);
                }
            }
            wp::assign(var_12, var_57);
            wp::assign(var_23, var_69);
            wp::adj_assign(var_23, var_69, adj_23, adj_69);
            wp::adj_assign(var_12, var_57, adj_12, adj_57);
            if (var_71) {
                if (var_75) {
                    wp::adj_array_store(var_values, var_38, var_77, adj_values, adj_38, adj_77);
                    // adj: values[i] = wp.identity(n=3, dtype=wp.float64)                        <L 63>
                }
                // adj: if row_idx == col_idx:                                                    <L 62>
                wp::adj_array_store(var_values, var_38, var_74, adj_values, adj_38, adj_74);
                wp::adj_mat_t(var_73, adj_73, adj_74);
                wp::adj_float64(var_72, adj_72, adj_73);
                // adj: values[i] = wp.mat33d(wp.float64(0))                                      <L 61>
            }
            // adj: if project_row or project_col:                                                <L 60>
            wp::adj_where(var_50, var_58, var_67, adj_50, adj_58, adj_67, adj_70);
            wp::adj_where(var_50, var_23, var_65, adj_50, adj_23, adj_65, adj_69);
            if (!var_50) {
                wp::adj_copy(var_68, adj_66, adj_67);
                wp::adj_load(var_66, adj_66, adj_68);
                wp::adj_address(var_x_target_reached, var_65, adj_x_target_reached, adj_65, adj_66);
                // adj: project_col = x_target_reached[soft_dof]                                  <L 59>
                wp::adj_sub(var_40, var_64, adj_40, adj_64, adj_65);
                wp::adj_mul(var_61, var_63, adj_61, adj_63, adj_64);
                wp::adj_extract(var_62, var_59, adj_60, adj_59, adj_61);
                wp::adj_load(var_60, adj_60, adj_62);
                adj_y_target_reached.shape = adj_60;
                // adj: soft_dof = col_idx - y_target_reached.shape[0] * 4                        <L 58>
            }
            wp::adj_where(var_50, var_55, var_43, adj_50, adj_55, adj_43, adj_58);
            wp::adj_where(var_50, var_53, var_12, adj_50, adj_53, adj_12, adj_57);
            if (var_50) {
                wp::adj_copy(var_56, adj_54, adj_55);
                wp::adj_load(var_54, adj_54, adj_56);
                wp::adj_address(var_y_target_reached, var_53, adj_y_target_reached, adj_53, adj_54);
                // adj: project_col = y_target_reached[body_id]                                   <L 56>
                wp::adj_int32(var_52, adj_52, adj_53);
                wp::adj_floordiv(var_40, var_51, adj_40, adj_51, adj_52);
                // adj: body_id = wp.int32(col_idx // 4)                                          <L 55>
            }
            wp::adj_mul(var_46, var_48, adj_46, adj_48, adj_49);
            wp::adj_extract(var_47, var_44, adj_45, adj_44, adj_46);
            wp::adj_load(var_45, adj_45, adj_47);
            adj_y_target_reached.shape = adj_45;
            // adj: if col_idx < y_target_reached.shape[0] * 4:                                   <L 54>
            adj_bool(var_42, adj_42, adj_43);
            // adj: project_col = wp.bool(0)                                                      <L 53>
            wp::adj_copy(var_41, adj_39, adj_40);
            wp::adj_load(var_39, adj_39, adj_41);
            wp::adj_address(var_columns, var_38, adj_columns, adj_38, adj_39);
            // adj: col_idx = columns[i]                                                          <L 52>
        	goto start_for_0;
        end_for_0:;
        wp::adj_range(var_36, var_37, adj_31, adj_34, adj_35);
        wp::adj_load(var_34, adj_34, adj_37);
        wp::adj_load(var_31, adj_31, adj_36);
        wp::adj_address(var_offsets, var_33, adj_offsets, adj_33, adj_34);
        wp::adj_add(var_0, var_32, adj_0, adj_32, adj_33);
        wp::adj_address(var_offsets, var_0, adj_offsets, adj_0, adj_31);
        // adj: for i in range(offsets[row_idx], offsets[row_idx + 1]):                           <L 51>
        if (var_27) {
            wp::adj_array_store(var_gradient, var_0, var_30, adj_gradient, adj_0, adj_30);
            wp::adj_vec_t(var_29, adj_29, adj_30);
            wp::adj_float64(var_28, adj_28, adj_29);
            // adj: gradient[row_idx] = wp.vec3d(wp.float64(0))                                   <L 50>
        }
        // adj: if project_row:                                                                   <L 49>
        wp::adj_where(var_9, var_16, var_25, adj_9, adj_16, adj_25, adj_27);
        if (!var_9) {
            wp::adj_copy(var_26, adj_24, adj_25);
            wp::adj_load(var_24, adj_24, adj_26);
            wp::adj_address(var_x_target_reached, var_23, adj_x_target_reached, adj_23, adj_24);
            // adj: project_row = x_target_reached[soft_dof]                                      <L 48>
            wp::adj_sub(var_0, var_22, adj_0, adj_22, adj_23);
            wp::adj_mul(var_19, var_21, adj_19, adj_21, adj_22);
            wp::adj_extract(var_20, var_17, adj_18, adj_17, adj_19);
            wp::adj_load(var_18, adj_18, adj_20);
            adj_y_target_reached.shape = adj_18;
            // adj: soft_dof = row_idx - y_target_reached.shape[0] * 4                            <L 47>
        }
        wp::adj_where(var_9, var_14, var_2, adj_9, adj_14, adj_2, adj_16);
        if (var_9) {
            wp::adj_copy(var_15, adj_13, adj_14);
            wp::adj_load(var_13, adj_13, adj_15);
            wp::adj_address(var_y_target_reached, var_12, adj_y_target_reached, adj_12, adj_13);
            // adj: project_row = y_target_reached[body_id]                                       <L 45>
            wp::adj_int32(var_11, adj_11, adj_12);
            wp::adj_floordiv(var_0, var_10, adj_0, adj_10, adj_11);
            // adj: body_id = wp.int32(row_idx // 4)                                              <L 44>
        }
        wp::adj_mul(var_5, var_7, adj_5, adj_7, adj_8);
        wp::adj_extract(var_6, var_3, adj_4, adj_3, adj_5);
        wp::adj_load(var_4, adj_4, adj_6);
        adj_y_target_reached.shape = adj_4;
        // adj: if row_idx < y_target_reached.shape[0] * 4:                                       <L 43>
        adj_bool(var_1, adj_1, adj_2);
        // adj: project_row = wp.bool(0)                                                          <L 42>
        // adj: row_idx = wp.tid()                                                                <L 41>
        // adj: def project_system_kernel(                                                        <L 33>
        continue;
    }
}

