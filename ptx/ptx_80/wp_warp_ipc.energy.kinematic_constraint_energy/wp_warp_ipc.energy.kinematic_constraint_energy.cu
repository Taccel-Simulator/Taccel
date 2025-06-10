
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

struct COOMatrix3x3_0df4b45d
{
    wp::array_t<wp::int32> rows;
    wp::array_t<wp::int32> cols;
    wp::array_t<wp::mat_t<3,3,wp::float64>> vals;
    wp::int32 capacity;
    wp::int32 size;
    wp::int32 n_rows;
    wp::int32 n_cols;


    COOMatrix3x3_0df4b45d() = default;
    CUDA_CALLABLE COOMatrix3x3_0df4b45d(wp::array_t<wp::int32> const& rows,
    wp::array_t<wp::int32> const& cols = {},
    wp::array_t<wp::mat_t<3,3,wp::float64>> const& vals = {},
    wp::int32 const& capacity = {},
    wp::int32 const& size = {},
    wp::int32 const& n_rows = {},
    wp::int32 const& n_cols = {})
        : rows{rows}
        , cols{cols}
        , vals{vals}
        , capacity{capacity}
        , size{size}
        , n_rows{n_rows}
        , n_cols{n_cols}

    {
    }

    CUDA_CALLABLE COOMatrix3x3_0df4b45d& operator += (const COOMatrix3x3_0df4b45d& rhs)
    {    capacity += rhs.capacity;
    size += rhs.size;
    n_rows += rhs.n_rows;
    n_cols += rhs.n_cols;

        return *this;}

};

static CUDA_CALLABLE void adj_COOMatrix3x3_0df4b45d(wp::array_t<wp::int32> const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::mat_t<3,3,wp::float64>> const&,
    wp::int32 const&,
    wp::int32 const&,
    wp::int32 const&,
    wp::int32 const&,
    wp::array_t<wp::int32> & adj_rows,
    wp::array_t<wp::int32> & adj_cols,
    wp::array_t<wp::mat_t<3,3,wp::float64>> & adj_vals,
    wp::int32 & adj_capacity,
    wp::int32 & adj_size,
    wp::int32 & adj_n_rows,
    wp::int32 & adj_n_cols,
    COOMatrix3x3_0df4b45d & adj_ret)
{
    adj_rows = adj_ret.rows;
    adj_cols = adj_ret.cols;
    adj_vals = adj_ret.vals;
    adj_capacity += adj_ret.capacity;
    adj_size += adj_ret.size;
    adj_n_rows += adj_ret.n_rows;
    adj_n_cols += adj_ret.n_cols;
}

CUDA_CALLABLE void adj_atomic_add(COOMatrix3x3_0df4b45d* p, COOMatrix3x3_0df4b45d t)
{
    wp::adj_atomic_add(&p->rows, t.rows);
    wp::adj_atomic_add(&p->cols, t.cols);
    wp::adj_atomic_add(&p->vals, t.vals);
    wp::adj_atomic_add(&p->capacity, t.capacity);
    wp::adj_atomic_add(&p->size, t.size);
    wp::adj_atomic_add(&p->n_rows, t.n_rows);
    wp::adj_atomic_add(&p->n_cols, t.n_cols);
}



// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/utils/wp_math.py:145
static CUDA_CALLABLE wp::float64 sqr_0(
    wp::float64 var_x)
{
    //---------
    // primal vars
    wp::float64 var_0;
    //---------
    // forward
    // def sqr(x: wp.float64):                                                                <L 146>
    // return x * x                                                                           <L 147>
    var_0 = wp::mul(var_x, var_x);
    return var_0;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/utils/wp_math.py:145
static CUDA_CALLABLE void adj_sqr_0(
    wp::float64 var_x,
    wp::float64 & adj_x,
    wp::float64 & adj_ret)
{
    //---------
    // primal vars
    wp::float64 var_0;
    //---------
    // dual vars
    wp::float64 adj_0 = {};
    //---------
    // forward
    // def sqr(x: wp.float64):                                                                <L 146>
    // return x * x                                                                           <L 147>
    var_0 = wp::mul(var_x, var_x);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_0 += adj_ret;
    wp::adj_mul(var_x, var_x, adj_x, adj_x, adj_0);
    // adj: return x * x                                                                      <L 147>
    // adj: def sqr(x: wp.float64):                                                           <L 146>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/utils/matrix.py:152
static CUDA_CALLABLE void COOMatrix3x3_atomic_add_0(
    COOMatrix3x3_0df4b45d var_mat,
    wp::int32 var_block_index,
    wp::mat_t<3,3,wp::float64> var_v)
{
    //---------
    // primal vars
    const bool var_0 = true;
    const wp::int32 var_1 = 0;
    bool var_2;
    wp::int32* var_3;
    bool var_4;
    wp::int32 var_5;
    bool var_6;
    const wp::str var_7 = "[COOMatrix OFB Error]\tblock_index: %d, size: %d!!!!!\n";
    wp::int32* var_8;
    wp::int32 var_9;
    wp::array_t<wp::mat_t<3,3,wp::float64>>* var_10;
    wp::mat_t<3,3,wp::float64> var_11;
    wp::array_t<wp::mat_t<3,3,wp::float64>> var_12;
    //---------
    // forward
    // def COOMatrix3x3_atomic_add(mat: COOMatrix3x3, block_index: wp.int32, v: wp.mat33d):       <L 153>
    // if ENABLE_COO_OFB:                                                                     <L 154>
    // if block_index < 0 or block_index >= mat.size:                                         <L 155>
    var_2 = (var_block_index < var_1);
    var_3 = &(var_mat.size);
    var_5 = wp::load(var_3);
    var_4 = (var_block_index >= var_5);
    var_6 = var_2 || var_4;
    if (var_6) {
        // wp.printf(                                                                         <L 156>
        // "[COOMatrix OFB Error]\tblock_index: %d, size: %d!!!!!\n",                         <L 157>
        // block_index,                                                                       <L 158>
        // mat.size,                                                                          <L 159>
        var_8 = &(var_mat.size);
        var_9 = wp::load(var_8);
        printf(var_7, var_block_index, var_9);
        // return                                                                             <L 161>
        return;
    }
    // wp.atomic_add(mat.vals, block_index, v)                                                <L 162>
    var_10 = &(var_mat.vals);
    var_12 = wp::load(var_10);
    var_11 = wp::atomic_add(var_12, var_block_index, var_v);
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/utils/matrix.py:152
static CUDA_CALLABLE void adj_COOMatrix3x3_atomic_add_0(
    COOMatrix3x3_0df4b45d var_mat,
    wp::int32 var_block_index,
    wp::mat_t<3,3,wp::float64> var_v,
    COOMatrix3x3_0df4b45d & adj_mat,
    wp::int32 & adj_block_index,
    wp::mat_t<3,3,wp::float64> & adj_v)
{
    //---------
    // primal vars
    const bool var_0 = true;
    const wp::int32 var_1 = 0;
    bool var_2;
    wp::int32* var_3;
    bool var_4;
    wp::int32 var_5;
    bool var_6;
    const wp::str var_7 = "[COOMatrix OFB Error]\tblock_index: %d, size: %d!!!!!\n";
    wp::int32* var_8;
    wp::int32 var_9;
    wp::array_t<wp::mat_t<3,3,wp::float64>>* var_10;
    wp::mat_t<3,3,wp::float64> var_11;
    wp::array_t<wp::mat_t<3,3,wp::float64>> var_12;
    //---------
    // dual vars
    bool adj_0 = {};
    wp::int32 adj_1 = {};
    bool adj_2 = {};
    wp::int32 adj_3 = {};
    bool adj_4 = {};
    wp::int32 adj_5 = {};
    bool adj_6 = {};
    wp::str adj_7 = {};
    wp::int32 adj_8 = {};
    wp::int32 adj_9 = {};
    wp::array_t<wp::mat_t<3,3,wp::float64>> adj_10 = {};
    wp::mat_t<3,3,wp::float64> adj_11 = {};
    wp::array_t<wp::mat_t<3,3,wp::float64>> adj_12 = {};
    //---------
    // forward
    // def COOMatrix3x3_atomic_add(mat: COOMatrix3x3, block_index: wp.int32, v: wp.mat33d):       <L 153>
    // if ENABLE_COO_OFB:                                                                     <L 154>
    // if block_index < 0 or block_index >= mat.size:                                         <L 155>
    var_2 = (var_block_index < var_1);
    var_3 = &(var_mat.size);
    var_5 = wp::load(var_3);
    var_4 = (var_block_index >= var_5);
    var_6 = var_2 || var_4;
    if (var_6) {
        // wp.printf(                                                                         <L 156>
        // "[COOMatrix OFB Error]\tblock_index: %d, size: %d!!!!!\n",                         <L 157>
        // block_index,                                                                       <L 158>
        // mat.size,                                                                          <L 159>
        var_8 = &(var_mat.size);
        var_9 = wp::load(var_8);
        printf(var_7, var_block_index, var_9);
        // return                                                                             <L 161>
        goto label0;
    }
    // wp.atomic_add(mat.vals, block_index, v)                                                <L 162>
    var_10 = &(var_mat.vals);
    var_12 = wp::load(var_10);
    // var_11 = wp::atomic_add(var_12, var_block_index, var_v);
    //---------
    // reverse
    wp::adj_atomic_add(var_12, var_block_index, var_v, adj_10, adj_block_index, adj_v, adj_11);
    wp::adj_load(var_10, adj_10, adj_12);
    adj_mat.vals = adj_10;
    // adj: wp.atomic_add(mat.vals, block_index, v)                                           <L 162>
    if (var_6) {
        label0:;
        // adj: return                                                                        <L 161>
        adj_printf(var_7, var_block_index, var_9, adj_7, adj_block_index, adj_8);
        wp::adj_load(var_8, adj_8, adj_9);
        adj_mat.size = adj_8;
        // adj: mat.size,                                                                     <L 159>
        // adj: block_index,                                                                  <L 158>
        // adj: "[COOMatrix OFB Error]\tblock_index: %d, size: %d!!!!!\n",                    <L 157>
        // adj: wp.printf(                                                                    <L 156>
    }
    wp::adj_load(var_3, adj_3, adj_5);
    adj_mat.size = adj_3;
    // adj: if block_index < 0 or block_index >= mat.size:                                    <L 155>
    // adj: if ENABLE_COO_OFB:                                                                <L 154>
    // adj: def COOMatrix3x3_atomic_add(mat: COOMatrix3x3, block_index: wp.int32, v: wp.mat33d):  <L 153>
    return;
}



extern "C" __global__ void compute_soft_kinematic_energy_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float64>> var_x,
    wp::array_t<bool> var_soft_has_constraint,
    wp::array_t<wp::vec_t<3,wp::float64>> var_soft_target_dof,
    wp::float64 var_weight,
    wp::array_t<wp::float64> var_soft_energy,
    wp::int32 var_affine_verts_num,
    wp::array_t<wp::float64> var_soft_verts_mass)
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
        bool* var_1;
        bool var_2;
        bool var_3;
        wp::float64* var_4;
        wp::float64 var_5;
        wp::float64 var_6;
        const wp::float32 var_7 = 0.5;
        wp::float64 var_8;
        wp::float64 var_9;
        wp::float64 var_10;
        wp::int32 var_11;
        wp::vec_t<3,wp::float64>* var_12;
        wp::vec_t<3,wp::float64>* var_13;
        wp::vec_t<3,wp::float64> var_14;
        wp::vec_t<3,wp::float64> var_15;
        wp::vec_t<3,wp::float64> var_16;
        wp::float64 var_17;
        wp::float64 var_18;
        wp::float64 var_19;
        //---------
        // forward
        // def compute_soft_kinematic_energy(                                                     <L 86>
        // tid = wp.tid()                                                                         <L 95>
        var_0 = builtin_tid1d();
        // has_constraint = soft_has_constraint[tid]                                              <L 96>
        var_1 = wp::address(var_soft_has_constraint, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // mass = soft_verts_mass[tid]                                                            <L 97>
        var_4 = wp::address(var_soft_verts_mass, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // if has_constraint:                                                                     <L 98>
        if (var_2) {
            // soft_energy[tid] += wp.float64(0.5) * weight * mass * wp.length_sq(x[tid + affine_verts_num] - soft_target_dof[tid])       <L 99>
            var_8 = wp::float64(var_7);
            var_9 = wp::mul(var_8, var_weight);
            var_10 = wp::mul(var_9, var_5);
            var_11 = wp::add(var_0, var_affine_verts_num);
            var_12 = wp::address(var_x, var_11);
            var_13 = wp::address(var_soft_target_dof, var_0);
            var_15 = wp::load(var_12);
            var_16 = wp::load(var_13);
            var_14 = wp::sub(var_15, var_16);
            var_17 = wp::length_sq(var_14);
            var_18 = wp::mul(var_10, var_17);
            var_19 = wp::atomic_add(var_soft_energy, var_0, var_18);
        }
    }
}



extern "C" __global__ void compute_soft_kinematic_energy_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float64>> var_x,
    wp::array_t<bool> var_soft_has_constraint,
    wp::array_t<wp::vec_t<3,wp::float64>> var_soft_target_dof,
    wp::float64 var_weight,
    wp::array_t<wp::float64> var_soft_energy,
    wp::int32 var_affine_verts_num,
    wp::array_t<wp::float64> var_soft_verts_mass,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_x,
    wp::array_t<bool> adj_soft_has_constraint,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_soft_target_dof,
    wp::float64 adj_weight,
    wp::array_t<wp::float64> adj_soft_energy,
    wp::int32 adj_affine_verts_num,
    wp::array_t<wp::float64> adj_soft_verts_mass)
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
        bool* var_1;
        bool var_2;
        bool var_3;
        wp::float64* var_4;
        wp::float64 var_5;
        wp::float64 var_6;
        const wp::float32 var_7 = 0.5;
        wp::float64 var_8;
        wp::float64 var_9;
        wp::float64 var_10;
        wp::int32 var_11;
        wp::vec_t<3,wp::float64>* var_12;
        wp::vec_t<3,wp::float64>* var_13;
        wp::vec_t<3,wp::float64> var_14;
        wp::vec_t<3,wp::float64> var_15;
        wp::vec_t<3,wp::float64> var_16;
        wp::float64 var_17;
        wp::float64 var_18;
        wp::float64 var_19;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        wp::float64 adj_4 = {};
        wp::float64 adj_5 = {};
        wp::float64 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::float64 adj_8 = {};
        wp::float64 adj_9 = {};
        wp::float64 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::vec_t<3,wp::float64> adj_12 = {};
        wp::vec_t<3,wp::float64> adj_13 = {};
        wp::vec_t<3,wp::float64> adj_14 = {};
        wp::vec_t<3,wp::float64> adj_15 = {};
        wp::vec_t<3,wp::float64> adj_16 = {};
        wp::float64 adj_17 = {};
        wp::float64 adj_18 = {};
        wp::float64 adj_19 = {};
        //---------
        // forward
        // def compute_soft_kinematic_energy(                                                     <L 86>
        // tid = wp.tid()                                                                         <L 95>
        var_0 = builtin_tid1d();
        // has_constraint = soft_has_constraint[tid]                                              <L 96>
        var_1 = wp::address(var_soft_has_constraint, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // mass = soft_verts_mass[tid]                                                            <L 97>
        var_4 = wp::address(var_soft_verts_mass, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // if has_constraint:                                                                     <L 98>
        if (var_2) {
            // soft_energy[tid] += wp.float64(0.5) * weight * mass * wp.length_sq(x[tid + affine_verts_num] - soft_target_dof[tid])       <L 99>
            var_8 = wp::float64(var_7);
            var_9 = wp::mul(var_8, var_weight);
            var_10 = wp::mul(var_9, var_5);
            var_11 = wp::add(var_0, var_affine_verts_num);
            var_12 = wp::address(var_x, var_11);
            var_13 = wp::address(var_soft_target_dof, var_0);
            var_15 = wp::load(var_12);
            var_16 = wp::load(var_13);
            var_14 = wp::sub(var_15, var_16);
            var_17 = wp::length_sq(var_14);
            var_18 = wp::mul(var_10, var_17);
            // var_19 = wp::atomic_add(var_soft_energy, var_0, var_18);
        }
        //---------
        // reverse
        if (var_2) {
            wp::adj_atomic_add(var_soft_energy, var_0, var_18, adj_soft_energy, adj_0, adj_18, adj_19);
            wp::adj_mul(var_10, var_17, adj_10, adj_17, adj_18);
            wp::adj_length_sq(var_14, adj_14, adj_17);
            wp::adj_sub(var_15, var_16, adj_12, adj_13, adj_14);
            wp::adj_load(var_13, adj_13, adj_16);
            wp::adj_load(var_12, adj_12, adj_15);
            wp::adj_address(var_soft_target_dof, var_0, adj_soft_target_dof, adj_0, adj_13);
            wp::adj_address(var_x, var_11, adj_x, adj_11, adj_12);
            wp::adj_add(var_0, var_affine_verts_num, adj_0, adj_affine_verts_num, adj_11);
            wp::adj_mul(var_9, var_5, adj_9, adj_5, adj_10);
            wp::adj_mul(var_8, var_weight, adj_8, adj_weight, adj_9);
            wp::adj_float64(var_7, adj_7, adj_8);
            // adj: soft_energy[tid] += wp.float64(0.5) * weight * mass * wp.length_sq(x[tid + affine_verts_num] - soft_target_dof[tid])  <L 99>
        }
        // adj: if has_constraint:                                                                <L 98>
        wp::adj_copy(var_6, adj_4, adj_5);
        wp::adj_load(var_4, adj_4, adj_6);
        wp::adj_address(var_soft_verts_mass, var_0, adj_soft_verts_mass, adj_0, adj_4);
        // adj: mass = soft_verts_mass[tid]                                                       <L 97>
        wp::adj_copy(var_3, adj_1, adj_2);
        wp::adj_load(var_1, adj_1, adj_3);
        wp::adj_address(var_soft_has_constraint, var_0, adj_soft_has_constraint, adj_0, adj_1);
        // adj: has_constraint = soft_has_constraint[tid]                                         <L 96>
        // adj: tid = wp.tid()                                                                    <L 95>
        // adj: def compute_soft_kinematic_energy(                                                <L 86>
        continue;
    }
}



extern "C" __global__ void compute_affine_kinematic_grad_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<12,wp::float64>> var_y,
    wp::array_t<bool> var_affine_has_constraint,
    wp::array_t<wp::vec_t<12,wp::float64>> var_affine_target_dof,
    wp::float64 var_weight,
    wp::array_t<wp::vec_t<12,wp::float64>> var_affine_grad,
    wp::array_t<wp::float64> var_mass_body,
    wp::array_t<wp::int32> var_body_env_id,
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
        wp::int32* var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        const wp::int32 var_4 = 1;
        bool var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        const wp::int32 var_10 = 2;
        bool var_11;
        wp::int32 var_12;
        bool var_13;
        wp::float64* var_14;
        wp::float64 var_15;
        wp::float64 var_16;
        bool* var_17;
        bool var_18;
        bool var_19;
        wp::float64 var_20;
        wp::vec_t<12,wp::float64>* var_21;
        wp::vec_t<12,wp::float64>* var_22;
        wp::vec_t<12,wp::float64> var_23;
        wp::vec_t<12,wp::float64> var_24;
        wp::vec_t<12,wp::float64> var_25;
        wp::vec_t<12,wp::float64> var_26;
        wp::vec_t<12,wp::float64> var_27;
        //---------
        // forward
        // def compute_affine_kinematic_grad(                                                     <L 143>
        // tid = wp.tid()                                                                         <L 153>
        var_0 = builtin_tid1d();
        // if env_states[body_env_id[tid]] == ENV_STATE_INVALID or env_states[body_env_id[tid]] == ENV_STATE_NEWTON_SOLVED:       <L 154>
        var_1 = wp::address(var_body_env_id, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::address(var_env_states, var_3);
        var_6 = wp::load(var_2);
        var_5 = (var_6 == var_4);
        var_7 = wp::address(var_body_env_id, var_0);
        var_9 = wp::load(var_7);
        var_8 = wp::address(var_env_states, var_9);
        var_12 = wp::load(var_8);
        var_11 = (var_12 == var_10);
        var_13 = var_5 || var_11;
        if (var_13) {
            // return                                                                             <L 155>
            continue;
        }
        // mass = mass_body[tid]                                                                  <L 156>
        var_14 = wp::address(var_mass_body, var_0);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // has_constraint = affine_has_constraint[tid]                                            <L 157>
        var_17 = wp::address(var_affine_has_constraint, var_0);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // if has_constraint:                                                                     <L 158>
        if (var_18) {
            // wp.atomic_add(affine_grad, tid, weight * mass * (y[tid] - affine_target_dof[tid]))       <L 159>
            var_20 = wp::mul(var_weight, var_15);
            var_21 = wp::address(var_y, var_0);
            var_22 = wp::address(var_affine_target_dof, var_0);
            var_24 = wp::load(var_21);
            var_25 = wp::load(var_22);
            var_23 = wp::sub(var_24, var_25);
            var_26 = wp::mul(var_20, var_23);
            var_27 = wp::atomic_add(var_affine_grad, var_0, var_26);
        }
    }
}



extern "C" __global__ void compute_affine_kinematic_grad_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<12,wp::float64>> var_y,
    wp::array_t<bool> var_affine_has_constraint,
    wp::array_t<wp::vec_t<12,wp::float64>> var_affine_target_dof,
    wp::float64 var_weight,
    wp::array_t<wp::vec_t<12,wp::float64>> var_affine_grad,
    wp::array_t<wp::float64> var_mass_body,
    wp::array_t<wp::int32> var_body_env_id,
    wp::array_t<wp::int32> var_env_states,
    wp::array_t<wp::vec_t<12,wp::float64>> adj_y,
    wp::array_t<bool> adj_affine_has_constraint,
    wp::array_t<wp::vec_t<12,wp::float64>> adj_affine_target_dof,
    wp::float64 adj_weight,
    wp::array_t<wp::vec_t<12,wp::float64>> adj_affine_grad,
    wp::array_t<wp::float64> adj_mass_body,
    wp::array_t<wp::int32> adj_body_env_id,
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
        wp::int32* var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        const wp::int32 var_4 = 1;
        bool var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        const wp::int32 var_10 = 2;
        bool var_11;
        wp::int32 var_12;
        bool var_13;
        wp::float64* var_14;
        wp::float64 var_15;
        wp::float64 var_16;
        bool* var_17;
        bool var_18;
        bool var_19;
        wp::float64 var_20;
        wp::vec_t<12,wp::float64>* var_21;
        wp::vec_t<12,wp::float64>* var_22;
        wp::vec_t<12,wp::float64> var_23;
        wp::vec_t<12,wp::float64> var_24;
        wp::vec_t<12,wp::float64> var_25;
        wp::vec_t<12,wp::float64> var_26;
        wp::vec_t<12,wp::float64> var_27;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        bool adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        bool adj_11 = {};
        wp::int32 adj_12 = {};
        bool adj_13 = {};
        wp::float64 adj_14 = {};
        wp::float64 adj_15 = {};
        wp::float64 adj_16 = {};
        bool adj_17 = {};
        bool adj_18 = {};
        bool adj_19 = {};
        wp::float64 adj_20 = {};
        wp::vec_t<12,wp::float64> adj_21 = {};
        wp::vec_t<12,wp::float64> adj_22 = {};
        wp::vec_t<12,wp::float64> adj_23 = {};
        wp::vec_t<12,wp::float64> adj_24 = {};
        wp::vec_t<12,wp::float64> adj_25 = {};
        wp::vec_t<12,wp::float64> adj_26 = {};
        wp::vec_t<12,wp::float64> adj_27 = {};
        //---------
        // forward
        // def compute_affine_kinematic_grad(                                                     <L 143>
        // tid = wp.tid()                                                                         <L 153>
        var_0 = builtin_tid1d();
        // if env_states[body_env_id[tid]] == ENV_STATE_INVALID or env_states[body_env_id[tid]] == ENV_STATE_NEWTON_SOLVED:       <L 154>
        var_1 = wp::address(var_body_env_id, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::address(var_env_states, var_3);
        var_6 = wp::load(var_2);
        var_5 = (var_6 == var_4);
        var_7 = wp::address(var_body_env_id, var_0);
        var_9 = wp::load(var_7);
        var_8 = wp::address(var_env_states, var_9);
        var_12 = wp::load(var_8);
        var_11 = (var_12 == var_10);
        var_13 = var_5 || var_11;
        if (var_13) {
            // return                                                                             <L 155>
            goto label0;
        }
        // mass = mass_body[tid]                                                                  <L 156>
        var_14 = wp::address(var_mass_body, var_0);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // has_constraint = affine_has_constraint[tid]                                            <L 157>
        var_17 = wp::address(var_affine_has_constraint, var_0);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // if has_constraint:                                                                     <L 158>
        if (var_18) {
            // wp.atomic_add(affine_grad, tid, weight * mass * (y[tid] - affine_target_dof[tid]))       <L 159>
            var_20 = wp::mul(var_weight, var_15);
            var_21 = wp::address(var_y, var_0);
            var_22 = wp::address(var_affine_target_dof, var_0);
            var_24 = wp::load(var_21);
            var_25 = wp::load(var_22);
            var_23 = wp::sub(var_24, var_25);
            var_26 = wp::mul(var_20, var_23);
            // var_27 = wp::atomic_add(var_affine_grad, var_0, var_26);
        }
        //---------
        // reverse
        if (var_18) {
            wp::adj_atomic_add(var_affine_grad, var_0, var_26, adj_affine_grad, adj_0, adj_26, adj_27);
            wp::adj_mul(var_20, var_23, adj_20, adj_23, adj_26);
            wp::adj_sub(var_24, var_25, adj_21, adj_22, adj_23);
            wp::adj_load(var_22, adj_22, adj_25);
            wp::adj_load(var_21, adj_21, adj_24);
            wp::adj_address(var_affine_target_dof, var_0, adj_affine_target_dof, adj_0, adj_22);
            wp::adj_address(var_y, var_0, adj_y, adj_0, adj_21);
            wp::adj_mul(var_weight, var_15, adj_weight, adj_15, adj_20);
            // adj: wp.atomic_add(affine_grad, tid, weight * mass * (y[tid] - affine_target_dof[tid]))  <L 159>
        }
        // adj: if has_constraint:                                                                <L 158>
        wp::adj_copy(var_19, adj_17, adj_18);
        wp::adj_load(var_17, adj_17, adj_19);
        wp::adj_address(var_affine_has_constraint, var_0, adj_affine_has_constraint, adj_0, adj_17);
        // adj: has_constraint = affine_has_constraint[tid]                                       <L 157>
        wp::adj_copy(var_16, adj_14, adj_15);
        wp::adj_load(var_14, adj_14, adj_16);
        wp::adj_address(var_mass_body, var_0, adj_mass_body, adj_0, adj_14);
        // adj: mass = mass_body[tid]                                                             <L 156>
        if (var_13) {
            label0:;
            // adj: return                                                                        <L 155>
        }
        wp::adj_load(var_8, adj_8, adj_12);
        wp::adj_address(var_env_states, var_9, adj_env_states, adj_7, adj_8);
        wp::adj_load(var_7, adj_7, adj_9);
        wp::adj_address(var_body_env_id, var_0, adj_body_env_id, adj_0, adj_7);
        wp::adj_load(var_2, adj_2, adj_6);
        wp::adj_address(var_env_states, var_3, adj_env_states, adj_1, adj_2);
        wp::adj_load(var_1, adj_1, adj_3);
        wp::adj_address(var_body_env_id, var_0, adj_body_env_id, adj_0, adj_1);
        // adj: if env_states[body_env_id[tid]] == ENV_STATE_INVALID or env_states[body_env_id[tid]] == ENV_STATE_NEWTON_SOLVED:  <L 154>
        // adj: tid = wp.tid()                                                                    <L 153>
        // adj: def compute_affine_kinematic_grad(                                                <L 143>
        continue;
    }
}



extern "C" __global__ void compute_affine_kinematic_energy_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<12,wp::float64>> var_y,
    wp::array_t<bool> var_affine_has_constraint,
    wp::array_t<wp::vec_t<12,wp::float64>> var_affine_target_dof,
    wp::float64 var_weight,
    wp::array_t<wp::float64> var_affine_energy,
    wp::array_t<wp::float64> var_mass_body)
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
        wp::float64* var_1;
        wp::float64 var_2;
        wp::float64 var_3;
        bool* var_4;
        bool var_5;
        bool var_6;
        const wp::int32 var_7 = 0;
        const wp::float32 var_8 = 0.5;
        wp::float64 var_9;
        wp::float64 var_10;
        wp::float64 var_11;
        wp::vec_t<12,wp::float64>* var_12;
        wp::float64 var_13;
        wp::vec_t<12,wp::float64> var_14;
        wp::vec_t<12,wp::float64>* var_15;
        wp::float64 var_16;
        wp::vec_t<12,wp::float64> var_17;
        wp::float64 var_18;
        wp::float64 var_19;
        wp::float64 var_20;
        wp::float64 var_21;
        const wp::int32 var_22 = 1;
        const wp::float32 var_23 = 0.5;
        wp::float64 var_24;
        wp::float64 var_25;
        wp::float64 var_26;
        wp::vec_t<12,wp::float64>* var_27;
        wp::float64 var_28;
        wp::vec_t<12,wp::float64> var_29;
        wp::vec_t<12,wp::float64>* var_30;
        wp::float64 var_31;
        wp::vec_t<12,wp::float64> var_32;
        wp::float64 var_33;
        wp::float64 var_34;
        wp::float64 var_35;
        wp::float64 var_36;
        const wp::int32 var_37 = 2;
        const wp::float32 var_38 = 0.5;
        wp::float64 var_39;
        wp::float64 var_40;
        wp::float64 var_41;
        wp::vec_t<12,wp::float64>* var_42;
        wp::float64 var_43;
        wp::vec_t<12,wp::float64> var_44;
        wp::vec_t<12,wp::float64>* var_45;
        wp::float64 var_46;
        wp::vec_t<12,wp::float64> var_47;
        wp::float64 var_48;
        wp::float64 var_49;
        wp::float64 var_50;
        wp::float64 var_51;
        const wp::int32 var_52 = 3;
        const wp::float32 var_53 = 0.5;
        wp::float64 var_54;
        wp::float64 var_55;
        wp::float64 var_56;
        wp::vec_t<12,wp::float64>* var_57;
        wp::float64 var_58;
        wp::vec_t<12,wp::float64> var_59;
        wp::vec_t<12,wp::float64>* var_60;
        wp::float64 var_61;
        wp::vec_t<12,wp::float64> var_62;
        wp::float64 var_63;
        wp::float64 var_64;
        wp::float64 var_65;
        wp::float64 var_66;
        const wp::int32 var_67 = 4;
        const wp::float32 var_68 = 0.5;
        wp::float64 var_69;
        wp::float64 var_70;
        wp::float64 var_71;
        wp::vec_t<12,wp::float64>* var_72;
        wp::float64 var_73;
        wp::vec_t<12,wp::float64> var_74;
        wp::vec_t<12,wp::float64>* var_75;
        wp::float64 var_76;
        wp::vec_t<12,wp::float64> var_77;
        wp::float64 var_78;
        wp::float64 var_79;
        wp::float64 var_80;
        wp::float64 var_81;
        const wp::int32 var_82 = 5;
        const wp::float32 var_83 = 0.5;
        wp::float64 var_84;
        wp::float64 var_85;
        wp::float64 var_86;
        wp::vec_t<12,wp::float64>* var_87;
        wp::float64 var_88;
        wp::vec_t<12,wp::float64> var_89;
        wp::vec_t<12,wp::float64>* var_90;
        wp::float64 var_91;
        wp::vec_t<12,wp::float64> var_92;
        wp::float64 var_93;
        wp::float64 var_94;
        wp::float64 var_95;
        wp::float64 var_96;
        const wp::int32 var_97 = 6;
        const wp::float32 var_98 = 0.5;
        wp::float64 var_99;
        wp::float64 var_100;
        wp::float64 var_101;
        wp::vec_t<12,wp::float64>* var_102;
        wp::float64 var_103;
        wp::vec_t<12,wp::float64> var_104;
        wp::vec_t<12,wp::float64>* var_105;
        wp::float64 var_106;
        wp::vec_t<12,wp::float64> var_107;
        wp::float64 var_108;
        wp::float64 var_109;
        wp::float64 var_110;
        wp::float64 var_111;
        const wp::int32 var_112 = 7;
        const wp::float32 var_113 = 0.5;
        wp::float64 var_114;
        wp::float64 var_115;
        wp::float64 var_116;
        wp::vec_t<12,wp::float64>* var_117;
        wp::float64 var_118;
        wp::vec_t<12,wp::float64> var_119;
        wp::vec_t<12,wp::float64>* var_120;
        wp::float64 var_121;
        wp::vec_t<12,wp::float64> var_122;
        wp::float64 var_123;
        wp::float64 var_124;
        wp::float64 var_125;
        wp::float64 var_126;
        const wp::int32 var_127 = 8;
        const wp::float32 var_128 = 0.5;
        wp::float64 var_129;
        wp::float64 var_130;
        wp::float64 var_131;
        wp::vec_t<12,wp::float64>* var_132;
        wp::float64 var_133;
        wp::vec_t<12,wp::float64> var_134;
        wp::vec_t<12,wp::float64>* var_135;
        wp::float64 var_136;
        wp::vec_t<12,wp::float64> var_137;
        wp::float64 var_138;
        wp::float64 var_139;
        wp::float64 var_140;
        wp::float64 var_141;
        const wp::int32 var_142 = 9;
        const wp::float32 var_143 = 0.5;
        wp::float64 var_144;
        wp::float64 var_145;
        wp::float64 var_146;
        wp::vec_t<12,wp::float64>* var_147;
        wp::float64 var_148;
        wp::vec_t<12,wp::float64> var_149;
        wp::vec_t<12,wp::float64>* var_150;
        wp::float64 var_151;
        wp::vec_t<12,wp::float64> var_152;
        wp::float64 var_153;
        wp::float64 var_154;
        wp::float64 var_155;
        wp::float64 var_156;
        const wp::int32 var_157 = 10;
        const wp::float32 var_158 = 0.5;
        wp::float64 var_159;
        wp::float64 var_160;
        wp::float64 var_161;
        wp::vec_t<12,wp::float64>* var_162;
        wp::float64 var_163;
        wp::vec_t<12,wp::float64> var_164;
        wp::vec_t<12,wp::float64>* var_165;
        wp::float64 var_166;
        wp::vec_t<12,wp::float64> var_167;
        wp::float64 var_168;
        wp::float64 var_169;
        wp::float64 var_170;
        wp::float64 var_171;
        const wp::int32 var_172 = 11;
        const wp::float32 var_173 = 0.5;
        wp::float64 var_174;
        wp::float64 var_175;
        wp::float64 var_176;
        wp::vec_t<12,wp::float64>* var_177;
        wp::float64 var_178;
        wp::vec_t<12,wp::float64> var_179;
        wp::vec_t<12,wp::float64>* var_180;
        wp::float64 var_181;
        wp::vec_t<12,wp::float64> var_182;
        wp::float64 var_183;
        wp::float64 var_184;
        wp::float64 var_185;
        wp::float64 var_186;
        //---------
        // forward
        // def compute_affine_kinematic_energy(                                                   <L 69>
        // tid = wp.tid()                                                                         <L 77>
        var_0 = builtin_tid1d();
        // mass = mass_body[tid]                                                                  <L 78>
        var_1 = wp::address(var_mass_body, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // has_constraint = affine_has_constraint[tid]                                            <L 79>
        var_4 = wp::address(var_affine_has_constraint, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // if has_constraint:                                                                     <L 80>
        if (var_5) {
            // for d in range(12):                                                                <L 81>
            // affine_energy[tid] += wp.float64(0.5) * weight * mass * sqr(y[tid][d] - affine_target_dof[tid][d])       <L 82>
            var_9 = wp::float64(var_8);
            var_10 = wp::mul(var_9, var_weight);
            var_11 = wp::mul(var_10, var_2);
            var_12 = wp::address(var_y, var_0);
            var_14 = wp::load(var_12);
            var_13 = wp::extract(var_14, var_7);
            var_15 = wp::address(var_affine_target_dof, var_0);
            var_17 = wp::load(var_15);
            var_16 = wp::extract(var_17, var_7);
            var_18 = wp::sub(var_13, var_16);
            var_19 = sqr_0(var_18);
            var_20 = wp::mul(var_11, var_19);
            var_21 = wp::atomic_add(var_affine_energy, var_0, var_20);
            var_24 = wp::float64(var_23);
            var_25 = wp::mul(var_24, var_weight);
            var_26 = wp::mul(var_25, var_2);
            var_27 = wp::address(var_y, var_0);
            var_29 = wp::load(var_27);
            var_28 = wp::extract(var_29, var_22);
            var_30 = wp::address(var_affine_target_dof, var_0);
            var_32 = wp::load(var_30);
            var_31 = wp::extract(var_32, var_22);
            var_33 = wp::sub(var_28, var_31);
            var_34 = sqr_0(var_33);
            var_35 = wp::mul(var_26, var_34);
            var_36 = wp::atomic_add(var_affine_energy, var_0, var_35);
            var_39 = wp::float64(var_38);
            var_40 = wp::mul(var_39, var_weight);
            var_41 = wp::mul(var_40, var_2);
            var_42 = wp::address(var_y, var_0);
            var_44 = wp::load(var_42);
            var_43 = wp::extract(var_44, var_37);
            var_45 = wp::address(var_affine_target_dof, var_0);
            var_47 = wp::load(var_45);
            var_46 = wp::extract(var_47, var_37);
            var_48 = wp::sub(var_43, var_46);
            var_49 = sqr_0(var_48);
            var_50 = wp::mul(var_41, var_49);
            var_51 = wp::atomic_add(var_affine_energy, var_0, var_50);
            var_54 = wp::float64(var_53);
            var_55 = wp::mul(var_54, var_weight);
            var_56 = wp::mul(var_55, var_2);
            var_57 = wp::address(var_y, var_0);
            var_59 = wp::load(var_57);
            var_58 = wp::extract(var_59, var_52);
            var_60 = wp::address(var_affine_target_dof, var_0);
            var_62 = wp::load(var_60);
            var_61 = wp::extract(var_62, var_52);
            var_63 = wp::sub(var_58, var_61);
            var_64 = sqr_0(var_63);
            var_65 = wp::mul(var_56, var_64);
            var_66 = wp::atomic_add(var_affine_energy, var_0, var_65);
            var_69 = wp::float64(var_68);
            var_70 = wp::mul(var_69, var_weight);
            var_71 = wp::mul(var_70, var_2);
            var_72 = wp::address(var_y, var_0);
            var_74 = wp::load(var_72);
            var_73 = wp::extract(var_74, var_67);
            var_75 = wp::address(var_affine_target_dof, var_0);
            var_77 = wp::load(var_75);
            var_76 = wp::extract(var_77, var_67);
            var_78 = wp::sub(var_73, var_76);
            var_79 = sqr_0(var_78);
            var_80 = wp::mul(var_71, var_79);
            var_81 = wp::atomic_add(var_affine_energy, var_0, var_80);
            var_84 = wp::float64(var_83);
            var_85 = wp::mul(var_84, var_weight);
            var_86 = wp::mul(var_85, var_2);
            var_87 = wp::address(var_y, var_0);
            var_89 = wp::load(var_87);
            var_88 = wp::extract(var_89, var_82);
            var_90 = wp::address(var_affine_target_dof, var_0);
            var_92 = wp::load(var_90);
            var_91 = wp::extract(var_92, var_82);
            var_93 = wp::sub(var_88, var_91);
            var_94 = sqr_0(var_93);
            var_95 = wp::mul(var_86, var_94);
            var_96 = wp::atomic_add(var_affine_energy, var_0, var_95);
            var_99 = wp::float64(var_98);
            var_100 = wp::mul(var_99, var_weight);
            var_101 = wp::mul(var_100, var_2);
            var_102 = wp::address(var_y, var_0);
            var_104 = wp::load(var_102);
            var_103 = wp::extract(var_104, var_97);
            var_105 = wp::address(var_affine_target_dof, var_0);
            var_107 = wp::load(var_105);
            var_106 = wp::extract(var_107, var_97);
            var_108 = wp::sub(var_103, var_106);
            var_109 = sqr_0(var_108);
            var_110 = wp::mul(var_101, var_109);
            var_111 = wp::atomic_add(var_affine_energy, var_0, var_110);
            var_114 = wp::float64(var_113);
            var_115 = wp::mul(var_114, var_weight);
            var_116 = wp::mul(var_115, var_2);
            var_117 = wp::address(var_y, var_0);
            var_119 = wp::load(var_117);
            var_118 = wp::extract(var_119, var_112);
            var_120 = wp::address(var_affine_target_dof, var_0);
            var_122 = wp::load(var_120);
            var_121 = wp::extract(var_122, var_112);
            var_123 = wp::sub(var_118, var_121);
            var_124 = sqr_0(var_123);
            var_125 = wp::mul(var_116, var_124);
            var_126 = wp::atomic_add(var_affine_energy, var_0, var_125);
            var_129 = wp::float64(var_128);
            var_130 = wp::mul(var_129, var_weight);
            var_131 = wp::mul(var_130, var_2);
            var_132 = wp::address(var_y, var_0);
            var_134 = wp::load(var_132);
            var_133 = wp::extract(var_134, var_127);
            var_135 = wp::address(var_affine_target_dof, var_0);
            var_137 = wp::load(var_135);
            var_136 = wp::extract(var_137, var_127);
            var_138 = wp::sub(var_133, var_136);
            var_139 = sqr_0(var_138);
            var_140 = wp::mul(var_131, var_139);
            var_141 = wp::atomic_add(var_affine_energy, var_0, var_140);
            var_144 = wp::float64(var_143);
            var_145 = wp::mul(var_144, var_weight);
            var_146 = wp::mul(var_145, var_2);
            var_147 = wp::address(var_y, var_0);
            var_149 = wp::load(var_147);
            var_148 = wp::extract(var_149, var_142);
            var_150 = wp::address(var_affine_target_dof, var_0);
            var_152 = wp::load(var_150);
            var_151 = wp::extract(var_152, var_142);
            var_153 = wp::sub(var_148, var_151);
            var_154 = sqr_0(var_153);
            var_155 = wp::mul(var_146, var_154);
            var_156 = wp::atomic_add(var_affine_energy, var_0, var_155);
            var_159 = wp::float64(var_158);
            var_160 = wp::mul(var_159, var_weight);
            var_161 = wp::mul(var_160, var_2);
            var_162 = wp::address(var_y, var_0);
            var_164 = wp::load(var_162);
            var_163 = wp::extract(var_164, var_157);
            var_165 = wp::address(var_affine_target_dof, var_0);
            var_167 = wp::load(var_165);
            var_166 = wp::extract(var_167, var_157);
            var_168 = wp::sub(var_163, var_166);
            var_169 = sqr_0(var_168);
            var_170 = wp::mul(var_161, var_169);
            var_171 = wp::atomic_add(var_affine_energy, var_0, var_170);
            var_174 = wp::float64(var_173);
            var_175 = wp::mul(var_174, var_weight);
            var_176 = wp::mul(var_175, var_2);
            var_177 = wp::address(var_y, var_0);
            var_179 = wp::load(var_177);
            var_178 = wp::extract(var_179, var_172);
            var_180 = wp::address(var_affine_target_dof, var_0);
            var_182 = wp::load(var_180);
            var_181 = wp::extract(var_182, var_172);
            var_183 = wp::sub(var_178, var_181);
            var_184 = sqr_0(var_183);
            var_185 = wp::mul(var_176, var_184);
            var_186 = wp::atomic_add(var_affine_energy, var_0, var_185);
        }
    }
}



extern "C" __global__ void compute_affine_kinematic_energy_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<12,wp::float64>> var_y,
    wp::array_t<bool> var_affine_has_constraint,
    wp::array_t<wp::vec_t<12,wp::float64>> var_affine_target_dof,
    wp::float64 var_weight,
    wp::array_t<wp::float64> var_affine_energy,
    wp::array_t<wp::float64> var_mass_body,
    wp::array_t<wp::vec_t<12,wp::float64>> adj_y,
    wp::array_t<bool> adj_affine_has_constraint,
    wp::array_t<wp::vec_t<12,wp::float64>> adj_affine_target_dof,
    wp::float64 adj_weight,
    wp::array_t<wp::float64> adj_affine_energy,
    wp::array_t<wp::float64> adj_mass_body)
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
        wp::float64* var_1;
        wp::float64 var_2;
        wp::float64 var_3;
        bool* var_4;
        bool var_5;
        bool var_6;
        const wp::int32 var_7 = 0;
        const wp::float32 var_8 = 0.5;
        wp::float64 var_9;
        wp::float64 var_10;
        wp::float64 var_11;
        wp::vec_t<12,wp::float64>* var_12;
        wp::float64 var_13;
        wp::vec_t<12,wp::float64> var_14;
        wp::vec_t<12,wp::float64>* var_15;
        wp::float64 var_16;
        wp::vec_t<12,wp::float64> var_17;
        wp::float64 var_18;
        wp::float64 var_19;
        wp::float64 var_20;
        wp::float64 var_21;
        const wp::int32 var_22 = 1;
        const wp::float32 var_23 = 0.5;
        wp::float64 var_24;
        wp::float64 var_25;
        wp::float64 var_26;
        wp::vec_t<12,wp::float64>* var_27;
        wp::float64 var_28;
        wp::vec_t<12,wp::float64> var_29;
        wp::vec_t<12,wp::float64>* var_30;
        wp::float64 var_31;
        wp::vec_t<12,wp::float64> var_32;
        wp::float64 var_33;
        wp::float64 var_34;
        wp::float64 var_35;
        wp::float64 var_36;
        const wp::int32 var_37 = 2;
        const wp::float32 var_38 = 0.5;
        wp::float64 var_39;
        wp::float64 var_40;
        wp::float64 var_41;
        wp::vec_t<12,wp::float64>* var_42;
        wp::float64 var_43;
        wp::vec_t<12,wp::float64> var_44;
        wp::vec_t<12,wp::float64>* var_45;
        wp::float64 var_46;
        wp::vec_t<12,wp::float64> var_47;
        wp::float64 var_48;
        wp::float64 var_49;
        wp::float64 var_50;
        wp::float64 var_51;
        const wp::int32 var_52 = 3;
        const wp::float32 var_53 = 0.5;
        wp::float64 var_54;
        wp::float64 var_55;
        wp::float64 var_56;
        wp::vec_t<12,wp::float64>* var_57;
        wp::float64 var_58;
        wp::vec_t<12,wp::float64> var_59;
        wp::vec_t<12,wp::float64>* var_60;
        wp::float64 var_61;
        wp::vec_t<12,wp::float64> var_62;
        wp::float64 var_63;
        wp::float64 var_64;
        wp::float64 var_65;
        wp::float64 var_66;
        const wp::int32 var_67 = 4;
        const wp::float32 var_68 = 0.5;
        wp::float64 var_69;
        wp::float64 var_70;
        wp::float64 var_71;
        wp::vec_t<12,wp::float64>* var_72;
        wp::float64 var_73;
        wp::vec_t<12,wp::float64> var_74;
        wp::vec_t<12,wp::float64>* var_75;
        wp::float64 var_76;
        wp::vec_t<12,wp::float64> var_77;
        wp::float64 var_78;
        wp::float64 var_79;
        wp::float64 var_80;
        wp::float64 var_81;
        const wp::int32 var_82 = 5;
        const wp::float32 var_83 = 0.5;
        wp::float64 var_84;
        wp::float64 var_85;
        wp::float64 var_86;
        wp::vec_t<12,wp::float64>* var_87;
        wp::float64 var_88;
        wp::vec_t<12,wp::float64> var_89;
        wp::vec_t<12,wp::float64>* var_90;
        wp::float64 var_91;
        wp::vec_t<12,wp::float64> var_92;
        wp::float64 var_93;
        wp::float64 var_94;
        wp::float64 var_95;
        wp::float64 var_96;
        const wp::int32 var_97 = 6;
        const wp::float32 var_98 = 0.5;
        wp::float64 var_99;
        wp::float64 var_100;
        wp::float64 var_101;
        wp::vec_t<12,wp::float64>* var_102;
        wp::float64 var_103;
        wp::vec_t<12,wp::float64> var_104;
        wp::vec_t<12,wp::float64>* var_105;
        wp::float64 var_106;
        wp::vec_t<12,wp::float64> var_107;
        wp::float64 var_108;
        wp::float64 var_109;
        wp::float64 var_110;
        wp::float64 var_111;
        const wp::int32 var_112 = 7;
        const wp::float32 var_113 = 0.5;
        wp::float64 var_114;
        wp::float64 var_115;
        wp::float64 var_116;
        wp::vec_t<12,wp::float64>* var_117;
        wp::float64 var_118;
        wp::vec_t<12,wp::float64> var_119;
        wp::vec_t<12,wp::float64>* var_120;
        wp::float64 var_121;
        wp::vec_t<12,wp::float64> var_122;
        wp::float64 var_123;
        wp::float64 var_124;
        wp::float64 var_125;
        wp::float64 var_126;
        const wp::int32 var_127 = 8;
        const wp::float32 var_128 = 0.5;
        wp::float64 var_129;
        wp::float64 var_130;
        wp::float64 var_131;
        wp::vec_t<12,wp::float64>* var_132;
        wp::float64 var_133;
        wp::vec_t<12,wp::float64> var_134;
        wp::vec_t<12,wp::float64>* var_135;
        wp::float64 var_136;
        wp::vec_t<12,wp::float64> var_137;
        wp::float64 var_138;
        wp::float64 var_139;
        wp::float64 var_140;
        wp::float64 var_141;
        const wp::int32 var_142 = 9;
        const wp::float32 var_143 = 0.5;
        wp::float64 var_144;
        wp::float64 var_145;
        wp::float64 var_146;
        wp::vec_t<12,wp::float64>* var_147;
        wp::float64 var_148;
        wp::vec_t<12,wp::float64> var_149;
        wp::vec_t<12,wp::float64>* var_150;
        wp::float64 var_151;
        wp::vec_t<12,wp::float64> var_152;
        wp::float64 var_153;
        wp::float64 var_154;
        wp::float64 var_155;
        wp::float64 var_156;
        const wp::int32 var_157 = 10;
        const wp::float32 var_158 = 0.5;
        wp::float64 var_159;
        wp::float64 var_160;
        wp::float64 var_161;
        wp::vec_t<12,wp::float64>* var_162;
        wp::float64 var_163;
        wp::vec_t<12,wp::float64> var_164;
        wp::vec_t<12,wp::float64>* var_165;
        wp::float64 var_166;
        wp::vec_t<12,wp::float64> var_167;
        wp::float64 var_168;
        wp::float64 var_169;
        wp::float64 var_170;
        wp::float64 var_171;
        const wp::int32 var_172 = 11;
        const wp::float32 var_173 = 0.5;
        wp::float64 var_174;
        wp::float64 var_175;
        wp::float64 var_176;
        wp::vec_t<12,wp::float64>* var_177;
        wp::float64 var_178;
        wp::vec_t<12,wp::float64> var_179;
        wp::vec_t<12,wp::float64>* var_180;
        wp::float64 var_181;
        wp::vec_t<12,wp::float64> var_182;
        wp::float64 var_183;
        wp::float64 var_184;
        wp::float64 var_185;
        wp::float64 var_186;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::float64 adj_1 = {};
        wp::float64 adj_2 = {};
        wp::float64 adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        bool adj_6 = {};
        wp::int32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float64 adj_9 = {};
        wp::float64 adj_10 = {};
        wp::float64 adj_11 = {};
        wp::vec_t<12,wp::float64> adj_12 = {};
        wp::float64 adj_13 = {};
        wp::vec_t<12,wp::float64> adj_14 = {};
        wp::vec_t<12,wp::float64> adj_15 = {};
        wp::float64 adj_16 = {};
        wp::vec_t<12,wp::float64> adj_17 = {};
        wp::float64 adj_18 = {};
        wp::float64 adj_19 = {};
        wp::float64 adj_20 = {};
        wp::float64 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float64 adj_24 = {};
        wp::float64 adj_25 = {};
        wp::float64 adj_26 = {};
        wp::vec_t<12,wp::float64> adj_27 = {};
        wp::float64 adj_28 = {};
        wp::vec_t<12,wp::float64> adj_29 = {};
        wp::vec_t<12,wp::float64> adj_30 = {};
        wp::float64 adj_31 = {};
        wp::vec_t<12,wp::float64> adj_32 = {};
        wp::float64 adj_33 = {};
        wp::float64 adj_34 = {};
        wp::float64 adj_35 = {};
        wp::float64 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::float64 adj_39 = {};
        wp::float64 adj_40 = {};
        wp::float64 adj_41 = {};
        wp::vec_t<12,wp::float64> adj_42 = {};
        wp::float64 adj_43 = {};
        wp::vec_t<12,wp::float64> adj_44 = {};
        wp::vec_t<12,wp::float64> adj_45 = {};
        wp::float64 adj_46 = {};
        wp::vec_t<12,wp::float64> adj_47 = {};
        wp::float64 adj_48 = {};
        wp::float64 adj_49 = {};
        wp::float64 adj_50 = {};
        wp::float64 adj_51 = {};
        wp::int32 adj_52 = {};
        wp::float32 adj_53 = {};
        wp::float64 adj_54 = {};
        wp::float64 adj_55 = {};
        wp::float64 adj_56 = {};
        wp::vec_t<12,wp::float64> adj_57 = {};
        wp::float64 adj_58 = {};
        wp::vec_t<12,wp::float64> adj_59 = {};
        wp::vec_t<12,wp::float64> adj_60 = {};
        wp::float64 adj_61 = {};
        wp::vec_t<12,wp::float64> adj_62 = {};
        wp::float64 adj_63 = {};
        wp::float64 adj_64 = {};
        wp::float64 adj_65 = {};
        wp::float64 adj_66 = {};
        wp::int32 adj_67 = {};
        wp::float32 adj_68 = {};
        wp::float64 adj_69 = {};
        wp::float64 adj_70 = {};
        wp::float64 adj_71 = {};
        wp::vec_t<12,wp::float64> adj_72 = {};
        wp::float64 adj_73 = {};
        wp::vec_t<12,wp::float64> adj_74 = {};
        wp::vec_t<12,wp::float64> adj_75 = {};
        wp::float64 adj_76 = {};
        wp::vec_t<12,wp::float64> adj_77 = {};
        wp::float64 adj_78 = {};
        wp::float64 adj_79 = {};
        wp::float64 adj_80 = {};
        wp::float64 adj_81 = {};
        wp::int32 adj_82 = {};
        wp::float32 adj_83 = {};
        wp::float64 adj_84 = {};
        wp::float64 adj_85 = {};
        wp::float64 adj_86 = {};
        wp::vec_t<12,wp::float64> adj_87 = {};
        wp::float64 adj_88 = {};
        wp::vec_t<12,wp::float64> adj_89 = {};
        wp::vec_t<12,wp::float64> adj_90 = {};
        wp::float64 adj_91 = {};
        wp::vec_t<12,wp::float64> adj_92 = {};
        wp::float64 adj_93 = {};
        wp::float64 adj_94 = {};
        wp::float64 adj_95 = {};
        wp::float64 adj_96 = {};
        wp::int32 adj_97 = {};
        wp::float32 adj_98 = {};
        wp::float64 adj_99 = {};
        wp::float64 adj_100 = {};
        wp::float64 adj_101 = {};
        wp::vec_t<12,wp::float64> adj_102 = {};
        wp::float64 adj_103 = {};
        wp::vec_t<12,wp::float64> adj_104 = {};
        wp::vec_t<12,wp::float64> adj_105 = {};
        wp::float64 adj_106 = {};
        wp::vec_t<12,wp::float64> adj_107 = {};
        wp::float64 adj_108 = {};
        wp::float64 adj_109 = {};
        wp::float64 adj_110 = {};
        wp::float64 adj_111 = {};
        wp::int32 adj_112 = {};
        wp::float32 adj_113 = {};
        wp::float64 adj_114 = {};
        wp::float64 adj_115 = {};
        wp::float64 adj_116 = {};
        wp::vec_t<12,wp::float64> adj_117 = {};
        wp::float64 adj_118 = {};
        wp::vec_t<12,wp::float64> adj_119 = {};
        wp::vec_t<12,wp::float64> adj_120 = {};
        wp::float64 adj_121 = {};
        wp::vec_t<12,wp::float64> adj_122 = {};
        wp::float64 adj_123 = {};
        wp::float64 adj_124 = {};
        wp::float64 adj_125 = {};
        wp::float64 adj_126 = {};
        wp::int32 adj_127 = {};
        wp::float32 adj_128 = {};
        wp::float64 adj_129 = {};
        wp::float64 adj_130 = {};
        wp::float64 adj_131 = {};
        wp::vec_t<12,wp::float64> adj_132 = {};
        wp::float64 adj_133 = {};
        wp::vec_t<12,wp::float64> adj_134 = {};
        wp::vec_t<12,wp::float64> adj_135 = {};
        wp::float64 adj_136 = {};
        wp::vec_t<12,wp::float64> adj_137 = {};
        wp::float64 adj_138 = {};
        wp::float64 adj_139 = {};
        wp::float64 adj_140 = {};
        wp::float64 adj_141 = {};
        wp::int32 adj_142 = {};
        wp::float32 adj_143 = {};
        wp::float64 adj_144 = {};
        wp::float64 adj_145 = {};
        wp::float64 adj_146 = {};
        wp::vec_t<12,wp::float64> adj_147 = {};
        wp::float64 adj_148 = {};
        wp::vec_t<12,wp::float64> adj_149 = {};
        wp::vec_t<12,wp::float64> adj_150 = {};
        wp::float64 adj_151 = {};
        wp::vec_t<12,wp::float64> adj_152 = {};
        wp::float64 adj_153 = {};
        wp::float64 adj_154 = {};
        wp::float64 adj_155 = {};
        wp::float64 adj_156 = {};
        wp::int32 adj_157 = {};
        wp::float32 adj_158 = {};
        wp::float64 adj_159 = {};
        wp::float64 adj_160 = {};
        wp::float64 adj_161 = {};
        wp::vec_t<12,wp::float64> adj_162 = {};
        wp::float64 adj_163 = {};
        wp::vec_t<12,wp::float64> adj_164 = {};
        wp::vec_t<12,wp::float64> adj_165 = {};
        wp::float64 adj_166 = {};
        wp::vec_t<12,wp::float64> adj_167 = {};
        wp::float64 adj_168 = {};
        wp::float64 adj_169 = {};
        wp::float64 adj_170 = {};
        wp::float64 adj_171 = {};
        wp::int32 adj_172 = {};
        wp::float32 adj_173 = {};
        wp::float64 adj_174 = {};
        wp::float64 adj_175 = {};
        wp::float64 adj_176 = {};
        wp::vec_t<12,wp::float64> adj_177 = {};
        wp::float64 adj_178 = {};
        wp::vec_t<12,wp::float64> adj_179 = {};
        wp::vec_t<12,wp::float64> adj_180 = {};
        wp::float64 adj_181 = {};
        wp::vec_t<12,wp::float64> adj_182 = {};
        wp::float64 adj_183 = {};
        wp::float64 adj_184 = {};
        wp::float64 adj_185 = {};
        wp::float64 adj_186 = {};
        //---------
        // forward
        // def compute_affine_kinematic_energy(                                                   <L 69>
        // tid = wp.tid()                                                                         <L 77>
        var_0 = builtin_tid1d();
        // mass = mass_body[tid]                                                                  <L 78>
        var_1 = wp::address(var_mass_body, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // has_constraint = affine_has_constraint[tid]                                            <L 79>
        var_4 = wp::address(var_affine_has_constraint, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // if has_constraint:                                                                     <L 80>
        if (var_5) {
            // for d in range(12):                                                                <L 81>
            // affine_energy[tid] += wp.float64(0.5) * weight * mass * sqr(y[tid][d] - affine_target_dof[tid][d])       <L 82>
            var_9 = wp::float64(var_8);
            var_10 = wp::mul(var_9, var_weight);
            var_11 = wp::mul(var_10, var_2);
            var_12 = wp::address(var_y, var_0);
            var_14 = wp::load(var_12);
            var_13 = wp::extract(var_14, var_7);
            var_15 = wp::address(var_affine_target_dof, var_0);
            var_17 = wp::load(var_15);
            var_16 = wp::extract(var_17, var_7);
            var_18 = wp::sub(var_13, var_16);
            var_19 = sqr_0(var_18);
            var_20 = wp::mul(var_11, var_19);
            // var_21 = wp::atomic_add(var_affine_energy, var_0, var_20);
            var_24 = wp::float64(var_23);
            var_25 = wp::mul(var_24, var_weight);
            var_26 = wp::mul(var_25, var_2);
            var_27 = wp::address(var_y, var_0);
            var_29 = wp::load(var_27);
            var_28 = wp::extract(var_29, var_22);
            var_30 = wp::address(var_affine_target_dof, var_0);
            var_32 = wp::load(var_30);
            var_31 = wp::extract(var_32, var_22);
            var_33 = wp::sub(var_28, var_31);
            var_34 = sqr_0(var_33);
            var_35 = wp::mul(var_26, var_34);
            // var_36 = wp::atomic_add(var_affine_energy, var_0, var_35);
            var_39 = wp::float64(var_38);
            var_40 = wp::mul(var_39, var_weight);
            var_41 = wp::mul(var_40, var_2);
            var_42 = wp::address(var_y, var_0);
            var_44 = wp::load(var_42);
            var_43 = wp::extract(var_44, var_37);
            var_45 = wp::address(var_affine_target_dof, var_0);
            var_47 = wp::load(var_45);
            var_46 = wp::extract(var_47, var_37);
            var_48 = wp::sub(var_43, var_46);
            var_49 = sqr_0(var_48);
            var_50 = wp::mul(var_41, var_49);
            // var_51 = wp::atomic_add(var_affine_energy, var_0, var_50);
            var_54 = wp::float64(var_53);
            var_55 = wp::mul(var_54, var_weight);
            var_56 = wp::mul(var_55, var_2);
            var_57 = wp::address(var_y, var_0);
            var_59 = wp::load(var_57);
            var_58 = wp::extract(var_59, var_52);
            var_60 = wp::address(var_affine_target_dof, var_0);
            var_62 = wp::load(var_60);
            var_61 = wp::extract(var_62, var_52);
            var_63 = wp::sub(var_58, var_61);
            var_64 = sqr_0(var_63);
            var_65 = wp::mul(var_56, var_64);
            // var_66 = wp::atomic_add(var_affine_energy, var_0, var_65);
            var_69 = wp::float64(var_68);
            var_70 = wp::mul(var_69, var_weight);
            var_71 = wp::mul(var_70, var_2);
            var_72 = wp::address(var_y, var_0);
            var_74 = wp::load(var_72);
            var_73 = wp::extract(var_74, var_67);
            var_75 = wp::address(var_affine_target_dof, var_0);
            var_77 = wp::load(var_75);
            var_76 = wp::extract(var_77, var_67);
            var_78 = wp::sub(var_73, var_76);
            var_79 = sqr_0(var_78);
            var_80 = wp::mul(var_71, var_79);
            // var_81 = wp::atomic_add(var_affine_energy, var_0, var_80);
            var_84 = wp::float64(var_83);
            var_85 = wp::mul(var_84, var_weight);
            var_86 = wp::mul(var_85, var_2);
            var_87 = wp::address(var_y, var_0);
            var_89 = wp::load(var_87);
            var_88 = wp::extract(var_89, var_82);
            var_90 = wp::address(var_affine_target_dof, var_0);
            var_92 = wp::load(var_90);
            var_91 = wp::extract(var_92, var_82);
            var_93 = wp::sub(var_88, var_91);
            var_94 = sqr_0(var_93);
            var_95 = wp::mul(var_86, var_94);
            // var_96 = wp::atomic_add(var_affine_energy, var_0, var_95);
            var_99 = wp::float64(var_98);
            var_100 = wp::mul(var_99, var_weight);
            var_101 = wp::mul(var_100, var_2);
            var_102 = wp::address(var_y, var_0);
            var_104 = wp::load(var_102);
            var_103 = wp::extract(var_104, var_97);
            var_105 = wp::address(var_affine_target_dof, var_0);
            var_107 = wp::load(var_105);
            var_106 = wp::extract(var_107, var_97);
            var_108 = wp::sub(var_103, var_106);
            var_109 = sqr_0(var_108);
            var_110 = wp::mul(var_101, var_109);
            // var_111 = wp::atomic_add(var_affine_energy, var_0, var_110);
            var_114 = wp::float64(var_113);
            var_115 = wp::mul(var_114, var_weight);
            var_116 = wp::mul(var_115, var_2);
            var_117 = wp::address(var_y, var_0);
            var_119 = wp::load(var_117);
            var_118 = wp::extract(var_119, var_112);
            var_120 = wp::address(var_affine_target_dof, var_0);
            var_122 = wp::load(var_120);
            var_121 = wp::extract(var_122, var_112);
            var_123 = wp::sub(var_118, var_121);
            var_124 = sqr_0(var_123);
            var_125 = wp::mul(var_116, var_124);
            // var_126 = wp::atomic_add(var_affine_energy, var_0, var_125);
            var_129 = wp::float64(var_128);
            var_130 = wp::mul(var_129, var_weight);
            var_131 = wp::mul(var_130, var_2);
            var_132 = wp::address(var_y, var_0);
            var_134 = wp::load(var_132);
            var_133 = wp::extract(var_134, var_127);
            var_135 = wp::address(var_affine_target_dof, var_0);
            var_137 = wp::load(var_135);
            var_136 = wp::extract(var_137, var_127);
            var_138 = wp::sub(var_133, var_136);
            var_139 = sqr_0(var_138);
            var_140 = wp::mul(var_131, var_139);
            // var_141 = wp::atomic_add(var_affine_energy, var_0, var_140);
            var_144 = wp::float64(var_143);
            var_145 = wp::mul(var_144, var_weight);
            var_146 = wp::mul(var_145, var_2);
            var_147 = wp::address(var_y, var_0);
            var_149 = wp::load(var_147);
            var_148 = wp::extract(var_149, var_142);
            var_150 = wp::address(var_affine_target_dof, var_0);
            var_152 = wp::load(var_150);
            var_151 = wp::extract(var_152, var_142);
            var_153 = wp::sub(var_148, var_151);
            var_154 = sqr_0(var_153);
            var_155 = wp::mul(var_146, var_154);
            // var_156 = wp::atomic_add(var_affine_energy, var_0, var_155);
            var_159 = wp::float64(var_158);
            var_160 = wp::mul(var_159, var_weight);
            var_161 = wp::mul(var_160, var_2);
            var_162 = wp::address(var_y, var_0);
            var_164 = wp::load(var_162);
            var_163 = wp::extract(var_164, var_157);
            var_165 = wp::address(var_affine_target_dof, var_0);
            var_167 = wp::load(var_165);
            var_166 = wp::extract(var_167, var_157);
            var_168 = wp::sub(var_163, var_166);
            var_169 = sqr_0(var_168);
            var_170 = wp::mul(var_161, var_169);
            // var_171 = wp::atomic_add(var_affine_energy, var_0, var_170);
            var_174 = wp::float64(var_173);
            var_175 = wp::mul(var_174, var_weight);
            var_176 = wp::mul(var_175, var_2);
            var_177 = wp::address(var_y, var_0);
            var_179 = wp::load(var_177);
            var_178 = wp::extract(var_179, var_172);
            var_180 = wp::address(var_affine_target_dof, var_0);
            var_182 = wp::load(var_180);
            var_181 = wp::extract(var_182, var_172);
            var_183 = wp::sub(var_178, var_181);
            var_184 = sqr_0(var_183);
            var_185 = wp::mul(var_176, var_184);
            // var_186 = wp::atomic_add(var_affine_energy, var_0, var_185);
        }
        //---------
        // reverse
        if (var_5) {
            wp::adj_atomic_add(var_affine_energy, var_0, var_185, adj_affine_energy, adj_0, adj_185, adj_186);
            wp::adj_mul(var_176, var_184, adj_176, adj_184, adj_185);
            adj_sqr_0(var_183, adj_183, adj_184);
            wp::adj_sub(var_178, var_181, adj_178, adj_181, adj_183);
            wp::adj_extract(var_182, var_172, adj_180, adj_172, adj_181);
            wp::adj_load(var_180, adj_180, adj_182);
            wp::adj_address(var_affine_target_dof, var_0, adj_affine_target_dof, adj_0, adj_180);
            wp::adj_extract(var_179, var_172, adj_177, adj_172, adj_178);
            wp::adj_load(var_177, adj_177, adj_179);
            wp::adj_address(var_y, var_0, adj_y, adj_0, adj_177);
            wp::adj_mul(var_175, var_2, adj_175, adj_2, adj_176);
            wp::adj_mul(var_174, var_weight, adj_174, adj_weight, adj_175);
            wp::adj_float64(var_173, adj_173, adj_174);
            wp::adj_atomic_add(var_affine_energy, var_0, var_170, adj_affine_energy, adj_0, adj_170, adj_171);
            wp::adj_mul(var_161, var_169, adj_161, adj_169, adj_170);
            adj_sqr_0(var_168, adj_168, adj_169);
            wp::adj_sub(var_163, var_166, adj_163, adj_166, adj_168);
            wp::adj_extract(var_167, var_157, adj_165, adj_157, adj_166);
            wp::adj_load(var_165, adj_165, adj_167);
            wp::adj_address(var_affine_target_dof, var_0, adj_affine_target_dof, adj_0, adj_165);
            wp::adj_extract(var_164, var_157, adj_162, adj_157, adj_163);
            wp::adj_load(var_162, adj_162, adj_164);
            wp::adj_address(var_y, var_0, adj_y, adj_0, adj_162);
            wp::adj_mul(var_160, var_2, adj_160, adj_2, adj_161);
            wp::adj_mul(var_159, var_weight, adj_159, adj_weight, adj_160);
            wp::adj_float64(var_158, adj_158, adj_159);
            wp::adj_atomic_add(var_affine_energy, var_0, var_155, adj_affine_energy, adj_0, adj_155, adj_156);
            wp::adj_mul(var_146, var_154, adj_146, adj_154, adj_155);
            adj_sqr_0(var_153, adj_153, adj_154);
            wp::adj_sub(var_148, var_151, adj_148, adj_151, adj_153);
            wp::adj_extract(var_152, var_142, adj_150, adj_142, adj_151);
            wp::adj_load(var_150, adj_150, adj_152);
            wp::adj_address(var_affine_target_dof, var_0, adj_affine_target_dof, adj_0, adj_150);
            wp::adj_extract(var_149, var_142, adj_147, adj_142, adj_148);
            wp::adj_load(var_147, adj_147, adj_149);
            wp::adj_address(var_y, var_0, adj_y, adj_0, adj_147);
            wp::adj_mul(var_145, var_2, adj_145, adj_2, adj_146);
            wp::adj_mul(var_144, var_weight, adj_144, adj_weight, adj_145);
            wp::adj_float64(var_143, adj_143, adj_144);
            wp::adj_atomic_add(var_affine_energy, var_0, var_140, adj_affine_energy, adj_0, adj_140, adj_141);
            wp::adj_mul(var_131, var_139, adj_131, adj_139, adj_140);
            adj_sqr_0(var_138, adj_138, adj_139);
            wp::adj_sub(var_133, var_136, adj_133, adj_136, adj_138);
            wp::adj_extract(var_137, var_127, adj_135, adj_127, adj_136);
            wp::adj_load(var_135, adj_135, adj_137);
            wp::adj_address(var_affine_target_dof, var_0, adj_affine_target_dof, adj_0, adj_135);
            wp::adj_extract(var_134, var_127, adj_132, adj_127, adj_133);
            wp::adj_load(var_132, adj_132, adj_134);
            wp::adj_address(var_y, var_0, adj_y, adj_0, adj_132);
            wp::adj_mul(var_130, var_2, adj_130, adj_2, adj_131);
            wp::adj_mul(var_129, var_weight, adj_129, adj_weight, adj_130);
            wp::adj_float64(var_128, adj_128, adj_129);
            wp::adj_atomic_add(var_affine_energy, var_0, var_125, adj_affine_energy, adj_0, adj_125, adj_126);
            wp::adj_mul(var_116, var_124, adj_116, adj_124, adj_125);
            adj_sqr_0(var_123, adj_123, adj_124);
            wp::adj_sub(var_118, var_121, adj_118, adj_121, adj_123);
            wp::adj_extract(var_122, var_112, adj_120, adj_112, adj_121);
            wp::adj_load(var_120, adj_120, adj_122);
            wp::adj_address(var_affine_target_dof, var_0, adj_affine_target_dof, adj_0, adj_120);
            wp::adj_extract(var_119, var_112, adj_117, adj_112, adj_118);
            wp::adj_load(var_117, adj_117, adj_119);
            wp::adj_address(var_y, var_0, adj_y, adj_0, adj_117);
            wp::adj_mul(var_115, var_2, adj_115, adj_2, adj_116);
            wp::adj_mul(var_114, var_weight, adj_114, adj_weight, adj_115);
            wp::adj_float64(var_113, adj_113, adj_114);
            wp::adj_atomic_add(var_affine_energy, var_0, var_110, adj_affine_energy, adj_0, adj_110, adj_111);
            wp::adj_mul(var_101, var_109, adj_101, adj_109, adj_110);
            adj_sqr_0(var_108, adj_108, adj_109);
            wp::adj_sub(var_103, var_106, adj_103, adj_106, adj_108);
            wp::adj_extract(var_107, var_97, adj_105, adj_97, adj_106);
            wp::adj_load(var_105, adj_105, adj_107);
            wp::adj_address(var_affine_target_dof, var_0, adj_affine_target_dof, adj_0, adj_105);
            wp::adj_extract(var_104, var_97, adj_102, adj_97, adj_103);
            wp::adj_load(var_102, adj_102, adj_104);
            wp::adj_address(var_y, var_0, adj_y, adj_0, adj_102);
            wp::adj_mul(var_100, var_2, adj_100, adj_2, adj_101);
            wp::adj_mul(var_99, var_weight, adj_99, adj_weight, adj_100);
            wp::adj_float64(var_98, adj_98, adj_99);
            wp::adj_atomic_add(var_affine_energy, var_0, var_95, adj_affine_energy, adj_0, adj_95, adj_96);
            wp::adj_mul(var_86, var_94, adj_86, adj_94, adj_95);
            adj_sqr_0(var_93, adj_93, adj_94);
            wp::adj_sub(var_88, var_91, adj_88, adj_91, adj_93);
            wp::adj_extract(var_92, var_82, adj_90, adj_82, adj_91);
            wp::adj_load(var_90, adj_90, adj_92);
            wp::adj_address(var_affine_target_dof, var_0, adj_affine_target_dof, adj_0, adj_90);
            wp::adj_extract(var_89, var_82, adj_87, adj_82, adj_88);
            wp::adj_load(var_87, adj_87, adj_89);
            wp::adj_address(var_y, var_0, adj_y, adj_0, adj_87);
            wp::adj_mul(var_85, var_2, adj_85, adj_2, adj_86);
            wp::adj_mul(var_84, var_weight, adj_84, adj_weight, adj_85);
            wp::adj_float64(var_83, adj_83, adj_84);
            wp::adj_atomic_add(var_affine_energy, var_0, var_80, adj_affine_energy, adj_0, adj_80, adj_81);
            wp::adj_mul(var_71, var_79, adj_71, adj_79, adj_80);
            adj_sqr_0(var_78, adj_78, adj_79);
            wp::adj_sub(var_73, var_76, adj_73, adj_76, adj_78);
            wp::adj_extract(var_77, var_67, adj_75, adj_67, adj_76);
            wp::adj_load(var_75, adj_75, adj_77);
            wp::adj_address(var_affine_target_dof, var_0, adj_affine_target_dof, adj_0, adj_75);
            wp::adj_extract(var_74, var_67, adj_72, adj_67, adj_73);
            wp::adj_load(var_72, adj_72, adj_74);
            wp::adj_address(var_y, var_0, adj_y, adj_0, adj_72);
            wp::adj_mul(var_70, var_2, adj_70, adj_2, adj_71);
            wp::adj_mul(var_69, var_weight, adj_69, adj_weight, adj_70);
            wp::adj_float64(var_68, adj_68, adj_69);
            wp::adj_atomic_add(var_affine_energy, var_0, var_65, adj_affine_energy, adj_0, adj_65, adj_66);
            wp::adj_mul(var_56, var_64, adj_56, adj_64, adj_65);
            adj_sqr_0(var_63, adj_63, adj_64);
            wp::adj_sub(var_58, var_61, adj_58, adj_61, adj_63);
            wp::adj_extract(var_62, var_52, adj_60, adj_52, adj_61);
            wp::adj_load(var_60, adj_60, adj_62);
            wp::adj_address(var_affine_target_dof, var_0, adj_affine_target_dof, adj_0, adj_60);
            wp::adj_extract(var_59, var_52, adj_57, adj_52, adj_58);
            wp::adj_load(var_57, adj_57, adj_59);
            wp::adj_address(var_y, var_0, adj_y, adj_0, adj_57);
            wp::adj_mul(var_55, var_2, adj_55, adj_2, adj_56);
            wp::adj_mul(var_54, var_weight, adj_54, adj_weight, adj_55);
            wp::adj_float64(var_53, adj_53, adj_54);
            wp::adj_atomic_add(var_affine_energy, var_0, var_50, adj_affine_energy, adj_0, adj_50, adj_51);
            wp::adj_mul(var_41, var_49, adj_41, adj_49, adj_50);
            adj_sqr_0(var_48, adj_48, adj_49);
            wp::adj_sub(var_43, var_46, adj_43, adj_46, adj_48);
            wp::adj_extract(var_47, var_37, adj_45, adj_37, adj_46);
            wp::adj_load(var_45, adj_45, adj_47);
            wp::adj_address(var_affine_target_dof, var_0, adj_affine_target_dof, adj_0, adj_45);
            wp::adj_extract(var_44, var_37, adj_42, adj_37, adj_43);
            wp::adj_load(var_42, adj_42, adj_44);
            wp::adj_address(var_y, var_0, adj_y, adj_0, adj_42);
            wp::adj_mul(var_40, var_2, adj_40, adj_2, adj_41);
            wp::adj_mul(var_39, var_weight, adj_39, adj_weight, adj_40);
            wp::adj_float64(var_38, adj_38, adj_39);
            wp::adj_atomic_add(var_affine_energy, var_0, var_35, adj_affine_energy, adj_0, adj_35, adj_36);
            wp::adj_mul(var_26, var_34, adj_26, adj_34, adj_35);
            adj_sqr_0(var_33, adj_33, adj_34);
            wp::adj_sub(var_28, var_31, adj_28, adj_31, adj_33);
            wp::adj_extract(var_32, var_22, adj_30, adj_22, adj_31);
            wp::adj_load(var_30, adj_30, adj_32);
            wp::adj_address(var_affine_target_dof, var_0, adj_affine_target_dof, adj_0, adj_30);
            wp::adj_extract(var_29, var_22, adj_27, adj_22, adj_28);
            wp::adj_load(var_27, adj_27, adj_29);
            wp::adj_address(var_y, var_0, adj_y, adj_0, adj_27);
            wp::adj_mul(var_25, var_2, adj_25, adj_2, adj_26);
            wp::adj_mul(var_24, var_weight, adj_24, adj_weight, adj_25);
            wp::adj_float64(var_23, adj_23, adj_24);
            wp::adj_atomic_add(var_affine_energy, var_0, var_20, adj_affine_energy, adj_0, adj_20, adj_21);
            wp::adj_mul(var_11, var_19, adj_11, adj_19, adj_20);
            adj_sqr_0(var_18, adj_18, adj_19);
            wp::adj_sub(var_13, var_16, adj_13, adj_16, adj_18);
            wp::adj_extract(var_17, var_7, adj_15, adj_7, adj_16);
            wp::adj_load(var_15, adj_15, adj_17);
            wp::adj_address(var_affine_target_dof, var_0, adj_affine_target_dof, adj_0, adj_15);
            wp::adj_extract(var_14, var_7, adj_12, adj_7, adj_13);
            wp::adj_load(var_12, adj_12, adj_14);
            wp::adj_address(var_y, var_0, adj_y, adj_0, adj_12);
            wp::adj_mul(var_10, var_2, adj_10, adj_2, adj_11);
            wp::adj_mul(var_9, var_weight, adj_9, adj_weight, adj_10);
            wp::adj_float64(var_8, adj_8, adj_9);
            // adj: affine_energy[tid] += wp.float64(0.5) * weight * mass * sqr(y[tid][d] - affine_target_dof[tid][d])  <L 82>
            // adj: for d in range(12):                                                           <L 81>
        }
        // adj: if has_constraint:                                                                <L 80>
        wp::adj_copy(var_6, adj_4, adj_5);
        wp::adj_load(var_4, adj_4, adj_6);
        wp::adj_address(var_affine_has_constraint, var_0, adj_affine_has_constraint, adj_0, adj_4);
        // adj: has_constraint = affine_has_constraint[tid]                                       <L 79>
        wp::adj_copy(var_3, adj_1, adj_2);
        wp::adj_load(var_1, adj_1, adj_3);
        wp::adj_address(var_mass_body, var_0, adj_mass_body, adj_0, adj_1);
        // adj: mass = mass_body[tid]                                                             <L 78>
        // adj: tid = wp.tid()                                                                    <L 77>
        // adj: def compute_affine_kinematic_energy(                                              <L 69>
        continue;
    }
}



extern "C" __global__ void compute_soft_kinematic_grad_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float64>> var_x,
    wp::array_t<bool> var_soft_has_constraint,
    wp::array_t<wp::vec_t<3,wp::float64>> var_soft_target_dof,
    wp::float64 var_weight,
    wp::array_t<wp::vec_t<3,wp::float64>> var_soft_grad,
    wp::int32 var_affine_verts_num,
    wp::array_t<wp::float64> var_soft_verts_mass,
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
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 1;
        bool var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        const wp::int32 var_12 = 2;
        bool var_13;
        wp::int32 var_14;
        bool var_15;
        wp::float64* var_16;
        wp::float64 var_17;
        wp::float64 var_18;
        bool* var_19;
        bool var_20;
        bool var_21;
        wp::float64 var_22;
        wp::int32 var_23;
        wp::vec_t<3,wp::float64>* var_24;
        wp::vec_t<3,wp::float64>* var_25;
        wp::vec_t<3,wp::float64> var_26;
        wp::vec_t<3,wp::float64> var_27;
        wp::vec_t<3,wp::float64> var_28;
        wp::vec_t<3,wp::float64> var_29;
        wp::vec_t<3,wp::float64> var_30;
        //---------
        // forward
        // def compute_soft_kinematic_grad(                                                       <L 163>
        // tid = wp.tid()                                                                         <L 174>
        var_0 = builtin_tid1d();
        // if env_states[node2env[tid + affine_verts_num]] == ENV_STATE_INVALID or env_states[node2env[tid + affine_verts_num]] == ENV_STATE_NEWTON_SOLVED:       <L 175>
        var_1 = wp::add(var_0, var_affine_verts_num);
        var_2 = wp::address(var_node2env, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::address(var_env_states, var_4);
        var_7 = wp::load(var_3);
        var_6 = (var_7 == var_5);
        var_8 = wp::add(var_0, var_affine_verts_num);
        var_9 = wp::address(var_node2env, var_8);
        var_11 = wp::load(var_9);
        var_10 = wp::address(var_env_states, var_11);
        var_14 = wp::load(var_10);
        var_13 = (var_14 == var_12);
        var_15 = var_6 || var_13;
        if (var_15) {
            // return                                                                             <L 176>
            continue;
        }
        // mass = soft_verts_mass[tid]                                                            <L 177>
        var_16 = wp::address(var_soft_verts_mass, var_0);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // has_constraint = soft_has_constraint[tid]                                              <L 178>
        var_19 = wp::address(var_soft_has_constraint, var_0);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // if has_constraint:                                                                     <L 179>
        if (var_20) {
            // wp.atomic_add(                                                                     <L 180>
            // soft_grad,                                                                         <L 181>
            // tid,                                                                               <L 182>
            // weight * mass * (x[tid + affine_verts_num] - soft_target_dof[tid]),                <L 183>
            var_22 = wp::mul(var_weight, var_17);
            var_23 = wp::add(var_0, var_affine_verts_num);
            var_24 = wp::address(var_x, var_23);
            var_25 = wp::address(var_soft_target_dof, var_0);
            var_27 = wp::load(var_24);
            var_28 = wp::load(var_25);
            var_26 = wp::sub(var_27, var_28);
            var_29 = wp::mul(var_22, var_26);
            var_30 = wp::atomic_add(var_soft_grad, var_0, var_29);
        }
    }
}



extern "C" __global__ void compute_soft_kinematic_grad_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float64>> var_x,
    wp::array_t<bool> var_soft_has_constraint,
    wp::array_t<wp::vec_t<3,wp::float64>> var_soft_target_dof,
    wp::float64 var_weight,
    wp::array_t<wp::vec_t<3,wp::float64>> var_soft_grad,
    wp::int32 var_affine_verts_num,
    wp::array_t<wp::float64> var_soft_verts_mass,
    wp::array_t<wp::int32> var_node2env,
    wp::array_t<wp::int32> var_env_states,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_x,
    wp::array_t<bool> adj_soft_has_constraint,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_soft_target_dof,
    wp::float64 adj_weight,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_soft_grad,
    wp::int32 adj_affine_verts_num,
    wp::array_t<wp::float64> adj_soft_verts_mass,
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
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 1;
        bool var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        const wp::int32 var_12 = 2;
        bool var_13;
        wp::int32 var_14;
        bool var_15;
        wp::float64* var_16;
        wp::float64 var_17;
        wp::float64 var_18;
        bool* var_19;
        bool var_20;
        bool var_21;
        wp::float64 var_22;
        wp::int32 var_23;
        wp::vec_t<3,wp::float64>* var_24;
        wp::vec_t<3,wp::float64>* var_25;
        wp::vec_t<3,wp::float64> var_26;
        wp::vec_t<3,wp::float64> var_27;
        wp::vec_t<3,wp::float64> var_28;
        wp::vec_t<3,wp::float64> var_29;
        wp::vec_t<3,wp::float64> var_30;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        bool adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        bool adj_13 = {};
        wp::int32 adj_14 = {};
        bool adj_15 = {};
        wp::float64 adj_16 = {};
        wp::float64 adj_17 = {};
        wp::float64 adj_18 = {};
        bool adj_19 = {};
        bool adj_20 = {};
        bool adj_21 = {};
        wp::float64 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::vec_t<3,wp::float64> adj_24 = {};
        wp::vec_t<3,wp::float64> adj_25 = {};
        wp::vec_t<3,wp::float64> adj_26 = {};
        wp::vec_t<3,wp::float64> adj_27 = {};
        wp::vec_t<3,wp::float64> adj_28 = {};
        wp::vec_t<3,wp::float64> adj_29 = {};
        wp::vec_t<3,wp::float64> adj_30 = {};
        //---------
        // forward
        // def compute_soft_kinematic_grad(                                                       <L 163>
        // tid = wp.tid()                                                                         <L 174>
        var_0 = builtin_tid1d();
        // if env_states[node2env[tid + affine_verts_num]] == ENV_STATE_INVALID or env_states[node2env[tid + affine_verts_num]] == ENV_STATE_NEWTON_SOLVED:       <L 175>
        var_1 = wp::add(var_0, var_affine_verts_num);
        var_2 = wp::address(var_node2env, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::address(var_env_states, var_4);
        var_7 = wp::load(var_3);
        var_6 = (var_7 == var_5);
        var_8 = wp::add(var_0, var_affine_verts_num);
        var_9 = wp::address(var_node2env, var_8);
        var_11 = wp::load(var_9);
        var_10 = wp::address(var_env_states, var_11);
        var_14 = wp::load(var_10);
        var_13 = (var_14 == var_12);
        var_15 = var_6 || var_13;
        if (var_15) {
            // return                                                                             <L 176>
            goto label0;
        }
        // mass = soft_verts_mass[tid]                                                            <L 177>
        var_16 = wp::address(var_soft_verts_mass, var_0);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // has_constraint = soft_has_constraint[tid]                                              <L 178>
        var_19 = wp::address(var_soft_has_constraint, var_0);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // if has_constraint:                                                                     <L 179>
        if (var_20) {
            // wp.atomic_add(                                                                     <L 180>
            // soft_grad,                                                                         <L 181>
            // tid,                                                                               <L 182>
            // weight * mass * (x[tid + affine_verts_num] - soft_target_dof[tid]),                <L 183>
            var_22 = wp::mul(var_weight, var_17);
            var_23 = wp::add(var_0, var_affine_verts_num);
            var_24 = wp::address(var_x, var_23);
            var_25 = wp::address(var_soft_target_dof, var_0);
            var_27 = wp::load(var_24);
            var_28 = wp::load(var_25);
            var_26 = wp::sub(var_27, var_28);
            var_29 = wp::mul(var_22, var_26);
            // var_30 = wp::atomic_add(var_soft_grad, var_0, var_29);
        }
        //---------
        // reverse
        if (var_20) {
            wp::adj_atomic_add(var_soft_grad, var_0, var_29, adj_soft_grad, adj_0, adj_29, adj_30);
            wp::adj_mul(var_22, var_26, adj_22, adj_26, adj_29);
            wp::adj_sub(var_27, var_28, adj_24, adj_25, adj_26);
            wp::adj_load(var_25, adj_25, adj_28);
            wp::adj_load(var_24, adj_24, adj_27);
            wp::adj_address(var_soft_target_dof, var_0, adj_soft_target_dof, adj_0, adj_25);
            wp::adj_address(var_x, var_23, adj_x, adj_23, adj_24);
            wp::adj_add(var_0, var_affine_verts_num, adj_0, adj_affine_verts_num, adj_23);
            wp::adj_mul(var_weight, var_17, adj_weight, adj_17, adj_22);
            // adj: weight * mass * (x[tid + affine_verts_num] - soft_target_dof[tid]),           <L 183>
            // adj: tid,                                                                          <L 182>
            // adj: soft_grad,                                                                    <L 181>
            // adj: wp.atomic_add(                                                                <L 180>
        }
        // adj: if has_constraint:                                                                <L 179>
        wp::adj_copy(var_21, adj_19, adj_20);
        wp::adj_load(var_19, adj_19, adj_21);
        wp::adj_address(var_soft_has_constraint, var_0, adj_soft_has_constraint, adj_0, adj_19);
        // adj: has_constraint = soft_has_constraint[tid]                                         <L 178>
        wp::adj_copy(var_18, adj_16, adj_17);
        wp::adj_load(var_16, adj_16, adj_18);
        wp::adj_address(var_soft_verts_mass, var_0, adj_soft_verts_mass, adj_0, adj_16);
        // adj: mass = soft_verts_mass[tid]                                                       <L 177>
        if (var_15) {
            label0:;
            // adj: return                                                                        <L 176>
        }
        wp::adj_load(var_10, adj_10, adj_14);
        wp::adj_address(var_env_states, var_11, adj_env_states, adj_9, adj_10);
        wp::adj_load(var_9, adj_9, adj_11);
        wp::adj_address(var_node2env, var_8, adj_node2env, adj_8, adj_9);
        wp::adj_add(var_0, var_affine_verts_num, adj_0, adj_affine_verts_num, adj_8);
        wp::adj_load(var_3, adj_3, adj_7);
        wp::adj_address(var_env_states, var_4, adj_env_states, adj_2, adj_3);
        wp::adj_load(var_2, adj_2, adj_4);
        wp::adj_address(var_node2env, var_1, adj_node2env, adj_1, adj_2);
        wp::adj_add(var_0, var_affine_verts_num, adj_0, adj_affine_verts_num, adj_1);
        // adj: if env_states[node2env[tid + affine_verts_num]] == ENV_STATE_INVALID or env_states[node2env[tid + affine_verts_num]] == ENV_STATE_NEWTON_SOLVED:  <L 175>
        // adj: tid = wp.tid()                                                                    <L 174>
        // adj: def compute_soft_kinematic_grad(                                                  <L 163>
        continue;
    }
}



extern "C" __global__ void compute_affine_kinematic_hess_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<bool> var_affine_has_constraint,
    wp::float64 var_weight,
    COOMatrix3x3_0df4b45d var_hess_affine_diag,
    wp::array_t<wp::float64> var_mass_body,
    wp::array_t<wp::int32> var_body_env_id,
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
        wp::int32* var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        const wp::int32 var_4 = 1;
        bool var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        const wp::int32 var_10 = 2;
        bool var_11;
        wp::int32 var_12;
        bool var_13;
        wp::float64* var_14;
        wp::float64 var_15;
        wp::float64 var_16;
        const wp::float32 var_17 = 0.0;
        wp::float64 var_18;
        bool* var_19;
        bool var_20;
        bool var_21;
        wp::float64 var_22;
        wp::mat_t<3,3,wp::float64> var_23;
        const wp::int32 var_24 = 0;
        const wp::int32 var_25 = 16;
        wp::int32 var_26;
        const wp::int32 var_27 = 4;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        const wp::int32 var_31 = 1;
        const wp::int32 var_32 = 16;
        wp::int32 var_33;
        const wp::int32 var_34 = 4;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        const wp::int32 var_38 = 2;
        const wp::int32 var_39 = 16;
        wp::int32 var_40;
        const wp::int32 var_41 = 4;
        wp::int32 var_42;
        wp::int32 var_43;
        wp::int32 var_44;
        const wp::int32 var_45 = 3;
        const wp::int32 var_46 = 16;
        wp::int32 var_47;
        const wp::int32 var_48 = 4;
        wp::int32 var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        //---------
        // forward
        // def compute_affine_kinematic_hess(                                                     <L 254>
        // tid = wp.tid()                                                                         <L 262>
        var_0 = builtin_tid1d();
        // if env_states[body_env_id[tid]] == ENV_STATE_INVALID or env_states[body_env_id[tid]] == ENV_STATE_NEWTON_SOLVED:       <L 263>
        var_1 = wp::address(var_body_env_id, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::address(var_env_states, var_3);
        var_6 = wp::load(var_2);
        var_5 = (var_6 == var_4);
        var_7 = wp::address(var_body_env_id, var_0);
        var_9 = wp::load(var_7);
        var_8 = wp::address(var_env_states, var_9);
        var_12 = wp::load(var_8);
        var_11 = (var_12 == var_10);
        var_13 = var_5 || var_11;
        if (var_13) {
            // return                                                                             <L 264>
            continue;
        }
        // mass = mass_body[tid]                                                                  <L 265>
        var_14 = wp::address(var_mass_body, var_0);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // _0 = wp.float64(0.0)                                                                   <L 266>
        var_18 = wp::float64(var_17);
        // has_constraint = affine_has_constraint[tid]                                            <L 267>
        var_19 = wp::address(var_affine_has_constraint, var_0);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // weighted_mass = mass * weight                                                          <L 268>
        var_22 = wp::mul(var_15, var_weight);
        // mat3 = wp.mat33d(weighted_mass, _0, _0, _0, weighted_mass, _0, _0, _0, weighted_mass)       <L 269>
        var_23 = wp::mat_t<3,3,wp::float64>(var_22, var_18, var_18, var_18, var_22, var_18, var_18, var_18, var_22);
        // if has_constraint:                                                                     <L 270>
        if (var_20) {
            // for bi in range(4):                                                                <L 271>
            // matrix.COOMatrix3x3_atomic_add(hess_affine_diag, tid * 16 + 4 * bi + bi, mat3)       <L 272>
            var_26 = wp::mul(var_0, var_25);
            var_28 = wp::mul(var_27, var_24);
            var_29 = wp::add(var_26, var_28);
            var_30 = wp::add(var_29, var_24);
            COOMatrix3x3_atomic_add_0(var_hess_affine_diag, var_30, var_23);
            var_33 = wp::mul(var_0, var_32);
            var_35 = wp::mul(var_34, var_31);
            var_36 = wp::add(var_33, var_35);
            var_37 = wp::add(var_36, var_31);
            COOMatrix3x3_atomic_add_0(var_hess_affine_diag, var_37, var_23);
            var_40 = wp::mul(var_0, var_39);
            var_42 = wp::mul(var_41, var_38);
            var_43 = wp::add(var_40, var_42);
            var_44 = wp::add(var_43, var_38);
            COOMatrix3x3_atomic_add_0(var_hess_affine_diag, var_44, var_23);
            var_47 = wp::mul(var_0, var_46);
            var_49 = wp::mul(var_48, var_45);
            var_50 = wp::add(var_47, var_49);
            var_51 = wp::add(var_50, var_45);
            COOMatrix3x3_atomic_add_0(var_hess_affine_diag, var_51, var_23);
        }
    }
}



extern "C" __global__ void compute_affine_kinematic_hess_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<bool> var_affine_has_constraint,
    wp::float64 var_weight,
    COOMatrix3x3_0df4b45d var_hess_affine_diag,
    wp::array_t<wp::float64> var_mass_body,
    wp::array_t<wp::int32> var_body_env_id,
    wp::array_t<wp::int32> var_env_states,
    wp::array_t<bool> adj_affine_has_constraint,
    wp::float64 adj_weight,
    COOMatrix3x3_0df4b45d adj_hess_affine_diag,
    wp::array_t<wp::float64> adj_mass_body,
    wp::array_t<wp::int32> adj_body_env_id,
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
        wp::int32* var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        const wp::int32 var_4 = 1;
        bool var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        const wp::int32 var_10 = 2;
        bool var_11;
        wp::int32 var_12;
        bool var_13;
        wp::float64* var_14;
        wp::float64 var_15;
        wp::float64 var_16;
        const wp::float32 var_17 = 0.0;
        wp::float64 var_18;
        bool* var_19;
        bool var_20;
        bool var_21;
        wp::float64 var_22;
        wp::mat_t<3,3,wp::float64> var_23;
        const wp::int32 var_24 = 0;
        const wp::int32 var_25 = 16;
        wp::int32 var_26;
        const wp::int32 var_27 = 4;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        const wp::int32 var_31 = 1;
        const wp::int32 var_32 = 16;
        wp::int32 var_33;
        const wp::int32 var_34 = 4;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        const wp::int32 var_38 = 2;
        const wp::int32 var_39 = 16;
        wp::int32 var_40;
        const wp::int32 var_41 = 4;
        wp::int32 var_42;
        wp::int32 var_43;
        wp::int32 var_44;
        const wp::int32 var_45 = 3;
        const wp::int32 var_46 = 16;
        wp::int32 var_47;
        const wp::int32 var_48 = 4;
        wp::int32 var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        bool adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        bool adj_11 = {};
        wp::int32 adj_12 = {};
        bool adj_13 = {};
        wp::float64 adj_14 = {};
        wp::float64 adj_15 = {};
        wp::float64 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float64 adj_18 = {};
        bool adj_19 = {};
        bool adj_20 = {};
        bool adj_21 = {};
        wp::float64 adj_22 = {};
        wp::mat_t<3,3,wp::float64> adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::int32 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::int32 adj_42 = {};
        wp::int32 adj_43 = {};
        wp::int32 adj_44 = {};
        wp::int32 adj_45 = {};
        wp::int32 adj_46 = {};
        wp::int32 adj_47 = {};
        wp::int32 adj_48 = {};
        wp::int32 adj_49 = {};
        wp::int32 adj_50 = {};
        wp::int32 adj_51 = {};
        //---------
        // forward
        // def compute_affine_kinematic_hess(                                                     <L 254>
        // tid = wp.tid()                                                                         <L 262>
        var_0 = builtin_tid1d();
        // if env_states[body_env_id[tid]] == ENV_STATE_INVALID or env_states[body_env_id[tid]] == ENV_STATE_NEWTON_SOLVED:       <L 263>
        var_1 = wp::address(var_body_env_id, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::address(var_env_states, var_3);
        var_6 = wp::load(var_2);
        var_5 = (var_6 == var_4);
        var_7 = wp::address(var_body_env_id, var_0);
        var_9 = wp::load(var_7);
        var_8 = wp::address(var_env_states, var_9);
        var_12 = wp::load(var_8);
        var_11 = (var_12 == var_10);
        var_13 = var_5 || var_11;
        if (var_13) {
            // return                                                                             <L 264>
            goto label0;
        }
        // mass = mass_body[tid]                                                                  <L 265>
        var_14 = wp::address(var_mass_body, var_0);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // _0 = wp.float64(0.0)                                                                   <L 266>
        var_18 = wp::float64(var_17);
        // has_constraint = affine_has_constraint[tid]                                            <L 267>
        var_19 = wp::address(var_affine_has_constraint, var_0);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // weighted_mass = mass * weight                                                          <L 268>
        var_22 = wp::mul(var_15, var_weight);
        // mat3 = wp.mat33d(weighted_mass, _0, _0, _0, weighted_mass, _0, _0, _0, weighted_mass)       <L 269>
        var_23 = wp::mat_t<3,3,wp::float64>(var_22, var_18, var_18, var_18, var_22, var_18, var_18, var_18, var_22);
        // if has_constraint:                                                                     <L 270>
        if (var_20) {
            // for bi in range(4):                                                                <L 271>
            // matrix.COOMatrix3x3_atomic_add(hess_affine_diag, tid * 16 + 4 * bi + bi, mat3)       <L 272>
            var_26 = wp::mul(var_0, var_25);
            var_28 = wp::mul(var_27, var_24);
            var_29 = wp::add(var_26, var_28);
            var_30 = wp::add(var_29, var_24);
            COOMatrix3x3_atomic_add_0(var_hess_affine_diag, var_30, var_23);
            var_33 = wp::mul(var_0, var_32);
            var_35 = wp::mul(var_34, var_31);
            var_36 = wp::add(var_33, var_35);
            var_37 = wp::add(var_36, var_31);
            COOMatrix3x3_atomic_add_0(var_hess_affine_diag, var_37, var_23);
            var_40 = wp::mul(var_0, var_39);
            var_42 = wp::mul(var_41, var_38);
            var_43 = wp::add(var_40, var_42);
            var_44 = wp::add(var_43, var_38);
            COOMatrix3x3_atomic_add_0(var_hess_affine_diag, var_44, var_23);
            var_47 = wp::mul(var_0, var_46);
            var_49 = wp::mul(var_48, var_45);
            var_50 = wp::add(var_47, var_49);
            var_51 = wp::add(var_50, var_45);
            COOMatrix3x3_atomic_add_0(var_hess_affine_diag, var_51, var_23);
        }
        //---------
        // reverse
        if (var_20) {
            adj_COOMatrix3x3_atomic_add_0(var_hess_affine_diag, var_51, var_23, adj_hess_affine_diag, adj_51, adj_23);
            wp::adj_add(var_50, var_45, adj_50, adj_45, adj_51);
            wp::adj_add(var_47, var_49, adj_47, adj_49, adj_50);
            wp::adj_mul(var_48, var_45, adj_48, adj_45, adj_49);
            wp::adj_mul(var_0, var_46, adj_0, adj_46, adj_47);
            adj_COOMatrix3x3_atomic_add_0(var_hess_affine_diag, var_44, var_23, adj_hess_affine_diag, adj_44, adj_23);
            wp::adj_add(var_43, var_38, adj_43, adj_38, adj_44);
            wp::adj_add(var_40, var_42, adj_40, adj_42, adj_43);
            wp::adj_mul(var_41, var_38, adj_41, adj_38, adj_42);
            wp::adj_mul(var_0, var_39, adj_0, adj_39, adj_40);
            adj_COOMatrix3x3_atomic_add_0(var_hess_affine_diag, var_37, var_23, adj_hess_affine_diag, adj_37, adj_23);
            wp::adj_add(var_36, var_31, adj_36, adj_31, adj_37);
            wp::adj_add(var_33, var_35, adj_33, adj_35, adj_36);
            wp::adj_mul(var_34, var_31, adj_34, adj_31, adj_35);
            wp::adj_mul(var_0, var_32, adj_0, adj_32, adj_33);
            adj_COOMatrix3x3_atomic_add_0(var_hess_affine_diag, var_30, var_23, adj_hess_affine_diag, adj_30, adj_23);
            wp::adj_add(var_29, var_24, adj_29, adj_24, adj_30);
            wp::adj_add(var_26, var_28, adj_26, adj_28, adj_29);
            wp::adj_mul(var_27, var_24, adj_27, adj_24, adj_28);
            wp::adj_mul(var_0, var_25, adj_0, adj_25, adj_26);
            // adj: matrix.COOMatrix3x3_atomic_add(hess_affine_diag, tid * 16 + 4 * bi + bi, mat3)  <L 272>
            // adj: for bi in range(4):                                                           <L 271>
        }
        // adj: if has_constraint:                                                                <L 270>
        wp::adj_mat_t(var_22, var_18, var_18, var_18, var_22, var_18, var_18, var_18, var_22, adj_22, adj_18, adj_18, adj_18, adj_22, adj_18, adj_18, adj_18, adj_22, adj_23);
        // adj: mat3 = wp.mat33d(weighted_mass, _0, _0, _0, weighted_mass, _0, _0, _0, weighted_mass)  <L 269>
        wp::adj_mul(var_15, var_weight, adj_15, adj_weight, adj_22);
        // adj: weighted_mass = mass * weight                                                     <L 268>
        wp::adj_copy(var_21, adj_19, adj_20);
        wp::adj_load(var_19, adj_19, adj_21);
        wp::adj_address(var_affine_has_constraint, var_0, adj_affine_has_constraint, adj_0, adj_19);
        // adj: has_constraint = affine_has_constraint[tid]                                       <L 267>
        wp::adj_float64(var_17, adj_17, adj_18);
        // adj: _0 = wp.float64(0.0)                                                              <L 266>
        wp::adj_copy(var_16, adj_14, adj_15);
        wp::adj_load(var_14, adj_14, adj_16);
        wp::adj_address(var_mass_body, var_0, adj_mass_body, adj_0, adj_14);
        // adj: mass = mass_body[tid]                                                             <L 265>
        if (var_13) {
            label0:;
            // adj: return                                                                        <L 264>
        }
        wp::adj_load(var_8, adj_8, adj_12);
        wp::adj_address(var_env_states, var_9, adj_env_states, adj_7, adj_8);
        wp::adj_load(var_7, adj_7, adj_9);
        wp::adj_address(var_body_env_id, var_0, adj_body_env_id, adj_0, adj_7);
        wp::adj_load(var_2, adj_2, adj_6);
        wp::adj_address(var_env_states, var_3, adj_env_states, adj_1, adj_2);
        wp::adj_load(var_1, adj_1, adj_3);
        wp::adj_address(var_body_env_id, var_0, adj_body_env_id, adj_0, adj_1);
        // adj: if env_states[body_env_id[tid]] == ENV_STATE_INVALID or env_states[body_env_id[tid]] == ENV_STATE_NEWTON_SOLVED:  <L 263>
        // adj: tid = wp.tid()                                                                    <L 262>
        // adj: def compute_affine_kinematic_hess(                                                <L 254>
        continue;
    }
}



extern "C" __global__ void init_affine_kinematic_target_kernel_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<bool> var_affine_has_constraint,
    wp::array_t<wp::vec_t<12,wp::float64>> var_affine_kinematic_target_pose,
    wp::array_t<wp::vec_t<12,wp::float64>> var_affine_target_dof,
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
        bool* var_1;
        const bool var_2 = false;
        bool var_3;
        bool var_4;
        wp::vec_t<12,wp::float64>* var_5;
        wp::vec_t<12,wp::float64> var_6;
        wp::vec_t<12,wp::float64> var_7;
        wp::vec_t<3,wp::float64>* var_8;
        wp::vec_t<3,wp::float64> var_9;
        wp::vec_t<3,wp::float64> var_10;
        const wp::float32 var_11 = 0.0;
        wp::float64 var_12;
        const wp::float32 var_13 = 1.0;
        wp::float64 var_14;
        wp::vec_t<3,wp::float64>* var_15;
        wp::vec_t<3,wp::float64> var_16;
        wp::vec_t<3,wp::float64> var_17;
        wp::vec_t<3,wp::float64> var_18;
        wp::vec_t<3,wp::float64> var_19;
        wp::vec_t<3,wp::float64> var_20;
        wp::vec_t<3,wp::float64> var_21;
        wp::vec_t<3,wp::float64> var_22;
        wp::vec_t<3,wp::float64> var_23;
        const wp::int32 var_24 = 0;
        wp::float64 var_25;
        const wp::int32 var_26 = 1;
        wp::float64 var_27;
        const wp::int32 var_28 = 2;
        wp::float64 var_29;
        const wp::int32 var_30 = 3;
        wp::float64 var_31;
        const wp::int32 var_32 = 4;
        wp::float64 var_33;
        const wp::int32 var_34 = 5;
        wp::float64 var_35;
        const wp::int32 var_36 = 6;
        wp::float64 var_37;
        const wp::int32 var_38 = 7;
        wp::float64 var_39;
        const wp::int32 var_40 = 8;
        wp::float64 var_41;
        wp::mat_t<3,3,wp::float64> var_42;
        const wp::int32 var_43 = 9;
        wp::float64 var_44;
        const wp::int32 var_45 = 10;
        wp::float64 var_46;
        const wp::int32 var_47 = 11;
        wp::float64 var_48;
        wp::vec_t<3,wp::float64> var_49;
        wp::vec_t<3,wp::float64> var_50;
        wp::vec_t<3,wp::float64> var_51;
        wp::vec_t<3,wp::float64> var_52;
        wp::vec_t<3,wp::float64> var_53;
        wp::vec_t<3,wp::float64> var_54;
        wp::vec_t<3,wp::float64> var_55;
        wp::vec_t<3,wp::float64> var_56;
        wp::vec_t<3,wp::float64> var_57;
        wp::vec_t<3,wp::float64> var_58;
        wp::vec_t<3,wp::float64> var_59;
        wp::vec_t<3,wp::float64> var_60;
        wp::vec_t<3,wp::float64> var_61;
        wp::vec_t<3,wp::float64> var_62;
        wp::vec_t<3,wp::float64> var_63;
        wp::vec_t<3,wp::float64> var_64;
        wp::vec_t<3,wp::float64> var_65;
        const wp::int32 var_66 = 0;
        wp::float64 var_67;
        const wp::int32 var_68 = 1;
        wp::float64 var_69;
        const wp::int32 var_70 = 2;
        wp::float64 var_71;
        const wp::int32 var_72 = 0;
        wp::float64 var_73;
        const wp::int32 var_74 = 1;
        wp::float64 var_75;
        const wp::int32 var_76 = 2;
        wp::float64 var_77;
        const wp::int32 var_78 = 0;
        wp::float64 var_79;
        const wp::int32 var_80 = 1;
        wp::float64 var_81;
        const wp::int32 var_82 = 2;
        wp::float64 var_83;
        const wp::int32 var_84 = 0;
        wp::float64 var_85;
        const wp::int32 var_86 = 1;
        wp::float64 var_87;
        const wp::int32 var_88 = 2;
        wp::float64 var_89;
        wp::vec_t<12,wp::float64> var_90;
        //---------
        // forward
        // def init_affine_kinematic_target_kernel(                                               <L 16>
        // tid = wp.tid()                                                                         <L 23>
        var_0 = builtin_tid1d();
        // if affine_has_constraint[tid] == False:                                                <L 24>
        var_1 = wp::address(var_affine_has_constraint, var_0);
        var_4 = wp::load(var_1);
        var_3 = (var_4 == var_2);
        if (var_3) {
            // return                                                                             <L 25>
            continue;
        }
        // target_state = affine_kinematic_target_pose[tid]                                       <L 26>
        var_5 = wp::address(var_affine_kinematic_target_pose, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // virtual_center = virtual_object_centers[tid]                                           <L 27>
        var_8 = wp::address(var_virtual_object_centers, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // _0 = wp.float64(0.0)                                                                   <L 28>
        var_12 = wp::float64(var_11);
        // _1 = wp.float64(1.0)                                                                   <L 29>
        var_14 = wp::float64(var_13);
        // rest_O = ABD_centers[tid]                                                              <L 30>
        var_15 = wp::address(var_ABD_centers, var_0);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // rest_A = rest_O + wp.vec3d(_1, _0, _0)                                                 <L 31>
        var_18 = wp::vec_t<3,wp::float64>(var_14, var_12, var_12);
        var_19 = wp::add(var_16, var_18);
        // rest_B = rest_O + wp.vec3d(_0, _1, _0)                                                 <L 32>
        var_20 = wp::vec_t<3,wp::float64>(var_12, var_14, var_12);
        var_21 = wp::add(var_16, var_20);
        // rest_C = rest_O + wp.vec3d(_0, _0, _1)                                                 <L 33>
        var_22 = wp::vec_t<3,wp::float64>(var_12, var_12, var_14);
        var_23 = wp::add(var_16, var_22);
        // R = wp.mat33d(                                                                         <L 34>
        // target_state[0],                                                                       <L 35>
        var_25 = wp::extract(var_6, var_24);
        // target_state[1],                                                                       <L 36>
        var_27 = wp::extract(var_6, var_26);
        // target_state[2],                                                                       <L 37>
        var_29 = wp::extract(var_6, var_28);
        // target_state[3],                                                                       <L 38>
        var_31 = wp::extract(var_6, var_30);
        // target_state[4],                                                                       <L 39>
        var_33 = wp::extract(var_6, var_32);
        // target_state[5],                                                                       <L 40>
        var_35 = wp::extract(var_6, var_34);
        // target_state[6],                                                                       <L 41>
        var_37 = wp::extract(var_6, var_36);
        // target_state[7],                                                                       <L 42>
        var_39 = wp::extract(var_6, var_38);
        // target_state[8],                                                                       <L 43>
        var_41 = wp::extract(var_6, var_40);
        var_42 = wp::mat_t<3,3,wp::float64>(var_25, var_27, var_29, var_31, var_33, var_35, var_37, var_39, var_41);
        // t = wp.vec3d(target_state[9], target_state[10], target_state[11])                      <L 45>
        var_44 = wp::extract(var_6, var_43);
        var_46 = wp::extract(var_6, var_45);
        var_48 = wp::extract(var_6, var_47);
        var_49 = wp::vec_t<3,wp::float64>(var_44, var_46, var_48);
        // O = R @ (rest_O - virtual_center) + virtual_center + t                                 <L 46>
        var_50 = wp::sub(var_16, var_9);
        var_51 = wp::mul(var_42, var_50);
        var_52 = wp::add(var_51, var_9);
        var_53 = wp::add(var_52, var_49);
        // A = R @ (rest_A - virtual_center) + virtual_center + t                                 <L 47>
        var_54 = wp::sub(var_19, var_9);
        var_55 = wp::mul(var_42, var_54);
        var_56 = wp::add(var_55, var_9);
        var_57 = wp::add(var_56, var_49);
        // B = R @ (rest_B - virtual_center) + virtual_center + t                                 <L 48>
        var_58 = wp::sub(var_21, var_9);
        var_59 = wp::mul(var_42, var_58);
        var_60 = wp::add(var_59, var_9);
        var_61 = wp::add(var_60, var_49);
        // C = R @ (rest_C - virtual_center) + virtual_center + t                                 <L 49>
        var_62 = wp::sub(var_23, var_9);
        var_63 = wp::mul(var_42, var_62);
        var_64 = wp::add(var_63, var_9);
        var_65 = wp::add(var_64, var_49);
        // affine_target_dof[tid] = vec12d(O[0], O[1], O[2], A[0], A[1], A[2], B[0], B[1], B[2], C[0], C[1], C[2])       <L 50>
        var_67 = wp::extract(var_53, var_66);
        var_69 = wp::extract(var_53, var_68);
        var_71 = wp::extract(var_53, var_70);
        var_73 = wp::extract(var_57, var_72);
        var_75 = wp::extract(var_57, var_74);
        var_77 = wp::extract(var_57, var_76);
        var_79 = wp::extract(var_61, var_78);
        var_81 = wp::extract(var_61, var_80);
        var_83 = wp::extract(var_61, var_82);
        var_85 = wp::extract(var_65, var_84);
        var_87 = wp::extract(var_65, var_86);
        var_89 = wp::extract(var_65, var_88);
        var_90 = wp::vec_t<12,wp::float64>({var_67, var_69, var_71, var_73, var_75, var_77, var_79, var_81, var_83, var_85, var_87, var_89});
        wp::array_store(var_affine_target_dof, var_0, var_90);
    }
}



extern "C" __global__ void init_affine_kinematic_target_kernel_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<bool> var_affine_has_constraint,
    wp::array_t<wp::vec_t<12,wp::float64>> var_affine_kinematic_target_pose,
    wp::array_t<wp::vec_t<12,wp::float64>> var_affine_target_dof,
    wp::array_t<wp::vec_t<3,wp::float64>> var_virtual_object_centers,
    wp::array_t<wp::vec_t<3,wp::float64>> var_ABD_centers,
    wp::array_t<bool> adj_affine_has_constraint,
    wp::array_t<wp::vec_t<12,wp::float64>> adj_affine_kinematic_target_pose,
    wp::array_t<wp::vec_t<12,wp::float64>> adj_affine_target_dof,
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
        bool* var_1;
        const bool var_2 = false;
        bool var_3;
        bool var_4;
        wp::vec_t<12,wp::float64>* var_5;
        wp::vec_t<12,wp::float64> var_6;
        wp::vec_t<12,wp::float64> var_7;
        wp::vec_t<3,wp::float64>* var_8;
        wp::vec_t<3,wp::float64> var_9;
        wp::vec_t<3,wp::float64> var_10;
        const wp::float32 var_11 = 0.0;
        wp::float64 var_12;
        const wp::float32 var_13 = 1.0;
        wp::float64 var_14;
        wp::vec_t<3,wp::float64>* var_15;
        wp::vec_t<3,wp::float64> var_16;
        wp::vec_t<3,wp::float64> var_17;
        wp::vec_t<3,wp::float64> var_18;
        wp::vec_t<3,wp::float64> var_19;
        wp::vec_t<3,wp::float64> var_20;
        wp::vec_t<3,wp::float64> var_21;
        wp::vec_t<3,wp::float64> var_22;
        wp::vec_t<3,wp::float64> var_23;
        const wp::int32 var_24 = 0;
        wp::float64 var_25;
        const wp::int32 var_26 = 1;
        wp::float64 var_27;
        const wp::int32 var_28 = 2;
        wp::float64 var_29;
        const wp::int32 var_30 = 3;
        wp::float64 var_31;
        const wp::int32 var_32 = 4;
        wp::float64 var_33;
        const wp::int32 var_34 = 5;
        wp::float64 var_35;
        const wp::int32 var_36 = 6;
        wp::float64 var_37;
        const wp::int32 var_38 = 7;
        wp::float64 var_39;
        const wp::int32 var_40 = 8;
        wp::float64 var_41;
        wp::mat_t<3,3,wp::float64> var_42;
        const wp::int32 var_43 = 9;
        wp::float64 var_44;
        const wp::int32 var_45 = 10;
        wp::float64 var_46;
        const wp::int32 var_47 = 11;
        wp::float64 var_48;
        wp::vec_t<3,wp::float64> var_49;
        wp::vec_t<3,wp::float64> var_50;
        wp::vec_t<3,wp::float64> var_51;
        wp::vec_t<3,wp::float64> var_52;
        wp::vec_t<3,wp::float64> var_53;
        wp::vec_t<3,wp::float64> var_54;
        wp::vec_t<3,wp::float64> var_55;
        wp::vec_t<3,wp::float64> var_56;
        wp::vec_t<3,wp::float64> var_57;
        wp::vec_t<3,wp::float64> var_58;
        wp::vec_t<3,wp::float64> var_59;
        wp::vec_t<3,wp::float64> var_60;
        wp::vec_t<3,wp::float64> var_61;
        wp::vec_t<3,wp::float64> var_62;
        wp::vec_t<3,wp::float64> var_63;
        wp::vec_t<3,wp::float64> var_64;
        wp::vec_t<3,wp::float64> var_65;
        const wp::int32 var_66 = 0;
        wp::float64 var_67;
        const wp::int32 var_68 = 1;
        wp::float64 var_69;
        const wp::int32 var_70 = 2;
        wp::float64 var_71;
        const wp::int32 var_72 = 0;
        wp::float64 var_73;
        const wp::int32 var_74 = 1;
        wp::float64 var_75;
        const wp::int32 var_76 = 2;
        wp::float64 var_77;
        const wp::int32 var_78 = 0;
        wp::float64 var_79;
        const wp::int32 var_80 = 1;
        wp::float64 var_81;
        const wp::int32 var_82 = 2;
        wp::float64 var_83;
        const wp::int32 var_84 = 0;
        wp::float64 var_85;
        const wp::int32 var_86 = 1;
        wp::float64 var_87;
        const wp::int32 var_88 = 2;
        wp::float64 var_89;
        wp::vec_t<12,wp::float64> var_90;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        wp::vec_t<12,wp::float64> adj_5 = {};
        wp::vec_t<12,wp::float64> adj_6 = {};
        wp::vec_t<12,wp::float64> adj_7 = {};
        wp::vec_t<3,wp::float64> adj_8 = {};
        wp::vec_t<3,wp::float64> adj_9 = {};
        wp::vec_t<3,wp::float64> adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float64 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float64 adj_14 = {};
        wp::vec_t<3,wp::float64> adj_15 = {};
        wp::vec_t<3,wp::float64> adj_16 = {};
        wp::vec_t<3,wp::float64> adj_17 = {};
        wp::vec_t<3,wp::float64> adj_18 = {};
        wp::vec_t<3,wp::float64> adj_19 = {};
        wp::vec_t<3,wp::float64> adj_20 = {};
        wp::vec_t<3,wp::float64> adj_21 = {};
        wp::vec_t<3,wp::float64> adj_22 = {};
        wp::vec_t<3,wp::float64> adj_23 = {};
        wp::int32 adj_24 = {};
        wp::float64 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::float64 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::float64 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::float64 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::float64 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::float64 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::float64 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::float64 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::float64 adj_41 = {};
        wp::mat_t<3,3,wp::float64> adj_42 = {};
        wp::int32 adj_43 = {};
        wp::float64 adj_44 = {};
        wp::int32 adj_45 = {};
        wp::float64 adj_46 = {};
        wp::int32 adj_47 = {};
        wp::float64 adj_48 = {};
        wp::vec_t<3,wp::float64> adj_49 = {};
        wp::vec_t<3,wp::float64> adj_50 = {};
        wp::vec_t<3,wp::float64> adj_51 = {};
        wp::vec_t<3,wp::float64> adj_52 = {};
        wp::vec_t<3,wp::float64> adj_53 = {};
        wp::vec_t<3,wp::float64> adj_54 = {};
        wp::vec_t<3,wp::float64> adj_55 = {};
        wp::vec_t<3,wp::float64> adj_56 = {};
        wp::vec_t<3,wp::float64> adj_57 = {};
        wp::vec_t<3,wp::float64> adj_58 = {};
        wp::vec_t<3,wp::float64> adj_59 = {};
        wp::vec_t<3,wp::float64> adj_60 = {};
        wp::vec_t<3,wp::float64> adj_61 = {};
        wp::vec_t<3,wp::float64> adj_62 = {};
        wp::vec_t<3,wp::float64> adj_63 = {};
        wp::vec_t<3,wp::float64> adj_64 = {};
        wp::vec_t<3,wp::float64> adj_65 = {};
        wp::int32 adj_66 = {};
        wp::float64 adj_67 = {};
        wp::int32 adj_68 = {};
        wp::float64 adj_69 = {};
        wp::int32 adj_70 = {};
        wp::float64 adj_71 = {};
        wp::int32 adj_72 = {};
        wp::float64 adj_73 = {};
        wp::int32 adj_74 = {};
        wp::float64 adj_75 = {};
        wp::int32 adj_76 = {};
        wp::float64 adj_77 = {};
        wp::int32 adj_78 = {};
        wp::float64 adj_79 = {};
        wp::int32 adj_80 = {};
        wp::float64 adj_81 = {};
        wp::int32 adj_82 = {};
        wp::float64 adj_83 = {};
        wp::int32 adj_84 = {};
        wp::float64 adj_85 = {};
        wp::int32 adj_86 = {};
        wp::float64 adj_87 = {};
        wp::int32 adj_88 = {};
        wp::float64 adj_89 = {};
        wp::vec_t<12,wp::float64> adj_90 = {};
        //---------
        // forward
        // def init_affine_kinematic_target_kernel(                                               <L 16>
        // tid = wp.tid()                                                                         <L 23>
        var_0 = builtin_tid1d();
        // if affine_has_constraint[tid] == False:                                                <L 24>
        var_1 = wp::address(var_affine_has_constraint, var_0);
        var_4 = wp::load(var_1);
        var_3 = (var_4 == var_2);
        if (var_3) {
            // return                                                                             <L 25>
            goto label0;
        }
        // target_state = affine_kinematic_target_pose[tid]                                       <L 26>
        var_5 = wp::address(var_affine_kinematic_target_pose, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // virtual_center = virtual_object_centers[tid]                                           <L 27>
        var_8 = wp::address(var_virtual_object_centers, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // _0 = wp.float64(0.0)                                                                   <L 28>
        var_12 = wp::float64(var_11);
        // _1 = wp.float64(1.0)                                                                   <L 29>
        var_14 = wp::float64(var_13);
        // rest_O = ABD_centers[tid]                                                              <L 30>
        var_15 = wp::address(var_ABD_centers, var_0);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // rest_A = rest_O + wp.vec3d(_1, _0, _0)                                                 <L 31>
        var_18 = wp::vec_t<3,wp::float64>(var_14, var_12, var_12);
        var_19 = wp::add(var_16, var_18);
        // rest_B = rest_O + wp.vec3d(_0, _1, _0)                                                 <L 32>
        var_20 = wp::vec_t<3,wp::float64>(var_12, var_14, var_12);
        var_21 = wp::add(var_16, var_20);
        // rest_C = rest_O + wp.vec3d(_0, _0, _1)                                                 <L 33>
        var_22 = wp::vec_t<3,wp::float64>(var_12, var_12, var_14);
        var_23 = wp::add(var_16, var_22);
        // R = wp.mat33d(                                                                         <L 34>
        // target_state[0],                                                                       <L 35>
        var_25 = wp::extract(var_6, var_24);
        // target_state[1],                                                                       <L 36>
        var_27 = wp::extract(var_6, var_26);
        // target_state[2],                                                                       <L 37>
        var_29 = wp::extract(var_6, var_28);
        // target_state[3],                                                                       <L 38>
        var_31 = wp::extract(var_6, var_30);
        // target_state[4],                                                                       <L 39>
        var_33 = wp::extract(var_6, var_32);
        // target_state[5],                                                                       <L 40>
        var_35 = wp::extract(var_6, var_34);
        // target_state[6],                                                                       <L 41>
        var_37 = wp::extract(var_6, var_36);
        // target_state[7],                                                                       <L 42>
        var_39 = wp::extract(var_6, var_38);
        // target_state[8],                                                                       <L 43>
        var_41 = wp::extract(var_6, var_40);
        var_42 = wp::mat_t<3,3,wp::float64>(var_25, var_27, var_29, var_31, var_33, var_35, var_37, var_39, var_41);
        // t = wp.vec3d(target_state[9], target_state[10], target_state[11])                      <L 45>
        var_44 = wp::extract(var_6, var_43);
        var_46 = wp::extract(var_6, var_45);
        var_48 = wp::extract(var_6, var_47);
        var_49 = wp::vec_t<3,wp::float64>(var_44, var_46, var_48);
        // O = R @ (rest_O - virtual_center) + virtual_center + t                                 <L 46>
        var_50 = wp::sub(var_16, var_9);
        var_51 = wp::mul(var_42, var_50);
        var_52 = wp::add(var_51, var_9);
        var_53 = wp::add(var_52, var_49);
        // A = R @ (rest_A - virtual_center) + virtual_center + t                                 <L 47>
        var_54 = wp::sub(var_19, var_9);
        var_55 = wp::mul(var_42, var_54);
        var_56 = wp::add(var_55, var_9);
        var_57 = wp::add(var_56, var_49);
        // B = R @ (rest_B - virtual_center) + virtual_center + t                                 <L 48>
        var_58 = wp::sub(var_21, var_9);
        var_59 = wp::mul(var_42, var_58);
        var_60 = wp::add(var_59, var_9);
        var_61 = wp::add(var_60, var_49);
        // C = R @ (rest_C - virtual_center) + virtual_center + t                                 <L 49>
        var_62 = wp::sub(var_23, var_9);
        var_63 = wp::mul(var_42, var_62);
        var_64 = wp::add(var_63, var_9);
        var_65 = wp::add(var_64, var_49);
        // affine_target_dof[tid] = vec12d(O[0], O[1], O[2], A[0], A[1], A[2], B[0], B[1], B[2], C[0], C[1], C[2])       <L 50>
        var_67 = wp::extract(var_53, var_66);
        var_69 = wp::extract(var_53, var_68);
        var_71 = wp::extract(var_53, var_70);
        var_73 = wp::extract(var_57, var_72);
        var_75 = wp::extract(var_57, var_74);
        var_77 = wp::extract(var_57, var_76);
        var_79 = wp::extract(var_61, var_78);
        var_81 = wp::extract(var_61, var_80);
        var_83 = wp::extract(var_61, var_82);
        var_85 = wp::extract(var_65, var_84);
        var_87 = wp::extract(var_65, var_86);
        var_89 = wp::extract(var_65, var_88);
        var_90 = wp::vec_t<12,wp::float64>({var_67, var_69, var_71, var_73, var_75, var_77, var_79, var_81, var_83, var_85, var_87, var_89});
        // wp::array_store(var_affine_target_dof, var_0, var_90);
        //---------
        // reverse
        wp::adj_array_store(var_affine_target_dof, var_0, var_90, adj_affine_target_dof, adj_0, adj_90);
        wp::adj_vec_t({var_67, var_69, var_71, var_73, var_75, var_77, var_79, var_81, var_83, var_85, var_87, var_89}, {&adj_67, &adj_69, &adj_71, &adj_73, &adj_75, &adj_77, &adj_79, &adj_81, &adj_83, &adj_85, &adj_87, &adj_89}, adj_90);
        wp::adj_extract(var_65, var_88, adj_65, adj_88, adj_89);
        wp::adj_extract(var_65, var_86, adj_65, adj_86, adj_87);
        wp::adj_extract(var_65, var_84, adj_65, adj_84, adj_85);
        wp::adj_extract(var_61, var_82, adj_61, adj_82, adj_83);
        wp::adj_extract(var_61, var_80, adj_61, adj_80, adj_81);
        wp::adj_extract(var_61, var_78, adj_61, adj_78, adj_79);
        wp::adj_extract(var_57, var_76, adj_57, adj_76, adj_77);
        wp::adj_extract(var_57, var_74, adj_57, adj_74, adj_75);
        wp::adj_extract(var_57, var_72, adj_57, adj_72, adj_73);
        wp::adj_extract(var_53, var_70, adj_53, adj_70, adj_71);
        wp::adj_extract(var_53, var_68, adj_53, adj_68, adj_69);
        wp::adj_extract(var_53, var_66, adj_53, adj_66, adj_67);
        // adj: affine_target_dof[tid] = vec12d(O[0], O[1], O[2], A[0], A[1], A[2], B[0], B[1], B[2], C[0], C[1], C[2])  <L 50>
        wp::adj_add(var_64, var_49, adj_64, adj_49, adj_65);
        wp::adj_add(var_63, var_9, adj_63, adj_9, adj_64);
        wp::adj_mul(var_42, var_62, adj_42, adj_62, adj_63);
        wp::adj_sub(var_23, var_9, adj_23, adj_9, adj_62);
        // adj: C = R @ (rest_C - virtual_center) + virtual_center + t                            <L 49>
        wp::adj_add(var_60, var_49, adj_60, adj_49, adj_61);
        wp::adj_add(var_59, var_9, adj_59, adj_9, adj_60);
        wp::adj_mul(var_42, var_58, adj_42, adj_58, adj_59);
        wp::adj_sub(var_21, var_9, adj_21, adj_9, adj_58);
        // adj: B = R @ (rest_B - virtual_center) + virtual_center + t                            <L 48>
        wp::adj_add(var_56, var_49, adj_56, adj_49, adj_57);
        wp::adj_add(var_55, var_9, adj_55, adj_9, adj_56);
        wp::adj_mul(var_42, var_54, adj_42, adj_54, adj_55);
        wp::adj_sub(var_19, var_9, adj_19, adj_9, adj_54);
        // adj: A = R @ (rest_A - virtual_center) + virtual_center + t                            <L 47>
        wp::adj_add(var_52, var_49, adj_52, adj_49, adj_53);
        wp::adj_add(var_51, var_9, adj_51, adj_9, adj_52);
        wp::adj_mul(var_42, var_50, adj_42, adj_50, adj_51);
        wp::adj_sub(var_16, var_9, adj_16, adj_9, adj_50);
        // adj: O = R @ (rest_O - virtual_center) + virtual_center + t                            <L 46>
        wp::adj_vec_t(var_44, var_46, var_48, adj_44, adj_46, adj_48, adj_49);
        wp::adj_extract(var_6, var_47, adj_6, adj_47, adj_48);
        wp::adj_extract(var_6, var_45, adj_6, adj_45, adj_46);
        wp::adj_extract(var_6, var_43, adj_6, adj_43, adj_44);
        // adj: t = wp.vec3d(target_state[9], target_state[10], target_state[11])                 <L 45>
        wp::adj_mat_t(var_25, var_27, var_29, var_31, var_33, var_35, var_37, var_39, var_41, adj_25, adj_27, adj_29, adj_31, adj_33, adj_35, adj_37, adj_39, adj_41, adj_42);
        wp::adj_extract(var_6, var_40, adj_6, adj_40, adj_41);
        // adj: target_state[8],                                                                  <L 43>
        wp::adj_extract(var_6, var_38, adj_6, adj_38, adj_39);
        // adj: target_state[7],                                                                  <L 42>
        wp::adj_extract(var_6, var_36, adj_6, adj_36, adj_37);
        // adj: target_state[6],                                                                  <L 41>
        wp::adj_extract(var_6, var_34, adj_6, adj_34, adj_35);
        // adj: target_state[5],                                                                  <L 40>
        wp::adj_extract(var_6, var_32, adj_6, adj_32, adj_33);
        // adj: target_state[4],                                                                  <L 39>
        wp::adj_extract(var_6, var_30, adj_6, adj_30, adj_31);
        // adj: target_state[3],                                                                  <L 38>
        wp::adj_extract(var_6, var_28, adj_6, adj_28, adj_29);
        // adj: target_state[2],                                                                  <L 37>
        wp::adj_extract(var_6, var_26, adj_6, adj_26, adj_27);
        // adj: target_state[1],                                                                  <L 36>
        wp::adj_extract(var_6, var_24, adj_6, adj_24, adj_25);
        // adj: target_state[0],                                                                  <L 35>
        // adj: R = wp.mat33d(                                                                    <L 34>
        wp::adj_add(var_16, var_22, adj_16, adj_22, adj_23);
        wp::adj_vec_t(var_12, var_12, var_14, adj_12, adj_12, adj_14, adj_22);
        // adj: rest_C = rest_O + wp.vec3d(_0, _0, _1)                                            <L 33>
        wp::adj_add(var_16, var_20, adj_16, adj_20, adj_21);
        wp::adj_vec_t(var_12, var_14, var_12, adj_12, adj_14, adj_12, adj_20);
        // adj: rest_B = rest_O + wp.vec3d(_0, _1, _0)                                            <L 32>
        wp::adj_add(var_16, var_18, adj_16, adj_18, adj_19);
        wp::adj_vec_t(var_14, var_12, var_12, adj_14, adj_12, adj_12, adj_18);
        // adj: rest_A = rest_O + wp.vec3d(_1, _0, _0)                                            <L 31>
        wp::adj_copy(var_17, adj_15, adj_16);
        wp::adj_load(var_15, adj_15, adj_17);
        wp::adj_address(var_ABD_centers, var_0, adj_ABD_centers, adj_0, adj_15);
        // adj: rest_O = ABD_centers[tid]                                                         <L 30>
        wp::adj_float64(var_13, adj_13, adj_14);
        // adj: _1 = wp.float64(1.0)                                                              <L 29>
        wp::adj_float64(var_11, adj_11, adj_12);
        // adj: _0 = wp.float64(0.0)                                                              <L 28>
        wp::adj_copy(var_10, adj_8, adj_9);
        wp::adj_load(var_8, adj_8, adj_10);
        wp::adj_address(var_virtual_object_centers, var_0, adj_virtual_object_centers, adj_0, adj_8);
        // adj: virtual_center = virtual_object_centers[tid]                                      <L 27>
        wp::adj_copy(var_7, adj_5, adj_6);
        wp::adj_load(var_5, adj_5, adj_7);
        wp::adj_address(var_affine_kinematic_target_pose, var_0, adj_affine_kinematic_target_pose, adj_0, adj_5);
        // adj: target_state = affine_kinematic_target_pose[tid]                                  <L 26>
        if (var_3) {
            label0:;
            // adj: return                                                                        <L 25>
        }
        wp::adj_load(var_1, adj_1, adj_4);
        wp::adj_address(var_affine_has_constraint, var_0, adj_affine_has_constraint, adj_0, adj_1);
        // adj: if affine_has_constraint[tid] == False:                                           <L 24>
        // adj: tid = wp.tid()                                                                    <L 23>
        // adj: def init_affine_kinematic_target_kernel(                                          <L 16>
        continue;
    }
}



extern "C" __global__ void compute_soft_kinematic_hess_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<bool> var_soft_has_constraint,
    wp::float64 var_weight,
    COOMatrix3x3_0df4b45d var_hess_soft_diag,
    wp::int32 var_affine_verts_num,
    wp::array_t<wp::float64> var_soft_verts_mass,
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
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 1;
        bool var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        const wp::int32 var_12 = 2;
        bool var_13;
        wp::int32 var_14;
        bool var_15;
        wp::float64* var_16;
        wp::float64 var_17;
        wp::float64 var_18;
        const wp::float32 var_19 = 0.0;
        wp::float64 var_20;
        bool* var_21;
        bool var_22;
        bool var_23;
        wp::float64 var_24;
        wp::mat_t<3,3,wp::float64> var_25;
        //---------
        // forward
        // def compute_soft_kinematic_hess(                                                       <L 276>
        // tid = wp.tid()                                                                         <L 285>
        var_0 = builtin_tid1d();
        // if env_states[node2env[tid + affine_verts_num]] == ENV_STATE_INVALID or env_states[node2env[tid + affine_verts_num]] == ENV_STATE_NEWTON_SOLVED:       <L 286>
        var_1 = wp::add(var_0, var_affine_verts_num);
        var_2 = wp::address(var_node2env, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::address(var_env_states, var_4);
        var_7 = wp::load(var_3);
        var_6 = (var_7 == var_5);
        var_8 = wp::add(var_0, var_affine_verts_num);
        var_9 = wp::address(var_node2env, var_8);
        var_11 = wp::load(var_9);
        var_10 = wp::address(var_env_states, var_11);
        var_14 = wp::load(var_10);
        var_13 = (var_14 == var_12);
        var_15 = var_6 || var_13;
        if (var_15) {
            // return                                                                             <L 287>
            continue;
        }
        // mass = soft_verts_mass[tid]                                                            <L 288>
        var_16 = wp::address(var_soft_verts_mass, var_0);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // _0 = wp.float64(0.0)                                                                   <L 289>
        var_20 = wp::float64(var_19);
        // has_constraint = soft_has_constraint[tid]                                              <L 290>
        var_21 = wp::address(var_soft_has_constraint, var_0);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // weighted_mass = mass * weight                                                          <L 291>
        var_24 = wp::mul(var_17, var_weight);
        // mat3 = wp.mat33d(weighted_mass, _0, _0, _0, weighted_mass, _0, _0, _0, weighted_mass)       <L 292>
        var_25 = wp::mat_t<3,3,wp::float64>(var_24, var_20, var_20, var_20, var_24, var_20, var_20, var_20, var_24);
        // if has_constraint:                                                                     <L 293>
        if (var_22) {
            // matrix.COOMatrix3x3_atomic_add(hess_soft_diag, tid, mat3)                          <L 294>
            COOMatrix3x3_atomic_add_0(var_hess_soft_diag, var_0, var_25);
        }
    }
}



extern "C" __global__ void compute_soft_kinematic_hess_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<bool> var_soft_has_constraint,
    wp::float64 var_weight,
    COOMatrix3x3_0df4b45d var_hess_soft_diag,
    wp::int32 var_affine_verts_num,
    wp::array_t<wp::float64> var_soft_verts_mass,
    wp::array_t<wp::int32> var_node2env,
    wp::array_t<wp::int32> var_env_states,
    wp::array_t<bool> adj_soft_has_constraint,
    wp::float64 adj_weight,
    COOMatrix3x3_0df4b45d adj_hess_soft_diag,
    wp::int32 adj_affine_verts_num,
    wp::array_t<wp::float64> adj_soft_verts_mass,
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
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 1;
        bool var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        const wp::int32 var_12 = 2;
        bool var_13;
        wp::int32 var_14;
        bool var_15;
        wp::float64* var_16;
        wp::float64 var_17;
        wp::float64 var_18;
        const wp::float32 var_19 = 0.0;
        wp::float64 var_20;
        bool* var_21;
        bool var_22;
        bool var_23;
        wp::float64 var_24;
        wp::mat_t<3,3,wp::float64> var_25;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        bool adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        bool adj_13 = {};
        wp::int32 adj_14 = {};
        bool adj_15 = {};
        wp::float64 adj_16 = {};
        wp::float64 adj_17 = {};
        wp::float64 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float64 adj_20 = {};
        bool adj_21 = {};
        bool adj_22 = {};
        bool adj_23 = {};
        wp::float64 adj_24 = {};
        wp::mat_t<3,3,wp::float64> adj_25 = {};
        //---------
        // forward
        // def compute_soft_kinematic_hess(                                                       <L 276>
        // tid = wp.tid()                                                                         <L 285>
        var_0 = builtin_tid1d();
        // if env_states[node2env[tid + affine_verts_num]] == ENV_STATE_INVALID or env_states[node2env[tid + affine_verts_num]] == ENV_STATE_NEWTON_SOLVED:       <L 286>
        var_1 = wp::add(var_0, var_affine_verts_num);
        var_2 = wp::address(var_node2env, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::address(var_env_states, var_4);
        var_7 = wp::load(var_3);
        var_6 = (var_7 == var_5);
        var_8 = wp::add(var_0, var_affine_verts_num);
        var_9 = wp::address(var_node2env, var_8);
        var_11 = wp::load(var_9);
        var_10 = wp::address(var_env_states, var_11);
        var_14 = wp::load(var_10);
        var_13 = (var_14 == var_12);
        var_15 = var_6 || var_13;
        if (var_15) {
            // return                                                                             <L 287>
            goto label0;
        }
        // mass = soft_verts_mass[tid]                                                            <L 288>
        var_16 = wp::address(var_soft_verts_mass, var_0);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // _0 = wp.float64(0.0)                                                                   <L 289>
        var_20 = wp::float64(var_19);
        // has_constraint = soft_has_constraint[tid]                                              <L 290>
        var_21 = wp::address(var_soft_has_constraint, var_0);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // weighted_mass = mass * weight                                                          <L 291>
        var_24 = wp::mul(var_17, var_weight);
        // mat3 = wp.mat33d(weighted_mass, _0, _0, _0, weighted_mass, _0, _0, _0, weighted_mass)       <L 292>
        var_25 = wp::mat_t<3,3,wp::float64>(var_24, var_20, var_20, var_20, var_24, var_20, var_20, var_20, var_24);
        // if has_constraint:                                                                     <L 293>
        if (var_22) {
            // matrix.COOMatrix3x3_atomic_add(hess_soft_diag, tid, mat3)                          <L 294>
            COOMatrix3x3_atomic_add_0(var_hess_soft_diag, var_0, var_25);
        }
        //---------
        // reverse
        if (var_22) {
            adj_COOMatrix3x3_atomic_add_0(var_hess_soft_diag, var_0, var_25, adj_hess_soft_diag, adj_0, adj_25);
            // adj: matrix.COOMatrix3x3_atomic_add(hess_soft_diag, tid, mat3)                     <L 294>
        }
        // adj: if has_constraint:                                                                <L 293>
        wp::adj_mat_t(var_24, var_20, var_20, var_20, var_24, var_20, var_20, var_20, var_24, adj_24, adj_20, adj_20, adj_20, adj_24, adj_20, adj_20, adj_20, adj_24, adj_25);
        // adj: mat3 = wp.mat33d(weighted_mass, _0, _0, _0, weighted_mass, _0, _0, _0, weighted_mass)  <L 292>
        wp::adj_mul(var_17, var_weight, adj_17, adj_weight, adj_24);
        // adj: weighted_mass = mass * weight                                                     <L 291>
        wp::adj_copy(var_23, adj_21, adj_22);
        wp::adj_load(var_21, adj_21, adj_23);
        wp::adj_address(var_soft_has_constraint, var_0, adj_soft_has_constraint, adj_0, adj_21);
        // adj: has_constraint = soft_has_constraint[tid]                                         <L 290>
        wp::adj_float64(var_19, adj_19, adj_20);
        // adj: _0 = wp.float64(0.0)                                                              <L 289>
        wp::adj_copy(var_18, adj_16, adj_17);
        wp::adj_load(var_16, adj_16, adj_18);
        wp::adj_address(var_soft_verts_mass, var_0, adj_soft_verts_mass, adj_0, adj_16);
        // adj: mass = soft_verts_mass[tid]                                                       <L 288>
        if (var_15) {
            label0:;
            // adj: return                                                                        <L 287>
        }
        wp::adj_load(var_10, adj_10, adj_14);
        wp::adj_address(var_env_states, var_11, adj_env_states, adj_9, adj_10);
        wp::adj_load(var_9, adj_9, adj_11);
        wp::adj_address(var_node2env, var_8, adj_node2env, adj_8, adj_9);
        wp::adj_add(var_0, var_affine_verts_num, adj_0, adj_affine_verts_num, adj_8);
        wp::adj_load(var_3, adj_3, adj_7);
        wp::adj_address(var_env_states, var_4, adj_env_states, adj_2, adj_3);
        wp::adj_load(var_2, adj_2, adj_4);
        wp::adj_address(var_node2env, var_1, adj_node2env, adj_1, adj_2);
        wp::adj_add(var_0, var_affine_verts_num, adj_0, adj_affine_verts_num, adj_1);
        // adj: if env_states[node2env[tid + affine_verts_num]] == ENV_STATE_INVALID or env_states[node2env[tid + affine_verts_num]] == ENV_STATE_NEWTON_SOLVED:  <L 286>
        // adj: tid = wp.tid()                                                                    <L 285>
        // adj: def compute_soft_kinematic_hess(                                                  <L 276>
        continue;
    }
}

