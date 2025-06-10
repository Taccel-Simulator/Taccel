
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

// /home/chang/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/utils/wp_math.py:187
static CUDA_CALLABLE wp::mat_t<3,3,wp::float64> col_stack3_1(
    wp::vec_t<3,wp::float64> var_vec0,
    wp::vec_t<3,wp::float64> var_vec1,
    wp::vec_t<3,wp::float64> var_vec2)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float64 var_1;
    const wp::int32 var_2 = 0;
    wp::float64 var_3;
    const wp::int32 var_4 = 0;
    wp::float64 var_5;
    const wp::int32 var_6 = 1;
    wp::float64 var_7;
    const wp::int32 var_8 = 1;
    wp::float64 var_9;
    const wp::int32 var_10 = 1;
    wp::float64 var_11;
    const wp::int32 var_12 = 2;
    wp::float64 var_13;
    const wp::int32 var_14 = 2;
    wp::float64 var_15;
    const wp::int32 var_16 = 2;
    wp::float64 var_17;
    wp::mat_t<3,3,wp::float64> var_18;
    //---------
    // forward
    // def col_stack3(vec0: wp.vec3d, vec1: wp.vec3d, vec2: wp.vec3d):                        <L 188>
    // return mat33d(vec0[0], vec1[0], vec2[0], vec0[1], vec1[1], vec2[1], vec0[2], vec1[2], vec2[2])       <L 189>
    var_1 = wp::extract(var_vec0, var_0);
    var_3 = wp::extract(var_vec1, var_2);
    var_5 = wp::extract(var_vec2, var_4);
    var_7 = wp::extract(var_vec0, var_6);
    var_9 = wp::extract(var_vec1, var_8);
    var_11 = wp::extract(var_vec2, var_10);
    var_13 = wp::extract(var_vec0, var_12);
    var_15 = wp::extract(var_vec1, var_14);
    var_17 = wp::extract(var_vec2, var_16);
    var_18 = wp::mat_t<3,3,wp::float64>(var_1, var_3, var_5, var_7, var_9, var_11, var_13, var_15, var_17);
    return var_18;
}


// /home/chang/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/utils/wp_math.py:187
static CUDA_CALLABLE void adj_col_stack3_1(
    wp::vec_t<3,wp::float64> var_vec0,
    wp::vec_t<3,wp::float64> var_vec1,
    wp::vec_t<3,wp::float64> var_vec2,
    wp::vec_t<3,wp::float64> & adj_vec0,
    wp::vec_t<3,wp::float64> & adj_vec1,
    wp::vec_t<3,wp::float64> & adj_vec2,
    wp::mat_t<3,3,wp::float64> & adj_ret)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float64 var_1;
    const wp::int32 var_2 = 0;
    wp::float64 var_3;
    const wp::int32 var_4 = 0;
    wp::float64 var_5;
    const wp::int32 var_6 = 1;
    wp::float64 var_7;
    const wp::int32 var_8 = 1;
    wp::float64 var_9;
    const wp::int32 var_10 = 1;
    wp::float64 var_11;
    const wp::int32 var_12 = 2;
    wp::float64 var_13;
    const wp::int32 var_14 = 2;
    wp::float64 var_15;
    const wp::int32 var_16 = 2;
    wp::float64 var_17;
    wp::mat_t<3,3,wp::float64> var_18;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::float64 adj_1 = {};
    wp::int32 adj_2 = {};
    wp::float64 adj_3 = {};
    wp::int32 adj_4 = {};
    wp::float64 adj_5 = {};
    wp::int32 adj_6 = {};
    wp::float64 adj_7 = {};
    wp::int32 adj_8 = {};
    wp::float64 adj_9 = {};
    wp::int32 adj_10 = {};
    wp::float64 adj_11 = {};
    wp::int32 adj_12 = {};
    wp::float64 adj_13 = {};
    wp::int32 adj_14 = {};
    wp::float64 adj_15 = {};
    wp::int32 adj_16 = {};
    wp::float64 adj_17 = {};
    wp::mat_t<3,3,wp::float64> adj_18 = {};
    //---------
    // forward
    // def col_stack3(vec0: wp.vec3d, vec1: wp.vec3d, vec2: wp.vec3d):                        <L 188>
    // return mat33d(vec0[0], vec1[0], vec2[0], vec0[1], vec1[1], vec2[1], vec0[2], vec1[2], vec2[2])       <L 189>
    var_1 = wp::extract(var_vec0, var_0);
    var_3 = wp::extract(var_vec1, var_2);
    var_5 = wp::extract(var_vec2, var_4);
    var_7 = wp::extract(var_vec0, var_6);
    var_9 = wp::extract(var_vec1, var_8);
    var_11 = wp::extract(var_vec2, var_10);
    var_13 = wp::extract(var_vec0, var_12);
    var_15 = wp::extract(var_vec1, var_14);
    var_17 = wp::extract(var_vec2, var_16);
    var_18 = wp::mat_t<3,3,wp::float64>(var_1, var_3, var_5, var_7, var_9, var_11, var_13, var_15, var_17);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_18 += adj_ret;
    wp::adj_mat_t(var_1, var_3, var_5, var_7, var_9, var_11, var_13, var_15, var_17, adj_1, adj_3, adj_5, adj_7, adj_9, adj_11, adj_13, adj_15, adj_17, adj_18);
    wp::adj_extract(var_vec2, var_16, adj_vec2, adj_16, adj_17);
    wp::adj_extract(var_vec1, var_14, adj_vec1, adj_14, adj_15);
    wp::adj_extract(var_vec0, var_12, adj_vec0, adj_12, adj_13);
    wp::adj_extract(var_vec2, var_10, adj_vec2, adj_10, adj_11);
    wp::adj_extract(var_vec1, var_8, adj_vec1, adj_8, adj_9);
    wp::adj_extract(var_vec0, var_6, adj_vec0, adj_6, adj_7);
    wp::adj_extract(var_vec2, var_4, adj_vec2, adj_4, adj_5);
    wp::adj_extract(var_vec1, var_2, adj_vec1, adj_2, adj_3);
    wp::adj_extract(var_vec0, var_0, adj_vec0, adj_0, adj_1);
    // adj: return mat33d(vec0[0], vec1[0], vec2[0], vec0[1], vec1[1], vec2[1], vec0[2], vec1[2], vec2[2])  <L 189>
    // adj: def col_stack3(vec0: wp.vec3d, vec1: wp.vec3d, vec2: wp.vec3d):                   <L 188>
    return;
}


// /home/chang/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/utils/wp_math.py:92
static CUDA_CALLABLE wp::vec_t<12,wp::float64> cat_4_vec3d_0(
    wp::vec_t<3,wp::float64> var_a,
    wp::vec_t<3,wp::float64> var_b,
    wp::vec_t<3,wp::float64> var_c,
    wp::vec_t<3,wp::float64> var_d)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float64 var_1;
    const wp::int32 var_2 = 1;
    wp::float64 var_3;
    const wp::int32 var_4 = 2;
    wp::float64 var_5;
    const wp::int32 var_6 = 0;
    wp::float64 var_7;
    const wp::int32 var_8 = 1;
    wp::float64 var_9;
    const wp::int32 var_10 = 2;
    wp::float64 var_11;
    const wp::int32 var_12 = 0;
    wp::float64 var_13;
    const wp::int32 var_14 = 1;
    wp::float64 var_15;
    const wp::int32 var_16 = 2;
    wp::float64 var_17;
    const wp::int32 var_18 = 0;
    wp::float64 var_19;
    const wp::int32 var_20 = 1;
    wp::float64 var_21;
    const wp::int32 var_22 = 2;
    wp::float64 var_23;
    wp::vec_t<12,wp::float64> var_24;
    //---------
    // forward
    // def cat_4_vec3d(                                                                       <L 93>
    // return vec12d(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], d[0], d[1], d[2])       <L 99>
    var_1 = wp::extract(var_a, var_0);
    var_3 = wp::extract(var_a, var_2);
    var_5 = wp::extract(var_a, var_4);
    var_7 = wp::extract(var_b, var_6);
    var_9 = wp::extract(var_b, var_8);
    var_11 = wp::extract(var_b, var_10);
    var_13 = wp::extract(var_c, var_12);
    var_15 = wp::extract(var_c, var_14);
    var_17 = wp::extract(var_c, var_16);
    var_19 = wp::extract(var_d, var_18);
    var_21 = wp::extract(var_d, var_20);
    var_23 = wp::extract(var_d, var_22);
    var_24 = wp::vec_t<12,wp::float64>({var_1, var_3, var_5, var_7, var_9, var_11, var_13, var_15, var_17, var_19, var_21, var_23});
    return var_24;
}


// /home/chang/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/utils/wp_math.py:92
static CUDA_CALLABLE void adj_cat_4_vec3d_0(
    wp::vec_t<3,wp::float64> var_a,
    wp::vec_t<3,wp::float64> var_b,
    wp::vec_t<3,wp::float64> var_c,
    wp::vec_t<3,wp::float64> var_d,
    wp::vec_t<3,wp::float64> & adj_a,
    wp::vec_t<3,wp::float64> & adj_b,
    wp::vec_t<3,wp::float64> & adj_c,
    wp::vec_t<3,wp::float64> & adj_d,
    wp::vec_t<12,wp::float64> & adj_ret)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float64 var_1;
    const wp::int32 var_2 = 1;
    wp::float64 var_3;
    const wp::int32 var_4 = 2;
    wp::float64 var_5;
    const wp::int32 var_6 = 0;
    wp::float64 var_7;
    const wp::int32 var_8 = 1;
    wp::float64 var_9;
    const wp::int32 var_10 = 2;
    wp::float64 var_11;
    const wp::int32 var_12 = 0;
    wp::float64 var_13;
    const wp::int32 var_14 = 1;
    wp::float64 var_15;
    const wp::int32 var_16 = 2;
    wp::float64 var_17;
    const wp::int32 var_18 = 0;
    wp::float64 var_19;
    const wp::int32 var_20 = 1;
    wp::float64 var_21;
    const wp::int32 var_22 = 2;
    wp::float64 var_23;
    wp::vec_t<12,wp::float64> var_24;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::float64 adj_1 = {};
    wp::int32 adj_2 = {};
    wp::float64 adj_3 = {};
    wp::int32 adj_4 = {};
    wp::float64 adj_5 = {};
    wp::int32 adj_6 = {};
    wp::float64 adj_7 = {};
    wp::int32 adj_8 = {};
    wp::float64 adj_9 = {};
    wp::int32 adj_10 = {};
    wp::float64 adj_11 = {};
    wp::int32 adj_12 = {};
    wp::float64 adj_13 = {};
    wp::int32 adj_14 = {};
    wp::float64 adj_15 = {};
    wp::int32 adj_16 = {};
    wp::float64 adj_17 = {};
    wp::int32 adj_18 = {};
    wp::float64 adj_19 = {};
    wp::int32 adj_20 = {};
    wp::float64 adj_21 = {};
    wp::int32 adj_22 = {};
    wp::float64 adj_23 = {};
    wp::vec_t<12,wp::float64> adj_24 = {};
    //---------
    // forward
    // def cat_4_vec3d(                                                                       <L 93>
    // return vec12d(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], d[0], d[1], d[2])       <L 99>
    var_1 = wp::extract(var_a, var_0);
    var_3 = wp::extract(var_a, var_2);
    var_5 = wp::extract(var_a, var_4);
    var_7 = wp::extract(var_b, var_6);
    var_9 = wp::extract(var_b, var_8);
    var_11 = wp::extract(var_b, var_10);
    var_13 = wp::extract(var_c, var_12);
    var_15 = wp::extract(var_c, var_14);
    var_17 = wp::extract(var_c, var_16);
    var_19 = wp::extract(var_d, var_18);
    var_21 = wp::extract(var_d, var_20);
    var_23 = wp::extract(var_d, var_22);
    var_24 = wp::vec_t<12,wp::float64>({var_1, var_3, var_5, var_7, var_9, var_11, var_13, var_15, var_17, var_19, var_21, var_23});
    goto label0;
    //---------
    // reverse
    label0:;
    adj_24 += adj_ret;
    wp::adj_vec_t({var_1, var_3, var_5, var_7, var_9, var_11, var_13, var_15, var_17, var_19, var_21, var_23}, {&adj_1, &adj_3, &adj_5, &adj_7, &adj_9, &adj_11, &adj_13, &adj_15, &adj_17, &adj_19, &adj_21, &adj_23}, adj_24);
    wp::adj_extract(var_d, var_22, adj_d, adj_22, adj_23);
    wp::adj_extract(var_d, var_20, adj_d, adj_20, adj_21);
    wp::adj_extract(var_d, var_18, adj_d, adj_18, adj_19);
    wp::adj_extract(var_c, var_16, adj_c, adj_16, adj_17);
    wp::adj_extract(var_c, var_14, adj_c, adj_14, adj_15);
    wp::adj_extract(var_c, var_12, adj_c, adj_12, adj_13);
    wp::adj_extract(var_b, var_10, adj_b, adj_10, adj_11);
    wp::adj_extract(var_b, var_8, adj_b, adj_8, adj_9);
    wp::adj_extract(var_b, var_6, adj_b, adj_6, adj_7);
    wp::adj_extract(var_a, var_4, adj_a, adj_4, adj_5);
    wp::adj_extract(var_a, var_2, adj_a, adj_2, adj_3);
    wp::adj_extract(var_a, var_0, adj_a, adj_0, adj_1);
    // adj: return vec12d(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], d[0], d[1], d[2])  <L 99>
    // adj: def cat_4_vec3d(                                                                  <L 93>
    return;
}



extern "C" __global__ void compute_body_force_energy_grad_affine_y_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<12,wp::float64>> var_gradient,
    wp::array_t<wp::vec_t<12,wp::float64>> var_hat_y,
    wp::array_t<wp::mat_t<4,4,wp::float64>> var_mass_matrix,
    wp::vec_t<3,wp::float64> var_gravity,
    wp::array_t<wp::vec_t<3,wp::float64>> var_affine_ext_force,
    wp::array_t<wp::vec_t<12,wp::float64>> var_affine_ext_y_force,
    wp::float64 var_scale,
    wp::array_t<bool> var_affine_has_constraint,
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
        bool* var_1;
        wp::int32* var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 1;
        bool var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        const wp::int32 var_11 = 2;
        bool var_12;
        wp::int32 var_13;
        bool var_14;
        bool var_15;
        bool var_16;
        wp::vec_t<12,wp::float64> var_17;
        const wp::int32 var_18 = 3;
        const wp::int32 var_19 = 1;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::mat_t<4,4,wp::float64>* var_22;
        wp::mat_t<4,4,wp::float64> var_23;
        wp::mat_t<4,4,wp::float64> var_24;
        wp::vec_t<3,wp::float64>* var_25;
        wp::vec_t<3,wp::float64> var_26;
        wp::vec_t<3,wp::float64> var_27;
        wp::range_t var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = 1;
        wp::int32 var_31;
        wp::range_t var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        wp::float64 var_35;
        wp::int32 var_36;
        wp::float64 var_37;
        wp::float64 var_38;
        wp::float64 var_39;
        wp::int32 var_40;
        wp::float64 var_41;
        wp::float64 var_42;
        wp::float64 var_43;
        wp::int32 var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        wp::vec_t<12,wp::float64>* var_47;
        wp::vec_t<12,wp::float64> var_48;
        wp::vec_t<12,wp::float64> var_49;
        wp::vec_t<12,wp::float64>* var_50;
        wp::vec_t<12,wp::float64> var_51;
        wp::vec_t<12,wp::float64> var_52;
        const wp::int32 var_53 = 9;
        wp::float64 var_54;
        const wp::int32 var_55 = 10;
        wp::float64 var_56;
        const wp::int32 var_57 = 11;
        wp::float64 var_58;
        wp::vec_t<3,wp::float64> var_59;
        const wp::int32 var_60 = 0;
        wp::float64 var_61;
        const wp::int32 var_62 = 1;
        wp::float64 var_63;
        const wp::int32 var_64 = 2;
        wp::float64 var_65;
        wp::vec_t<3,wp::float64> var_66;
        const wp::int32 var_67 = 3;
        wp::float64 var_68;
        const wp::int32 var_69 = 4;
        wp::float64 var_70;
        const wp::int32 var_71 = 5;
        wp::float64 var_72;
        wp::vec_t<3,wp::float64> var_73;
        const wp::int32 var_74 = 6;
        wp::float64 var_75;
        const wp::int32 var_76 = 7;
        wp::float64 var_77;
        const wp::int32 var_78 = 8;
        wp::float64 var_79;
        wp::vec_t<3,wp::float64> var_80;
        wp::vec_t<3,wp::float64> var_81;
        wp::vec_t<3,wp::float64> var_82;
        wp::vec_t<3,wp::float64> var_83;
        wp::mat_t<3,3,wp::float64> var_84;
        const wp::int32 var_85 = 0;
        wp::float64 var_86;
        const wp::int32 var_87 = 1;
        wp::float64 var_88;
        const wp::int32 var_89 = 2;
        wp::float64 var_90;
        wp::vec_t<3,wp::float64> var_91;
        wp::vec_t<3,wp::float64> var_92;
        const wp::int32 var_93 = 3;
        wp::float64 var_94;
        const wp::int32 var_95 = 4;
        wp::float64 var_96;
        const wp::int32 var_97 = 5;
        wp::float64 var_98;
        wp::vec_t<3,wp::float64> var_99;
        wp::vec_t<3,wp::float64> var_100;
        const wp::int32 var_101 = 6;
        wp::float64 var_102;
        const wp::int32 var_103 = 7;
        wp::float64 var_104;
        const wp::int32 var_105 = 8;
        wp::float64 var_106;
        wp::vec_t<3,wp::float64> var_107;
        wp::vec_t<3,wp::float64> var_108;
        const wp::int32 var_109 = 9;
        wp::float64 var_110;
        const wp::int32 var_111 = 10;
        wp::float64 var_112;
        const wp::int32 var_113 = 11;
        wp::float64 var_114;
        wp::vec_t<3,wp::float64> var_115;
        wp::vec_t<3,wp::float64> var_116;
        wp::vec_t<12,wp::float64> var_117;
        const wp::int32 var_118 = 0;
        wp::float64 var_119;
        wp::float64 var_120;
        wp::float64 var_121;
        const wp::int32 var_122 = 1;
        wp::float64 var_123;
        wp::float64 var_124;
        wp::float64 var_125;
        const wp::int32 var_126 = 2;
        wp::float64 var_127;
        wp::float64 var_128;
        wp::float64 var_129;
        const wp::int32 var_130 = 3;
        wp::float64 var_131;
        wp::float64 var_132;
        wp::float64 var_133;
        const wp::int32 var_134 = 4;
        wp::float64 var_135;
        wp::float64 var_136;
        wp::float64 var_137;
        const wp::int32 var_138 = 5;
        wp::float64 var_139;
        wp::float64 var_140;
        wp::float64 var_141;
        const wp::int32 var_142 = 6;
        wp::float64 var_143;
        wp::float64 var_144;
        wp::float64 var_145;
        const wp::int32 var_146 = 7;
        wp::float64 var_147;
        wp::float64 var_148;
        wp::float64 var_149;
        const wp::int32 var_150 = 8;
        wp::float64 var_151;
        wp::float64 var_152;
        wp::float64 var_153;
        const wp::int32 var_154 = 9;
        wp::float64 var_155;
        wp::float64 var_156;
        wp::float64 var_157;
        const wp::int32 var_158 = 10;
        wp::float64 var_159;
        wp::float64 var_160;
        wp::float64 var_161;
        const wp::int32 var_162 = 11;
        wp::float64 var_163;
        wp::float64 var_164;
        wp::float64 var_165;
        wp::vec_t<12,wp::float64> var_166;
        //---------
        // forward
        // def compute_body_force_energy_grad_affine_y(                                           <L 72>
        // tid = wp.tid()                                                                         <L 84>
        var_0 = builtin_tid1d();
        // if affine_has_constraint[tid] or (                                                     <L 85>
        var_1 = wp::address(var_affine_has_constraint, var_0);
        // (env_states[body_env_id[tid]] == ENV_STATE_INVALID) or (env_states[body_env_id[tid]] == ENV_STATE_NEWTON_SOLVED)       <L 86>
        var_2 = wp::address(var_body_env_id, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::address(var_env_states, var_4);
        var_7 = wp::load(var_3);
        var_6 = (var_7 == var_5);
        var_8 = wp::address(var_body_env_id, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::address(var_env_states, var_10);
        var_13 = wp::load(var_9);
        var_12 = (var_13 == var_11);
        var_14 = var_6 || var_12;
        var_15 = wp::load(var_1);
        var_16 = var_15 || var_14;
        if (var_16) {
            // return                                                                             <L 88>
            continue;
        }
        // local_gradient = vec12d()                                                              <L 90>
        var_17 = wp::vec_t<12,wp::float64>();
        // dim = 3                                                                                <L 91>
        // n_dof = (dim + 1) * dim                                                                <L 92>
        var_20 = wp::add(var_18, var_19);
        var_21 = wp::mul(var_20, var_18);
        // mass = mass_matrix[tid]                                                                <L 93>
        var_22 = wp::address(var_mass_matrix, var_0);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // ext_force = affine_ext_force[tid]                                                      <L 94>
        var_25 = wp::address(var_affine_ext_force, var_0);
        var_27 = wp::load(var_25);
        var_26 = wp::copy(var_27);
        // for i in range(n_dof):                                                                 <L 95>
        var_28 = wp::range(var_21);
        start_for_1:;
            if (iter_cmp(var_28) == 0) goto end_for_1;
            var_29 = wp::iter_next(var_28);
            // for j in range(dim + 1):                                                           <L 96>
            var_31 = wp::add(var_18, var_30);
            var_32 = wp::range(var_31);
            start_for_3:;
                if (iter_cmp(var_32) == 0) goto end_for_3;
                var_33 = wp::iter_next(var_32);
                // local_gradient[j * dim + i % dim] += -(gravity[i % dim] + ext_force[i % dim]) * mass[(i / dim), j] * scale       <L 97>
                var_34 = wp::mod(var_29, var_18);
                var_35 = wp::extract(var_gravity, var_34);
                var_36 = wp::mod(var_29, var_18);
                var_37 = wp::extract(var_26, var_36);
                var_38 = wp::add(var_35, var_37);
                var_39 = wp::neg(var_38);
                var_40 = wp::div(var_29, var_18);
                var_41 = wp::extract(var_23, var_40, var_33);
                var_42 = wp::mul(var_39, var_41);
                var_43 = wp::mul(var_42, var_scale);
                var_44 = wp::mul(var_33, var_18);
                var_45 = wp::mod(var_29, var_18);
                var_46 = wp::add(var_44, var_45);
                wp::add_inplace(var_17, var_46, var_43);
                goto start_for_3;
            end_for_3:;
            goto start_for_1;
        end_for_1:;
        // rest_y_force = affine_ext_y_force[tid]                                                 <L 98>
        var_47 = wp::address(var_affine_ext_y_force, var_0);
        var_49 = wp::load(var_47);
        var_48 = wp::copy(var_49);
        // hat_yi = hat_y[tid]                                                                    <L 99>
        var_50 = wp::address(var_hat_y, var_0);
        var_52 = wp::load(var_50);
        var_51 = wp::copy(var_52);
        // O = wp.vec3d(hat_yi[9], hat_yi[10], hat_yi[11])                                        <L 100>
        var_54 = wp::extract(var_51, var_53);
        var_56 = wp::extract(var_51, var_55);
        var_58 = wp::extract(var_51, var_57);
        var_59 = wp::vec_t<3,wp::float64>(var_54, var_56, var_58);
        // A = wp.vec3d(hat_yi[0], hat_yi[1], hat_yi[2])                                          <L 101>
        var_61 = wp::extract(var_51, var_60);
        var_63 = wp::extract(var_51, var_62);
        var_65 = wp::extract(var_51, var_64);
        var_66 = wp::vec_t<3,wp::float64>(var_61, var_63, var_65);
        // B = wp.vec3d(hat_yi[3], hat_yi[4], hat_yi[5])                                          <L 102>
        var_68 = wp::extract(var_51, var_67);
        var_70 = wp::extract(var_51, var_69);
        var_72 = wp::extract(var_51, var_71);
        var_73 = wp::vec_t<3,wp::float64>(var_68, var_70, var_72);
        // C = wp.vec3d(hat_yi[6], hat_yi[7], hat_yi[8])                                          <L 103>
        var_75 = wp::extract(var_51, var_74);
        var_77 = wp::extract(var_51, var_76);
        var_79 = wp::extract(var_51, var_78);
        var_80 = wp::vec_t<3,wp::float64>(var_75, var_77, var_79);
        // R = col_stack3(A - O, B - O, C - O)                                                    <L 104>
        var_81 = wp::sub(var_66, var_59);
        var_82 = wp::sub(var_73, var_59);
        var_83 = wp::sub(var_80, var_59);
        var_84 = col_stack3_1(var_81, var_82, var_83);
        // y_force = cat_4_vec3d(                                                                 <L 105>
        // R * wp.vec3d(rest_y_force[0], rest_y_force[1], rest_y_force[2]),                       <L 106>
        var_86 = wp::extract(var_48, var_85);
        var_88 = wp::extract(var_48, var_87);
        var_90 = wp::extract(var_48, var_89);
        var_91 = wp::vec_t<3,wp::float64>(var_86, var_88, var_90);
        var_92 = wp::mul(var_84, var_91);
        // R * wp.vec3d(rest_y_force[3], rest_y_force[4], rest_y_force[5]),                       <L 107>
        var_94 = wp::extract(var_48, var_93);
        var_96 = wp::extract(var_48, var_95);
        var_98 = wp::extract(var_48, var_97);
        var_99 = wp::vec_t<3,wp::float64>(var_94, var_96, var_98);
        var_100 = wp::mul(var_84, var_99);
        // R * wp.vec3d(rest_y_force[6], rest_y_force[7], rest_y_force[8]),                       <L 108>
        var_102 = wp::extract(var_48, var_101);
        var_104 = wp::extract(var_48, var_103);
        var_106 = wp::extract(var_48, var_105);
        var_107 = wp::vec_t<3,wp::float64>(var_102, var_104, var_106);
        var_108 = wp::mul(var_84, var_107);
        // R * wp.vec3d(rest_y_force[9], rest_y_force[10], rest_y_force[11]),                     <L 109>
        var_110 = wp::extract(var_48, var_109);
        var_112 = wp::extract(var_48, var_111);
        var_114 = wp::extract(var_48, var_113);
        var_115 = wp::vec_t<3,wp::float64>(var_110, var_112, var_114);
        var_116 = wp::mul(var_84, var_115);
        var_117 = cat_4_vec3d_0(var_92, var_100, var_108, var_116);
        // for d in range(12):                                                                    <L 111>
        // local_gradient[d] += -y_force[d] * scale                                               <L 112>
        var_119 = wp::extract(var_117, var_118);
        var_120 = wp::neg(var_119);
        var_121 = wp::mul(var_120, var_scale);
        wp::add_inplace(var_17, var_118, var_121);
        var_123 = wp::extract(var_117, var_122);
        var_124 = wp::neg(var_123);
        var_125 = wp::mul(var_124, var_scale);
        wp::add_inplace(var_17, var_122, var_125);
        var_127 = wp::extract(var_117, var_126);
        var_128 = wp::neg(var_127);
        var_129 = wp::mul(var_128, var_scale);
        wp::add_inplace(var_17, var_126, var_129);
        var_131 = wp::extract(var_117, var_130);
        var_132 = wp::neg(var_131);
        var_133 = wp::mul(var_132, var_scale);
        wp::add_inplace(var_17, var_130, var_133);
        var_135 = wp::extract(var_117, var_134);
        var_136 = wp::neg(var_135);
        var_137 = wp::mul(var_136, var_scale);
        wp::add_inplace(var_17, var_134, var_137);
        var_139 = wp::extract(var_117, var_138);
        var_140 = wp::neg(var_139);
        var_141 = wp::mul(var_140, var_scale);
        wp::add_inplace(var_17, var_138, var_141);
        var_143 = wp::extract(var_117, var_142);
        var_144 = wp::neg(var_143);
        var_145 = wp::mul(var_144, var_scale);
        wp::add_inplace(var_17, var_142, var_145);
        var_147 = wp::extract(var_117, var_146);
        var_148 = wp::neg(var_147);
        var_149 = wp::mul(var_148, var_scale);
        wp::add_inplace(var_17, var_146, var_149);
        var_151 = wp::extract(var_117, var_150);
        var_152 = wp::neg(var_151);
        var_153 = wp::mul(var_152, var_scale);
        wp::add_inplace(var_17, var_150, var_153);
        var_155 = wp::extract(var_117, var_154);
        var_156 = wp::neg(var_155);
        var_157 = wp::mul(var_156, var_scale);
        wp::add_inplace(var_17, var_154, var_157);
        var_159 = wp::extract(var_117, var_158);
        var_160 = wp::neg(var_159);
        var_161 = wp::mul(var_160, var_scale);
        wp::add_inplace(var_17, var_158, var_161);
        var_163 = wp::extract(var_117, var_162);
        var_164 = wp::neg(var_163);
        var_165 = wp::mul(var_164, var_scale);
        wp::add_inplace(var_17, var_162, var_165);
        // gradient[tid] += local_gradient                                                        <L 114>
        var_166 = wp::atomic_add(var_gradient, var_0, var_17);
    }
}



extern "C" __global__ void compute_body_force_energy_grad_affine_y_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<12,wp::float64>> var_gradient,
    wp::array_t<wp::vec_t<12,wp::float64>> var_hat_y,
    wp::array_t<wp::mat_t<4,4,wp::float64>> var_mass_matrix,
    wp::vec_t<3,wp::float64> var_gravity,
    wp::array_t<wp::vec_t<3,wp::float64>> var_affine_ext_force,
    wp::array_t<wp::vec_t<12,wp::float64>> var_affine_ext_y_force,
    wp::float64 var_scale,
    wp::array_t<bool> var_affine_has_constraint,
    wp::array_t<wp::int32> var_body_env_id,
    wp::array_t<wp::int32> var_env_states,
    wp::array_t<wp::vec_t<12,wp::float64>> adj_gradient,
    wp::array_t<wp::vec_t<12,wp::float64>> adj_hat_y,
    wp::array_t<wp::mat_t<4,4,wp::float64>> adj_mass_matrix,
    wp::vec_t<3,wp::float64> adj_gravity,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_affine_ext_force,
    wp::array_t<wp::vec_t<12,wp::float64>> adj_affine_ext_y_force,
    wp::float64 adj_scale,
    wp::array_t<bool> adj_affine_has_constraint,
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
        bool* var_1;
        wp::int32* var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 1;
        bool var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        const wp::int32 var_11 = 2;
        bool var_12;
        wp::int32 var_13;
        bool var_14;
        bool var_15;
        bool var_16;
        wp::vec_t<12,wp::float64> var_17;
        const wp::int32 var_18 = 3;
        const wp::int32 var_19 = 1;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::mat_t<4,4,wp::float64>* var_22;
        wp::mat_t<4,4,wp::float64> var_23;
        wp::mat_t<4,4,wp::float64> var_24;
        wp::vec_t<3,wp::float64>* var_25;
        wp::vec_t<3,wp::float64> var_26;
        wp::vec_t<3,wp::float64> var_27;
        wp::range_t var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = 1;
        wp::int32 var_31;
        wp::range_t var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        wp::float64 var_35;
        wp::int32 var_36;
        wp::float64 var_37;
        wp::float64 var_38;
        wp::float64 var_39;
        wp::int32 var_40;
        wp::float64 var_41;
        wp::float64 var_42;
        wp::float64 var_43;
        wp::int32 var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        wp::vec_t<12,wp::float64>* var_47;
        wp::vec_t<12,wp::float64> var_48;
        wp::vec_t<12,wp::float64> var_49;
        wp::vec_t<12,wp::float64>* var_50;
        wp::vec_t<12,wp::float64> var_51;
        wp::vec_t<12,wp::float64> var_52;
        const wp::int32 var_53 = 9;
        wp::float64 var_54;
        const wp::int32 var_55 = 10;
        wp::float64 var_56;
        const wp::int32 var_57 = 11;
        wp::float64 var_58;
        wp::vec_t<3,wp::float64> var_59;
        const wp::int32 var_60 = 0;
        wp::float64 var_61;
        const wp::int32 var_62 = 1;
        wp::float64 var_63;
        const wp::int32 var_64 = 2;
        wp::float64 var_65;
        wp::vec_t<3,wp::float64> var_66;
        const wp::int32 var_67 = 3;
        wp::float64 var_68;
        const wp::int32 var_69 = 4;
        wp::float64 var_70;
        const wp::int32 var_71 = 5;
        wp::float64 var_72;
        wp::vec_t<3,wp::float64> var_73;
        const wp::int32 var_74 = 6;
        wp::float64 var_75;
        const wp::int32 var_76 = 7;
        wp::float64 var_77;
        const wp::int32 var_78 = 8;
        wp::float64 var_79;
        wp::vec_t<3,wp::float64> var_80;
        wp::vec_t<3,wp::float64> var_81;
        wp::vec_t<3,wp::float64> var_82;
        wp::vec_t<3,wp::float64> var_83;
        wp::mat_t<3,3,wp::float64> var_84;
        const wp::int32 var_85 = 0;
        wp::float64 var_86;
        const wp::int32 var_87 = 1;
        wp::float64 var_88;
        const wp::int32 var_89 = 2;
        wp::float64 var_90;
        wp::vec_t<3,wp::float64> var_91;
        wp::vec_t<3,wp::float64> var_92;
        const wp::int32 var_93 = 3;
        wp::float64 var_94;
        const wp::int32 var_95 = 4;
        wp::float64 var_96;
        const wp::int32 var_97 = 5;
        wp::float64 var_98;
        wp::vec_t<3,wp::float64> var_99;
        wp::vec_t<3,wp::float64> var_100;
        const wp::int32 var_101 = 6;
        wp::float64 var_102;
        const wp::int32 var_103 = 7;
        wp::float64 var_104;
        const wp::int32 var_105 = 8;
        wp::float64 var_106;
        wp::vec_t<3,wp::float64> var_107;
        wp::vec_t<3,wp::float64> var_108;
        const wp::int32 var_109 = 9;
        wp::float64 var_110;
        const wp::int32 var_111 = 10;
        wp::float64 var_112;
        const wp::int32 var_113 = 11;
        wp::float64 var_114;
        wp::vec_t<3,wp::float64> var_115;
        wp::vec_t<3,wp::float64> var_116;
        wp::vec_t<12,wp::float64> var_117;
        const wp::int32 var_118 = 0;
        wp::float64 var_119;
        wp::float64 var_120;
        wp::float64 var_121;
        const wp::int32 var_122 = 1;
        wp::float64 var_123;
        wp::float64 var_124;
        wp::float64 var_125;
        const wp::int32 var_126 = 2;
        wp::float64 var_127;
        wp::float64 var_128;
        wp::float64 var_129;
        const wp::int32 var_130 = 3;
        wp::float64 var_131;
        wp::float64 var_132;
        wp::float64 var_133;
        const wp::int32 var_134 = 4;
        wp::float64 var_135;
        wp::float64 var_136;
        wp::float64 var_137;
        const wp::int32 var_138 = 5;
        wp::float64 var_139;
        wp::float64 var_140;
        wp::float64 var_141;
        const wp::int32 var_142 = 6;
        wp::float64 var_143;
        wp::float64 var_144;
        wp::float64 var_145;
        const wp::int32 var_146 = 7;
        wp::float64 var_147;
        wp::float64 var_148;
        wp::float64 var_149;
        const wp::int32 var_150 = 8;
        wp::float64 var_151;
        wp::float64 var_152;
        wp::float64 var_153;
        const wp::int32 var_154 = 9;
        wp::float64 var_155;
        wp::float64 var_156;
        wp::float64 var_157;
        const wp::int32 var_158 = 10;
        wp::float64 var_159;
        wp::float64 var_160;
        wp::float64 var_161;
        const wp::int32 var_162 = 11;
        wp::float64 var_163;
        wp::float64 var_164;
        wp::float64 var_165;
        wp::vec_t<12,wp::float64> var_166;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
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
        bool adj_12 = {};
        wp::int32 adj_13 = {};
        bool adj_14 = {};
        bool adj_15 = {};
        bool adj_16 = {};
        wp::vec_t<12,wp::float64> adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::mat_t<4,4,wp::float64> adj_22 = {};
        wp::mat_t<4,4,wp::float64> adj_23 = {};
        wp::mat_t<4,4,wp::float64> adj_24 = {};
        wp::vec_t<3,wp::float64> adj_25 = {};
        wp::vec_t<3,wp::float64> adj_26 = {};
        wp::vec_t<3,wp::float64> adj_27 = {};
        wp::range_t adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::range_t adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::float64 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::float64 adj_37 = {};
        wp::float64 adj_38 = {};
        wp::float64 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::float64 adj_41 = {};
        wp::float64 adj_42 = {};
        wp::float64 adj_43 = {};
        wp::int32 adj_44 = {};
        wp::int32 adj_45 = {};
        wp::int32 adj_46 = {};
        wp::vec_t<12,wp::float64> adj_47 = {};
        wp::vec_t<12,wp::float64> adj_48 = {};
        wp::vec_t<12,wp::float64> adj_49 = {};
        wp::vec_t<12,wp::float64> adj_50 = {};
        wp::vec_t<12,wp::float64> adj_51 = {};
        wp::vec_t<12,wp::float64> adj_52 = {};
        wp::int32 adj_53 = {};
        wp::float64 adj_54 = {};
        wp::int32 adj_55 = {};
        wp::float64 adj_56 = {};
        wp::int32 adj_57 = {};
        wp::float64 adj_58 = {};
        wp::vec_t<3,wp::float64> adj_59 = {};
        wp::int32 adj_60 = {};
        wp::float64 adj_61 = {};
        wp::int32 adj_62 = {};
        wp::float64 adj_63 = {};
        wp::int32 adj_64 = {};
        wp::float64 adj_65 = {};
        wp::vec_t<3,wp::float64> adj_66 = {};
        wp::int32 adj_67 = {};
        wp::float64 adj_68 = {};
        wp::int32 adj_69 = {};
        wp::float64 adj_70 = {};
        wp::int32 adj_71 = {};
        wp::float64 adj_72 = {};
        wp::vec_t<3,wp::float64> adj_73 = {};
        wp::int32 adj_74 = {};
        wp::float64 adj_75 = {};
        wp::int32 adj_76 = {};
        wp::float64 adj_77 = {};
        wp::int32 adj_78 = {};
        wp::float64 adj_79 = {};
        wp::vec_t<3,wp::float64> adj_80 = {};
        wp::vec_t<3,wp::float64> adj_81 = {};
        wp::vec_t<3,wp::float64> adj_82 = {};
        wp::vec_t<3,wp::float64> adj_83 = {};
        wp::mat_t<3,3,wp::float64> adj_84 = {};
        wp::int32 adj_85 = {};
        wp::float64 adj_86 = {};
        wp::int32 adj_87 = {};
        wp::float64 adj_88 = {};
        wp::int32 adj_89 = {};
        wp::float64 adj_90 = {};
        wp::vec_t<3,wp::float64> adj_91 = {};
        wp::vec_t<3,wp::float64> adj_92 = {};
        wp::int32 adj_93 = {};
        wp::float64 adj_94 = {};
        wp::int32 adj_95 = {};
        wp::float64 adj_96 = {};
        wp::int32 adj_97 = {};
        wp::float64 adj_98 = {};
        wp::vec_t<3,wp::float64> adj_99 = {};
        wp::vec_t<3,wp::float64> adj_100 = {};
        wp::int32 adj_101 = {};
        wp::float64 adj_102 = {};
        wp::int32 adj_103 = {};
        wp::float64 adj_104 = {};
        wp::int32 adj_105 = {};
        wp::float64 adj_106 = {};
        wp::vec_t<3,wp::float64> adj_107 = {};
        wp::vec_t<3,wp::float64> adj_108 = {};
        wp::int32 adj_109 = {};
        wp::float64 adj_110 = {};
        wp::int32 adj_111 = {};
        wp::float64 adj_112 = {};
        wp::int32 adj_113 = {};
        wp::float64 adj_114 = {};
        wp::vec_t<3,wp::float64> adj_115 = {};
        wp::vec_t<3,wp::float64> adj_116 = {};
        wp::vec_t<12,wp::float64> adj_117 = {};
        wp::int32 adj_118 = {};
        wp::float64 adj_119 = {};
        wp::float64 adj_120 = {};
        wp::float64 adj_121 = {};
        wp::int32 adj_122 = {};
        wp::float64 adj_123 = {};
        wp::float64 adj_124 = {};
        wp::float64 adj_125 = {};
        wp::int32 adj_126 = {};
        wp::float64 adj_127 = {};
        wp::float64 adj_128 = {};
        wp::float64 adj_129 = {};
        wp::int32 adj_130 = {};
        wp::float64 adj_131 = {};
        wp::float64 adj_132 = {};
        wp::float64 adj_133 = {};
        wp::int32 adj_134 = {};
        wp::float64 adj_135 = {};
        wp::float64 adj_136 = {};
        wp::float64 adj_137 = {};
        wp::int32 adj_138 = {};
        wp::float64 adj_139 = {};
        wp::float64 adj_140 = {};
        wp::float64 adj_141 = {};
        wp::int32 adj_142 = {};
        wp::float64 adj_143 = {};
        wp::float64 adj_144 = {};
        wp::float64 adj_145 = {};
        wp::int32 adj_146 = {};
        wp::float64 adj_147 = {};
        wp::float64 adj_148 = {};
        wp::float64 adj_149 = {};
        wp::int32 adj_150 = {};
        wp::float64 adj_151 = {};
        wp::float64 adj_152 = {};
        wp::float64 adj_153 = {};
        wp::int32 adj_154 = {};
        wp::float64 adj_155 = {};
        wp::float64 adj_156 = {};
        wp::float64 adj_157 = {};
        wp::int32 adj_158 = {};
        wp::float64 adj_159 = {};
        wp::float64 adj_160 = {};
        wp::float64 adj_161 = {};
        wp::int32 adj_162 = {};
        wp::float64 adj_163 = {};
        wp::float64 adj_164 = {};
        wp::float64 adj_165 = {};
        wp::vec_t<12,wp::float64> adj_166 = {};
        //---------
        // forward
        // def compute_body_force_energy_grad_affine_y(                                           <L 72>
        // tid = wp.tid()                                                                         <L 84>
        var_0 = builtin_tid1d();
        // if affine_has_constraint[tid] or (                                                     <L 85>
        var_1 = wp::address(var_affine_has_constraint, var_0);
        // (env_states[body_env_id[tid]] == ENV_STATE_INVALID) or (env_states[body_env_id[tid]] == ENV_STATE_NEWTON_SOLVED)       <L 86>
        var_2 = wp::address(var_body_env_id, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::address(var_env_states, var_4);
        var_7 = wp::load(var_3);
        var_6 = (var_7 == var_5);
        var_8 = wp::address(var_body_env_id, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::address(var_env_states, var_10);
        var_13 = wp::load(var_9);
        var_12 = (var_13 == var_11);
        var_14 = var_6 || var_12;
        var_15 = wp::load(var_1);
        var_16 = var_15 || var_14;
        if (var_16) {
            // return                                                                             <L 88>
            goto label0;
        }
        // local_gradient = vec12d()                                                              <L 90>
        var_17 = wp::vec_t<12,wp::float64>();
        // dim = 3                                                                                <L 91>
        // n_dof = (dim + 1) * dim                                                                <L 92>
        var_20 = wp::add(var_18, var_19);
        var_21 = wp::mul(var_20, var_18);
        // mass = mass_matrix[tid]                                                                <L 93>
        var_22 = wp::address(var_mass_matrix, var_0);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // ext_force = affine_ext_force[tid]                                                      <L 94>
        var_25 = wp::address(var_affine_ext_force, var_0);
        var_27 = wp::load(var_25);
        var_26 = wp::copy(var_27);
        // for i in range(n_dof):                                                                 <L 95>
        var_28 = wp::range(var_21);
        // rest_y_force = affine_ext_y_force[tid]                                                 <L 98>
        var_47 = wp::address(var_affine_ext_y_force, var_0);
        var_49 = wp::load(var_47);
        var_48 = wp::copy(var_49);
        // hat_yi = hat_y[tid]                                                                    <L 99>
        var_50 = wp::address(var_hat_y, var_0);
        var_52 = wp::load(var_50);
        var_51 = wp::copy(var_52);
        // O = wp.vec3d(hat_yi[9], hat_yi[10], hat_yi[11])                                        <L 100>
        var_54 = wp::extract(var_51, var_53);
        var_56 = wp::extract(var_51, var_55);
        var_58 = wp::extract(var_51, var_57);
        var_59 = wp::vec_t<3,wp::float64>(var_54, var_56, var_58);
        // A = wp.vec3d(hat_yi[0], hat_yi[1], hat_yi[2])                                          <L 101>
        var_61 = wp::extract(var_51, var_60);
        var_63 = wp::extract(var_51, var_62);
        var_65 = wp::extract(var_51, var_64);
        var_66 = wp::vec_t<3,wp::float64>(var_61, var_63, var_65);
        // B = wp.vec3d(hat_yi[3], hat_yi[4], hat_yi[5])                                          <L 102>
        var_68 = wp::extract(var_51, var_67);
        var_70 = wp::extract(var_51, var_69);
        var_72 = wp::extract(var_51, var_71);
        var_73 = wp::vec_t<3,wp::float64>(var_68, var_70, var_72);
        // C = wp.vec3d(hat_yi[6], hat_yi[7], hat_yi[8])                                          <L 103>
        var_75 = wp::extract(var_51, var_74);
        var_77 = wp::extract(var_51, var_76);
        var_79 = wp::extract(var_51, var_78);
        var_80 = wp::vec_t<3,wp::float64>(var_75, var_77, var_79);
        // R = col_stack3(A - O, B - O, C - O)                                                    <L 104>
        var_81 = wp::sub(var_66, var_59);
        var_82 = wp::sub(var_73, var_59);
        var_83 = wp::sub(var_80, var_59);
        var_84 = col_stack3_1(var_81, var_82, var_83);
        // y_force = cat_4_vec3d(                                                                 <L 105>
        // R * wp.vec3d(rest_y_force[0], rest_y_force[1], rest_y_force[2]),                       <L 106>
        var_86 = wp::extract(var_48, var_85);
        var_88 = wp::extract(var_48, var_87);
        var_90 = wp::extract(var_48, var_89);
        var_91 = wp::vec_t<3,wp::float64>(var_86, var_88, var_90);
        var_92 = wp::mul(var_84, var_91);
        // R * wp.vec3d(rest_y_force[3], rest_y_force[4], rest_y_force[5]),                       <L 107>
        var_94 = wp::extract(var_48, var_93);
        var_96 = wp::extract(var_48, var_95);
        var_98 = wp::extract(var_48, var_97);
        var_99 = wp::vec_t<3,wp::float64>(var_94, var_96, var_98);
        var_100 = wp::mul(var_84, var_99);
        // R * wp.vec3d(rest_y_force[6], rest_y_force[7], rest_y_force[8]),                       <L 108>
        var_102 = wp::extract(var_48, var_101);
        var_104 = wp::extract(var_48, var_103);
        var_106 = wp::extract(var_48, var_105);
        var_107 = wp::vec_t<3,wp::float64>(var_102, var_104, var_106);
        var_108 = wp::mul(var_84, var_107);
        // R * wp.vec3d(rest_y_force[9], rest_y_force[10], rest_y_force[11]),                     <L 109>
        var_110 = wp::extract(var_48, var_109);
        var_112 = wp::extract(var_48, var_111);
        var_114 = wp::extract(var_48, var_113);
        var_115 = wp::vec_t<3,wp::float64>(var_110, var_112, var_114);
        var_116 = wp::mul(var_84, var_115);
        var_117 = cat_4_vec3d_0(var_92, var_100, var_108, var_116);
        // for d in range(12):                                                                    <L 111>
        // local_gradient[d] += -y_force[d] * scale                                               <L 112>
        var_119 = wp::extract(var_117, var_118);
        var_120 = wp::neg(var_119);
        var_121 = wp::mul(var_120, var_scale);
        wp::add_inplace(var_17, var_118, var_121);
        var_123 = wp::extract(var_117, var_122);
        var_124 = wp::neg(var_123);
        var_125 = wp::mul(var_124, var_scale);
        wp::add_inplace(var_17, var_122, var_125);
        var_127 = wp::extract(var_117, var_126);
        var_128 = wp::neg(var_127);
        var_129 = wp::mul(var_128, var_scale);
        wp::add_inplace(var_17, var_126, var_129);
        var_131 = wp::extract(var_117, var_130);
        var_132 = wp::neg(var_131);
        var_133 = wp::mul(var_132, var_scale);
        wp::add_inplace(var_17, var_130, var_133);
        var_135 = wp::extract(var_117, var_134);
        var_136 = wp::neg(var_135);
        var_137 = wp::mul(var_136, var_scale);
        wp::add_inplace(var_17, var_134, var_137);
        var_139 = wp::extract(var_117, var_138);
        var_140 = wp::neg(var_139);
        var_141 = wp::mul(var_140, var_scale);
        wp::add_inplace(var_17, var_138, var_141);
        var_143 = wp::extract(var_117, var_142);
        var_144 = wp::neg(var_143);
        var_145 = wp::mul(var_144, var_scale);
        wp::add_inplace(var_17, var_142, var_145);
        var_147 = wp::extract(var_117, var_146);
        var_148 = wp::neg(var_147);
        var_149 = wp::mul(var_148, var_scale);
        wp::add_inplace(var_17, var_146, var_149);
        var_151 = wp::extract(var_117, var_150);
        var_152 = wp::neg(var_151);
        var_153 = wp::mul(var_152, var_scale);
        wp::add_inplace(var_17, var_150, var_153);
        var_155 = wp::extract(var_117, var_154);
        var_156 = wp::neg(var_155);
        var_157 = wp::mul(var_156, var_scale);
        wp::add_inplace(var_17, var_154, var_157);
        var_159 = wp::extract(var_117, var_158);
        var_160 = wp::neg(var_159);
        var_161 = wp::mul(var_160, var_scale);
        wp::add_inplace(var_17, var_158, var_161);
        var_163 = wp::extract(var_117, var_162);
        var_164 = wp::neg(var_163);
        var_165 = wp::mul(var_164, var_scale);
        wp::add_inplace(var_17, var_162, var_165);
        // gradient[tid] += local_gradient                                                        <L 114>
        // var_166 = wp::atomic_add(var_gradient, var_0, var_17);
        //---------
        // reverse
        wp::adj_atomic_add(var_gradient, var_0, var_17, adj_gradient, adj_0, adj_17, adj_166);
        // adj: gradient[tid] += local_gradient                                                   <L 114>
        wp::adj_add_inplace(var_17, var_162, var_165, adj_17, adj_162, adj_165);
        wp::adj_mul(var_164, var_scale, adj_164, adj_scale, adj_165);
        wp::adj_neg(var_163, adj_163, adj_164);
        wp::adj_extract(var_117, var_162, adj_117, adj_162, adj_163);
        wp::adj_add_inplace(var_17, var_158, var_161, adj_17, adj_158, adj_161);
        wp::adj_mul(var_160, var_scale, adj_160, adj_scale, adj_161);
        wp::adj_neg(var_159, adj_159, adj_160);
        wp::adj_extract(var_117, var_158, adj_117, adj_158, adj_159);
        wp::adj_add_inplace(var_17, var_154, var_157, adj_17, adj_154, adj_157);
        wp::adj_mul(var_156, var_scale, adj_156, adj_scale, adj_157);
        wp::adj_neg(var_155, adj_155, adj_156);
        wp::adj_extract(var_117, var_154, adj_117, adj_154, adj_155);
        wp::adj_add_inplace(var_17, var_150, var_153, adj_17, adj_150, adj_153);
        wp::adj_mul(var_152, var_scale, adj_152, adj_scale, adj_153);
        wp::adj_neg(var_151, adj_151, adj_152);
        wp::adj_extract(var_117, var_150, adj_117, adj_150, adj_151);
        wp::adj_add_inplace(var_17, var_146, var_149, adj_17, adj_146, adj_149);
        wp::adj_mul(var_148, var_scale, adj_148, adj_scale, adj_149);
        wp::adj_neg(var_147, adj_147, adj_148);
        wp::adj_extract(var_117, var_146, adj_117, adj_146, adj_147);
        wp::adj_add_inplace(var_17, var_142, var_145, adj_17, adj_142, adj_145);
        wp::adj_mul(var_144, var_scale, adj_144, adj_scale, adj_145);
        wp::adj_neg(var_143, adj_143, adj_144);
        wp::adj_extract(var_117, var_142, adj_117, adj_142, adj_143);
        wp::adj_add_inplace(var_17, var_138, var_141, adj_17, adj_138, adj_141);
        wp::adj_mul(var_140, var_scale, adj_140, adj_scale, adj_141);
        wp::adj_neg(var_139, adj_139, adj_140);
        wp::adj_extract(var_117, var_138, adj_117, adj_138, adj_139);
        wp::adj_add_inplace(var_17, var_134, var_137, adj_17, adj_134, adj_137);
        wp::adj_mul(var_136, var_scale, adj_136, adj_scale, adj_137);
        wp::adj_neg(var_135, adj_135, adj_136);
        wp::adj_extract(var_117, var_134, adj_117, adj_134, adj_135);
        wp::adj_add_inplace(var_17, var_130, var_133, adj_17, adj_130, adj_133);
        wp::adj_mul(var_132, var_scale, adj_132, adj_scale, adj_133);
        wp::adj_neg(var_131, adj_131, adj_132);
        wp::adj_extract(var_117, var_130, adj_117, adj_130, adj_131);
        wp::adj_add_inplace(var_17, var_126, var_129, adj_17, adj_126, adj_129);
        wp::adj_mul(var_128, var_scale, adj_128, adj_scale, adj_129);
        wp::adj_neg(var_127, adj_127, adj_128);
        wp::adj_extract(var_117, var_126, adj_117, adj_126, adj_127);
        wp::adj_add_inplace(var_17, var_122, var_125, adj_17, adj_122, adj_125);
        wp::adj_mul(var_124, var_scale, adj_124, adj_scale, adj_125);
        wp::adj_neg(var_123, adj_123, adj_124);
        wp::adj_extract(var_117, var_122, adj_117, adj_122, adj_123);
        wp::adj_add_inplace(var_17, var_118, var_121, adj_17, adj_118, adj_121);
        wp::adj_mul(var_120, var_scale, adj_120, adj_scale, adj_121);
        wp::adj_neg(var_119, adj_119, adj_120);
        wp::adj_extract(var_117, var_118, adj_117, adj_118, adj_119);
        // adj: local_gradient[d] += -y_force[d] * scale                                          <L 112>
        // adj: for d in range(12):                                                               <L 111>
        adj_cat_4_vec3d_0(var_92, var_100, var_108, var_116, adj_92, adj_100, adj_108, adj_116, adj_117);
        wp::adj_mul(var_84, var_115, adj_84, adj_115, adj_116);
        wp::adj_vec_t(var_110, var_112, var_114, adj_110, adj_112, adj_114, adj_115);
        wp::adj_extract(var_48, var_113, adj_48, adj_113, adj_114);
        wp::adj_extract(var_48, var_111, adj_48, adj_111, adj_112);
        wp::adj_extract(var_48, var_109, adj_48, adj_109, adj_110);
        // adj: R * wp.vec3d(rest_y_force[9], rest_y_force[10], rest_y_force[11]),                <L 109>
        wp::adj_mul(var_84, var_107, adj_84, adj_107, adj_108);
        wp::adj_vec_t(var_102, var_104, var_106, adj_102, adj_104, adj_106, adj_107);
        wp::adj_extract(var_48, var_105, adj_48, adj_105, adj_106);
        wp::adj_extract(var_48, var_103, adj_48, adj_103, adj_104);
        wp::adj_extract(var_48, var_101, adj_48, adj_101, adj_102);
        // adj: R * wp.vec3d(rest_y_force[6], rest_y_force[7], rest_y_force[8]),                  <L 108>
        wp::adj_mul(var_84, var_99, adj_84, adj_99, adj_100);
        wp::adj_vec_t(var_94, var_96, var_98, adj_94, adj_96, adj_98, adj_99);
        wp::adj_extract(var_48, var_97, adj_48, adj_97, adj_98);
        wp::adj_extract(var_48, var_95, adj_48, adj_95, adj_96);
        wp::adj_extract(var_48, var_93, adj_48, adj_93, adj_94);
        // adj: R * wp.vec3d(rest_y_force[3], rest_y_force[4], rest_y_force[5]),                  <L 107>
        wp::adj_mul(var_84, var_91, adj_84, adj_91, adj_92);
        wp::adj_vec_t(var_86, var_88, var_90, adj_86, adj_88, adj_90, adj_91);
        wp::adj_extract(var_48, var_89, adj_48, adj_89, adj_90);
        wp::adj_extract(var_48, var_87, adj_48, adj_87, adj_88);
        wp::adj_extract(var_48, var_85, adj_48, adj_85, adj_86);
        // adj: R * wp.vec3d(rest_y_force[0], rest_y_force[1], rest_y_force[2]),                  <L 106>
        // adj: y_force = cat_4_vec3d(                                                            <L 105>
        adj_col_stack3_1(var_81, var_82, var_83, adj_81, adj_82, adj_83, adj_84);
        wp::adj_sub(var_80, var_59, adj_80, adj_59, adj_83);
        wp::adj_sub(var_73, var_59, adj_73, adj_59, adj_82);
        wp::adj_sub(var_66, var_59, adj_66, adj_59, adj_81);
        // adj: R = col_stack3(A - O, B - O, C - O)                                               <L 104>
        wp::adj_vec_t(var_75, var_77, var_79, adj_75, adj_77, adj_79, adj_80);
        wp::adj_extract(var_51, var_78, adj_51, adj_78, adj_79);
        wp::adj_extract(var_51, var_76, adj_51, adj_76, adj_77);
        wp::adj_extract(var_51, var_74, adj_51, adj_74, adj_75);
        // adj: C = wp.vec3d(hat_yi[6], hat_yi[7], hat_yi[8])                                     <L 103>
        wp::adj_vec_t(var_68, var_70, var_72, adj_68, adj_70, adj_72, adj_73);
        wp::adj_extract(var_51, var_71, adj_51, adj_71, adj_72);
        wp::adj_extract(var_51, var_69, adj_51, adj_69, adj_70);
        wp::adj_extract(var_51, var_67, adj_51, adj_67, adj_68);
        // adj: B = wp.vec3d(hat_yi[3], hat_yi[4], hat_yi[5])                                     <L 102>
        wp::adj_vec_t(var_61, var_63, var_65, adj_61, adj_63, adj_65, adj_66);
        wp::adj_extract(var_51, var_64, adj_51, adj_64, adj_65);
        wp::adj_extract(var_51, var_62, adj_51, adj_62, adj_63);
        wp::adj_extract(var_51, var_60, adj_51, adj_60, adj_61);
        // adj: A = wp.vec3d(hat_yi[0], hat_yi[1], hat_yi[2])                                     <L 101>
        wp::adj_vec_t(var_54, var_56, var_58, adj_54, adj_56, adj_58, adj_59);
        wp::adj_extract(var_51, var_57, adj_51, adj_57, adj_58);
        wp::adj_extract(var_51, var_55, adj_51, adj_55, adj_56);
        wp::adj_extract(var_51, var_53, adj_51, adj_53, adj_54);
        // adj: O = wp.vec3d(hat_yi[9], hat_yi[10], hat_yi[11])                                   <L 100>
        wp::adj_copy(var_52, adj_50, adj_51);
        wp::adj_load(var_50, adj_50, adj_52);
        wp::adj_address(var_hat_y, var_0, adj_hat_y, adj_0, adj_50);
        // adj: hat_yi = hat_y[tid]                                                               <L 99>
        wp::adj_copy(var_49, adj_47, adj_48);
        wp::adj_load(var_47, adj_47, adj_49);
        wp::adj_address(var_affine_ext_y_force, var_0, adj_affine_ext_y_force, adj_0, adj_47);
        // adj: rest_y_force = affine_ext_y_force[tid]                                            <L 98>
        var_28 = wp::iter_reverse(var_28);
        start_for_1:;
            if (iter_cmp(var_28) == 0) goto end_for_1;
            var_29 = wp::iter_next(var_28);
        	adj_30 = {};
        	adj_31 = {};
        	adj_32 = {};
            // for j in range(dim + 1):                                                           <L 96>
            var_31 = wp::add(var_18, var_30);
            var_32 = wp::range(var_31);
            var_32 = wp::iter_reverse(var_32);
            start_for_3:;
                if (iter_cmp(var_32) == 0) goto end_for_3;
                var_33 = wp::iter_next(var_32);
            	adj_34 = {};
            	adj_35 = {};
            	adj_36 = {};
            	adj_37 = {};
            	adj_38 = {};
            	adj_39 = {};
            	adj_40 = {};
            	adj_41 = {};
            	adj_42 = {};
            	adj_43 = {};
            	adj_44 = {};
            	adj_45 = {};
            	adj_46 = {};
                // local_gradient[j * dim + i % dim] += -(gravity[i % dim] + ext_force[i % dim]) * mass[(i / dim), j] * scale       <L 97>
                var_34 = wp::mod(var_29, var_18);
                var_35 = wp::extract(var_gravity, var_34);
                var_36 = wp::mod(var_29, var_18);
                var_37 = wp::extract(var_26, var_36);
                var_38 = wp::add(var_35, var_37);
                var_39 = wp::neg(var_38);
                var_40 = wp::div(var_29, var_18);
                var_41 = wp::extract(var_23, var_40, var_33);
                var_42 = wp::mul(var_39, var_41);
                var_43 = wp::mul(var_42, var_scale);
                var_44 = wp::mul(var_33, var_18);
                var_45 = wp::mod(var_29, var_18);
                var_46 = wp::add(var_44, var_45);
                wp::add_inplace(var_17, var_46, var_43);
                wp::adj_add_inplace(var_17, var_46, var_43, adj_17, adj_46, adj_43);
                wp::adj_add(var_44, var_45, adj_44, adj_45, adj_46);
                wp::adj_mod(var_29, var_18, adj_29, adj_18, adj_45);
                wp::adj_mul(var_33, var_18, adj_33, adj_18, adj_44);
                wp::adj_mul(var_42, var_scale, adj_42, adj_scale, adj_43);
                wp::adj_mul(var_39, var_41, adj_39, adj_41, adj_42);
                wp::adj_extract(var_23, var_40, var_33, adj_23, adj_40, adj_33, adj_41);
                wp::adj_div(var_29, var_18, var_40, adj_29, adj_18, adj_40);
                wp::adj_neg(var_38, adj_38, adj_39);
                wp::adj_add(var_35, var_37, adj_35, adj_37, adj_38);
                wp::adj_extract(var_26, var_36, adj_26, adj_36, adj_37);
                wp::adj_mod(var_29, var_18, adj_29, adj_18, adj_36);
                wp::adj_extract(var_gravity, var_34, adj_gravity, adj_34, adj_35);
                wp::adj_mod(var_29, var_18, adj_29, adj_18, adj_34);
                // adj: local_gradient[j * dim + i % dim] += -(gravity[i % dim] + ext_force[i % dim]) * mass[(i / dim), j] * scale  <L 97>
            	goto start_for_3;
            end_for_3:;
            wp::adj_range(var_31, adj_31, adj_32);
            wp::adj_add(var_18, var_30, adj_18, adj_30, adj_31);
            // adj: for j in range(dim + 1):                                                      <L 96>
        	goto start_for_1;
        end_for_1:;
        wp::adj_range(var_21, adj_21, adj_28);
        // adj: for i in range(n_dof):                                                            <L 95>
        wp::adj_copy(var_27, adj_25, adj_26);
        wp::adj_load(var_25, adj_25, adj_27);
        wp::adj_address(var_affine_ext_force, var_0, adj_affine_ext_force, adj_0, adj_25);
        // adj: ext_force = affine_ext_force[tid]                                                 <L 94>
        wp::adj_copy(var_24, adj_22, adj_23);
        wp::adj_load(var_22, adj_22, adj_24);
        wp::adj_address(var_mass_matrix, var_0, adj_mass_matrix, adj_0, adj_22);
        // adj: mass = mass_matrix[tid]                                                           <L 93>
        wp::adj_mul(var_20, var_18, adj_20, adj_18, adj_21);
        wp::adj_add(var_18, var_19, adj_18, adj_19, adj_20);
        // adj: n_dof = (dim + 1) * dim                                                           <L 92>
        // adj: dim = 3                                                                           <L 91>
        // adj: local_gradient = vec12d()                                                         <L 90>
        if (var_16) {
            label0:;
            // adj: return                                                                        <L 88>
        }
        wp::adj_load(var_1, adj_1, adj_15);
        wp::adj_load(var_9, adj_9, adj_13);
        wp::adj_address(var_env_states, var_10, adj_env_states, adj_8, adj_9);
        wp::adj_load(var_8, adj_8, adj_10);
        wp::adj_address(var_body_env_id, var_0, adj_body_env_id, adj_0, adj_8);
        wp::adj_load(var_3, adj_3, adj_7);
        wp::adj_address(var_env_states, var_4, adj_env_states, adj_2, adj_3);
        wp::adj_load(var_2, adj_2, adj_4);
        wp::adj_address(var_body_env_id, var_0, adj_body_env_id, adj_0, adj_2);
        // adj: (env_states[body_env_id[tid]] == ENV_STATE_INVALID) or (env_states[body_env_id[tid]] == ENV_STATE_NEWTON_SOLVED)  <L 86>
        wp::adj_address(var_affine_has_constraint, var_0, adj_affine_has_constraint, adj_0, adj_1);
        // adj: if affine_has_constraint[tid] or (                                                <L 85>
        // adj: tid = wp.tid()                                                                    <L 84>
        // adj: def compute_body_force_energy_grad_affine_y(                                      <L 72>
        continue;
    }
}



extern "C" __global__ void compute_body_force_energy_val_affine_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float64> var_energy,
    wp::array_t<wp::vec_t<12,wp::float64>> var_y,
    wp::array_t<wp::vec_t<12,wp::float64>> var_hat_y,
    wp::array_t<wp::mat_t<4,4,wp::float64>> var_mass_matrix,
    wp::vec_t<3,wp::float64> var_gravity,
    wp::array_t<wp::vec_t<3,wp::float64>> var_affine_ext_force,
    wp::array_t<wp::vec_t<12,wp::float64>> var_affine_ext_y_force,
    wp::float64 var_scale,
    wp::array_t<bool> var_affine_has_constraint)
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
        const wp::int32 var_4 = 3;
        const wp::int32 var_5 = 1;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::mat_t<4,4,wp::float64>* var_8;
        wp::mat_t<4,4,wp::float64> var_9;
        wp::mat_t<4,4,wp::float64> var_10;
        wp::vec_t<3,wp::float64>* var_11;
        wp::vec_t<3,wp::float64> var_12;
        wp::vec_t<3,wp::float64> var_13;
        wp::range_t var_14;
        wp::int32 var_15;
        const wp::int32 var_16 = 1;
        wp::int32 var_17;
        wp::range_t var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::float64 var_21;
        wp::int32 var_22;
        wp::float64 var_23;
        wp::float64 var_24;
        wp::float64 var_25;
        wp::vec_t<12,wp::float64>* var_26;
        wp::float64 var_27;
        wp::vec_t<12,wp::float64> var_28;
        wp::float64 var_29;
        wp::int32 var_30;
        wp::float64 var_31;
        wp::float64 var_32;
        wp::float64 var_33;
        wp::float64 var_34;
        wp::vec_t<12,wp::float64>* var_35;
        wp::vec_t<12,wp::float64> var_36;
        wp::vec_t<12,wp::float64> var_37;
        wp::vec_t<12,wp::float64>* var_38;
        wp::vec_t<12,wp::float64> var_39;
        wp::vec_t<12,wp::float64> var_40;
        wp::vec_t<12,wp::float64>* var_41;
        wp::vec_t<12,wp::float64> var_42;
        wp::vec_t<12,wp::float64> var_43;
        const wp::int32 var_44 = 9;
        wp::float64 var_45;
        const wp::int32 var_46 = 10;
        wp::float64 var_47;
        const wp::int32 var_48 = 11;
        wp::float64 var_49;
        wp::vec_t<3,wp::float64> var_50;
        const wp::int32 var_51 = 0;
        wp::float64 var_52;
        const wp::int32 var_53 = 1;
        wp::float64 var_54;
        const wp::int32 var_55 = 2;
        wp::float64 var_56;
        wp::vec_t<3,wp::float64> var_57;
        const wp::int32 var_58 = 3;
        wp::float64 var_59;
        const wp::int32 var_60 = 4;
        wp::float64 var_61;
        const wp::int32 var_62 = 5;
        wp::float64 var_63;
        wp::vec_t<3,wp::float64> var_64;
        const wp::int32 var_65 = 6;
        wp::float64 var_66;
        const wp::int32 var_67 = 7;
        wp::float64 var_68;
        const wp::int32 var_69 = 8;
        wp::float64 var_70;
        wp::vec_t<3,wp::float64> var_71;
        wp::vec_t<3,wp::float64> var_72;
        wp::vec_t<3,wp::float64> var_73;
        wp::vec_t<3,wp::float64> var_74;
        wp::mat_t<3,3,wp::float64> var_75;
        const wp::int32 var_76 = 0;
        wp::float64 var_77;
        const wp::int32 var_78 = 1;
        wp::float64 var_79;
        const wp::int32 var_80 = 2;
        wp::float64 var_81;
        wp::vec_t<3,wp::float64> var_82;
        wp::vec_t<3,wp::float64> var_83;
        const wp::int32 var_84 = 3;
        wp::float64 var_85;
        const wp::int32 var_86 = 4;
        wp::float64 var_87;
        const wp::int32 var_88 = 5;
        wp::float64 var_89;
        wp::vec_t<3,wp::float64> var_90;
        wp::vec_t<3,wp::float64> var_91;
        const wp::int32 var_92 = 6;
        wp::float64 var_93;
        const wp::int32 var_94 = 7;
        wp::float64 var_95;
        const wp::int32 var_96 = 8;
        wp::float64 var_97;
        wp::vec_t<3,wp::float64> var_98;
        wp::vec_t<3,wp::float64> var_99;
        const wp::int32 var_100 = 9;
        wp::float64 var_101;
        const wp::int32 var_102 = 10;
        wp::float64 var_103;
        const wp::int32 var_104 = 11;
        wp::float64 var_105;
        wp::vec_t<3,wp::float64> var_106;
        wp::vec_t<3,wp::float64> var_107;
        wp::vec_t<12,wp::float64> var_108;
        const wp::int32 var_109 = 0;
        wp::float64 var_110;
        wp::float64 var_111;
        wp::float64 var_112;
        wp::float64 var_113;
        wp::float64 var_114;
        wp::float64 var_115;
        const wp::int32 var_116 = 1;
        wp::float64 var_117;
        wp::float64 var_118;
        wp::float64 var_119;
        wp::float64 var_120;
        wp::float64 var_121;
        wp::float64 var_122;
        const wp::int32 var_123 = 2;
        wp::float64 var_124;
        wp::float64 var_125;
        wp::float64 var_126;
        wp::float64 var_127;
        wp::float64 var_128;
        wp::float64 var_129;
        const wp::int32 var_130 = 3;
        wp::float64 var_131;
        wp::float64 var_132;
        wp::float64 var_133;
        wp::float64 var_134;
        wp::float64 var_135;
        wp::float64 var_136;
        const wp::int32 var_137 = 4;
        wp::float64 var_138;
        wp::float64 var_139;
        wp::float64 var_140;
        wp::float64 var_141;
        wp::float64 var_142;
        wp::float64 var_143;
        const wp::int32 var_144 = 5;
        wp::float64 var_145;
        wp::float64 var_146;
        wp::float64 var_147;
        wp::float64 var_148;
        wp::float64 var_149;
        wp::float64 var_150;
        const wp::int32 var_151 = 6;
        wp::float64 var_152;
        wp::float64 var_153;
        wp::float64 var_154;
        wp::float64 var_155;
        wp::float64 var_156;
        wp::float64 var_157;
        const wp::int32 var_158 = 7;
        wp::float64 var_159;
        wp::float64 var_160;
        wp::float64 var_161;
        wp::float64 var_162;
        wp::float64 var_163;
        wp::float64 var_164;
        const wp::int32 var_165 = 8;
        wp::float64 var_166;
        wp::float64 var_167;
        wp::float64 var_168;
        wp::float64 var_169;
        wp::float64 var_170;
        wp::float64 var_171;
        const wp::int32 var_172 = 9;
        wp::float64 var_173;
        wp::float64 var_174;
        wp::float64 var_175;
        wp::float64 var_176;
        wp::float64 var_177;
        wp::float64 var_178;
        const wp::int32 var_179 = 10;
        wp::float64 var_180;
        wp::float64 var_181;
        wp::float64 var_182;
        wp::float64 var_183;
        wp::float64 var_184;
        wp::float64 var_185;
        const wp::int32 var_186 = 11;
        wp::float64 var_187;
        wp::float64 var_188;
        wp::float64 var_189;
        wp::float64 var_190;
        wp::float64 var_191;
        wp::float64 var_192;
        //---------
        // forward
        // def compute_body_force_energy_val_affine(                                              <L 16>
        // tid = wp.tid()                                                                         <L 27>
        var_0 = builtin_tid1d();
        // if affine_has_constraint[tid]:                                                         <L 28>
        var_1 = wp::address(var_affine_has_constraint, var_0);
        var_2 = wp::load(var_1);
        if (var_2) {
            // return                                                                             <L 29>
            continue;
        }
        var_3 = wp::load(var_1);
        // dim = 3                                                                                <L 30>
        // n_dof = (dim + 1) * dim                                                                <L 31>
        var_6 = wp::add(var_4, var_5);
        var_7 = wp::mul(var_6, var_4);
        // mass = mass_matrix[tid]                                                                <L 32>
        var_8 = wp::address(var_mass_matrix, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // ext_force = affine_ext_force[tid]                                                      <L 33>
        var_11 = wp::address(var_affine_ext_force, var_0);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // for i in range(n_dof):                                                                 <L 34>
        var_14 = wp::range(var_7);
        start_for_1:;
            if (iter_cmp(var_14) == 0) goto end_for_1;
            var_15 = wp::iter_next(var_14);
            // for j in range(dim + 1):                                                           <L 35>
            var_17 = wp::add(var_4, var_16);
            var_18 = wp::range(var_17);
            start_for_3:;
                if (iter_cmp(var_18) == 0) goto end_for_3;
                var_19 = wp::iter_next(var_18);
                // energy[tid] += -(gravity[i % dim] + ext_force[i % dim]) * y[tid][i] * mass[(i / dim), j] * scale       <L 36>
                var_20 = wp::mod(var_15, var_4);
                var_21 = wp::extract(var_gravity, var_20);
                var_22 = wp::mod(var_15, var_4);
                var_23 = wp::extract(var_12, var_22);
                var_24 = wp::add(var_21, var_23);
                var_25 = wp::neg(var_24);
                var_26 = wp::address(var_y, var_0);
                var_28 = wp::load(var_26);
                var_27 = wp::extract(var_28, var_15);
                var_29 = wp::mul(var_25, var_27);
                var_30 = wp::div(var_15, var_4);
                var_31 = wp::extract(var_9, var_30, var_19);
                var_32 = wp::mul(var_29, var_31);
                var_33 = wp::mul(var_32, var_scale);
                var_34 = wp::atomic_add(var_energy, var_0, var_33);
                goto start_for_3;
            end_for_3:;
            goto start_for_1;
        end_for_1:;
        // rest_y_force = affine_ext_y_force[tid]                                                 <L 37>
        var_35 = wp::address(var_affine_ext_y_force, var_0);
        var_37 = wp::load(var_35);
        var_36 = wp::copy(var_37);
        // yi = y[tid]                                                                            <L 38>
        var_38 = wp::address(var_y, var_0);
        var_40 = wp::load(var_38);
        var_39 = wp::copy(var_40);
        // hat_yi = hat_y[tid]                                                                    <L 39>
        var_41 = wp::address(var_hat_y, var_0);
        var_43 = wp::load(var_41);
        var_42 = wp::copy(var_43);
        // O = wp.vec3d(hat_yi[9], hat_yi[10], hat_yi[11])                                        <L 40>
        var_45 = wp::extract(var_42, var_44);
        var_47 = wp::extract(var_42, var_46);
        var_49 = wp::extract(var_42, var_48);
        var_50 = wp::vec_t<3,wp::float64>(var_45, var_47, var_49);
        // A = wp.vec3d(hat_yi[0], hat_yi[1], hat_yi[2])                                          <L 41>
        var_52 = wp::extract(var_42, var_51);
        var_54 = wp::extract(var_42, var_53);
        var_56 = wp::extract(var_42, var_55);
        var_57 = wp::vec_t<3,wp::float64>(var_52, var_54, var_56);
        // B = wp.vec3d(hat_yi[3], hat_yi[4], hat_yi[5])                                          <L 42>
        var_59 = wp::extract(var_42, var_58);
        var_61 = wp::extract(var_42, var_60);
        var_63 = wp::extract(var_42, var_62);
        var_64 = wp::vec_t<3,wp::float64>(var_59, var_61, var_63);
        // C = wp.vec3d(hat_yi[6], hat_yi[7], hat_yi[8])                                          <L 43>
        var_66 = wp::extract(var_42, var_65);
        var_68 = wp::extract(var_42, var_67);
        var_70 = wp::extract(var_42, var_69);
        var_71 = wp::vec_t<3,wp::float64>(var_66, var_68, var_70);
        // R = col_stack3(A - O, B - O, C - O)                                                    <L 44>
        var_72 = wp::sub(var_57, var_50);
        var_73 = wp::sub(var_64, var_50);
        var_74 = wp::sub(var_71, var_50);
        var_75 = col_stack3_1(var_72, var_73, var_74);
        // y_force = cat_4_vec3d(                                                                 <L 45>
        // R * wp.vec3d(rest_y_force[0], rest_y_force[1], rest_y_force[2]),                       <L 46>
        var_77 = wp::extract(var_36, var_76);
        var_79 = wp::extract(var_36, var_78);
        var_81 = wp::extract(var_36, var_80);
        var_82 = wp::vec_t<3,wp::float64>(var_77, var_79, var_81);
        var_83 = wp::mul(var_75, var_82);
        // R * wp.vec3d(rest_y_force[3], rest_y_force[4], rest_y_force[5]),                       <L 47>
        var_85 = wp::extract(var_36, var_84);
        var_87 = wp::extract(var_36, var_86);
        var_89 = wp::extract(var_36, var_88);
        var_90 = wp::vec_t<3,wp::float64>(var_85, var_87, var_89);
        var_91 = wp::mul(var_75, var_90);
        // R * wp.vec3d(rest_y_force[6], rest_y_force[7], rest_y_force[8]),                       <L 48>
        var_93 = wp::extract(var_36, var_92);
        var_95 = wp::extract(var_36, var_94);
        var_97 = wp::extract(var_36, var_96);
        var_98 = wp::vec_t<3,wp::float64>(var_93, var_95, var_97);
        var_99 = wp::mul(var_75, var_98);
        // R * wp.vec3d(rest_y_force[9], rest_y_force[10], rest_y_force[11]),                     <L 49>
        var_101 = wp::extract(var_36, var_100);
        var_103 = wp::extract(var_36, var_102);
        var_105 = wp::extract(var_36, var_104);
        var_106 = wp::vec_t<3,wp::float64>(var_101, var_103, var_105);
        var_107 = wp::mul(var_75, var_106);
        var_108 = cat_4_vec3d_0(var_83, var_91, var_99, var_107);
        // for d in range(12):                                                                    <L 51>
        // energy[tid] += -y_force[d] * yi[d] * scale                                             <L 52>
        var_110 = wp::extract(var_108, var_109);
        var_111 = wp::neg(var_110);
        var_112 = wp::extract(var_39, var_109);
        var_113 = wp::mul(var_111, var_112);
        var_114 = wp::mul(var_113, var_scale);
        var_115 = wp::atomic_add(var_energy, var_0, var_114);
        var_117 = wp::extract(var_108, var_116);
        var_118 = wp::neg(var_117);
        var_119 = wp::extract(var_39, var_116);
        var_120 = wp::mul(var_118, var_119);
        var_121 = wp::mul(var_120, var_scale);
        var_122 = wp::atomic_add(var_energy, var_0, var_121);
        var_124 = wp::extract(var_108, var_123);
        var_125 = wp::neg(var_124);
        var_126 = wp::extract(var_39, var_123);
        var_127 = wp::mul(var_125, var_126);
        var_128 = wp::mul(var_127, var_scale);
        var_129 = wp::atomic_add(var_energy, var_0, var_128);
        var_131 = wp::extract(var_108, var_130);
        var_132 = wp::neg(var_131);
        var_133 = wp::extract(var_39, var_130);
        var_134 = wp::mul(var_132, var_133);
        var_135 = wp::mul(var_134, var_scale);
        var_136 = wp::atomic_add(var_energy, var_0, var_135);
        var_138 = wp::extract(var_108, var_137);
        var_139 = wp::neg(var_138);
        var_140 = wp::extract(var_39, var_137);
        var_141 = wp::mul(var_139, var_140);
        var_142 = wp::mul(var_141, var_scale);
        var_143 = wp::atomic_add(var_energy, var_0, var_142);
        var_145 = wp::extract(var_108, var_144);
        var_146 = wp::neg(var_145);
        var_147 = wp::extract(var_39, var_144);
        var_148 = wp::mul(var_146, var_147);
        var_149 = wp::mul(var_148, var_scale);
        var_150 = wp::atomic_add(var_energy, var_0, var_149);
        var_152 = wp::extract(var_108, var_151);
        var_153 = wp::neg(var_152);
        var_154 = wp::extract(var_39, var_151);
        var_155 = wp::mul(var_153, var_154);
        var_156 = wp::mul(var_155, var_scale);
        var_157 = wp::atomic_add(var_energy, var_0, var_156);
        var_159 = wp::extract(var_108, var_158);
        var_160 = wp::neg(var_159);
        var_161 = wp::extract(var_39, var_158);
        var_162 = wp::mul(var_160, var_161);
        var_163 = wp::mul(var_162, var_scale);
        var_164 = wp::atomic_add(var_energy, var_0, var_163);
        var_166 = wp::extract(var_108, var_165);
        var_167 = wp::neg(var_166);
        var_168 = wp::extract(var_39, var_165);
        var_169 = wp::mul(var_167, var_168);
        var_170 = wp::mul(var_169, var_scale);
        var_171 = wp::atomic_add(var_energy, var_0, var_170);
        var_173 = wp::extract(var_108, var_172);
        var_174 = wp::neg(var_173);
        var_175 = wp::extract(var_39, var_172);
        var_176 = wp::mul(var_174, var_175);
        var_177 = wp::mul(var_176, var_scale);
        var_178 = wp::atomic_add(var_energy, var_0, var_177);
        var_180 = wp::extract(var_108, var_179);
        var_181 = wp::neg(var_180);
        var_182 = wp::extract(var_39, var_179);
        var_183 = wp::mul(var_181, var_182);
        var_184 = wp::mul(var_183, var_scale);
        var_185 = wp::atomic_add(var_energy, var_0, var_184);
        var_187 = wp::extract(var_108, var_186);
        var_188 = wp::neg(var_187);
        var_189 = wp::extract(var_39, var_186);
        var_190 = wp::mul(var_188, var_189);
        var_191 = wp::mul(var_190, var_scale);
        var_192 = wp::atomic_add(var_energy, var_0, var_191);
    }
}



extern "C" __global__ void compute_body_force_energy_val_affine_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float64> var_energy,
    wp::array_t<wp::vec_t<12,wp::float64>> var_y,
    wp::array_t<wp::vec_t<12,wp::float64>> var_hat_y,
    wp::array_t<wp::mat_t<4,4,wp::float64>> var_mass_matrix,
    wp::vec_t<3,wp::float64> var_gravity,
    wp::array_t<wp::vec_t<3,wp::float64>> var_affine_ext_force,
    wp::array_t<wp::vec_t<12,wp::float64>> var_affine_ext_y_force,
    wp::float64 var_scale,
    wp::array_t<bool> var_affine_has_constraint,
    wp::array_t<wp::float64> adj_energy,
    wp::array_t<wp::vec_t<12,wp::float64>> adj_y,
    wp::array_t<wp::vec_t<12,wp::float64>> adj_hat_y,
    wp::array_t<wp::mat_t<4,4,wp::float64>> adj_mass_matrix,
    wp::vec_t<3,wp::float64> adj_gravity,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_affine_ext_force,
    wp::array_t<wp::vec_t<12,wp::float64>> adj_affine_ext_y_force,
    wp::float64 adj_scale,
    wp::array_t<bool> adj_affine_has_constraint)
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
        const wp::int32 var_4 = 3;
        const wp::int32 var_5 = 1;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::mat_t<4,4,wp::float64>* var_8;
        wp::mat_t<4,4,wp::float64> var_9;
        wp::mat_t<4,4,wp::float64> var_10;
        wp::vec_t<3,wp::float64>* var_11;
        wp::vec_t<3,wp::float64> var_12;
        wp::vec_t<3,wp::float64> var_13;
        wp::range_t var_14;
        wp::int32 var_15;
        const wp::int32 var_16 = 1;
        wp::int32 var_17;
        wp::range_t var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::float64 var_21;
        wp::int32 var_22;
        wp::float64 var_23;
        wp::float64 var_24;
        wp::float64 var_25;
        wp::vec_t<12,wp::float64>* var_26;
        wp::float64 var_27;
        wp::vec_t<12,wp::float64> var_28;
        wp::float64 var_29;
        wp::int32 var_30;
        wp::float64 var_31;
        wp::float64 var_32;
        wp::float64 var_33;
        wp::float64 var_34;
        wp::vec_t<12,wp::float64>* var_35;
        wp::vec_t<12,wp::float64> var_36;
        wp::vec_t<12,wp::float64> var_37;
        wp::vec_t<12,wp::float64>* var_38;
        wp::vec_t<12,wp::float64> var_39;
        wp::vec_t<12,wp::float64> var_40;
        wp::vec_t<12,wp::float64>* var_41;
        wp::vec_t<12,wp::float64> var_42;
        wp::vec_t<12,wp::float64> var_43;
        const wp::int32 var_44 = 9;
        wp::float64 var_45;
        const wp::int32 var_46 = 10;
        wp::float64 var_47;
        const wp::int32 var_48 = 11;
        wp::float64 var_49;
        wp::vec_t<3,wp::float64> var_50;
        const wp::int32 var_51 = 0;
        wp::float64 var_52;
        const wp::int32 var_53 = 1;
        wp::float64 var_54;
        const wp::int32 var_55 = 2;
        wp::float64 var_56;
        wp::vec_t<3,wp::float64> var_57;
        const wp::int32 var_58 = 3;
        wp::float64 var_59;
        const wp::int32 var_60 = 4;
        wp::float64 var_61;
        const wp::int32 var_62 = 5;
        wp::float64 var_63;
        wp::vec_t<3,wp::float64> var_64;
        const wp::int32 var_65 = 6;
        wp::float64 var_66;
        const wp::int32 var_67 = 7;
        wp::float64 var_68;
        const wp::int32 var_69 = 8;
        wp::float64 var_70;
        wp::vec_t<3,wp::float64> var_71;
        wp::vec_t<3,wp::float64> var_72;
        wp::vec_t<3,wp::float64> var_73;
        wp::vec_t<3,wp::float64> var_74;
        wp::mat_t<3,3,wp::float64> var_75;
        const wp::int32 var_76 = 0;
        wp::float64 var_77;
        const wp::int32 var_78 = 1;
        wp::float64 var_79;
        const wp::int32 var_80 = 2;
        wp::float64 var_81;
        wp::vec_t<3,wp::float64> var_82;
        wp::vec_t<3,wp::float64> var_83;
        const wp::int32 var_84 = 3;
        wp::float64 var_85;
        const wp::int32 var_86 = 4;
        wp::float64 var_87;
        const wp::int32 var_88 = 5;
        wp::float64 var_89;
        wp::vec_t<3,wp::float64> var_90;
        wp::vec_t<3,wp::float64> var_91;
        const wp::int32 var_92 = 6;
        wp::float64 var_93;
        const wp::int32 var_94 = 7;
        wp::float64 var_95;
        const wp::int32 var_96 = 8;
        wp::float64 var_97;
        wp::vec_t<3,wp::float64> var_98;
        wp::vec_t<3,wp::float64> var_99;
        const wp::int32 var_100 = 9;
        wp::float64 var_101;
        const wp::int32 var_102 = 10;
        wp::float64 var_103;
        const wp::int32 var_104 = 11;
        wp::float64 var_105;
        wp::vec_t<3,wp::float64> var_106;
        wp::vec_t<3,wp::float64> var_107;
        wp::vec_t<12,wp::float64> var_108;
        const wp::int32 var_109 = 0;
        wp::float64 var_110;
        wp::float64 var_111;
        wp::float64 var_112;
        wp::float64 var_113;
        wp::float64 var_114;
        wp::float64 var_115;
        const wp::int32 var_116 = 1;
        wp::float64 var_117;
        wp::float64 var_118;
        wp::float64 var_119;
        wp::float64 var_120;
        wp::float64 var_121;
        wp::float64 var_122;
        const wp::int32 var_123 = 2;
        wp::float64 var_124;
        wp::float64 var_125;
        wp::float64 var_126;
        wp::float64 var_127;
        wp::float64 var_128;
        wp::float64 var_129;
        const wp::int32 var_130 = 3;
        wp::float64 var_131;
        wp::float64 var_132;
        wp::float64 var_133;
        wp::float64 var_134;
        wp::float64 var_135;
        wp::float64 var_136;
        const wp::int32 var_137 = 4;
        wp::float64 var_138;
        wp::float64 var_139;
        wp::float64 var_140;
        wp::float64 var_141;
        wp::float64 var_142;
        wp::float64 var_143;
        const wp::int32 var_144 = 5;
        wp::float64 var_145;
        wp::float64 var_146;
        wp::float64 var_147;
        wp::float64 var_148;
        wp::float64 var_149;
        wp::float64 var_150;
        const wp::int32 var_151 = 6;
        wp::float64 var_152;
        wp::float64 var_153;
        wp::float64 var_154;
        wp::float64 var_155;
        wp::float64 var_156;
        wp::float64 var_157;
        const wp::int32 var_158 = 7;
        wp::float64 var_159;
        wp::float64 var_160;
        wp::float64 var_161;
        wp::float64 var_162;
        wp::float64 var_163;
        wp::float64 var_164;
        const wp::int32 var_165 = 8;
        wp::float64 var_166;
        wp::float64 var_167;
        wp::float64 var_168;
        wp::float64 var_169;
        wp::float64 var_170;
        wp::float64 var_171;
        const wp::int32 var_172 = 9;
        wp::float64 var_173;
        wp::float64 var_174;
        wp::float64 var_175;
        wp::float64 var_176;
        wp::float64 var_177;
        wp::float64 var_178;
        const wp::int32 var_179 = 10;
        wp::float64 var_180;
        wp::float64 var_181;
        wp::float64 var_182;
        wp::float64 var_183;
        wp::float64 var_184;
        wp::float64 var_185;
        const wp::int32 var_186 = 11;
        wp::float64 var_187;
        wp::float64 var_188;
        wp::float64 var_189;
        wp::float64 var_190;
        wp::float64 var_191;
        wp::float64 var_192;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::mat_t<4,4,wp::float64> adj_8 = {};
        wp::mat_t<4,4,wp::float64> adj_9 = {};
        wp::mat_t<4,4,wp::float64> adj_10 = {};
        wp::vec_t<3,wp::float64> adj_11 = {};
        wp::vec_t<3,wp::float64> adj_12 = {};
        wp::vec_t<3,wp::float64> adj_13 = {};
        wp::range_t adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::range_t adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::float64 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::float64 adj_23 = {};
        wp::float64 adj_24 = {};
        wp::float64 adj_25 = {};
        wp::vec_t<12,wp::float64> adj_26 = {};
        wp::float64 adj_27 = {};
        wp::vec_t<12,wp::float64> adj_28 = {};
        wp::float64 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::float64 adj_31 = {};
        wp::float64 adj_32 = {};
        wp::float64 adj_33 = {};
        wp::float64 adj_34 = {};
        wp::vec_t<12,wp::float64> adj_35 = {};
        wp::vec_t<12,wp::float64> adj_36 = {};
        wp::vec_t<12,wp::float64> adj_37 = {};
        wp::vec_t<12,wp::float64> adj_38 = {};
        wp::vec_t<12,wp::float64> adj_39 = {};
        wp::vec_t<12,wp::float64> adj_40 = {};
        wp::vec_t<12,wp::float64> adj_41 = {};
        wp::vec_t<12,wp::float64> adj_42 = {};
        wp::vec_t<12,wp::float64> adj_43 = {};
        wp::int32 adj_44 = {};
        wp::float64 adj_45 = {};
        wp::int32 adj_46 = {};
        wp::float64 adj_47 = {};
        wp::int32 adj_48 = {};
        wp::float64 adj_49 = {};
        wp::vec_t<3,wp::float64> adj_50 = {};
        wp::int32 adj_51 = {};
        wp::float64 adj_52 = {};
        wp::int32 adj_53 = {};
        wp::float64 adj_54 = {};
        wp::int32 adj_55 = {};
        wp::float64 adj_56 = {};
        wp::vec_t<3,wp::float64> adj_57 = {};
        wp::int32 adj_58 = {};
        wp::float64 adj_59 = {};
        wp::int32 adj_60 = {};
        wp::float64 adj_61 = {};
        wp::int32 adj_62 = {};
        wp::float64 adj_63 = {};
        wp::vec_t<3,wp::float64> adj_64 = {};
        wp::int32 adj_65 = {};
        wp::float64 adj_66 = {};
        wp::int32 adj_67 = {};
        wp::float64 adj_68 = {};
        wp::int32 adj_69 = {};
        wp::float64 adj_70 = {};
        wp::vec_t<3,wp::float64> adj_71 = {};
        wp::vec_t<3,wp::float64> adj_72 = {};
        wp::vec_t<3,wp::float64> adj_73 = {};
        wp::vec_t<3,wp::float64> adj_74 = {};
        wp::mat_t<3,3,wp::float64> adj_75 = {};
        wp::int32 adj_76 = {};
        wp::float64 adj_77 = {};
        wp::int32 adj_78 = {};
        wp::float64 adj_79 = {};
        wp::int32 adj_80 = {};
        wp::float64 adj_81 = {};
        wp::vec_t<3,wp::float64> adj_82 = {};
        wp::vec_t<3,wp::float64> adj_83 = {};
        wp::int32 adj_84 = {};
        wp::float64 adj_85 = {};
        wp::int32 adj_86 = {};
        wp::float64 adj_87 = {};
        wp::int32 adj_88 = {};
        wp::float64 adj_89 = {};
        wp::vec_t<3,wp::float64> adj_90 = {};
        wp::vec_t<3,wp::float64> adj_91 = {};
        wp::int32 adj_92 = {};
        wp::float64 adj_93 = {};
        wp::int32 adj_94 = {};
        wp::float64 adj_95 = {};
        wp::int32 adj_96 = {};
        wp::float64 adj_97 = {};
        wp::vec_t<3,wp::float64> adj_98 = {};
        wp::vec_t<3,wp::float64> adj_99 = {};
        wp::int32 adj_100 = {};
        wp::float64 adj_101 = {};
        wp::int32 adj_102 = {};
        wp::float64 adj_103 = {};
        wp::int32 adj_104 = {};
        wp::float64 adj_105 = {};
        wp::vec_t<3,wp::float64> adj_106 = {};
        wp::vec_t<3,wp::float64> adj_107 = {};
        wp::vec_t<12,wp::float64> adj_108 = {};
        wp::int32 adj_109 = {};
        wp::float64 adj_110 = {};
        wp::float64 adj_111 = {};
        wp::float64 adj_112 = {};
        wp::float64 adj_113 = {};
        wp::float64 adj_114 = {};
        wp::float64 adj_115 = {};
        wp::int32 adj_116 = {};
        wp::float64 adj_117 = {};
        wp::float64 adj_118 = {};
        wp::float64 adj_119 = {};
        wp::float64 adj_120 = {};
        wp::float64 adj_121 = {};
        wp::float64 adj_122 = {};
        wp::int32 adj_123 = {};
        wp::float64 adj_124 = {};
        wp::float64 adj_125 = {};
        wp::float64 adj_126 = {};
        wp::float64 adj_127 = {};
        wp::float64 adj_128 = {};
        wp::float64 adj_129 = {};
        wp::int32 adj_130 = {};
        wp::float64 adj_131 = {};
        wp::float64 adj_132 = {};
        wp::float64 adj_133 = {};
        wp::float64 adj_134 = {};
        wp::float64 adj_135 = {};
        wp::float64 adj_136 = {};
        wp::int32 adj_137 = {};
        wp::float64 adj_138 = {};
        wp::float64 adj_139 = {};
        wp::float64 adj_140 = {};
        wp::float64 adj_141 = {};
        wp::float64 adj_142 = {};
        wp::float64 adj_143 = {};
        wp::int32 adj_144 = {};
        wp::float64 adj_145 = {};
        wp::float64 adj_146 = {};
        wp::float64 adj_147 = {};
        wp::float64 adj_148 = {};
        wp::float64 adj_149 = {};
        wp::float64 adj_150 = {};
        wp::int32 adj_151 = {};
        wp::float64 adj_152 = {};
        wp::float64 adj_153 = {};
        wp::float64 adj_154 = {};
        wp::float64 adj_155 = {};
        wp::float64 adj_156 = {};
        wp::float64 adj_157 = {};
        wp::int32 adj_158 = {};
        wp::float64 adj_159 = {};
        wp::float64 adj_160 = {};
        wp::float64 adj_161 = {};
        wp::float64 adj_162 = {};
        wp::float64 adj_163 = {};
        wp::float64 adj_164 = {};
        wp::int32 adj_165 = {};
        wp::float64 adj_166 = {};
        wp::float64 adj_167 = {};
        wp::float64 adj_168 = {};
        wp::float64 adj_169 = {};
        wp::float64 adj_170 = {};
        wp::float64 adj_171 = {};
        wp::int32 adj_172 = {};
        wp::float64 adj_173 = {};
        wp::float64 adj_174 = {};
        wp::float64 adj_175 = {};
        wp::float64 adj_176 = {};
        wp::float64 adj_177 = {};
        wp::float64 adj_178 = {};
        wp::int32 adj_179 = {};
        wp::float64 adj_180 = {};
        wp::float64 adj_181 = {};
        wp::float64 adj_182 = {};
        wp::float64 adj_183 = {};
        wp::float64 adj_184 = {};
        wp::float64 adj_185 = {};
        wp::int32 adj_186 = {};
        wp::float64 adj_187 = {};
        wp::float64 adj_188 = {};
        wp::float64 adj_189 = {};
        wp::float64 adj_190 = {};
        wp::float64 adj_191 = {};
        wp::float64 adj_192 = {};
        //---------
        // forward
        // def compute_body_force_energy_val_affine(                                              <L 16>
        // tid = wp.tid()                                                                         <L 27>
        var_0 = builtin_tid1d();
        // if affine_has_constraint[tid]:                                                         <L 28>
        var_1 = wp::address(var_affine_has_constraint, var_0);
        var_2 = wp::load(var_1);
        if (var_2) {
            // return                                                                             <L 29>
            goto label0;
        }
        var_3 = wp::load(var_1);
        // dim = 3                                                                                <L 30>
        // n_dof = (dim + 1) * dim                                                                <L 31>
        var_6 = wp::add(var_4, var_5);
        var_7 = wp::mul(var_6, var_4);
        // mass = mass_matrix[tid]                                                                <L 32>
        var_8 = wp::address(var_mass_matrix, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // ext_force = affine_ext_force[tid]                                                      <L 33>
        var_11 = wp::address(var_affine_ext_force, var_0);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // for i in range(n_dof):                                                                 <L 34>
        var_14 = wp::range(var_7);
        // rest_y_force = affine_ext_y_force[tid]                                                 <L 37>
        var_35 = wp::address(var_affine_ext_y_force, var_0);
        var_37 = wp::load(var_35);
        var_36 = wp::copy(var_37);
        // yi = y[tid]                                                                            <L 38>
        var_38 = wp::address(var_y, var_0);
        var_40 = wp::load(var_38);
        var_39 = wp::copy(var_40);
        // hat_yi = hat_y[tid]                                                                    <L 39>
        var_41 = wp::address(var_hat_y, var_0);
        var_43 = wp::load(var_41);
        var_42 = wp::copy(var_43);
        // O = wp.vec3d(hat_yi[9], hat_yi[10], hat_yi[11])                                        <L 40>
        var_45 = wp::extract(var_42, var_44);
        var_47 = wp::extract(var_42, var_46);
        var_49 = wp::extract(var_42, var_48);
        var_50 = wp::vec_t<3,wp::float64>(var_45, var_47, var_49);
        // A = wp.vec3d(hat_yi[0], hat_yi[1], hat_yi[2])                                          <L 41>
        var_52 = wp::extract(var_42, var_51);
        var_54 = wp::extract(var_42, var_53);
        var_56 = wp::extract(var_42, var_55);
        var_57 = wp::vec_t<3,wp::float64>(var_52, var_54, var_56);
        // B = wp.vec3d(hat_yi[3], hat_yi[4], hat_yi[5])                                          <L 42>
        var_59 = wp::extract(var_42, var_58);
        var_61 = wp::extract(var_42, var_60);
        var_63 = wp::extract(var_42, var_62);
        var_64 = wp::vec_t<3,wp::float64>(var_59, var_61, var_63);
        // C = wp.vec3d(hat_yi[6], hat_yi[7], hat_yi[8])                                          <L 43>
        var_66 = wp::extract(var_42, var_65);
        var_68 = wp::extract(var_42, var_67);
        var_70 = wp::extract(var_42, var_69);
        var_71 = wp::vec_t<3,wp::float64>(var_66, var_68, var_70);
        // R = col_stack3(A - O, B - O, C - O)                                                    <L 44>
        var_72 = wp::sub(var_57, var_50);
        var_73 = wp::sub(var_64, var_50);
        var_74 = wp::sub(var_71, var_50);
        var_75 = col_stack3_1(var_72, var_73, var_74);
        // y_force = cat_4_vec3d(                                                                 <L 45>
        // R * wp.vec3d(rest_y_force[0], rest_y_force[1], rest_y_force[2]),                       <L 46>
        var_77 = wp::extract(var_36, var_76);
        var_79 = wp::extract(var_36, var_78);
        var_81 = wp::extract(var_36, var_80);
        var_82 = wp::vec_t<3,wp::float64>(var_77, var_79, var_81);
        var_83 = wp::mul(var_75, var_82);
        // R * wp.vec3d(rest_y_force[3], rest_y_force[4], rest_y_force[5]),                       <L 47>
        var_85 = wp::extract(var_36, var_84);
        var_87 = wp::extract(var_36, var_86);
        var_89 = wp::extract(var_36, var_88);
        var_90 = wp::vec_t<3,wp::float64>(var_85, var_87, var_89);
        var_91 = wp::mul(var_75, var_90);
        // R * wp.vec3d(rest_y_force[6], rest_y_force[7], rest_y_force[8]),                       <L 48>
        var_93 = wp::extract(var_36, var_92);
        var_95 = wp::extract(var_36, var_94);
        var_97 = wp::extract(var_36, var_96);
        var_98 = wp::vec_t<3,wp::float64>(var_93, var_95, var_97);
        var_99 = wp::mul(var_75, var_98);
        // R * wp.vec3d(rest_y_force[9], rest_y_force[10], rest_y_force[11]),                     <L 49>
        var_101 = wp::extract(var_36, var_100);
        var_103 = wp::extract(var_36, var_102);
        var_105 = wp::extract(var_36, var_104);
        var_106 = wp::vec_t<3,wp::float64>(var_101, var_103, var_105);
        var_107 = wp::mul(var_75, var_106);
        var_108 = cat_4_vec3d_0(var_83, var_91, var_99, var_107);
        // for d in range(12):                                                                    <L 51>
        // energy[tid] += -y_force[d] * yi[d] * scale                                             <L 52>
        var_110 = wp::extract(var_108, var_109);
        var_111 = wp::neg(var_110);
        var_112 = wp::extract(var_39, var_109);
        var_113 = wp::mul(var_111, var_112);
        var_114 = wp::mul(var_113, var_scale);
        // var_115 = wp::atomic_add(var_energy, var_0, var_114);
        var_117 = wp::extract(var_108, var_116);
        var_118 = wp::neg(var_117);
        var_119 = wp::extract(var_39, var_116);
        var_120 = wp::mul(var_118, var_119);
        var_121 = wp::mul(var_120, var_scale);
        // var_122 = wp::atomic_add(var_energy, var_0, var_121);
        var_124 = wp::extract(var_108, var_123);
        var_125 = wp::neg(var_124);
        var_126 = wp::extract(var_39, var_123);
        var_127 = wp::mul(var_125, var_126);
        var_128 = wp::mul(var_127, var_scale);
        // var_129 = wp::atomic_add(var_energy, var_0, var_128);
        var_131 = wp::extract(var_108, var_130);
        var_132 = wp::neg(var_131);
        var_133 = wp::extract(var_39, var_130);
        var_134 = wp::mul(var_132, var_133);
        var_135 = wp::mul(var_134, var_scale);
        // var_136 = wp::atomic_add(var_energy, var_0, var_135);
        var_138 = wp::extract(var_108, var_137);
        var_139 = wp::neg(var_138);
        var_140 = wp::extract(var_39, var_137);
        var_141 = wp::mul(var_139, var_140);
        var_142 = wp::mul(var_141, var_scale);
        // var_143 = wp::atomic_add(var_energy, var_0, var_142);
        var_145 = wp::extract(var_108, var_144);
        var_146 = wp::neg(var_145);
        var_147 = wp::extract(var_39, var_144);
        var_148 = wp::mul(var_146, var_147);
        var_149 = wp::mul(var_148, var_scale);
        // var_150 = wp::atomic_add(var_energy, var_0, var_149);
        var_152 = wp::extract(var_108, var_151);
        var_153 = wp::neg(var_152);
        var_154 = wp::extract(var_39, var_151);
        var_155 = wp::mul(var_153, var_154);
        var_156 = wp::mul(var_155, var_scale);
        // var_157 = wp::atomic_add(var_energy, var_0, var_156);
        var_159 = wp::extract(var_108, var_158);
        var_160 = wp::neg(var_159);
        var_161 = wp::extract(var_39, var_158);
        var_162 = wp::mul(var_160, var_161);
        var_163 = wp::mul(var_162, var_scale);
        // var_164 = wp::atomic_add(var_energy, var_0, var_163);
        var_166 = wp::extract(var_108, var_165);
        var_167 = wp::neg(var_166);
        var_168 = wp::extract(var_39, var_165);
        var_169 = wp::mul(var_167, var_168);
        var_170 = wp::mul(var_169, var_scale);
        // var_171 = wp::atomic_add(var_energy, var_0, var_170);
        var_173 = wp::extract(var_108, var_172);
        var_174 = wp::neg(var_173);
        var_175 = wp::extract(var_39, var_172);
        var_176 = wp::mul(var_174, var_175);
        var_177 = wp::mul(var_176, var_scale);
        // var_178 = wp::atomic_add(var_energy, var_0, var_177);
        var_180 = wp::extract(var_108, var_179);
        var_181 = wp::neg(var_180);
        var_182 = wp::extract(var_39, var_179);
        var_183 = wp::mul(var_181, var_182);
        var_184 = wp::mul(var_183, var_scale);
        // var_185 = wp::atomic_add(var_energy, var_0, var_184);
        var_187 = wp::extract(var_108, var_186);
        var_188 = wp::neg(var_187);
        var_189 = wp::extract(var_39, var_186);
        var_190 = wp::mul(var_188, var_189);
        var_191 = wp::mul(var_190, var_scale);
        // var_192 = wp::atomic_add(var_energy, var_0, var_191);
        //---------
        // reverse
        wp::adj_atomic_add(var_energy, var_0, var_191, adj_energy, adj_0, adj_191, adj_192);
        wp::adj_mul(var_190, var_scale, adj_190, adj_scale, adj_191);
        wp::adj_mul(var_188, var_189, adj_188, adj_189, adj_190);
        wp::adj_extract(var_39, var_186, adj_39, adj_186, adj_189);
        wp::adj_neg(var_187, adj_187, adj_188);
        wp::adj_extract(var_108, var_186, adj_108, adj_186, adj_187);
        wp::adj_atomic_add(var_energy, var_0, var_184, adj_energy, adj_0, adj_184, adj_185);
        wp::adj_mul(var_183, var_scale, adj_183, adj_scale, adj_184);
        wp::adj_mul(var_181, var_182, adj_181, adj_182, adj_183);
        wp::adj_extract(var_39, var_179, adj_39, adj_179, adj_182);
        wp::adj_neg(var_180, adj_180, adj_181);
        wp::adj_extract(var_108, var_179, adj_108, adj_179, adj_180);
        wp::adj_atomic_add(var_energy, var_0, var_177, adj_energy, adj_0, adj_177, adj_178);
        wp::adj_mul(var_176, var_scale, adj_176, adj_scale, adj_177);
        wp::adj_mul(var_174, var_175, adj_174, adj_175, adj_176);
        wp::adj_extract(var_39, var_172, adj_39, adj_172, adj_175);
        wp::adj_neg(var_173, adj_173, adj_174);
        wp::adj_extract(var_108, var_172, adj_108, adj_172, adj_173);
        wp::adj_atomic_add(var_energy, var_0, var_170, adj_energy, adj_0, adj_170, adj_171);
        wp::adj_mul(var_169, var_scale, adj_169, adj_scale, adj_170);
        wp::adj_mul(var_167, var_168, adj_167, adj_168, adj_169);
        wp::adj_extract(var_39, var_165, adj_39, adj_165, adj_168);
        wp::adj_neg(var_166, adj_166, adj_167);
        wp::adj_extract(var_108, var_165, adj_108, adj_165, adj_166);
        wp::adj_atomic_add(var_energy, var_0, var_163, adj_energy, adj_0, adj_163, adj_164);
        wp::adj_mul(var_162, var_scale, adj_162, adj_scale, adj_163);
        wp::adj_mul(var_160, var_161, adj_160, adj_161, adj_162);
        wp::adj_extract(var_39, var_158, adj_39, adj_158, adj_161);
        wp::adj_neg(var_159, adj_159, adj_160);
        wp::adj_extract(var_108, var_158, adj_108, adj_158, adj_159);
        wp::adj_atomic_add(var_energy, var_0, var_156, adj_energy, adj_0, adj_156, adj_157);
        wp::adj_mul(var_155, var_scale, adj_155, adj_scale, adj_156);
        wp::adj_mul(var_153, var_154, adj_153, adj_154, adj_155);
        wp::adj_extract(var_39, var_151, adj_39, adj_151, adj_154);
        wp::adj_neg(var_152, adj_152, adj_153);
        wp::adj_extract(var_108, var_151, adj_108, adj_151, adj_152);
        wp::adj_atomic_add(var_energy, var_0, var_149, adj_energy, adj_0, adj_149, adj_150);
        wp::adj_mul(var_148, var_scale, adj_148, adj_scale, adj_149);
        wp::adj_mul(var_146, var_147, adj_146, adj_147, adj_148);
        wp::adj_extract(var_39, var_144, adj_39, adj_144, adj_147);
        wp::adj_neg(var_145, adj_145, adj_146);
        wp::adj_extract(var_108, var_144, adj_108, adj_144, adj_145);
        wp::adj_atomic_add(var_energy, var_0, var_142, adj_energy, adj_0, adj_142, adj_143);
        wp::adj_mul(var_141, var_scale, adj_141, adj_scale, adj_142);
        wp::adj_mul(var_139, var_140, adj_139, adj_140, adj_141);
        wp::adj_extract(var_39, var_137, adj_39, adj_137, adj_140);
        wp::adj_neg(var_138, adj_138, adj_139);
        wp::adj_extract(var_108, var_137, adj_108, adj_137, adj_138);
        wp::adj_atomic_add(var_energy, var_0, var_135, adj_energy, adj_0, adj_135, adj_136);
        wp::adj_mul(var_134, var_scale, adj_134, adj_scale, adj_135);
        wp::adj_mul(var_132, var_133, adj_132, adj_133, adj_134);
        wp::adj_extract(var_39, var_130, adj_39, adj_130, adj_133);
        wp::adj_neg(var_131, adj_131, adj_132);
        wp::adj_extract(var_108, var_130, adj_108, adj_130, adj_131);
        wp::adj_atomic_add(var_energy, var_0, var_128, adj_energy, adj_0, adj_128, adj_129);
        wp::adj_mul(var_127, var_scale, adj_127, adj_scale, adj_128);
        wp::adj_mul(var_125, var_126, adj_125, adj_126, adj_127);
        wp::adj_extract(var_39, var_123, adj_39, adj_123, adj_126);
        wp::adj_neg(var_124, adj_124, adj_125);
        wp::adj_extract(var_108, var_123, adj_108, adj_123, adj_124);
        wp::adj_atomic_add(var_energy, var_0, var_121, adj_energy, adj_0, adj_121, adj_122);
        wp::adj_mul(var_120, var_scale, adj_120, adj_scale, adj_121);
        wp::adj_mul(var_118, var_119, adj_118, adj_119, adj_120);
        wp::adj_extract(var_39, var_116, adj_39, adj_116, adj_119);
        wp::adj_neg(var_117, adj_117, adj_118);
        wp::adj_extract(var_108, var_116, adj_108, adj_116, adj_117);
        wp::adj_atomic_add(var_energy, var_0, var_114, adj_energy, adj_0, adj_114, adj_115);
        wp::adj_mul(var_113, var_scale, adj_113, adj_scale, adj_114);
        wp::adj_mul(var_111, var_112, adj_111, adj_112, adj_113);
        wp::adj_extract(var_39, var_109, adj_39, adj_109, adj_112);
        wp::adj_neg(var_110, adj_110, adj_111);
        wp::adj_extract(var_108, var_109, adj_108, adj_109, adj_110);
        // adj: energy[tid] += -y_force[d] * yi[d] * scale                                        <L 52>
        // adj: for d in range(12):                                                               <L 51>
        adj_cat_4_vec3d_0(var_83, var_91, var_99, var_107, adj_83, adj_91, adj_99, adj_107, adj_108);
        wp::adj_mul(var_75, var_106, adj_75, adj_106, adj_107);
        wp::adj_vec_t(var_101, var_103, var_105, adj_101, adj_103, adj_105, adj_106);
        wp::adj_extract(var_36, var_104, adj_36, adj_104, adj_105);
        wp::adj_extract(var_36, var_102, adj_36, adj_102, adj_103);
        wp::adj_extract(var_36, var_100, adj_36, adj_100, adj_101);
        // adj: R * wp.vec3d(rest_y_force[9], rest_y_force[10], rest_y_force[11]),                <L 49>
        wp::adj_mul(var_75, var_98, adj_75, adj_98, adj_99);
        wp::adj_vec_t(var_93, var_95, var_97, adj_93, adj_95, adj_97, adj_98);
        wp::adj_extract(var_36, var_96, adj_36, adj_96, adj_97);
        wp::adj_extract(var_36, var_94, adj_36, adj_94, adj_95);
        wp::adj_extract(var_36, var_92, adj_36, adj_92, adj_93);
        // adj: R * wp.vec3d(rest_y_force[6], rest_y_force[7], rest_y_force[8]),                  <L 48>
        wp::adj_mul(var_75, var_90, adj_75, adj_90, adj_91);
        wp::adj_vec_t(var_85, var_87, var_89, adj_85, adj_87, adj_89, adj_90);
        wp::adj_extract(var_36, var_88, adj_36, adj_88, adj_89);
        wp::adj_extract(var_36, var_86, adj_36, adj_86, adj_87);
        wp::adj_extract(var_36, var_84, adj_36, adj_84, adj_85);
        // adj: R * wp.vec3d(rest_y_force[3], rest_y_force[4], rest_y_force[5]),                  <L 47>
        wp::adj_mul(var_75, var_82, adj_75, adj_82, adj_83);
        wp::adj_vec_t(var_77, var_79, var_81, adj_77, adj_79, adj_81, adj_82);
        wp::adj_extract(var_36, var_80, adj_36, adj_80, adj_81);
        wp::adj_extract(var_36, var_78, adj_36, adj_78, adj_79);
        wp::adj_extract(var_36, var_76, adj_36, adj_76, adj_77);
        // adj: R * wp.vec3d(rest_y_force[0], rest_y_force[1], rest_y_force[2]),                  <L 46>
        // adj: y_force = cat_4_vec3d(                                                            <L 45>
        adj_col_stack3_1(var_72, var_73, var_74, adj_72, adj_73, adj_74, adj_75);
        wp::adj_sub(var_71, var_50, adj_71, adj_50, adj_74);
        wp::adj_sub(var_64, var_50, adj_64, adj_50, adj_73);
        wp::adj_sub(var_57, var_50, adj_57, adj_50, adj_72);
        // adj: R = col_stack3(A - O, B - O, C - O)                                               <L 44>
        wp::adj_vec_t(var_66, var_68, var_70, adj_66, adj_68, adj_70, adj_71);
        wp::adj_extract(var_42, var_69, adj_42, adj_69, adj_70);
        wp::adj_extract(var_42, var_67, adj_42, adj_67, adj_68);
        wp::adj_extract(var_42, var_65, adj_42, adj_65, adj_66);
        // adj: C = wp.vec3d(hat_yi[6], hat_yi[7], hat_yi[8])                                     <L 43>
        wp::adj_vec_t(var_59, var_61, var_63, adj_59, adj_61, adj_63, adj_64);
        wp::adj_extract(var_42, var_62, adj_42, adj_62, adj_63);
        wp::adj_extract(var_42, var_60, adj_42, adj_60, adj_61);
        wp::adj_extract(var_42, var_58, adj_42, adj_58, adj_59);
        // adj: B = wp.vec3d(hat_yi[3], hat_yi[4], hat_yi[5])                                     <L 42>
        wp::adj_vec_t(var_52, var_54, var_56, adj_52, adj_54, adj_56, adj_57);
        wp::adj_extract(var_42, var_55, adj_42, adj_55, adj_56);
        wp::adj_extract(var_42, var_53, adj_42, adj_53, adj_54);
        wp::adj_extract(var_42, var_51, adj_42, adj_51, adj_52);
        // adj: A = wp.vec3d(hat_yi[0], hat_yi[1], hat_yi[2])                                     <L 41>
        wp::adj_vec_t(var_45, var_47, var_49, adj_45, adj_47, adj_49, adj_50);
        wp::adj_extract(var_42, var_48, adj_42, adj_48, adj_49);
        wp::adj_extract(var_42, var_46, adj_42, adj_46, adj_47);
        wp::adj_extract(var_42, var_44, adj_42, adj_44, adj_45);
        // adj: O = wp.vec3d(hat_yi[9], hat_yi[10], hat_yi[11])                                   <L 40>
        wp::adj_copy(var_43, adj_41, adj_42);
        wp::adj_load(var_41, adj_41, adj_43);
        wp::adj_address(var_hat_y, var_0, adj_hat_y, adj_0, adj_41);
        // adj: hat_yi = hat_y[tid]                                                               <L 39>
        wp::adj_copy(var_40, adj_38, adj_39);
        wp::adj_load(var_38, adj_38, adj_40);
        wp::adj_address(var_y, var_0, adj_y, adj_0, adj_38);
        // adj: yi = y[tid]                                                                       <L 38>
        wp::adj_copy(var_37, adj_35, adj_36);
        wp::adj_load(var_35, adj_35, adj_37);
        wp::adj_address(var_affine_ext_y_force, var_0, adj_affine_ext_y_force, adj_0, adj_35);
        // adj: rest_y_force = affine_ext_y_force[tid]                                            <L 37>
        var_14 = wp::iter_reverse(var_14);
        start_for_1:;
            if (iter_cmp(var_14) == 0) goto end_for_1;
            var_15 = wp::iter_next(var_14);
        	adj_16 = {};
        	adj_17 = {};
        	adj_18 = {};
            // for j in range(dim + 1):                                                           <L 35>
            var_17 = wp::add(var_4, var_16);
            var_18 = wp::range(var_17);
            var_18 = wp::iter_reverse(var_18);
            start_for_3:;
                if (iter_cmp(var_18) == 0) goto end_for_3;
                var_19 = wp::iter_next(var_18);
            	adj_20 = {};
            	adj_21 = {};
            	adj_22 = {};
            	adj_23 = {};
            	adj_24 = {};
            	adj_25 = {};
            	adj_26 = {};
            	adj_27 = {};
            	adj_28 = {};
            	adj_29 = {};
            	adj_30 = {};
            	adj_31 = {};
            	adj_32 = {};
            	adj_33 = {};
            	adj_34 = {};
                // energy[tid] += -(gravity[i % dim] + ext_force[i % dim]) * y[tid][i] * mass[(i / dim), j] * scale       <L 36>
                var_20 = wp::mod(var_15, var_4);
                var_21 = wp::extract(var_gravity, var_20);
                var_22 = wp::mod(var_15, var_4);
                var_23 = wp::extract(var_12, var_22);
                var_24 = wp::add(var_21, var_23);
                var_25 = wp::neg(var_24);
                var_26 = wp::address(var_y, var_0);
                var_28 = wp::load(var_26);
                var_27 = wp::extract(var_28, var_15);
                var_29 = wp::mul(var_25, var_27);
                var_30 = wp::div(var_15, var_4);
                var_31 = wp::extract(var_9, var_30, var_19);
                var_32 = wp::mul(var_29, var_31);
                var_33 = wp::mul(var_32, var_scale);
                // var_34 = wp::atomic_add(var_energy, var_0, var_33);
                wp::adj_atomic_add(var_energy, var_0, var_33, adj_energy, adj_0, adj_33, adj_34);
                wp::adj_mul(var_32, var_scale, adj_32, adj_scale, adj_33);
                wp::adj_mul(var_29, var_31, adj_29, adj_31, adj_32);
                wp::adj_extract(var_9, var_30, var_19, adj_9, adj_30, adj_19, adj_31);
                wp::adj_div(var_15, var_4, var_30, adj_15, adj_4, adj_30);
                wp::adj_mul(var_25, var_27, adj_25, adj_27, adj_29);
                wp::adj_extract(var_28, var_15, adj_26, adj_15, adj_27);
                wp::adj_load(var_26, adj_26, adj_28);
                wp::adj_address(var_y, var_0, adj_y, adj_0, adj_26);
                wp::adj_neg(var_24, adj_24, adj_25);
                wp::adj_add(var_21, var_23, adj_21, adj_23, adj_24);
                wp::adj_extract(var_12, var_22, adj_12, adj_22, adj_23);
                wp::adj_mod(var_15, var_4, adj_15, adj_4, adj_22);
                wp::adj_extract(var_gravity, var_20, adj_gravity, adj_20, adj_21);
                wp::adj_mod(var_15, var_4, adj_15, adj_4, adj_20);
                // adj: energy[tid] += -(gravity[i % dim] + ext_force[i % dim]) * y[tid][i] * mass[(i / dim), j] * scale  <L 36>
            	goto start_for_3;
            end_for_3:;
            wp::adj_range(var_17, adj_17, adj_18);
            wp::adj_add(var_4, var_16, adj_4, adj_16, adj_17);
            // adj: for j in range(dim + 1):                                                      <L 35>
        	goto start_for_1;
        end_for_1:;
        wp::adj_range(var_7, adj_7, adj_14);
        // adj: for i in range(n_dof):                                                            <L 34>
        wp::adj_copy(var_13, adj_11, adj_12);
        wp::adj_load(var_11, adj_11, adj_13);
        wp::adj_address(var_affine_ext_force, var_0, adj_affine_ext_force, adj_0, adj_11);
        // adj: ext_force = affine_ext_force[tid]                                                 <L 33>
        wp::adj_copy(var_10, adj_8, adj_9);
        wp::adj_load(var_8, adj_8, adj_10);
        wp::adj_address(var_mass_matrix, var_0, adj_mass_matrix, adj_0, adj_8);
        // adj: mass = mass_matrix[tid]                                                           <L 32>
        wp::adj_mul(var_6, var_4, adj_6, adj_4, adj_7);
        wp::adj_add(var_4, var_5, adj_4, adj_5, adj_6);
        // adj: n_dof = (dim + 1) * dim                                                           <L 31>
        // adj: dim = 3                                                                           <L 30>
        if (var_3) {
        wp::adj_load(var_1, adj_1, adj_3);
            label0:;
            // adj: return                                                                        <L 29>
        }
        wp::adj_load(var_1, adj_1, adj_2);
        wp::adj_address(var_affine_has_constraint, var_0, adj_affine_has_constraint, adj_0, adj_1);
        // adj: if affine_has_constraint[tid]:                                                    <L 28>
        // adj: tid = wp.tid()                                                                    <L 27>
        // adj: def compute_body_force_energy_val_affine(                                         <L 16>
        continue;
    }
}



extern "C" __global__ void compute_body_force_energy_grad_soft_x_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float64>> var_gradient,
    wp::array_t<wp::float64> var_soft_verts_mass,
    wp::vec_t<3,wp::float64> var_gravity,
    wp::float64 var_scale,
    wp::int32 var_affine_verts_num,
    wp::array_t<bool> var_soft_has_constraint,
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
        bool* var_1;
        wp::int32 var_2;
        wp::int32* var_3;
        wp::int32* var_4;
        wp::int32 var_5;
        const wp::int32 var_6 = 1;
        bool var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 2;
        bool var_14;
        wp::int32 var_15;
        bool var_16;
        bool var_17;
        bool var_18;
        wp::vec_t<3,wp::float64> var_19;
        wp::float64* var_20;
        wp::vec_t<3,wp::float64> var_21;
        wp::float64 var_22;
        wp::vec_t<3,wp::float64> var_23;
        wp::vec_t<3,wp::float64> var_24;
        //---------
        // forward
        // def compute_body_force_energy_grad_soft_x(                                             <L 118>
        // tid = wp.tid()                                                                         <L 128>
        var_0 = builtin_tid1d();
        // if soft_has_constraint[tid] or (                                                       <L 129>
        var_1 = wp::address(var_soft_has_constraint, var_0);
        // (env_states[node2env[tid + affine_verts_num]] == ENV_STATE_INVALID)                    <L 130>
        var_2 = wp::add(var_0, var_affine_verts_num);
        var_3 = wp::address(var_node2env, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::address(var_env_states, var_5);
        var_8 = wp::load(var_4);
        var_7 = (var_8 == var_6);
        // or (env_states[node2env[tid + affine_verts_num]] == ENV_STATE_NEWTON_SOLVED)           <L 131>
        var_9 = wp::add(var_0, var_affine_verts_num);
        var_10 = wp::address(var_node2env, var_9);
        var_12 = wp::load(var_10);
        var_11 = wp::address(var_env_states, var_12);
        var_15 = wp::load(var_11);
        var_14 = (var_15 == var_13);
        var_16 = var_7 || var_14;
        var_17 = wp::load(var_1);
        var_18 = var_17 || var_16;
        if (var_18) {
            // return                                                                             <L 133>
            continue;
        }
        // wp.atomic_add(gradient, tid, -gravity * soft_verts_mass[tid] * scale)                  <L 134>
        var_19 = wp::neg(var_gravity);
        var_20 = wp::address(var_soft_verts_mass, var_0);
        var_22 = wp::load(var_20);
        var_21 = wp::mul(var_19, var_22);
        var_23 = wp::mul(var_21, var_scale);
        var_24 = wp::atomic_add(var_gradient, var_0, var_23);
    }
}



extern "C" __global__ void compute_body_force_energy_grad_soft_x_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float64>> var_gradient,
    wp::array_t<wp::float64> var_soft_verts_mass,
    wp::vec_t<3,wp::float64> var_gravity,
    wp::float64 var_scale,
    wp::int32 var_affine_verts_num,
    wp::array_t<bool> var_soft_has_constraint,
    wp::array_t<wp::int32> var_node2env,
    wp::array_t<wp::int32> var_env_states,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_gradient,
    wp::array_t<wp::float64> adj_soft_verts_mass,
    wp::vec_t<3,wp::float64> adj_gravity,
    wp::float64 adj_scale,
    wp::int32 adj_affine_verts_num,
    wp::array_t<bool> adj_soft_has_constraint,
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
        bool* var_1;
        wp::int32 var_2;
        wp::int32* var_3;
        wp::int32* var_4;
        wp::int32 var_5;
        const wp::int32 var_6 = 1;
        bool var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 2;
        bool var_14;
        wp::int32 var_15;
        bool var_16;
        bool var_17;
        bool var_18;
        wp::vec_t<3,wp::float64> var_19;
        wp::float64* var_20;
        wp::vec_t<3,wp::float64> var_21;
        wp::float64 var_22;
        wp::vec_t<3,wp::float64> var_23;
        wp::vec_t<3,wp::float64> var_24;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        bool adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        bool adj_14 = {};
        wp::int32 adj_15 = {};
        bool adj_16 = {};
        bool adj_17 = {};
        bool adj_18 = {};
        wp::vec_t<3,wp::float64> adj_19 = {};
        wp::float64 adj_20 = {};
        wp::vec_t<3,wp::float64> adj_21 = {};
        wp::float64 adj_22 = {};
        wp::vec_t<3,wp::float64> adj_23 = {};
        wp::vec_t<3,wp::float64> adj_24 = {};
        //---------
        // forward
        // def compute_body_force_energy_grad_soft_x(                                             <L 118>
        // tid = wp.tid()                                                                         <L 128>
        var_0 = builtin_tid1d();
        // if soft_has_constraint[tid] or (                                                       <L 129>
        var_1 = wp::address(var_soft_has_constraint, var_0);
        // (env_states[node2env[tid + affine_verts_num]] == ENV_STATE_INVALID)                    <L 130>
        var_2 = wp::add(var_0, var_affine_verts_num);
        var_3 = wp::address(var_node2env, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::address(var_env_states, var_5);
        var_8 = wp::load(var_4);
        var_7 = (var_8 == var_6);
        // or (env_states[node2env[tid + affine_verts_num]] == ENV_STATE_NEWTON_SOLVED)           <L 131>
        var_9 = wp::add(var_0, var_affine_verts_num);
        var_10 = wp::address(var_node2env, var_9);
        var_12 = wp::load(var_10);
        var_11 = wp::address(var_env_states, var_12);
        var_15 = wp::load(var_11);
        var_14 = (var_15 == var_13);
        var_16 = var_7 || var_14;
        var_17 = wp::load(var_1);
        var_18 = var_17 || var_16;
        if (var_18) {
            // return                                                                             <L 133>
            goto label0;
        }
        // wp.atomic_add(gradient, tid, -gravity * soft_verts_mass[tid] * scale)                  <L 134>
        var_19 = wp::neg(var_gravity);
        var_20 = wp::address(var_soft_verts_mass, var_0);
        var_22 = wp::load(var_20);
        var_21 = wp::mul(var_19, var_22);
        var_23 = wp::mul(var_21, var_scale);
        // var_24 = wp::atomic_add(var_gradient, var_0, var_23);
        //---------
        // reverse
        wp::adj_atomic_add(var_gradient, var_0, var_23, adj_gradient, adj_0, adj_23, adj_24);
        wp::adj_mul(var_21, var_scale, adj_21, adj_scale, adj_23);
        wp::adj_mul(var_19, var_22, adj_19, adj_20, adj_21);
        wp::adj_load(var_20, adj_20, adj_22);
        wp::adj_address(var_soft_verts_mass, var_0, adj_soft_verts_mass, adj_0, adj_20);
        wp::adj_neg(var_gravity, adj_gravity, adj_19);
        // adj: wp.atomic_add(gradient, tid, -gravity * soft_verts_mass[tid] * scale)             <L 134>
        if (var_18) {
            label0:;
            // adj: return                                                                        <L 133>
        }
        wp::adj_load(var_1, adj_1, adj_17);
        wp::adj_load(var_11, adj_11, adj_15);
        wp::adj_address(var_env_states, var_12, adj_env_states, adj_10, adj_11);
        wp::adj_load(var_10, adj_10, adj_12);
        wp::adj_address(var_node2env, var_9, adj_node2env, adj_9, adj_10);
        wp::adj_add(var_0, var_affine_verts_num, adj_0, adj_affine_verts_num, adj_9);
        // adj: or (env_states[node2env[tid + affine_verts_num]] == ENV_STATE_NEWTON_SOLVED)      <L 131>
        wp::adj_load(var_4, adj_4, adj_8);
        wp::adj_address(var_env_states, var_5, adj_env_states, adj_3, adj_4);
        wp::adj_load(var_3, adj_3, adj_5);
        wp::adj_address(var_node2env, var_2, adj_node2env, adj_2, adj_3);
        wp::adj_add(var_0, var_affine_verts_num, adj_0, adj_affine_verts_num, adj_2);
        // adj: (env_states[node2env[tid + affine_verts_num]] == ENV_STATE_INVALID)               <L 130>
        wp::adj_address(var_soft_has_constraint, var_0, adj_soft_has_constraint, adj_0, adj_1);
        // adj: if soft_has_constraint[tid] or (                                                  <L 129>
        // adj: tid = wp.tid()                                                                    <L 128>
        // adj: def compute_body_force_energy_grad_soft_x(                                        <L 118>
        continue;
    }
}



extern "C" __global__ void compute_body_force_energy_val_soft_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float64> var_energy,
    wp::array_t<wp::vec_t<3,wp::float64>> var_x,
    wp::array_t<wp::float64> var_soft_verts_mass,
    wp::vec_t<3,wp::float64> var_gravity,
    wp::float64 var_scale,
    wp::int32 var_affine_verts_num,
    wp::array_t<bool> var_soft_has_constraint)
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
        wp::int32 var_4;
        wp::vec_t<3,wp::float64>* var_5;
        wp::float64 var_6;
        wp::vec_t<3,wp::float64> var_7;
        wp::float64 var_8;
        wp::float64 var_9;
        wp::float64* var_10;
        wp::float64 var_11;
        wp::float64 var_12;
        wp::float64 var_13;
        //---------
        // forward
        // def compute_body_force_energy_val_soft(                                                <L 56>
        // tid = wp.tid()                                                                         <L 65>
        var_0 = builtin_tid1d();
        // if soft_has_constraint[tid]:                                                           <L 66>
        var_1 = wp::address(var_soft_has_constraint, var_0);
        var_2 = wp::load(var_1);
        if (var_2) {
            // return                                                                             <L 67>
            continue;
        }
        var_3 = wp::load(var_1);
        // energy[tid] += -wp.dot(gravity, x[tid + affine_verts_num]) * scale * soft_verts_mass[tid]       <L 68>
        var_4 = wp::add(var_0, var_affine_verts_num);
        var_5 = wp::address(var_x, var_4);
        var_7 = wp::load(var_5);
        var_6 = wp::dot(var_gravity, var_7);
        var_8 = wp::neg(var_6);
        var_9 = wp::mul(var_8, var_scale);
        var_10 = wp::address(var_soft_verts_mass, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::mul(var_9, var_12);
        var_13 = wp::atomic_add(var_energy, var_0, var_11);
    }
}



extern "C" __global__ void compute_body_force_energy_val_soft_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float64> var_energy,
    wp::array_t<wp::vec_t<3,wp::float64>> var_x,
    wp::array_t<wp::float64> var_soft_verts_mass,
    wp::vec_t<3,wp::float64> var_gravity,
    wp::float64 var_scale,
    wp::int32 var_affine_verts_num,
    wp::array_t<bool> var_soft_has_constraint,
    wp::array_t<wp::float64> adj_energy,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_x,
    wp::array_t<wp::float64> adj_soft_verts_mass,
    wp::vec_t<3,wp::float64> adj_gravity,
    wp::float64 adj_scale,
    wp::int32 adj_affine_verts_num,
    wp::array_t<bool> adj_soft_has_constraint)
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
        wp::int32 var_4;
        wp::vec_t<3,wp::float64>* var_5;
        wp::float64 var_6;
        wp::vec_t<3,wp::float64> var_7;
        wp::float64 var_8;
        wp::float64 var_9;
        wp::float64* var_10;
        wp::float64 var_11;
        wp::float64 var_12;
        wp::float64 var_13;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        wp::int32 adj_4 = {};
        wp::vec_t<3,wp::float64> adj_5 = {};
        wp::float64 adj_6 = {};
        wp::vec_t<3,wp::float64> adj_7 = {};
        wp::float64 adj_8 = {};
        wp::float64 adj_9 = {};
        wp::float64 adj_10 = {};
        wp::float64 adj_11 = {};
        wp::float64 adj_12 = {};
        wp::float64 adj_13 = {};
        //---------
        // forward
        // def compute_body_force_energy_val_soft(                                                <L 56>
        // tid = wp.tid()                                                                         <L 65>
        var_0 = builtin_tid1d();
        // if soft_has_constraint[tid]:                                                           <L 66>
        var_1 = wp::address(var_soft_has_constraint, var_0);
        var_2 = wp::load(var_1);
        if (var_2) {
            // return                                                                             <L 67>
            goto label0;
        }
        var_3 = wp::load(var_1);
        // energy[tid] += -wp.dot(gravity, x[tid + affine_verts_num]) * scale * soft_verts_mass[tid]       <L 68>
        var_4 = wp::add(var_0, var_affine_verts_num);
        var_5 = wp::address(var_x, var_4);
        var_7 = wp::load(var_5);
        var_6 = wp::dot(var_gravity, var_7);
        var_8 = wp::neg(var_6);
        var_9 = wp::mul(var_8, var_scale);
        var_10 = wp::address(var_soft_verts_mass, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::mul(var_9, var_12);
        // var_13 = wp::atomic_add(var_energy, var_0, var_11);
        //---------
        // reverse
        wp::adj_atomic_add(var_energy, var_0, var_11, adj_energy, adj_0, adj_11, adj_13);
        wp::adj_mul(var_9, var_12, adj_9, adj_10, adj_11);
        wp::adj_load(var_10, adj_10, adj_12);
        wp::adj_address(var_soft_verts_mass, var_0, adj_soft_verts_mass, adj_0, adj_10);
        wp::adj_mul(var_8, var_scale, adj_8, adj_scale, adj_9);
        wp::adj_neg(var_6, adj_6, adj_8);
        wp::adj_dot(var_gravity, var_7, adj_gravity, adj_5, adj_6);
        wp::adj_load(var_5, adj_5, adj_7);
        wp::adj_address(var_x, var_4, adj_x, adj_4, adj_5);
        wp::adj_add(var_0, var_affine_verts_num, adj_0, adj_affine_verts_num, adj_4);
        // adj: energy[tid] += -wp.dot(gravity, x[tid + affine_verts_num]) * scale * soft_verts_mass[tid]  <L 68>
        if (var_3) {
        wp::adj_load(var_1, adj_1, adj_3);
            label0:;
            // adj: return                                                                        <L 67>
        }
        wp::adj_load(var_1, adj_1, adj_2);
        wp::adj_address(var_soft_has_constraint, var_0, adj_soft_has_constraint, adj_0, adj_1);
        // adj: if soft_has_constraint[tid]:                                                      <L 66>
        // adj: tid = wp.tid()                                                                    <L 65>
        // adj: def compute_body_force_energy_val_soft(                                           <L 56>
        continue;
    }
}

