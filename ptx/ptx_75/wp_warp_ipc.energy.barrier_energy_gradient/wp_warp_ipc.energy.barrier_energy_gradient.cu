
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

// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/barrier.py:13
static CUDA_CALLABLE wp::float64 barrier_gradient_0(
    wp::float64 var_d2,
    wp::float64 var_dHat2,
    wp::float64 var_kappa)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    wp::float64 var_1;
    bool var_2;
    wp::float64 var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    wp::float64 var_6;
    wp::float64 var_7;
    const wp::float32 var_8 = 2.0;
    wp::float64 var_9;
    wp::float64 var_10;
    wp::float64 var_11;
    wp::float64 var_12;
    wp::float64 var_13;
    wp::float64 var_14;
    wp::float64 var_15;
    wp::float64 var_16;
    wp::float64 var_17;
    wp::float64 var_18;
    wp::float64 var_19;
    //---------
    // forward
    // def barrier_gradient(d2: wp.float64, dHat2: wp.float64, kappa: wp.float64):            <L 14>
    // grad = wp.float64(0.0)                                                                 <L 15>
    var_1 = wp::float64(var_0);
    // if d2 < dHat2:                                                                         <L 16>
    var_2 = (var_d2 < var_dHat2);
    if (var_2) {
        // t2 = d2 - dHat2                                                                    <L 17>
        var_3 = wp::sub(var_d2, var_dHat2);
        // grad = kappa * ((t2 / dHat2) * wp.log(d2 / dHat2) * -wp.float64(2.0) / dHat2 - ((t2 / dHat2) * (t2 / dHat2)) / d2)       <L 18>
        var_4 = wp::div(var_3, var_dHat2);
        var_5 = wp::div(var_d2, var_dHat2);
        var_6 = wp::log(var_5);
        var_7 = wp::mul(var_4, var_6);
        var_9 = wp::float64(var_8);
        var_10 = wp::neg(var_9);
        var_11 = wp::mul(var_7, var_10);
        var_12 = wp::div(var_11, var_dHat2);
        var_13 = wp::div(var_3, var_dHat2);
        var_14 = wp::div(var_3, var_dHat2);
        var_15 = wp::mul(var_13, var_14);
        var_16 = wp::div(var_15, var_d2);
        var_17 = wp::sub(var_12, var_16);
        var_18 = wp::mul(var_kappa, var_17);
    }
    var_19 = wp::where(var_2, var_18, var_1);
    // return grad                                                                            <L 19>
    return var_19;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/barrier.py:13
static CUDA_CALLABLE void adj_barrier_gradient_0(
    wp::float64 var_d2,
    wp::float64 var_dHat2,
    wp::float64 var_kappa,
    wp::float64 & adj_d2,
    wp::float64 & adj_dHat2,
    wp::float64 & adj_kappa,
    wp::float64 & adj_ret)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    wp::float64 var_1;
    bool var_2;
    wp::float64 var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    wp::float64 var_6;
    wp::float64 var_7;
    const wp::float32 var_8 = 2.0;
    wp::float64 var_9;
    wp::float64 var_10;
    wp::float64 var_11;
    wp::float64 var_12;
    wp::float64 var_13;
    wp::float64 var_14;
    wp::float64 var_15;
    wp::float64 var_16;
    wp::float64 var_17;
    wp::float64 var_18;
    wp::float64 var_19;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    wp::float64 adj_1 = {};
    bool adj_2 = {};
    wp::float64 adj_3 = {};
    wp::float64 adj_4 = {};
    wp::float64 adj_5 = {};
    wp::float64 adj_6 = {};
    wp::float64 adj_7 = {};
    wp::float32 adj_8 = {};
    wp::float64 adj_9 = {};
    wp::float64 adj_10 = {};
    wp::float64 adj_11 = {};
    wp::float64 adj_12 = {};
    wp::float64 adj_13 = {};
    wp::float64 adj_14 = {};
    wp::float64 adj_15 = {};
    wp::float64 adj_16 = {};
    wp::float64 adj_17 = {};
    wp::float64 adj_18 = {};
    wp::float64 adj_19 = {};
    //---------
    // forward
    // def barrier_gradient(d2: wp.float64, dHat2: wp.float64, kappa: wp.float64):            <L 14>
    // grad = wp.float64(0.0)                                                                 <L 15>
    var_1 = wp::float64(var_0);
    // if d2 < dHat2:                                                                         <L 16>
    var_2 = (var_d2 < var_dHat2);
    if (var_2) {
        // t2 = d2 - dHat2                                                                    <L 17>
        var_3 = wp::sub(var_d2, var_dHat2);
        // grad = kappa * ((t2 / dHat2) * wp.log(d2 / dHat2) * -wp.float64(2.0) / dHat2 - ((t2 / dHat2) * (t2 / dHat2)) / d2)       <L 18>
        var_4 = wp::div(var_3, var_dHat2);
        var_5 = wp::div(var_d2, var_dHat2);
        var_6 = wp::log(var_5);
        var_7 = wp::mul(var_4, var_6);
        var_9 = wp::float64(var_8);
        var_10 = wp::neg(var_9);
        var_11 = wp::mul(var_7, var_10);
        var_12 = wp::div(var_11, var_dHat2);
        var_13 = wp::div(var_3, var_dHat2);
        var_14 = wp::div(var_3, var_dHat2);
        var_15 = wp::mul(var_13, var_14);
        var_16 = wp::div(var_15, var_d2);
        var_17 = wp::sub(var_12, var_16);
        var_18 = wp::mul(var_kappa, var_17);
    }
    var_19 = wp::where(var_2, var_18, var_1);
    // return grad                                                                            <L 19>
    goto label0;
    //---------
    // reverse
    label0:;
    adj_19 += adj_ret;
    // adj: return grad                                                                       <L 19>
    wp::adj_where(var_2, var_18, var_1, adj_2, adj_18, adj_1, adj_19);
    if (var_2) {
        wp::adj_mul(var_kappa, var_17, adj_kappa, adj_17, adj_18);
        wp::adj_sub(var_12, var_16, adj_12, adj_16, adj_17);
        wp::adj_div(var_15, var_d2, var_16, adj_15, adj_d2, adj_16);
        wp::adj_mul(var_13, var_14, adj_13, adj_14, adj_15);
        wp::adj_div(var_3, var_dHat2, var_14, adj_3, adj_dHat2, adj_14);
        wp::adj_div(var_3, var_dHat2, var_13, adj_3, adj_dHat2, adj_13);
        wp::adj_div(var_11, var_dHat2, var_12, adj_11, adj_dHat2, adj_12);
        wp::adj_mul(var_7, var_10, adj_7, adj_10, adj_11);
        wp::adj_neg(var_9, adj_9, adj_10);
        wp::adj_float64(var_8, adj_8, adj_9);
        wp::adj_mul(var_4, var_6, adj_4, adj_6, adj_7);
        wp::adj_log(var_5, adj_5, adj_6);
        wp::adj_div(var_d2, var_dHat2, var_5, adj_d2, adj_dHat2, adj_5);
        wp::adj_div(var_3, var_dHat2, var_4, adj_3, adj_dHat2, adj_4);
        // adj: grad = kappa * ((t2 / dHat2) * wp.log(d2 / dHat2) * -wp.float64(2.0) / dHat2 - ((t2 / dHat2) * (t2 / dHat2)) / d2)  <L 18>
        wp::adj_sub(var_d2, var_dHat2, adj_d2, adj_dHat2, adj_3);
        // adj: t2 = d2 - dHat2                                                               <L 17>
    }
    // adj: if d2 < dHat2:                                                                    <L 16>
    wp::adj_float64(var_0, adj_0, adj_1);
    // adj: grad = wp.float64(0.0)                                                            <L 15>
    // adj: def barrier_gradient(d2: wp.float64, dHat2: wp.float64, kappa: wp.float64):       <L 14>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/friction.py:14
static CUDA_CALLABLE wp::float64 f1_div_vbarnorm_0(
    wp::float64 var_vbarnorm,
    wp::float64 var_epsv)
{
    //---------
    // primal vars
    bool var_0;
    const wp::float32 var_1 = 1.0;
    wp::float64 var_2;
    wp::float64 var_3;
    wp::float64 var_4;
    const wp::float32 var_5 = 2.0;
    wp::float64 var_6;
    wp::float64 var_7;
    wp::float64 var_8;
    wp::float64 var_9;
    wp::float64 var_10;
    //---------
    // forward
    // def f1_div_vbarnorm(vbarnorm: wp.float64, epsv: wp.float64) -> wp.float64:             <L 15>
    // if vbarnorm >= epsv:                                                                   <L 16>
    var_0 = (var_vbarnorm >= var_epsv);
    if (var_0) {
        // return wp.float64(1.0) / vbarnorm                                                  <L 17>
        var_2 = wp::float64(var_1);
        var_3 = wp::div(var_2, var_vbarnorm);
        return var_3;
    }
    if (!var_0) {
        // return (-vbarnorm + wp.float64(2.0) * epsv) / (epsv * epsv)                        <L 19>
        var_4 = wp::neg(var_vbarnorm);
        var_6 = wp::float64(var_5);
        var_7 = wp::mul(var_6, var_epsv);
        var_8 = wp::add(var_4, var_7);
        var_9 = wp::mul(var_epsv, var_epsv);
        var_10 = wp::div(var_8, var_9);
        return var_10;
    }
    return {};
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/friction.py:14
static CUDA_CALLABLE void adj_f1_div_vbarnorm_0(
    wp::float64 var_vbarnorm,
    wp::float64 var_epsv,
    wp::float64 & adj_vbarnorm,
    wp::float64 & adj_epsv,
    wp::float64 & adj_ret)
{
    //---------
    // primal vars
    bool var_0;
    const wp::float32 var_1 = 1.0;
    wp::float64 var_2;
    wp::float64 var_3;
    wp::float64 var_4;
    const wp::float32 var_5 = 2.0;
    wp::float64 var_6;
    wp::float64 var_7;
    wp::float64 var_8;
    wp::float64 var_9;
    wp::float64 var_10;
    //---------
    // dual vars
    bool adj_0 = {};
    wp::float32 adj_1 = {};
    wp::float64 adj_2 = {};
    wp::float64 adj_3 = {};
    wp::float64 adj_4 = {};
    wp::float32 adj_5 = {};
    wp::float64 adj_6 = {};
    wp::float64 adj_7 = {};
    wp::float64 adj_8 = {};
    wp::float64 adj_9 = {};
    wp::float64 adj_10 = {};
    //---------
    // forward
    // def f1_div_vbarnorm(vbarnorm: wp.float64, epsv: wp.float64) -> wp.float64:             <L 15>
    // if vbarnorm >= epsv:                                                                   <L 16>
    var_0 = (var_vbarnorm >= var_epsv);
    if (var_0) {
        // return wp.float64(1.0) / vbarnorm                                                  <L 17>
        var_2 = wp::float64(var_1);
        var_3 = wp::div(var_2, var_vbarnorm);
        goto label0;
    }
    if (!var_0) {
        // return (-vbarnorm + wp.float64(2.0) * epsv) / (epsv * epsv)                        <L 19>
        var_4 = wp::neg(var_vbarnorm);
        var_6 = wp::float64(var_5);
        var_7 = wp::mul(var_6, var_epsv);
        var_8 = wp::add(var_4, var_7);
        var_9 = wp::mul(var_epsv, var_epsv);
        var_10 = wp::div(var_8, var_9);
        goto label1;
    }
    //---------
    // reverse
    if (!var_0) {
        label1:;
        adj_10 += adj_ret;
        wp::adj_div(var_8, var_9, var_10, adj_8, adj_9, adj_10);
        wp::adj_mul(var_epsv, var_epsv, adj_epsv, adj_epsv, adj_9);
        wp::adj_add(var_4, var_7, adj_4, adj_7, adj_8);
        wp::adj_mul(var_6, var_epsv, adj_6, adj_epsv, adj_7);
        wp::adj_float64(var_5, adj_5, adj_6);
        wp::adj_neg(var_vbarnorm, adj_vbarnorm, adj_4);
        // adj: return (-vbarnorm + wp.float64(2.0) * epsv) / (epsv * epsv)                   <L 19>
    }
    if (var_0) {
        label0:;
        adj_3 += adj_ret;
        wp::adj_div(var_2, var_vbarnorm, var_3, adj_2, adj_vbarnorm, adj_3);
        wp::adj_float64(var_1, adj_1, adj_2);
        // adj: return wp.float64(1.0) / vbarnorm                                             <L 17>
    }
    // adj: if vbarnorm >= epsv:                                                              <L 16>
    // adj: def f1_div_vbarnorm(vbarnorm: wp.float64, epsv: wp.float64) -> wp.float64:        <L 15>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/friction.py:45
static CUDA_CALLABLE wp::vec_t<3,wp::float64> friction_gradient_0(
    wp::vec_t<3,wp::float64> var_n,
    wp::vec_t<3,wp::float64> var_x,
    wp::vec_t<3,wp::float64> var_hat_x,
    wp::float64 var_hat_h,
    wp::float64 var_epsv,
    wp::float64 var_coeff)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 3;
    wp::mat_t<3,3,wp::float64> var_1;
    wp::mat_t<3,3,wp::float64> var_2;
    wp::mat_t<3,3,wp::float64> var_3;
    wp::mat_t<3,3,wp::float64> var_4;
    wp::vec_t<3,wp::float64> var_5;
    wp::vec_t<3,wp::float64> var_6;
    wp::vec_t<3,wp::float64> var_7;
    wp::float64 var_8;
    wp::float64 var_9;
    wp::float64 var_10;
    wp::mat_t<3,3,wp::float64> var_11;
    wp::vec_t<3,wp::float64> var_12;
    //---------
    // forward
    // def friction_gradient(                                                                 <L 46>
    // tanOperator = wp.identity(n=3, dtype=wp.float64) - wp.outer(n, n)                      <L 54>
    var_1 = wp::identity<3,wp::float64>();
    var_2 = wp::outer(var_n, var_n);
    var_3 = wp::sub(var_1, var_2);
    // vbar = wp.transpose(tanOperator) @ (x - hat_x) / hat_h                                 <L 55>
    var_4 = wp::transpose(var_3);
    var_5 = wp::sub(var_x, var_hat_x);
    var_6 = wp::mul(var_4, var_5);
    var_7 = wp::div(var_6, var_hat_h);
    // vbarnorm = wp.length(vbar)                                                             <L 56>
    var_8 = wp::length(var_7);
    // f1 = f1_div_vbarnorm(vbarnorm, epsv)                                                   <L 57>
    var_9 = f1_div_vbarnorm_0(var_8, var_epsv);
    // return coeff * f1 * tanOperator @ vbar                                                 <L 58>
    var_10 = wp::mul(var_coeff, var_9);
    var_11 = wp::mul(var_10, var_3);
    var_12 = wp::mul(var_11, var_7);
    return var_12;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/friction.py:45
static CUDA_CALLABLE void adj_friction_gradient_0(
    wp::vec_t<3,wp::float64> var_n,
    wp::vec_t<3,wp::float64> var_x,
    wp::vec_t<3,wp::float64> var_hat_x,
    wp::float64 var_hat_h,
    wp::float64 var_epsv,
    wp::float64 var_coeff,
    wp::vec_t<3,wp::float64> & adj_n,
    wp::vec_t<3,wp::float64> & adj_x,
    wp::vec_t<3,wp::float64> & adj_hat_x,
    wp::float64 & adj_hat_h,
    wp::float64 & adj_epsv,
    wp::float64 & adj_coeff,
    wp::vec_t<3,wp::float64> & adj_ret)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 3;
    wp::mat_t<3,3,wp::float64> var_1;
    wp::mat_t<3,3,wp::float64> var_2;
    wp::mat_t<3,3,wp::float64> var_3;
    wp::mat_t<3,3,wp::float64> var_4;
    wp::vec_t<3,wp::float64> var_5;
    wp::vec_t<3,wp::float64> var_6;
    wp::vec_t<3,wp::float64> var_7;
    wp::float64 var_8;
    wp::float64 var_9;
    wp::float64 var_10;
    wp::mat_t<3,3,wp::float64> var_11;
    wp::vec_t<3,wp::float64> var_12;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::mat_t<3,3,wp::float64> adj_1 = {};
    wp::mat_t<3,3,wp::float64> adj_2 = {};
    wp::mat_t<3,3,wp::float64> adj_3 = {};
    wp::mat_t<3,3,wp::float64> adj_4 = {};
    wp::vec_t<3,wp::float64> adj_5 = {};
    wp::vec_t<3,wp::float64> adj_6 = {};
    wp::vec_t<3,wp::float64> adj_7 = {};
    wp::float64 adj_8 = {};
    wp::float64 adj_9 = {};
    wp::float64 adj_10 = {};
    wp::mat_t<3,3,wp::float64> adj_11 = {};
    wp::vec_t<3,wp::float64> adj_12 = {};
    //---------
    // forward
    // def friction_gradient(                                                                 <L 46>
    // tanOperator = wp.identity(n=3, dtype=wp.float64) - wp.outer(n, n)                      <L 54>
    var_1 = wp::identity<3,wp::float64>();
    var_2 = wp::outer(var_n, var_n);
    var_3 = wp::sub(var_1, var_2);
    // vbar = wp.transpose(tanOperator) @ (x - hat_x) / hat_h                                 <L 55>
    var_4 = wp::transpose(var_3);
    var_5 = wp::sub(var_x, var_hat_x);
    var_6 = wp::mul(var_4, var_5);
    var_7 = wp::div(var_6, var_hat_h);
    // vbarnorm = wp.length(vbar)                                                             <L 56>
    var_8 = wp::length(var_7);
    // f1 = f1_div_vbarnorm(vbarnorm, epsv)                                                   <L 57>
    var_9 = f1_div_vbarnorm_0(var_8, var_epsv);
    // return coeff * f1 * tanOperator @ vbar                                                 <L 58>
    var_10 = wp::mul(var_coeff, var_9);
    var_11 = wp::mul(var_10, var_3);
    var_12 = wp::mul(var_11, var_7);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_12 += adj_ret;
    wp::adj_mul(var_11, var_7, adj_11, adj_7, adj_12);
    wp::adj_mul(var_10, var_3, adj_10, adj_3, adj_11);
    wp::adj_mul(var_coeff, var_9, adj_coeff, adj_9, adj_10);
    // adj: return coeff * f1 * tanOperator @ vbar                                            <L 58>
    adj_f1_div_vbarnorm_0(var_8, var_epsv, adj_8, adj_epsv, adj_9);
    // adj: f1 = f1_div_vbarnorm(vbarnorm, epsv)                                              <L 57>
    wp::adj_length(var_7, var_8, adj_7, adj_8);
    // adj: vbarnorm = wp.length(vbar)                                                        <L 56>
    wp::adj_div(var_6, var_hat_h, adj_6, adj_hat_h, adj_7);
    wp::adj_mul(var_4, var_5, adj_4, adj_5, adj_6);
    wp::adj_sub(var_x, var_hat_x, adj_x, adj_hat_x, adj_5);
    wp::adj_transpose(var_3, adj_3, adj_4);
    // adj: vbar = wp.transpose(tanOperator) @ (x - hat_x) / hat_h                            <L 55>
    wp::adj_sub(var_1, var_2, adj_1, adj_2, adj_3);
    wp::adj_outer(var_n, var_n, adj_n, adj_n, adj_2);
    // adj: tanOperator = wp.identity(n=3, dtype=wp.float64) - wp.outer(n, n)                 <L 54>
    // adj: def friction_gradient(                                                            <L 46>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/distance_type.py:4
static CUDA_CALLABLE void solveLdlt2D_0(
    wp::float64 var_A00,
    wp::float64 var_A01,
    wp::float64 var_A11,
    wp::float64 var_b0,
    wp::float64 var_b1,
    wp::float64 & ret_0,
    wp::float64 & ret_1)
{
    //---------
    // primal vars
    wp::float64 var_0;
    wp::float64 var_1;
    wp::float64 var_2;
    wp::float64 var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    wp::float64 var_6;
    wp::float64 var_7;
    wp::float64 var_8;
    wp::float64 var_9;
    wp::float64 var_10;
    wp::float64 var_11;
    wp::float64 var_12;
    //---------
    // forward
    // def solveLdlt2D(A00: wp.float64, A01: wp.float64, A11: wp.float64, b0: wp.float64, b1: wp.float64):       <L 5>
    // D00 = A00                                                                              <L 6>
    var_0 = wp::copy(var_A00);
    // L01 = A01 / D00                                                                        <L 7>
    var_1 = wp::div(var_A01, var_0);
    // D11 = A11 - L01 * L01 * D00                                                            <L 8>
    var_2 = wp::mul(var_1, var_1);
    var_3 = wp::mul(var_2, var_0);
    var_4 = wp::sub(var_A11, var_3);
    // y0 = b0                                                                                <L 10>
    var_5 = wp::copy(var_b0);
    // y1 = b1 - L01 * y0                                                                     <L 11>
    var_6 = wp::mul(var_1, var_5);
    var_7 = wp::sub(var_b1, var_6);
    // x1 = y1 / D11                                                                          <L 13>
    var_8 = wp::div(var_7, var_4);
    // x0 = (y0 - D00 * L01 * x1) / D00                                                       <L 14>
    var_9 = wp::mul(var_0, var_1);
    var_10 = wp::mul(var_9, var_8);
    var_11 = wp::sub(var_5, var_10);
    var_12 = wp::div(var_11, var_0);
    // return x0, x1                                                                          <L 15>
    ret_0 = var_12;
    ret_1 = var_8;
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/distance_type.py:4
static CUDA_CALLABLE void adj_solveLdlt2D_0(
    wp::float64 var_A00,
    wp::float64 var_A01,
    wp::float64 var_A11,
    wp::float64 var_b0,
    wp::float64 var_b1,
    wp::float64 & ret_0,
    wp::float64 & ret_1,
    wp::float64 & adj_A00,
    wp::float64 & adj_A01,
    wp::float64 & adj_A11,
    wp::float64 & adj_b0,
    wp::float64 & adj_b1,
    wp::float64 & adj_ret_0,
    wp::float64 & adj_ret_1)
{
    //---------
    // primal vars
    wp::float64 var_0;
    wp::float64 var_1;
    wp::float64 var_2;
    wp::float64 var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    wp::float64 var_6;
    wp::float64 var_7;
    wp::float64 var_8;
    wp::float64 var_9;
    wp::float64 var_10;
    wp::float64 var_11;
    wp::float64 var_12;
    //---------
    // dual vars
    wp::float64 adj_0 = {};
    wp::float64 adj_1 = {};
    wp::float64 adj_2 = {};
    wp::float64 adj_3 = {};
    wp::float64 adj_4 = {};
    wp::float64 adj_5 = {};
    wp::float64 adj_6 = {};
    wp::float64 adj_7 = {};
    wp::float64 adj_8 = {};
    wp::float64 adj_9 = {};
    wp::float64 adj_10 = {};
    wp::float64 adj_11 = {};
    wp::float64 adj_12 = {};
    //---------
    // forward
    // def solveLdlt2D(A00: wp.float64, A01: wp.float64, A11: wp.float64, b0: wp.float64, b1: wp.float64):       <L 5>
    // D00 = A00                                                                              <L 6>
    var_0 = wp::copy(var_A00);
    // L01 = A01 / D00                                                                        <L 7>
    var_1 = wp::div(var_A01, var_0);
    // D11 = A11 - L01 * L01 * D00                                                            <L 8>
    var_2 = wp::mul(var_1, var_1);
    var_3 = wp::mul(var_2, var_0);
    var_4 = wp::sub(var_A11, var_3);
    // y0 = b0                                                                                <L 10>
    var_5 = wp::copy(var_b0);
    // y1 = b1 - L01 * y0                                                                     <L 11>
    var_6 = wp::mul(var_1, var_5);
    var_7 = wp::sub(var_b1, var_6);
    // x1 = y1 / D11                                                                          <L 13>
    var_8 = wp::div(var_7, var_4);
    // x0 = (y0 - D00 * L01 * x1) / D00                                                       <L 14>
    var_9 = wp::mul(var_0, var_1);
    var_10 = wp::mul(var_9, var_8);
    var_11 = wp::sub(var_5, var_10);
    var_12 = wp::div(var_11, var_0);
    // return x0, x1                                                                          <L 15>
    ret_0 = var_12;
    ret_1 = var_8;
    goto label0;
    //---------
    // reverse
    label0:;
    adj_8 += adj_ret_1;
    adj_12 += adj_ret_0;
    // adj: return x0, x1                                                                     <L 15>
    wp::adj_div(var_11, var_0, var_12, adj_11, adj_0, adj_12);
    wp::adj_sub(var_5, var_10, adj_5, adj_10, adj_11);
    wp::adj_mul(var_9, var_8, adj_9, adj_8, adj_10);
    wp::adj_mul(var_0, var_1, adj_0, adj_1, adj_9);
    // adj: x0 = (y0 - D00 * L01 * x1) / D00                                                  <L 14>
    wp::adj_div(var_7, var_4, var_8, adj_7, adj_4, adj_8);
    // adj: x1 = y1 / D11                                                                     <L 13>
    wp::adj_sub(var_b1, var_6, adj_b1, adj_6, adj_7);
    wp::adj_mul(var_1, var_5, adj_1, adj_5, adj_6);
    // adj: y1 = b1 - L01 * y0                                                                <L 11>
    wp::adj_copy(var_b0, adj_b0, adj_5);
    // adj: y0 = b0                                                                           <L 10>
    wp::adj_sub(var_A11, var_3, adj_A11, adj_3, adj_4);
    wp::adj_mul(var_2, var_0, adj_2, adj_0, adj_3);
    wp::adj_mul(var_1, var_1, adj_1, adj_1, adj_2);
    // adj: D11 = A11 - L01 * L01 * D00                                                       <L 8>
    wp::adj_div(var_A01, var_0, var_1, adj_A01, adj_0, adj_1);
    // adj: L01 = A01 / D00                                                                   <L 7>
    wp::adj_copy(var_A00, adj_A00, adj_0);
    // adj: D00 = A00                                                                         <L 6>
    // adj: def solveLdlt2D(A00: wp.float64, A01: wp.float64, A11: wp.float64, b0: wp.float64, b1: wp.float64):  <L 5>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/distance_type.py:18
static CUDA_CALLABLE wp::int32 point_triangle_distance_type_0(
    wp::vec_t<3,wp::float64> var_p,
    wp::vec_t<3,wp::float64> var_t0,
    wp::vec_t<3,wp::float64> var_t1,
    wp::vec_t<3,wp::float64> var_t2)
{
    //---------
    // primal vars
    wp::vec_t<3,wp::float64> var_0;
    wp::vec_t<3,wp::float64> var_1;
    wp::vec_t<3,wp::float64> var_2;
    wp::vec_t<3,wp::float64> var_3;
    wp::vec_t<4,wp::float64> var_4;
    wp::vec_t<2,wp::float64> var_5;
    const wp::int32 var_6 = 0;
    wp::float64 var_7;
    const wp::int32 var_8 = 0;
    wp::float64 var_9;
    wp::float64 var_10;
    const wp::int32 var_11 = 1;
    wp::float64 var_12;
    const wp::int32 var_13 = 1;
    wp::float64 var_14;
    wp::float64 var_15;
    wp::float64 var_16;
    const wp::int32 var_17 = 2;
    wp::float64 var_18;
    const wp::int32 var_19 = 2;
    wp::float64 var_20;
    wp::float64 var_21;
    wp::float64 var_22;
    const wp::int32 var_23 = 0;
    const wp::int32 var_24 = 0;
    wp::float64 var_25;
    const wp::int32 var_26 = 0;
    wp::float64 var_27;
    wp::float64 var_28;
    const wp::int32 var_29 = 1;
    wp::float64 var_30;
    const wp::int32 var_31 = 1;
    wp::float64 var_32;
    wp::float64 var_33;
    wp::float64 var_34;
    const wp::int32 var_35 = 2;
    wp::float64 var_36;
    const wp::int32 var_37 = 2;
    wp::float64 var_38;
    wp::float64 var_39;
    wp::float64 var_40;
    const wp::int32 var_41 = 2;
    const wp::int32 var_42 = 2;
    wp::float64 var_43;
    const wp::int32 var_44 = 1;
    const wp::int32 var_45 = 0;
    wp::float64 var_46;
    const wp::int32 var_47 = 0;
    wp::float64 var_48;
    wp::float64 var_49;
    const wp::int32 var_50 = 1;
    wp::float64 var_51;
    const wp::int32 var_52 = 1;
    wp::float64 var_53;
    wp::float64 var_54;
    wp::float64 var_55;
    const wp::int32 var_56 = 2;
    wp::float64 var_57;
    const wp::int32 var_58 = 2;
    wp::float64 var_59;
    wp::float64 var_60;
    wp::float64 var_61;
    const wp::int32 var_62 = 3;
    wp::vec_t<3,wp::float64> var_63;
    wp::float64 var_64;
    const wp::int32 var_65 = 0;
    wp::float64 var_66;
    const wp::int32 var_67 = 1;
    const wp::int32 var_68 = 0;
    wp::float64 var_69;
    const wp::int32 var_70 = 1;
    wp::float64 var_71;
    const wp::int32 var_72 = 3;
    wp::float64 var_73;
    const wp::int32 var_74 = 0;
    wp::float64 var_75;
    const wp::int32 var_76 = 1;
    wp::float64 var_77;
    wp::float64 var_78;
    wp::float64 var_79;
    const wp::float32 var_80 = 0.0;
    wp::float64 var_81;
    bool var_82;
    const wp::float32 var_83 = 1.0;
    wp::float64 var_84;
    bool var_85;
    const wp::float32 var_86 = 0.0;
    wp::float64 var_87;
    bool var_88;
    bool var_89;
    const wp::int32 var_90 = 3;
    wp::vec_t<3,wp::float64> var_91;
    wp::vec_t<3,wp::float64> var_92;
    const wp::int32 var_93 = 0;
    wp::float64 var_94;
    const wp::int32 var_95 = 0;
    wp::float64 var_96;
    wp::float64 var_97;
    const wp::int32 var_98 = 1;
    wp::float64 var_99;
    const wp::int32 var_100 = 1;
    wp::float64 var_101;
    wp::float64 var_102;
    wp::float64 var_103;
    const wp::int32 var_104 = 2;
    wp::float64 var_105;
    const wp::int32 var_106 = 2;
    wp::float64 var_107;
    wp::float64 var_108;
    wp::float64 var_109;
    const wp::int32 var_110 = 0;
    const wp::int32 var_111 = 0;
    wp::float64 var_112;
    const wp::int32 var_113 = 0;
    wp::float64 var_114;
    wp::float64 var_115;
    const wp::int32 var_116 = 1;
    wp::float64 var_117;
    const wp::int32 var_118 = 1;
    wp::float64 var_119;
    wp::float64 var_120;
    wp::float64 var_121;
    const wp::int32 var_122 = 2;
    wp::float64 var_123;
    const wp::int32 var_124 = 2;
    wp::float64 var_125;
    wp::float64 var_126;
    wp::float64 var_127;
    const wp::int32 var_128 = 2;
    const wp::int32 var_129 = 2;
    wp::float64 var_130;
    const wp::int32 var_131 = 1;
    const wp::int32 var_132 = 0;
    wp::float64 var_133;
    const wp::int32 var_134 = 0;
    wp::float64 var_135;
    wp::float64 var_136;
    const wp::int32 var_137 = 1;
    wp::float64 var_138;
    const wp::int32 var_139 = 1;
    wp::float64 var_140;
    wp::float64 var_141;
    wp::float64 var_142;
    const wp::int32 var_143 = 2;
    wp::float64 var_144;
    const wp::int32 var_145 = 2;
    wp::float64 var_146;
    wp::float64 var_147;
    wp::float64 var_148;
    const wp::int32 var_149 = 3;
    wp::vec_t<3,wp::float64> var_150;
    wp::float64 var_151;
    const wp::int32 var_152 = 0;
    wp::float64 var_153;
    const wp::int32 var_154 = 1;
    const wp::int32 var_155 = 0;
    wp::float64 var_156;
    const wp::int32 var_157 = 1;
    wp::float64 var_158;
    const wp::int32 var_159 = 3;
    wp::float64 var_160;
    const wp::int32 var_161 = 0;
    wp::float64 var_162;
    const wp::int32 var_163 = 1;
    wp::float64 var_164;
    wp::float64 var_165;
    wp::float64 var_166;
    const wp::float32 var_167 = 0.0;
    wp::float64 var_168;
    bool var_169;
    const wp::float32 var_170 = 1.0;
    wp::float64 var_171;
    bool var_172;
    const wp::float32 var_173 = 0.0;
    wp::float64 var_174;
    bool var_175;
    bool var_176;
    const wp::int32 var_177 = 4;
    wp::vec_t<3,wp::float64> var_178;
    wp::vec_t<3,wp::float64> var_179;
    const wp::int32 var_180 = 0;
    wp::float64 var_181;
    const wp::int32 var_182 = 0;
    wp::float64 var_183;
    wp::float64 var_184;
    const wp::int32 var_185 = 1;
    wp::float64 var_186;
    const wp::int32 var_187 = 1;
    wp::float64 var_188;
    wp::float64 var_189;
    wp::float64 var_190;
    const wp::int32 var_191 = 2;
    wp::float64 var_192;
    const wp::int32 var_193 = 2;
    wp::float64 var_194;
    wp::float64 var_195;
    wp::float64 var_196;
    const wp::int32 var_197 = 0;
    const wp::int32 var_198 = 0;
    wp::float64 var_199;
    const wp::int32 var_200 = 0;
    wp::float64 var_201;
    wp::float64 var_202;
    const wp::int32 var_203 = 1;
    wp::float64 var_204;
    const wp::int32 var_205 = 1;
    wp::float64 var_206;
    wp::float64 var_207;
    wp::float64 var_208;
    const wp::int32 var_209 = 2;
    wp::float64 var_210;
    const wp::int32 var_211 = 2;
    wp::float64 var_212;
    wp::float64 var_213;
    wp::float64 var_214;
    const wp::int32 var_215 = 2;
    const wp::int32 var_216 = 2;
    wp::float64 var_217;
    const wp::int32 var_218 = 1;
    const wp::int32 var_219 = 0;
    wp::float64 var_220;
    const wp::int32 var_221 = 0;
    wp::float64 var_222;
    wp::float64 var_223;
    const wp::int32 var_224 = 1;
    wp::float64 var_225;
    const wp::int32 var_226 = 1;
    wp::float64 var_227;
    wp::float64 var_228;
    wp::float64 var_229;
    const wp::int32 var_230 = 2;
    wp::float64 var_231;
    const wp::int32 var_232 = 2;
    wp::float64 var_233;
    wp::float64 var_234;
    wp::float64 var_235;
    const wp::int32 var_236 = 3;
    wp::vec_t<3,wp::float64> var_237;
    wp::float64 var_238;
    const wp::int32 var_239 = 0;
    wp::float64 var_240;
    const wp::int32 var_241 = 1;
    const wp::int32 var_242 = 0;
    wp::float64 var_243;
    const wp::int32 var_244 = 1;
    wp::float64 var_245;
    const wp::int32 var_246 = 3;
    wp::float64 var_247;
    const wp::int32 var_248 = 0;
    wp::float64 var_249;
    const wp::int32 var_250 = 1;
    wp::float64 var_251;
    wp::float64 var_252;
    wp::float64 var_253;
    const wp::float32 var_254 = 0.0;
    wp::float64 var_255;
    bool var_256;
    const wp::float32 var_257 = 1.0;
    wp::float64 var_258;
    bool var_259;
    const wp::float32 var_260 = 0.0;
    wp::float64 var_261;
    bool var_262;
    bool var_263;
    const wp::int32 var_264 = 5;
    const wp::float32 var_265 = 0.0;
    wp::float64 var_266;
    bool var_267;
    const wp::float32 var_268 = 1.0;
    wp::float64 var_269;
    bool var_270;
    bool var_271;
    const wp::int32 var_272 = 0;
    const wp::float32 var_273 = 0.0;
    wp::float64 var_274;
    bool var_275;
    const wp::float32 var_276 = 1.0;
    wp::float64 var_277;
    bool var_278;
    bool var_279;
    const wp::int32 var_280 = 1;
    const wp::float32 var_281 = 0.0;
    wp::float64 var_282;
    bool var_283;
    const wp::float32 var_284 = 1.0;
    wp::float64 var_285;
    bool var_286;
    bool var_287;
    const wp::int32 var_288 = 2;
    const wp::int32 var_289 = 6;
    wp::vec_t<3,wp::float64> var_290;
    wp::vec_t<3,wp::float64> var_291;
    wp::vec_t<3,wp::float64> var_292;
    wp::vec_t<3,wp::float64> var_293;
    wp::vec_t<3,wp::float64> var_294;
    wp::vec_t<3,wp::float64> var_295;
    //---------
    // forward
    // def point_triangle_distance_type(p: wp.vec3d, t0: wp.vec3d, t1: wp.vec3d, t2: wp.vec3d) -> int:       <L 19>
    // basis_row0 = t1 - t0                                                                   <L 20>
    var_0 = wp::sub(var_t1, var_t0);
    // basis_row1 = t2 - t0                                                                   <L 21>
    var_1 = wp::sub(var_t2, var_t0);
    // nVec = wp.cross(basis_row0, basis_row1)                                                <L 22>
    var_2 = wp::cross(var_0, var_1);
    // basis_row1 = wp.cross(basis_row0, nVec)                                                <L 23>
    var_3 = wp::cross(var_0, var_2);
    // sys = wp.vec4d()                                                                       <L 25>
    var_4 = wp::vec_t<4,wp::float64>();
    // rhs = wp.vec2d()                                                                       <L 26>
    var_5 = wp::vec_t<2,wp::float64>();
    // sys[0] = basis_row0[0] * basis_row0[0] + basis_row0[1] * basis_row0[1] + basis_row0[2] * basis_row0[2]       <L 27>
    var_7 = wp::extract(var_0, var_6);
    var_9 = wp::extract(var_0, var_8);
    var_10 = wp::mul(var_7, var_9);
    var_12 = wp::extract(var_0, var_11);
    var_14 = wp::extract(var_0, var_13);
    var_15 = wp::mul(var_12, var_14);
    var_16 = wp::add(var_10, var_15);
    var_18 = wp::extract(var_0, var_17);
    var_20 = wp::extract(var_0, var_19);
    var_21 = wp::mul(var_18, var_20);
    var_22 = wp::add(var_16, var_21);
    wp::assign_inplace(var_4, var_23, var_22);
    // sys[2] = basis_row0[0] * basis_row1[0] + basis_row0[1] * basis_row1[1] + basis_row0[2] * basis_row1[2]       <L 28>
    var_25 = wp::extract(var_0, var_24);
    var_27 = wp::extract(var_3, var_26);
    var_28 = wp::mul(var_25, var_27);
    var_30 = wp::extract(var_0, var_29);
    var_32 = wp::extract(var_3, var_31);
    var_33 = wp::mul(var_30, var_32);
    var_34 = wp::add(var_28, var_33);
    var_36 = wp::extract(var_0, var_35);
    var_38 = wp::extract(var_3, var_37);
    var_39 = wp::mul(var_36, var_38);
    var_40 = wp::add(var_34, var_39);
    wp::assign_inplace(var_4, var_41, var_40);
    // sys[1] = sys[2]                                                                        <L 29>
    var_43 = wp::extract(var_4, var_42);
    wp::assign_inplace(var_4, var_44, var_43);
    // sys[3] = basis_row1[0] * basis_row1[0] + basis_row1[1] * basis_row1[1] + basis_row1[2] * basis_row1[2]       <L 30>
    var_46 = wp::extract(var_3, var_45);
    var_48 = wp::extract(var_3, var_47);
    var_49 = wp::mul(var_46, var_48);
    var_51 = wp::extract(var_3, var_50);
    var_53 = wp::extract(var_3, var_52);
    var_54 = wp::mul(var_51, var_53);
    var_55 = wp::add(var_49, var_54);
    var_57 = wp::extract(var_3, var_56);
    var_59 = wp::extract(var_3, var_58);
    var_60 = wp::mul(var_57, var_59);
    var_61 = wp::add(var_55, var_60);
    wp::assign_inplace(var_4, var_62, var_61);
    // b = p - t0                                                                             <L 32>
    var_63 = wp::sub(var_p, var_t0);
    // rhs[0] = wp.dot(basis_row0, b)                                                         <L 33>
    var_64 = wp::dot(var_0, var_63);
    wp::assign_inplace(var_5, var_65, var_64);
    // rhs[1] = wp.dot(basis_row1, b)                                                         <L 34>
    var_66 = wp::dot(var_3, var_63);
    wp::assign_inplace(var_5, var_67, var_66);
    // param_col00, param_col01 = solveLdlt2D(sys[0], sys[1], sys[3], rhs[0], rhs[1])         <L 35>
    var_69 = wp::extract(var_4, var_68);
    var_71 = wp::extract(var_4, var_70);
    var_73 = wp::extract(var_4, var_72);
    var_75 = wp::extract(var_5, var_74);
    var_77 = wp::extract(var_5, var_76);
    solveLdlt2D_0(var_69, var_71, var_73, var_75, var_77, var_78, var_79);
    // if param_col00 > wp.float64(0.0) and param_col00 < wp.float64(1.0) and param_col01 >= wp.float64(0.0):       <L 36>
    var_81 = wp::float64(var_80);
    var_82 = (var_78 > var_81);
    var_84 = wp::float64(var_83);
    var_85 = (var_78 < var_84);
    var_87 = wp::float64(var_86);
    var_88 = (var_79 >= var_87);
    var_89 = var_82 && var_85 && var_88;
    if (var_89) {
        // return 3  # PE t0t1                                                                <L 37>
        return var_90;
    }
    if (!var_89) {
        // basis_row0 = t2 - t1                                                               <L 39>
        var_91 = wp::sub(var_t2, var_t1);
        // basis_row1 = wp.cross(basis_row0, nVec)                                            <L 40>
        var_92 = wp::cross(var_91, var_2);
        // sys[0] = basis_row0[0] * basis_row0[0] + basis_row0[1] * basis_row0[1] + basis_row0[2] * basis_row0[2]       <L 42>
        var_94 = wp::extract(var_91, var_93);
        var_96 = wp::extract(var_91, var_95);
        var_97 = wp::mul(var_94, var_96);
        var_99 = wp::extract(var_91, var_98);
        var_101 = wp::extract(var_91, var_100);
        var_102 = wp::mul(var_99, var_101);
        var_103 = wp::add(var_97, var_102);
        var_105 = wp::extract(var_91, var_104);
        var_107 = wp::extract(var_91, var_106);
        var_108 = wp::mul(var_105, var_107);
        var_109 = wp::add(var_103, var_108);
        wp::assign_inplace(var_4, var_110, var_109);
        // sys[2] = basis_row0[0] * basis_row1[0] + basis_row0[1] * basis_row1[1] + basis_row0[2] * basis_row1[2]       <L 43>
        var_112 = wp::extract(var_91, var_111);
        var_114 = wp::extract(var_92, var_113);
        var_115 = wp::mul(var_112, var_114);
        var_117 = wp::extract(var_91, var_116);
        var_119 = wp::extract(var_92, var_118);
        var_120 = wp::mul(var_117, var_119);
        var_121 = wp::add(var_115, var_120);
        var_123 = wp::extract(var_91, var_122);
        var_125 = wp::extract(var_92, var_124);
        var_126 = wp::mul(var_123, var_125);
        var_127 = wp::add(var_121, var_126);
        wp::assign_inplace(var_4, var_128, var_127);
        // sys[1] = sys[2]                                                                    <L 44>
        var_130 = wp::extract(var_4, var_129);
        wp::assign_inplace(var_4, var_131, var_130);
        // sys[3] = basis_row1[0] * basis_row1[0] + basis_row1[1] * basis_row1[1] + basis_row1[2] * basis_row1[2]       <L 45>
        var_133 = wp::extract(var_92, var_132);
        var_135 = wp::extract(var_92, var_134);
        var_136 = wp::mul(var_133, var_135);
        var_138 = wp::extract(var_92, var_137);
        var_140 = wp::extract(var_92, var_139);
        var_141 = wp::mul(var_138, var_140);
        var_142 = wp::add(var_136, var_141);
        var_144 = wp::extract(var_92, var_143);
        var_146 = wp::extract(var_92, var_145);
        var_147 = wp::mul(var_144, var_146);
        var_148 = wp::add(var_142, var_147);
        wp::assign_inplace(var_4, var_149, var_148);
        // b = p - t1                                                                         <L 46>
        var_150 = wp::sub(var_p, var_t1);
        // rhs[0] = wp.dot(basis_row0, b)                                                     <L 47>
        var_151 = wp::dot(var_91, var_150);
        wp::assign_inplace(var_5, var_152, var_151);
        // rhs[1] = wp.dot(basis_row1, b)                                                     <L 48>
        var_153 = wp::dot(var_92, var_150);
        wp::assign_inplace(var_5, var_154, var_153);
        // param_col10, param_col11 = solveLdlt2D(sys[0], sys[1], sys[3], rhs[0], rhs[1])       <L 49>
        var_156 = wp::extract(var_4, var_155);
        var_158 = wp::extract(var_4, var_157);
        var_160 = wp::extract(var_4, var_159);
        var_162 = wp::extract(var_5, var_161);
        var_164 = wp::extract(var_5, var_163);
        solveLdlt2D_0(var_156, var_158, var_160, var_162, var_164, var_165, var_166);
        // if param_col10 > wp.float64(0.0) and param_col10 < wp.float64(1.0) and param_col11 >= wp.float64(0.0):       <L 50>
        var_168 = wp::float64(var_167);
        var_169 = (var_165 > var_168);
        var_171 = wp::float64(var_170);
        var_172 = (var_165 < var_171);
        var_174 = wp::float64(var_173);
        var_175 = (var_166 >= var_174);
        var_176 = var_169 && var_172 && var_175;
        if (var_176) {
            // return 4  # PE t1t2                                                            <L 51>
            return var_177;
        }
        if (!var_176) {
            // basis_row0 = t0 - t2                                                           <L 53>
            var_178 = wp::sub(var_t0, var_t2);
            // basis_row1 = wp.cross(basis_row0, nVec)                                        <L 54>
            var_179 = wp::cross(var_178, var_2);
            // sys[0] = basis_row0[0] * basis_row0[0] + basis_row0[1] * basis_row0[1] + basis_row0[2] * basis_row0[2]       <L 55>
            var_181 = wp::extract(var_178, var_180);
            var_183 = wp::extract(var_178, var_182);
            var_184 = wp::mul(var_181, var_183);
            var_186 = wp::extract(var_178, var_185);
            var_188 = wp::extract(var_178, var_187);
            var_189 = wp::mul(var_186, var_188);
            var_190 = wp::add(var_184, var_189);
            var_192 = wp::extract(var_178, var_191);
            var_194 = wp::extract(var_178, var_193);
            var_195 = wp::mul(var_192, var_194);
            var_196 = wp::add(var_190, var_195);
            wp::assign_inplace(var_4, var_197, var_196);
            // sys[2] = basis_row0[0] * basis_row1[0] + basis_row0[1] * basis_row1[1] + basis_row0[2] * basis_row1[2]       <L 56>
            var_199 = wp::extract(var_178, var_198);
            var_201 = wp::extract(var_179, var_200);
            var_202 = wp::mul(var_199, var_201);
            var_204 = wp::extract(var_178, var_203);
            var_206 = wp::extract(var_179, var_205);
            var_207 = wp::mul(var_204, var_206);
            var_208 = wp::add(var_202, var_207);
            var_210 = wp::extract(var_178, var_209);
            var_212 = wp::extract(var_179, var_211);
            var_213 = wp::mul(var_210, var_212);
            var_214 = wp::add(var_208, var_213);
            wp::assign_inplace(var_4, var_215, var_214);
            // sys[1] = sys[2]                                                                <L 57>
            var_217 = wp::extract(var_4, var_216);
            wp::assign_inplace(var_4, var_218, var_217);
            // sys[3] = basis_row1[0] * basis_row1[0] + basis_row1[1] * basis_row1[1] + basis_row1[2] * basis_row1[2]       <L 58>
            var_220 = wp::extract(var_179, var_219);
            var_222 = wp::extract(var_179, var_221);
            var_223 = wp::mul(var_220, var_222);
            var_225 = wp::extract(var_179, var_224);
            var_227 = wp::extract(var_179, var_226);
            var_228 = wp::mul(var_225, var_227);
            var_229 = wp::add(var_223, var_228);
            var_231 = wp::extract(var_179, var_230);
            var_233 = wp::extract(var_179, var_232);
            var_234 = wp::mul(var_231, var_233);
            var_235 = wp::add(var_229, var_234);
            wp::assign_inplace(var_4, var_236, var_235);
            // b = p - t2                                                                     <L 59>
            var_237 = wp::sub(var_p, var_t2);
            // rhs[0] = wp.dot(basis_row0, b)                                                 <L 60>
            var_238 = wp::dot(var_178, var_237);
            wp::assign_inplace(var_5, var_239, var_238);
            // rhs[1] = wp.dot(basis_row1, b)                                                 <L 61>
            var_240 = wp::dot(var_179, var_237);
            wp::assign_inplace(var_5, var_241, var_240);
            // param_col20, param_col21 = solveLdlt2D(sys[0], sys[1], sys[3], rhs[0], rhs[1])       <L 62>
            var_243 = wp::extract(var_4, var_242);
            var_245 = wp::extract(var_4, var_244);
            var_247 = wp::extract(var_4, var_246);
            var_249 = wp::extract(var_5, var_248);
            var_251 = wp::extract(var_5, var_250);
            solveLdlt2D_0(var_243, var_245, var_247, var_249, var_251, var_252, var_253);
            // if param_col20 > wp.float64(0.0) and param_col20 < wp.float64(1.0) and param_col21 >= wp.float64(0.0):       <L 64>
            var_255 = wp::float64(var_254);
            var_256 = (var_252 > var_255);
            var_258 = wp::float64(var_257);
            var_259 = (var_252 < var_258);
            var_261 = wp::float64(var_260);
            var_262 = (var_253 >= var_261);
            var_263 = var_256 && var_259 && var_262;
            if (var_263) {
                // return 5  # PE t2t0                                                        <L 65>
                return var_264;
            }
            if (!var_263) {
                // if param_col00 <= wp.float64(0.0) and param_col20 >= wp.float64(1.0):       <L 67>
                var_266 = wp::float64(var_265);
                var_267 = (var_78 <= var_266);
                var_269 = wp::float64(var_268);
                var_270 = (var_252 >= var_269);
                var_271 = var_267 && var_270;
                if (var_271) {
                    // return 0  # PP t0                                                      <L 68>
                    return var_272;
                }
                if (!var_271) {
                    // elif param_col10 <= wp.float64(0.0) and param_col00 >= wp.float64(1.0):       <L 69>
                    var_274 = wp::float64(var_273);
                    var_275 = (var_165 <= var_274);
                    var_277 = wp::float64(var_276);
                    var_278 = (var_78 >= var_277);
                    var_279 = var_275 && var_278;
                    if (var_279) {
                        // return 1  # PP t1                                                  <L 70>
                        return var_280;
                    }
                    if (!var_279) {
                        // elif param_col20 <= wp.float64(0.0) and param_col10 >= wp.float64(1.0):       <L 71>
                        var_282 = wp::float64(var_281);
                        var_283 = (var_252 <= var_282);
                        var_285 = wp::float64(var_284);
                        var_286 = (var_165 >= var_285);
                        var_287 = var_283 && var_286;
                        if (var_287) {
                            // return 2  # PP t2                                              <L 72>
                            return var_288;
                        }
                        if (!var_287) {
                            // return 6  # PT                                                 <L 74>
                            return var_289;
                        }
                    }
                }
            }
        }
        var_290 = wp::where(var_176, var_91, var_178);
        var_291 = wp::where(var_176, var_92, var_179);
        var_292 = wp::where(var_176, var_150, var_237);
    }
    var_293 = wp::where(var_89, var_0, var_290);
    var_294 = wp::where(var_89, var_3, var_291);
    var_295 = wp::where(var_89, var_63, var_292);
    return {};
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/distance_type.py:18
static CUDA_CALLABLE void adj_point_triangle_distance_type_0(
    wp::vec_t<3,wp::float64> var_p,
    wp::vec_t<3,wp::float64> var_t0,
    wp::vec_t<3,wp::float64> var_t1,
    wp::vec_t<3,wp::float64> var_t2,
    wp::vec_t<3,wp::float64> & adj_p,
    wp::vec_t<3,wp::float64> & adj_t0,
    wp::vec_t<3,wp::float64> & adj_t1,
    wp::vec_t<3,wp::float64> & adj_t2,
    wp::int32 & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3,wp::float64> var_0;
    wp::vec_t<3,wp::float64> var_1;
    wp::vec_t<3,wp::float64> var_2;
    wp::vec_t<3,wp::float64> var_3;
    wp::vec_t<4,wp::float64> var_4;
    wp::vec_t<2,wp::float64> var_5;
    const wp::int32 var_6 = 0;
    wp::float64 var_7;
    const wp::int32 var_8 = 0;
    wp::float64 var_9;
    wp::float64 var_10;
    const wp::int32 var_11 = 1;
    wp::float64 var_12;
    const wp::int32 var_13 = 1;
    wp::float64 var_14;
    wp::float64 var_15;
    wp::float64 var_16;
    const wp::int32 var_17 = 2;
    wp::float64 var_18;
    const wp::int32 var_19 = 2;
    wp::float64 var_20;
    wp::float64 var_21;
    wp::float64 var_22;
    const wp::int32 var_23 = 0;
    const wp::int32 var_24 = 0;
    wp::float64 var_25;
    const wp::int32 var_26 = 0;
    wp::float64 var_27;
    wp::float64 var_28;
    const wp::int32 var_29 = 1;
    wp::float64 var_30;
    const wp::int32 var_31 = 1;
    wp::float64 var_32;
    wp::float64 var_33;
    wp::float64 var_34;
    const wp::int32 var_35 = 2;
    wp::float64 var_36;
    const wp::int32 var_37 = 2;
    wp::float64 var_38;
    wp::float64 var_39;
    wp::float64 var_40;
    const wp::int32 var_41 = 2;
    const wp::int32 var_42 = 2;
    wp::float64 var_43;
    const wp::int32 var_44 = 1;
    const wp::int32 var_45 = 0;
    wp::float64 var_46;
    const wp::int32 var_47 = 0;
    wp::float64 var_48;
    wp::float64 var_49;
    const wp::int32 var_50 = 1;
    wp::float64 var_51;
    const wp::int32 var_52 = 1;
    wp::float64 var_53;
    wp::float64 var_54;
    wp::float64 var_55;
    const wp::int32 var_56 = 2;
    wp::float64 var_57;
    const wp::int32 var_58 = 2;
    wp::float64 var_59;
    wp::float64 var_60;
    wp::float64 var_61;
    const wp::int32 var_62 = 3;
    wp::vec_t<3,wp::float64> var_63;
    wp::float64 var_64;
    const wp::int32 var_65 = 0;
    wp::float64 var_66;
    const wp::int32 var_67 = 1;
    const wp::int32 var_68 = 0;
    wp::float64 var_69;
    const wp::int32 var_70 = 1;
    wp::float64 var_71;
    const wp::int32 var_72 = 3;
    wp::float64 var_73;
    const wp::int32 var_74 = 0;
    wp::float64 var_75;
    const wp::int32 var_76 = 1;
    wp::float64 var_77;
    wp::float64 var_78;
    wp::float64 var_79;
    const wp::float32 var_80 = 0.0;
    wp::float64 var_81;
    bool var_82;
    const wp::float32 var_83 = 1.0;
    wp::float64 var_84;
    bool var_85;
    const wp::float32 var_86 = 0.0;
    wp::float64 var_87;
    bool var_88;
    bool var_89;
    const wp::int32 var_90 = 3;
    wp::vec_t<3,wp::float64> var_91;
    wp::vec_t<3,wp::float64> var_92;
    const wp::int32 var_93 = 0;
    wp::float64 var_94;
    const wp::int32 var_95 = 0;
    wp::float64 var_96;
    wp::float64 var_97;
    const wp::int32 var_98 = 1;
    wp::float64 var_99;
    const wp::int32 var_100 = 1;
    wp::float64 var_101;
    wp::float64 var_102;
    wp::float64 var_103;
    const wp::int32 var_104 = 2;
    wp::float64 var_105;
    const wp::int32 var_106 = 2;
    wp::float64 var_107;
    wp::float64 var_108;
    wp::float64 var_109;
    const wp::int32 var_110 = 0;
    const wp::int32 var_111 = 0;
    wp::float64 var_112;
    const wp::int32 var_113 = 0;
    wp::float64 var_114;
    wp::float64 var_115;
    const wp::int32 var_116 = 1;
    wp::float64 var_117;
    const wp::int32 var_118 = 1;
    wp::float64 var_119;
    wp::float64 var_120;
    wp::float64 var_121;
    const wp::int32 var_122 = 2;
    wp::float64 var_123;
    const wp::int32 var_124 = 2;
    wp::float64 var_125;
    wp::float64 var_126;
    wp::float64 var_127;
    const wp::int32 var_128 = 2;
    const wp::int32 var_129 = 2;
    wp::float64 var_130;
    const wp::int32 var_131 = 1;
    const wp::int32 var_132 = 0;
    wp::float64 var_133;
    const wp::int32 var_134 = 0;
    wp::float64 var_135;
    wp::float64 var_136;
    const wp::int32 var_137 = 1;
    wp::float64 var_138;
    const wp::int32 var_139 = 1;
    wp::float64 var_140;
    wp::float64 var_141;
    wp::float64 var_142;
    const wp::int32 var_143 = 2;
    wp::float64 var_144;
    const wp::int32 var_145 = 2;
    wp::float64 var_146;
    wp::float64 var_147;
    wp::float64 var_148;
    const wp::int32 var_149 = 3;
    wp::vec_t<3,wp::float64> var_150;
    wp::float64 var_151;
    const wp::int32 var_152 = 0;
    wp::float64 var_153;
    const wp::int32 var_154 = 1;
    const wp::int32 var_155 = 0;
    wp::float64 var_156;
    const wp::int32 var_157 = 1;
    wp::float64 var_158;
    const wp::int32 var_159 = 3;
    wp::float64 var_160;
    const wp::int32 var_161 = 0;
    wp::float64 var_162;
    const wp::int32 var_163 = 1;
    wp::float64 var_164;
    wp::float64 var_165;
    wp::float64 var_166;
    const wp::float32 var_167 = 0.0;
    wp::float64 var_168;
    bool var_169;
    const wp::float32 var_170 = 1.0;
    wp::float64 var_171;
    bool var_172;
    const wp::float32 var_173 = 0.0;
    wp::float64 var_174;
    bool var_175;
    bool var_176;
    const wp::int32 var_177 = 4;
    wp::vec_t<3,wp::float64> var_178;
    wp::vec_t<3,wp::float64> var_179;
    const wp::int32 var_180 = 0;
    wp::float64 var_181;
    const wp::int32 var_182 = 0;
    wp::float64 var_183;
    wp::float64 var_184;
    const wp::int32 var_185 = 1;
    wp::float64 var_186;
    const wp::int32 var_187 = 1;
    wp::float64 var_188;
    wp::float64 var_189;
    wp::float64 var_190;
    const wp::int32 var_191 = 2;
    wp::float64 var_192;
    const wp::int32 var_193 = 2;
    wp::float64 var_194;
    wp::float64 var_195;
    wp::float64 var_196;
    const wp::int32 var_197 = 0;
    const wp::int32 var_198 = 0;
    wp::float64 var_199;
    const wp::int32 var_200 = 0;
    wp::float64 var_201;
    wp::float64 var_202;
    const wp::int32 var_203 = 1;
    wp::float64 var_204;
    const wp::int32 var_205 = 1;
    wp::float64 var_206;
    wp::float64 var_207;
    wp::float64 var_208;
    const wp::int32 var_209 = 2;
    wp::float64 var_210;
    const wp::int32 var_211 = 2;
    wp::float64 var_212;
    wp::float64 var_213;
    wp::float64 var_214;
    const wp::int32 var_215 = 2;
    const wp::int32 var_216 = 2;
    wp::float64 var_217;
    const wp::int32 var_218 = 1;
    const wp::int32 var_219 = 0;
    wp::float64 var_220;
    const wp::int32 var_221 = 0;
    wp::float64 var_222;
    wp::float64 var_223;
    const wp::int32 var_224 = 1;
    wp::float64 var_225;
    const wp::int32 var_226 = 1;
    wp::float64 var_227;
    wp::float64 var_228;
    wp::float64 var_229;
    const wp::int32 var_230 = 2;
    wp::float64 var_231;
    const wp::int32 var_232 = 2;
    wp::float64 var_233;
    wp::float64 var_234;
    wp::float64 var_235;
    const wp::int32 var_236 = 3;
    wp::vec_t<3,wp::float64> var_237;
    wp::float64 var_238;
    const wp::int32 var_239 = 0;
    wp::float64 var_240;
    const wp::int32 var_241 = 1;
    const wp::int32 var_242 = 0;
    wp::float64 var_243;
    const wp::int32 var_244 = 1;
    wp::float64 var_245;
    const wp::int32 var_246 = 3;
    wp::float64 var_247;
    const wp::int32 var_248 = 0;
    wp::float64 var_249;
    const wp::int32 var_250 = 1;
    wp::float64 var_251;
    wp::float64 var_252;
    wp::float64 var_253;
    const wp::float32 var_254 = 0.0;
    wp::float64 var_255;
    bool var_256;
    const wp::float32 var_257 = 1.0;
    wp::float64 var_258;
    bool var_259;
    const wp::float32 var_260 = 0.0;
    wp::float64 var_261;
    bool var_262;
    bool var_263;
    const wp::int32 var_264 = 5;
    const wp::float32 var_265 = 0.0;
    wp::float64 var_266;
    bool var_267;
    const wp::float32 var_268 = 1.0;
    wp::float64 var_269;
    bool var_270;
    bool var_271;
    const wp::int32 var_272 = 0;
    const wp::float32 var_273 = 0.0;
    wp::float64 var_274;
    bool var_275;
    const wp::float32 var_276 = 1.0;
    wp::float64 var_277;
    bool var_278;
    bool var_279;
    const wp::int32 var_280 = 1;
    const wp::float32 var_281 = 0.0;
    wp::float64 var_282;
    bool var_283;
    const wp::float32 var_284 = 1.0;
    wp::float64 var_285;
    bool var_286;
    bool var_287;
    const wp::int32 var_288 = 2;
    const wp::int32 var_289 = 6;
    wp::vec_t<3,wp::float64> var_290;
    wp::vec_t<3,wp::float64> var_291;
    wp::vec_t<3,wp::float64> var_292;
    wp::vec_t<3,wp::float64> var_293;
    wp::vec_t<3,wp::float64> var_294;
    wp::vec_t<3,wp::float64> var_295;
    //---------
    // dual vars
    wp::vec_t<3,wp::float64> adj_0 = {};
    wp::vec_t<3,wp::float64> adj_1 = {};
    wp::vec_t<3,wp::float64> adj_2 = {};
    wp::vec_t<3,wp::float64> adj_3 = {};
    wp::vec_t<4,wp::float64> adj_4 = {};
    wp::vec_t<2,wp::float64> adj_5 = {};
    wp::int32 adj_6 = {};
    wp::float64 adj_7 = {};
    wp::int32 adj_8 = {};
    wp::float64 adj_9 = {};
    wp::float64 adj_10 = {};
    wp::int32 adj_11 = {};
    wp::float64 adj_12 = {};
    wp::int32 adj_13 = {};
    wp::float64 adj_14 = {};
    wp::float64 adj_15 = {};
    wp::float64 adj_16 = {};
    wp::int32 adj_17 = {};
    wp::float64 adj_18 = {};
    wp::int32 adj_19 = {};
    wp::float64 adj_20 = {};
    wp::float64 adj_21 = {};
    wp::float64 adj_22 = {};
    wp::int32 adj_23 = {};
    wp::int32 adj_24 = {};
    wp::float64 adj_25 = {};
    wp::int32 adj_26 = {};
    wp::float64 adj_27 = {};
    wp::float64 adj_28 = {};
    wp::int32 adj_29 = {};
    wp::float64 adj_30 = {};
    wp::int32 adj_31 = {};
    wp::float64 adj_32 = {};
    wp::float64 adj_33 = {};
    wp::float64 adj_34 = {};
    wp::int32 adj_35 = {};
    wp::float64 adj_36 = {};
    wp::int32 adj_37 = {};
    wp::float64 adj_38 = {};
    wp::float64 adj_39 = {};
    wp::float64 adj_40 = {};
    wp::int32 adj_41 = {};
    wp::int32 adj_42 = {};
    wp::float64 adj_43 = {};
    wp::int32 adj_44 = {};
    wp::int32 adj_45 = {};
    wp::float64 adj_46 = {};
    wp::int32 adj_47 = {};
    wp::float64 adj_48 = {};
    wp::float64 adj_49 = {};
    wp::int32 adj_50 = {};
    wp::float64 adj_51 = {};
    wp::int32 adj_52 = {};
    wp::float64 adj_53 = {};
    wp::float64 adj_54 = {};
    wp::float64 adj_55 = {};
    wp::int32 adj_56 = {};
    wp::float64 adj_57 = {};
    wp::int32 adj_58 = {};
    wp::float64 adj_59 = {};
    wp::float64 adj_60 = {};
    wp::float64 adj_61 = {};
    wp::int32 adj_62 = {};
    wp::vec_t<3,wp::float64> adj_63 = {};
    wp::float64 adj_64 = {};
    wp::int32 adj_65 = {};
    wp::float64 adj_66 = {};
    wp::int32 adj_67 = {};
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
    wp::float64 adj_78 = {};
    wp::float64 adj_79 = {};
    wp::float32 adj_80 = {};
    wp::float64 adj_81 = {};
    bool adj_82 = {};
    wp::float32 adj_83 = {};
    wp::float64 adj_84 = {};
    bool adj_85 = {};
    wp::float32 adj_86 = {};
    wp::float64 adj_87 = {};
    bool adj_88 = {};
    bool adj_89 = {};
    wp::int32 adj_90 = {};
    wp::vec_t<3,wp::float64> adj_91 = {};
    wp::vec_t<3,wp::float64> adj_92 = {};
    wp::int32 adj_93 = {};
    wp::float64 adj_94 = {};
    wp::int32 adj_95 = {};
    wp::float64 adj_96 = {};
    wp::float64 adj_97 = {};
    wp::int32 adj_98 = {};
    wp::float64 adj_99 = {};
    wp::int32 adj_100 = {};
    wp::float64 adj_101 = {};
    wp::float64 adj_102 = {};
    wp::float64 adj_103 = {};
    wp::int32 adj_104 = {};
    wp::float64 adj_105 = {};
    wp::int32 adj_106 = {};
    wp::float64 adj_107 = {};
    wp::float64 adj_108 = {};
    wp::float64 adj_109 = {};
    wp::int32 adj_110 = {};
    wp::int32 adj_111 = {};
    wp::float64 adj_112 = {};
    wp::int32 adj_113 = {};
    wp::float64 adj_114 = {};
    wp::float64 adj_115 = {};
    wp::int32 adj_116 = {};
    wp::float64 adj_117 = {};
    wp::int32 adj_118 = {};
    wp::float64 adj_119 = {};
    wp::float64 adj_120 = {};
    wp::float64 adj_121 = {};
    wp::int32 adj_122 = {};
    wp::float64 adj_123 = {};
    wp::int32 adj_124 = {};
    wp::float64 adj_125 = {};
    wp::float64 adj_126 = {};
    wp::float64 adj_127 = {};
    wp::int32 adj_128 = {};
    wp::int32 adj_129 = {};
    wp::float64 adj_130 = {};
    wp::int32 adj_131 = {};
    wp::int32 adj_132 = {};
    wp::float64 adj_133 = {};
    wp::int32 adj_134 = {};
    wp::float64 adj_135 = {};
    wp::float64 adj_136 = {};
    wp::int32 adj_137 = {};
    wp::float64 adj_138 = {};
    wp::int32 adj_139 = {};
    wp::float64 adj_140 = {};
    wp::float64 adj_141 = {};
    wp::float64 adj_142 = {};
    wp::int32 adj_143 = {};
    wp::float64 adj_144 = {};
    wp::int32 adj_145 = {};
    wp::float64 adj_146 = {};
    wp::float64 adj_147 = {};
    wp::float64 adj_148 = {};
    wp::int32 adj_149 = {};
    wp::vec_t<3,wp::float64> adj_150 = {};
    wp::float64 adj_151 = {};
    wp::int32 adj_152 = {};
    wp::float64 adj_153 = {};
    wp::int32 adj_154 = {};
    wp::int32 adj_155 = {};
    wp::float64 adj_156 = {};
    wp::int32 adj_157 = {};
    wp::float64 adj_158 = {};
    wp::int32 adj_159 = {};
    wp::float64 adj_160 = {};
    wp::int32 adj_161 = {};
    wp::float64 adj_162 = {};
    wp::int32 adj_163 = {};
    wp::float64 adj_164 = {};
    wp::float64 adj_165 = {};
    wp::float64 adj_166 = {};
    wp::float32 adj_167 = {};
    wp::float64 adj_168 = {};
    bool adj_169 = {};
    wp::float32 adj_170 = {};
    wp::float64 adj_171 = {};
    bool adj_172 = {};
    wp::float32 adj_173 = {};
    wp::float64 adj_174 = {};
    bool adj_175 = {};
    bool adj_176 = {};
    wp::int32 adj_177 = {};
    wp::vec_t<3,wp::float64> adj_178 = {};
    wp::vec_t<3,wp::float64> adj_179 = {};
    wp::int32 adj_180 = {};
    wp::float64 adj_181 = {};
    wp::int32 adj_182 = {};
    wp::float64 adj_183 = {};
    wp::float64 adj_184 = {};
    wp::int32 adj_185 = {};
    wp::float64 adj_186 = {};
    wp::int32 adj_187 = {};
    wp::float64 adj_188 = {};
    wp::float64 adj_189 = {};
    wp::float64 adj_190 = {};
    wp::int32 adj_191 = {};
    wp::float64 adj_192 = {};
    wp::int32 adj_193 = {};
    wp::float64 adj_194 = {};
    wp::float64 adj_195 = {};
    wp::float64 adj_196 = {};
    wp::int32 adj_197 = {};
    wp::int32 adj_198 = {};
    wp::float64 adj_199 = {};
    wp::int32 adj_200 = {};
    wp::float64 adj_201 = {};
    wp::float64 adj_202 = {};
    wp::int32 adj_203 = {};
    wp::float64 adj_204 = {};
    wp::int32 adj_205 = {};
    wp::float64 adj_206 = {};
    wp::float64 adj_207 = {};
    wp::float64 adj_208 = {};
    wp::int32 adj_209 = {};
    wp::float64 adj_210 = {};
    wp::int32 adj_211 = {};
    wp::float64 adj_212 = {};
    wp::float64 adj_213 = {};
    wp::float64 adj_214 = {};
    wp::int32 adj_215 = {};
    wp::int32 adj_216 = {};
    wp::float64 adj_217 = {};
    wp::int32 adj_218 = {};
    wp::int32 adj_219 = {};
    wp::float64 adj_220 = {};
    wp::int32 adj_221 = {};
    wp::float64 adj_222 = {};
    wp::float64 adj_223 = {};
    wp::int32 adj_224 = {};
    wp::float64 adj_225 = {};
    wp::int32 adj_226 = {};
    wp::float64 adj_227 = {};
    wp::float64 adj_228 = {};
    wp::float64 adj_229 = {};
    wp::int32 adj_230 = {};
    wp::float64 adj_231 = {};
    wp::int32 adj_232 = {};
    wp::float64 adj_233 = {};
    wp::float64 adj_234 = {};
    wp::float64 adj_235 = {};
    wp::int32 adj_236 = {};
    wp::vec_t<3,wp::float64> adj_237 = {};
    wp::float64 adj_238 = {};
    wp::int32 adj_239 = {};
    wp::float64 adj_240 = {};
    wp::int32 adj_241 = {};
    wp::int32 adj_242 = {};
    wp::float64 adj_243 = {};
    wp::int32 adj_244 = {};
    wp::float64 adj_245 = {};
    wp::int32 adj_246 = {};
    wp::float64 adj_247 = {};
    wp::int32 adj_248 = {};
    wp::float64 adj_249 = {};
    wp::int32 adj_250 = {};
    wp::float64 adj_251 = {};
    wp::float64 adj_252 = {};
    wp::float64 adj_253 = {};
    wp::float32 adj_254 = {};
    wp::float64 adj_255 = {};
    bool adj_256 = {};
    wp::float32 adj_257 = {};
    wp::float64 adj_258 = {};
    bool adj_259 = {};
    wp::float32 adj_260 = {};
    wp::float64 adj_261 = {};
    bool adj_262 = {};
    bool adj_263 = {};
    wp::int32 adj_264 = {};
    wp::float32 adj_265 = {};
    wp::float64 adj_266 = {};
    bool adj_267 = {};
    wp::float32 adj_268 = {};
    wp::float64 adj_269 = {};
    bool adj_270 = {};
    bool adj_271 = {};
    wp::int32 adj_272 = {};
    wp::float32 adj_273 = {};
    wp::float64 adj_274 = {};
    bool adj_275 = {};
    wp::float32 adj_276 = {};
    wp::float64 adj_277 = {};
    bool adj_278 = {};
    bool adj_279 = {};
    wp::int32 adj_280 = {};
    wp::float32 adj_281 = {};
    wp::float64 adj_282 = {};
    bool adj_283 = {};
    wp::float32 adj_284 = {};
    wp::float64 adj_285 = {};
    bool adj_286 = {};
    bool adj_287 = {};
    wp::int32 adj_288 = {};
    wp::int32 adj_289 = {};
    wp::vec_t<3,wp::float64> adj_290 = {};
    wp::vec_t<3,wp::float64> adj_291 = {};
    wp::vec_t<3,wp::float64> adj_292 = {};
    wp::vec_t<3,wp::float64> adj_293 = {};
    wp::vec_t<3,wp::float64> adj_294 = {};
    wp::vec_t<3,wp::float64> adj_295 = {};
    //---------
    // forward
    // def point_triangle_distance_type(p: wp.vec3d, t0: wp.vec3d, t1: wp.vec3d, t2: wp.vec3d) -> int:       <L 19>
    // basis_row0 = t1 - t0                                                                   <L 20>
    var_0 = wp::sub(var_t1, var_t0);
    // basis_row1 = t2 - t0                                                                   <L 21>
    var_1 = wp::sub(var_t2, var_t0);
    // nVec = wp.cross(basis_row0, basis_row1)                                                <L 22>
    var_2 = wp::cross(var_0, var_1);
    // basis_row1 = wp.cross(basis_row0, nVec)                                                <L 23>
    var_3 = wp::cross(var_0, var_2);
    // sys = wp.vec4d()                                                                       <L 25>
    var_4 = wp::vec_t<4,wp::float64>();
    // rhs = wp.vec2d()                                                                       <L 26>
    var_5 = wp::vec_t<2,wp::float64>();
    // sys[0] = basis_row0[0] * basis_row0[0] + basis_row0[1] * basis_row0[1] + basis_row0[2] * basis_row0[2]       <L 27>
    var_7 = wp::extract(var_0, var_6);
    var_9 = wp::extract(var_0, var_8);
    var_10 = wp::mul(var_7, var_9);
    var_12 = wp::extract(var_0, var_11);
    var_14 = wp::extract(var_0, var_13);
    var_15 = wp::mul(var_12, var_14);
    var_16 = wp::add(var_10, var_15);
    var_18 = wp::extract(var_0, var_17);
    var_20 = wp::extract(var_0, var_19);
    var_21 = wp::mul(var_18, var_20);
    var_22 = wp::add(var_16, var_21);
    wp::assign_inplace(var_4, var_23, var_22);
    // sys[2] = basis_row0[0] * basis_row1[0] + basis_row0[1] * basis_row1[1] + basis_row0[2] * basis_row1[2]       <L 28>
    var_25 = wp::extract(var_0, var_24);
    var_27 = wp::extract(var_3, var_26);
    var_28 = wp::mul(var_25, var_27);
    var_30 = wp::extract(var_0, var_29);
    var_32 = wp::extract(var_3, var_31);
    var_33 = wp::mul(var_30, var_32);
    var_34 = wp::add(var_28, var_33);
    var_36 = wp::extract(var_0, var_35);
    var_38 = wp::extract(var_3, var_37);
    var_39 = wp::mul(var_36, var_38);
    var_40 = wp::add(var_34, var_39);
    wp::assign_inplace(var_4, var_41, var_40);
    // sys[1] = sys[2]                                                                        <L 29>
    var_43 = wp::extract(var_4, var_42);
    wp::assign_inplace(var_4, var_44, var_43);
    // sys[3] = basis_row1[0] * basis_row1[0] + basis_row1[1] * basis_row1[1] + basis_row1[2] * basis_row1[2]       <L 30>
    var_46 = wp::extract(var_3, var_45);
    var_48 = wp::extract(var_3, var_47);
    var_49 = wp::mul(var_46, var_48);
    var_51 = wp::extract(var_3, var_50);
    var_53 = wp::extract(var_3, var_52);
    var_54 = wp::mul(var_51, var_53);
    var_55 = wp::add(var_49, var_54);
    var_57 = wp::extract(var_3, var_56);
    var_59 = wp::extract(var_3, var_58);
    var_60 = wp::mul(var_57, var_59);
    var_61 = wp::add(var_55, var_60);
    wp::assign_inplace(var_4, var_62, var_61);
    // b = p - t0                                                                             <L 32>
    var_63 = wp::sub(var_p, var_t0);
    // rhs[0] = wp.dot(basis_row0, b)                                                         <L 33>
    var_64 = wp::dot(var_0, var_63);
    wp::assign_inplace(var_5, var_65, var_64);
    // rhs[1] = wp.dot(basis_row1, b)                                                         <L 34>
    var_66 = wp::dot(var_3, var_63);
    wp::assign_inplace(var_5, var_67, var_66);
    // param_col00, param_col01 = solveLdlt2D(sys[0], sys[1], sys[3], rhs[0], rhs[1])         <L 35>
    var_69 = wp::extract(var_4, var_68);
    var_71 = wp::extract(var_4, var_70);
    var_73 = wp::extract(var_4, var_72);
    var_75 = wp::extract(var_5, var_74);
    var_77 = wp::extract(var_5, var_76);
    solveLdlt2D_0(var_69, var_71, var_73, var_75, var_77, var_78, var_79);
    // if param_col00 > wp.float64(0.0) and param_col00 < wp.float64(1.0) and param_col01 >= wp.float64(0.0):       <L 36>
    var_81 = wp::float64(var_80);
    var_82 = (var_78 > var_81);
    var_84 = wp::float64(var_83);
    var_85 = (var_78 < var_84);
    var_87 = wp::float64(var_86);
    var_88 = (var_79 >= var_87);
    var_89 = var_82 && var_85 && var_88;
    if (var_89) {
        // return 3  # PE t0t1                                                                <L 37>
        goto label0;
    }
    if (!var_89) {
        // basis_row0 = t2 - t1                                                               <L 39>
        var_91 = wp::sub(var_t2, var_t1);
        // basis_row1 = wp.cross(basis_row0, nVec)                                            <L 40>
        var_92 = wp::cross(var_91, var_2);
        // sys[0] = basis_row0[0] * basis_row0[0] + basis_row0[1] * basis_row0[1] + basis_row0[2] * basis_row0[2]       <L 42>
        var_94 = wp::extract(var_91, var_93);
        var_96 = wp::extract(var_91, var_95);
        var_97 = wp::mul(var_94, var_96);
        var_99 = wp::extract(var_91, var_98);
        var_101 = wp::extract(var_91, var_100);
        var_102 = wp::mul(var_99, var_101);
        var_103 = wp::add(var_97, var_102);
        var_105 = wp::extract(var_91, var_104);
        var_107 = wp::extract(var_91, var_106);
        var_108 = wp::mul(var_105, var_107);
        var_109 = wp::add(var_103, var_108);
        wp::assign_inplace(var_4, var_110, var_109);
        // sys[2] = basis_row0[0] * basis_row1[0] + basis_row0[1] * basis_row1[1] + basis_row0[2] * basis_row1[2]       <L 43>
        var_112 = wp::extract(var_91, var_111);
        var_114 = wp::extract(var_92, var_113);
        var_115 = wp::mul(var_112, var_114);
        var_117 = wp::extract(var_91, var_116);
        var_119 = wp::extract(var_92, var_118);
        var_120 = wp::mul(var_117, var_119);
        var_121 = wp::add(var_115, var_120);
        var_123 = wp::extract(var_91, var_122);
        var_125 = wp::extract(var_92, var_124);
        var_126 = wp::mul(var_123, var_125);
        var_127 = wp::add(var_121, var_126);
        wp::assign_inplace(var_4, var_128, var_127);
        // sys[1] = sys[2]                                                                    <L 44>
        var_130 = wp::extract(var_4, var_129);
        wp::assign_inplace(var_4, var_131, var_130);
        // sys[3] = basis_row1[0] * basis_row1[0] + basis_row1[1] * basis_row1[1] + basis_row1[2] * basis_row1[2]       <L 45>
        var_133 = wp::extract(var_92, var_132);
        var_135 = wp::extract(var_92, var_134);
        var_136 = wp::mul(var_133, var_135);
        var_138 = wp::extract(var_92, var_137);
        var_140 = wp::extract(var_92, var_139);
        var_141 = wp::mul(var_138, var_140);
        var_142 = wp::add(var_136, var_141);
        var_144 = wp::extract(var_92, var_143);
        var_146 = wp::extract(var_92, var_145);
        var_147 = wp::mul(var_144, var_146);
        var_148 = wp::add(var_142, var_147);
        wp::assign_inplace(var_4, var_149, var_148);
        // b = p - t1                                                                         <L 46>
        var_150 = wp::sub(var_p, var_t1);
        // rhs[0] = wp.dot(basis_row0, b)                                                     <L 47>
        var_151 = wp::dot(var_91, var_150);
        wp::assign_inplace(var_5, var_152, var_151);
        // rhs[1] = wp.dot(basis_row1, b)                                                     <L 48>
        var_153 = wp::dot(var_92, var_150);
        wp::assign_inplace(var_5, var_154, var_153);
        // param_col10, param_col11 = solveLdlt2D(sys[0], sys[1], sys[3], rhs[0], rhs[1])       <L 49>
        var_156 = wp::extract(var_4, var_155);
        var_158 = wp::extract(var_4, var_157);
        var_160 = wp::extract(var_4, var_159);
        var_162 = wp::extract(var_5, var_161);
        var_164 = wp::extract(var_5, var_163);
        solveLdlt2D_0(var_156, var_158, var_160, var_162, var_164, var_165, var_166);
        // if param_col10 > wp.float64(0.0) and param_col10 < wp.float64(1.0) and param_col11 >= wp.float64(0.0):       <L 50>
        var_168 = wp::float64(var_167);
        var_169 = (var_165 > var_168);
        var_171 = wp::float64(var_170);
        var_172 = (var_165 < var_171);
        var_174 = wp::float64(var_173);
        var_175 = (var_166 >= var_174);
        var_176 = var_169 && var_172 && var_175;
        if (var_176) {
            // return 4  # PE t1t2                                                            <L 51>
            goto label1;
        }
        if (!var_176) {
            // basis_row0 = t0 - t2                                                           <L 53>
            var_178 = wp::sub(var_t0, var_t2);
            // basis_row1 = wp.cross(basis_row0, nVec)                                        <L 54>
            var_179 = wp::cross(var_178, var_2);
            // sys[0] = basis_row0[0] * basis_row0[0] + basis_row0[1] * basis_row0[1] + basis_row0[2] * basis_row0[2]       <L 55>
            var_181 = wp::extract(var_178, var_180);
            var_183 = wp::extract(var_178, var_182);
            var_184 = wp::mul(var_181, var_183);
            var_186 = wp::extract(var_178, var_185);
            var_188 = wp::extract(var_178, var_187);
            var_189 = wp::mul(var_186, var_188);
            var_190 = wp::add(var_184, var_189);
            var_192 = wp::extract(var_178, var_191);
            var_194 = wp::extract(var_178, var_193);
            var_195 = wp::mul(var_192, var_194);
            var_196 = wp::add(var_190, var_195);
            wp::assign_inplace(var_4, var_197, var_196);
            // sys[2] = basis_row0[0] * basis_row1[0] + basis_row0[1] * basis_row1[1] + basis_row0[2] * basis_row1[2]       <L 56>
            var_199 = wp::extract(var_178, var_198);
            var_201 = wp::extract(var_179, var_200);
            var_202 = wp::mul(var_199, var_201);
            var_204 = wp::extract(var_178, var_203);
            var_206 = wp::extract(var_179, var_205);
            var_207 = wp::mul(var_204, var_206);
            var_208 = wp::add(var_202, var_207);
            var_210 = wp::extract(var_178, var_209);
            var_212 = wp::extract(var_179, var_211);
            var_213 = wp::mul(var_210, var_212);
            var_214 = wp::add(var_208, var_213);
            wp::assign_inplace(var_4, var_215, var_214);
            // sys[1] = sys[2]                                                                <L 57>
            var_217 = wp::extract(var_4, var_216);
            wp::assign_inplace(var_4, var_218, var_217);
            // sys[3] = basis_row1[0] * basis_row1[0] + basis_row1[1] * basis_row1[1] + basis_row1[2] * basis_row1[2]       <L 58>
            var_220 = wp::extract(var_179, var_219);
            var_222 = wp::extract(var_179, var_221);
            var_223 = wp::mul(var_220, var_222);
            var_225 = wp::extract(var_179, var_224);
            var_227 = wp::extract(var_179, var_226);
            var_228 = wp::mul(var_225, var_227);
            var_229 = wp::add(var_223, var_228);
            var_231 = wp::extract(var_179, var_230);
            var_233 = wp::extract(var_179, var_232);
            var_234 = wp::mul(var_231, var_233);
            var_235 = wp::add(var_229, var_234);
            wp::assign_inplace(var_4, var_236, var_235);
            // b = p - t2                                                                     <L 59>
            var_237 = wp::sub(var_p, var_t2);
            // rhs[0] = wp.dot(basis_row0, b)                                                 <L 60>
            var_238 = wp::dot(var_178, var_237);
            wp::assign_inplace(var_5, var_239, var_238);
            // rhs[1] = wp.dot(basis_row1, b)                                                 <L 61>
            var_240 = wp::dot(var_179, var_237);
            wp::assign_inplace(var_5, var_241, var_240);
            // param_col20, param_col21 = solveLdlt2D(sys[0], sys[1], sys[3], rhs[0], rhs[1])       <L 62>
            var_243 = wp::extract(var_4, var_242);
            var_245 = wp::extract(var_4, var_244);
            var_247 = wp::extract(var_4, var_246);
            var_249 = wp::extract(var_5, var_248);
            var_251 = wp::extract(var_5, var_250);
            solveLdlt2D_0(var_243, var_245, var_247, var_249, var_251, var_252, var_253);
            // if param_col20 > wp.float64(0.0) and param_col20 < wp.float64(1.0) and param_col21 >= wp.float64(0.0):       <L 64>
            var_255 = wp::float64(var_254);
            var_256 = (var_252 > var_255);
            var_258 = wp::float64(var_257);
            var_259 = (var_252 < var_258);
            var_261 = wp::float64(var_260);
            var_262 = (var_253 >= var_261);
            var_263 = var_256 && var_259 && var_262;
            if (var_263) {
                // return 5  # PE t2t0                                                        <L 65>
                goto label2;
            }
            if (!var_263) {
                // if param_col00 <= wp.float64(0.0) and param_col20 >= wp.float64(1.0):       <L 67>
                var_266 = wp::float64(var_265);
                var_267 = (var_78 <= var_266);
                var_269 = wp::float64(var_268);
                var_270 = (var_252 >= var_269);
                var_271 = var_267 && var_270;
                if (var_271) {
                    // return 0  # PP t0                                                      <L 68>
                    goto label3;
                }
                if (!var_271) {
                    // elif param_col10 <= wp.float64(0.0) and param_col00 >= wp.float64(1.0):       <L 69>
                    var_274 = wp::float64(var_273);
                    var_275 = (var_165 <= var_274);
                    var_277 = wp::float64(var_276);
                    var_278 = (var_78 >= var_277);
                    var_279 = var_275 && var_278;
                    if (var_279) {
                        // return 1  # PP t1                                                  <L 70>
                        goto label4;
                    }
                    if (!var_279) {
                        // elif param_col20 <= wp.float64(0.0) and param_col10 >= wp.float64(1.0):       <L 71>
                        var_282 = wp::float64(var_281);
                        var_283 = (var_252 <= var_282);
                        var_285 = wp::float64(var_284);
                        var_286 = (var_165 >= var_285);
                        var_287 = var_283 && var_286;
                        if (var_287) {
                            // return 2  # PP t2                                              <L 72>
                            goto label5;
                        }
                        if (!var_287) {
                            // return 6  # PT                                                 <L 74>
                            goto label6;
                        }
                    }
                }
            }
        }
        var_290 = wp::where(var_176, var_91, var_178);
        var_291 = wp::where(var_176, var_92, var_179);
        var_292 = wp::where(var_176, var_150, var_237);
    }
    var_293 = wp::where(var_89, var_0, var_290);
    var_294 = wp::where(var_89, var_3, var_291);
    var_295 = wp::where(var_89, var_63, var_292);
    //---------
    // reverse
    wp::adj_where(var_89, var_63, var_292, adj_89, adj_63, adj_292, adj_295);
    wp::adj_where(var_89, var_3, var_291, adj_89, adj_3, adj_291, adj_294);
    wp::adj_where(var_89, var_0, var_290, adj_89, adj_0, adj_290, adj_293);
    if (!var_89) {
        wp::adj_where(var_176, var_150, var_237, adj_176, adj_150, adj_237, adj_292);
        wp::adj_where(var_176, var_92, var_179, adj_176, adj_92, adj_179, adj_291);
        wp::adj_where(var_176, var_91, var_178, adj_176, adj_91, adj_178, adj_290);
        if (!var_176) {
            if (!var_263) {
                if (!var_271) {
                    if (!var_279) {
                        if (!var_287) {
                            label6:;
                            adj_289 += adj_ret;
                            // adj: return 6  # PT                                            <L 74>
                        }
                        if (var_287) {
                            label5:;
                            adj_288 += adj_ret;
                            // adj: return 2  # PP t2                                         <L 72>
                        }
                        wp::adj_float64(var_284, adj_284, adj_285);
                        wp::adj_float64(var_281, adj_281, adj_282);
                        // adj: elif param_col20 <= wp.float64(0.0) and param_col10 >= wp.float64(1.0):  <L 71>
                    }
                    if (var_279) {
                        label4:;
                        adj_280 += adj_ret;
                        // adj: return 1  # PP t1                                             <L 70>
                    }
                    wp::adj_float64(var_276, adj_276, adj_277);
                    wp::adj_float64(var_273, adj_273, adj_274);
                    // adj: elif param_col10 <= wp.float64(0.0) and param_col00 >= wp.float64(1.0):  <L 69>
                }
                if (var_271) {
                    label3:;
                    adj_272 += adj_ret;
                    // adj: return 0  # PP t0                                                 <L 68>
                }
                wp::adj_float64(var_268, adj_268, adj_269);
                wp::adj_float64(var_265, adj_265, adj_266);
                // adj: if param_col00 <= wp.float64(0.0) and param_col20 >= wp.float64(1.0):  <L 67>
            }
            if (var_263) {
                label2:;
                adj_264 += adj_ret;
                // adj: return 5  # PE t2t0                                                   <L 65>
            }
            wp::adj_float64(var_260, adj_260, adj_261);
            wp::adj_float64(var_257, adj_257, adj_258);
            wp::adj_float64(var_254, adj_254, adj_255);
            // adj: if param_col20 > wp.float64(0.0) and param_col20 < wp.float64(1.0) and param_col21 >= wp.float64(0.0):  <L 64>
            adj_solveLdlt2D_0(var_243, var_245, var_247, var_249, var_251, var_252, var_253, adj_243, adj_245, adj_247, adj_249, adj_251, adj_252, adj_253);
            wp::adj_extract(var_5, var_250, adj_5, adj_250, adj_251);
            wp::adj_extract(var_5, var_248, adj_5, adj_248, adj_249);
            wp::adj_extract(var_4, var_246, adj_4, adj_246, adj_247);
            wp::adj_extract(var_4, var_244, adj_4, adj_244, adj_245);
            wp::adj_extract(var_4, var_242, adj_4, adj_242, adj_243);
            // adj: param_col20, param_col21 = solveLdlt2D(sys[0], sys[1], sys[3], rhs[0], rhs[1])  <L 62>
            wp::adj_assign_inplace(var_5, var_241, var_240, adj_5, adj_241, adj_240);
            wp::adj_dot(var_179, var_237, adj_179, adj_237, adj_240);
            // adj: rhs[1] = wp.dot(basis_row1, b)                                            <L 61>
            wp::adj_assign_inplace(var_5, var_239, var_238, adj_5, adj_239, adj_238);
            wp::adj_dot(var_178, var_237, adj_178, adj_237, adj_238);
            // adj: rhs[0] = wp.dot(basis_row0, b)                                            <L 60>
            wp::adj_sub(var_p, var_t2, adj_p, adj_t2, adj_237);
            // adj: b = p - t2                                                                <L 59>
            wp::adj_assign_inplace(var_4, var_236, var_235, adj_4, adj_236, adj_235);
            wp::adj_add(var_229, var_234, adj_229, adj_234, adj_235);
            wp::adj_mul(var_231, var_233, adj_231, adj_233, adj_234);
            wp::adj_extract(var_179, var_232, adj_179, adj_232, adj_233);
            wp::adj_extract(var_179, var_230, adj_179, adj_230, adj_231);
            wp::adj_add(var_223, var_228, adj_223, adj_228, adj_229);
            wp::adj_mul(var_225, var_227, adj_225, adj_227, adj_228);
            wp::adj_extract(var_179, var_226, adj_179, adj_226, adj_227);
            wp::adj_extract(var_179, var_224, adj_179, adj_224, adj_225);
            wp::adj_mul(var_220, var_222, adj_220, adj_222, adj_223);
            wp::adj_extract(var_179, var_221, adj_179, adj_221, adj_222);
            wp::adj_extract(var_179, var_219, adj_179, adj_219, adj_220);
            // adj: sys[3] = basis_row1[0] * basis_row1[0] + basis_row1[1] * basis_row1[1] + basis_row1[2] * basis_row1[2]  <L 58>
            wp::adj_assign_inplace(var_4, var_218, var_217, adj_4, adj_218, adj_217);
            wp::adj_extract(var_4, var_216, adj_4, adj_216, adj_217);
            // adj: sys[1] = sys[2]                                                           <L 57>
            wp::adj_assign_inplace(var_4, var_215, var_214, adj_4, adj_215, adj_214);
            wp::adj_add(var_208, var_213, adj_208, adj_213, adj_214);
            wp::adj_mul(var_210, var_212, adj_210, adj_212, adj_213);
            wp::adj_extract(var_179, var_211, adj_179, adj_211, adj_212);
            wp::adj_extract(var_178, var_209, adj_178, adj_209, adj_210);
            wp::adj_add(var_202, var_207, adj_202, adj_207, adj_208);
            wp::adj_mul(var_204, var_206, adj_204, adj_206, adj_207);
            wp::adj_extract(var_179, var_205, adj_179, adj_205, adj_206);
            wp::adj_extract(var_178, var_203, adj_178, adj_203, adj_204);
            wp::adj_mul(var_199, var_201, adj_199, adj_201, adj_202);
            wp::adj_extract(var_179, var_200, adj_179, adj_200, adj_201);
            wp::adj_extract(var_178, var_198, adj_178, adj_198, adj_199);
            // adj: sys[2] = basis_row0[0] * basis_row1[0] + basis_row0[1] * basis_row1[1] + basis_row0[2] * basis_row1[2]  <L 56>
            wp::adj_assign_inplace(var_4, var_197, var_196, adj_4, adj_197, adj_196);
            wp::adj_add(var_190, var_195, adj_190, adj_195, adj_196);
            wp::adj_mul(var_192, var_194, adj_192, adj_194, adj_195);
            wp::adj_extract(var_178, var_193, adj_178, adj_193, adj_194);
            wp::adj_extract(var_178, var_191, adj_178, adj_191, adj_192);
            wp::adj_add(var_184, var_189, adj_184, adj_189, adj_190);
            wp::adj_mul(var_186, var_188, adj_186, adj_188, adj_189);
            wp::adj_extract(var_178, var_187, adj_178, adj_187, adj_188);
            wp::adj_extract(var_178, var_185, adj_178, adj_185, adj_186);
            wp::adj_mul(var_181, var_183, adj_181, adj_183, adj_184);
            wp::adj_extract(var_178, var_182, adj_178, adj_182, adj_183);
            wp::adj_extract(var_178, var_180, adj_178, adj_180, adj_181);
            // adj: sys[0] = basis_row0[0] * basis_row0[0] + basis_row0[1] * basis_row0[1] + basis_row0[2] * basis_row0[2]  <L 55>
            wp::adj_cross(var_178, var_2, adj_178, adj_2, adj_179);
            // adj: basis_row1 = wp.cross(basis_row0, nVec)                                   <L 54>
            wp::adj_sub(var_t0, var_t2, adj_t0, adj_t2, adj_178);
            // adj: basis_row0 = t0 - t2                                                      <L 53>
        }
        if (var_176) {
            label1:;
            adj_177 += adj_ret;
            // adj: return 4  # PE t1t2                                                       <L 51>
        }
        wp::adj_float64(var_173, adj_173, adj_174);
        wp::adj_float64(var_170, adj_170, adj_171);
        wp::adj_float64(var_167, adj_167, adj_168);
        // adj: if param_col10 > wp.float64(0.0) and param_col10 < wp.float64(1.0) and param_col11 >= wp.float64(0.0):  <L 50>
        adj_solveLdlt2D_0(var_156, var_158, var_160, var_162, var_164, var_165, var_166, adj_156, adj_158, adj_160, adj_162, adj_164, adj_165, adj_166);
        wp::adj_extract(var_5, var_163, adj_5, adj_163, adj_164);
        wp::adj_extract(var_5, var_161, adj_5, adj_161, adj_162);
        wp::adj_extract(var_4, var_159, adj_4, adj_159, adj_160);
        wp::adj_extract(var_4, var_157, adj_4, adj_157, adj_158);
        wp::adj_extract(var_4, var_155, adj_4, adj_155, adj_156);
        // adj: param_col10, param_col11 = solveLdlt2D(sys[0], sys[1], sys[3], rhs[0], rhs[1])  <L 49>
        wp::adj_assign_inplace(var_5, var_154, var_153, adj_5, adj_154, adj_153);
        wp::adj_dot(var_92, var_150, adj_92, adj_150, adj_153);
        // adj: rhs[1] = wp.dot(basis_row1, b)                                                <L 48>
        wp::adj_assign_inplace(var_5, var_152, var_151, adj_5, adj_152, adj_151);
        wp::adj_dot(var_91, var_150, adj_91, adj_150, adj_151);
        // adj: rhs[0] = wp.dot(basis_row0, b)                                                <L 47>
        wp::adj_sub(var_p, var_t1, adj_p, adj_t1, adj_150);
        // adj: b = p - t1                                                                    <L 46>
        wp::adj_assign_inplace(var_4, var_149, var_148, adj_4, adj_149, adj_148);
        wp::adj_add(var_142, var_147, adj_142, adj_147, adj_148);
        wp::adj_mul(var_144, var_146, adj_144, adj_146, adj_147);
        wp::adj_extract(var_92, var_145, adj_92, adj_145, adj_146);
        wp::adj_extract(var_92, var_143, adj_92, adj_143, adj_144);
        wp::adj_add(var_136, var_141, adj_136, adj_141, adj_142);
        wp::adj_mul(var_138, var_140, adj_138, adj_140, adj_141);
        wp::adj_extract(var_92, var_139, adj_92, adj_139, adj_140);
        wp::adj_extract(var_92, var_137, adj_92, adj_137, adj_138);
        wp::adj_mul(var_133, var_135, adj_133, adj_135, adj_136);
        wp::adj_extract(var_92, var_134, adj_92, adj_134, adj_135);
        wp::adj_extract(var_92, var_132, adj_92, adj_132, adj_133);
        // adj: sys[3] = basis_row1[0] * basis_row1[0] + basis_row1[1] * basis_row1[1] + basis_row1[2] * basis_row1[2]  <L 45>
        wp::adj_assign_inplace(var_4, var_131, var_130, adj_4, adj_131, adj_130);
        wp::adj_extract(var_4, var_129, adj_4, adj_129, adj_130);
        // adj: sys[1] = sys[2]                                                               <L 44>
        wp::adj_assign_inplace(var_4, var_128, var_127, adj_4, adj_128, adj_127);
        wp::adj_add(var_121, var_126, adj_121, adj_126, adj_127);
        wp::adj_mul(var_123, var_125, adj_123, adj_125, adj_126);
        wp::adj_extract(var_92, var_124, adj_92, adj_124, adj_125);
        wp::adj_extract(var_91, var_122, adj_91, adj_122, adj_123);
        wp::adj_add(var_115, var_120, adj_115, adj_120, adj_121);
        wp::adj_mul(var_117, var_119, adj_117, adj_119, adj_120);
        wp::adj_extract(var_92, var_118, adj_92, adj_118, adj_119);
        wp::adj_extract(var_91, var_116, adj_91, adj_116, adj_117);
        wp::adj_mul(var_112, var_114, adj_112, adj_114, adj_115);
        wp::adj_extract(var_92, var_113, adj_92, adj_113, adj_114);
        wp::adj_extract(var_91, var_111, adj_91, adj_111, adj_112);
        // adj: sys[2] = basis_row0[0] * basis_row1[0] + basis_row0[1] * basis_row1[1] + basis_row0[2] * basis_row1[2]  <L 43>
        wp::adj_assign_inplace(var_4, var_110, var_109, adj_4, adj_110, adj_109);
        wp::adj_add(var_103, var_108, adj_103, adj_108, adj_109);
        wp::adj_mul(var_105, var_107, adj_105, adj_107, adj_108);
        wp::adj_extract(var_91, var_106, adj_91, adj_106, adj_107);
        wp::adj_extract(var_91, var_104, adj_91, adj_104, adj_105);
        wp::adj_add(var_97, var_102, adj_97, adj_102, adj_103);
        wp::adj_mul(var_99, var_101, adj_99, adj_101, adj_102);
        wp::adj_extract(var_91, var_100, adj_91, adj_100, adj_101);
        wp::adj_extract(var_91, var_98, adj_91, adj_98, adj_99);
        wp::adj_mul(var_94, var_96, adj_94, adj_96, adj_97);
        wp::adj_extract(var_91, var_95, adj_91, adj_95, adj_96);
        wp::adj_extract(var_91, var_93, adj_91, adj_93, adj_94);
        // adj: sys[0] = basis_row0[0] * basis_row0[0] + basis_row0[1] * basis_row0[1] + basis_row0[2] * basis_row0[2]  <L 42>
        wp::adj_cross(var_91, var_2, adj_91, adj_2, adj_92);
        // adj: basis_row1 = wp.cross(basis_row0, nVec)                                       <L 40>
        wp::adj_sub(var_t2, var_t1, adj_t2, adj_t1, adj_91);
        // adj: basis_row0 = t2 - t1                                                          <L 39>
    }
    if (var_89) {
        label0:;
        adj_90 += adj_ret;
        // adj: return 3  # PE t0t1                                                           <L 37>
    }
    wp::adj_float64(var_86, adj_86, adj_87);
    wp::adj_float64(var_83, adj_83, adj_84);
    wp::adj_float64(var_80, adj_80, adj_81);
    // adj: if param_col00 > wp.float64(0.0) and param_col00 < wp.float64(1.0) and param_col01 >= wp.float64(0.0):  <L 36>
    adj_solveLdlt2D_0(var_69, var_71, var_73, var_75, var_77, var_78, var_79, adj_69, adj_71, adj_73, adj_75, adj_77, adj_78, adj_79);
    wp::adj_extract(var_5, var_76, adj_5, adj_76, adj_77);
    wp::adj_extract(var_5, var_74, adj_5, adj_74, adj_75);
    wp::adj_extract(var_4, var_72, adj_4, adj_72, adj_73);
    wp::adj_extract(var_4, var_70, adj_4, adj_70, adj_71);
    wp::adj_extract(var_4, var_68, adj_4, adj_68, adj_69);
    // adj: param_col00, param_col01 = solveLdlt2D(sys[0], sys[1], sys[3], rhs[0], rhs[1])    <L 35>
    wp::adj_assign_inplace(var_5, var_67, var_66, adj_5, adj_67, adj_66);
    wp::adj_dot(var_3, var_63, adj_3, adj_63, adj_66);
    // adj: rhs[1] = wp.dot(basis_row1, b)                                                    <L 34>
    wp::adj_assign_inplace(var_5, var_65, var_64, adj_5, adj_65, adj_64);
    wp::adj_dot(var_0, var_63, adj_0, adj_63, adj_64);
    // adj: rhs[0] = wp.dot(basis_row0, b)                                                    <L 33>
    wp::adj_sub(var_p, var_t0, adj_p, adj_t0, adj_63);
    // adj: b = p - t0                                                                        <L 32>
    wp::adj_assign_inplace(var_4, var_62, var_61, adj_4, adj_62, adj_61);
    wp::adj_add(var_55, var_60, adj_55, adj_60, adj_61);
    wp::adj_mul(var_57, var_59, adj_57, adj_59, adj_60);
    wp::adj_extract(var_3, var_58, adj_3, adj_58, adj_59);
    wp::adj_extract(var_3, var_56, adj_3, adj_56, adj_57);
    wp::adj_add(var_49, var_54, adj_49, adj_54, adj_55);
    wp::adj_mul(var_51, var_53, adj_51, adj_53, adj_54);
    wp::adj_extract(var_3, var_52, adj_3, adj_52, adj_53);
    wp::adj_extract(var_3, var_50, adj_3, adj_50, adj_51);
    wp::adj_mul(var_46, var_48, adj_46, adj_48, adj_49);
    wp::adj_extract(var_3, var_47, adj_3, adj_47, adj_48);
    wp::adj_extract(var_3, var_45, adj_3, adj_45, adj_46);
    // adj: sys[3] = basis_row1[0] * basis_row1[0] + basis_row1[1] * basis_row1[1] + basis_row1[2] * basis_row1[2]  <L 30>
    wp::adj_assign_inplace(var_4, var_44, var_43, adj_4, adj_44, adj_43);
    wp::adj_extract(var_4, var_42, adj_4, adj_42, adj_43);
    // adj: sys[1] = sys[2]                                                                   <L 29>
    wp::adj_assign_inplace(var_4, var_41, var_40, adj_4, adj_41, adj_40);
    wp::adj_add(var_34, var_39, adj_34, adj_39, adj_40);
    wp::adj_mul(var_36, var_38, adj_36, adj_38, adj_39);
    wp::adj_extract(var_3, var_37, adj_3, adj_37, adj_38);
    wp::adj_extract(var_0, var_35, adj_0, adj_35, adj_36);
    wp::adj_add(var_28, var_33, adj_28, adj_33, adj_34);
    wp::adj_mul(var_30, var_32, adj_30, adj_32, adj_33);
    wp::adj_extract(var_3, var_31, adj_3, adj_31, adj_32);
    wp::adj_extract(var_0, var_29, adj_0, adj_29, adj_30);
    wp::adj_mul(var_25, var_27, adj_25, adj_27, adj_28);
    wp::adj_extract(var_3, var_26, adj_3, adj_26, adj_27);
    wp::adj_extract(var_0, var_24, adj_0, adj_24, adj_25);
    // adj: sys[2] = basis_row0[0] * basis_row1[0] + basis_row0[1] * basis_row1[1] + basis_row0[2] * basis_row1[2]  <L 28>
    wp::adj_assign_inplace(var_4, var_23, var_22, adj_4, adj_23, adj_22);
    wp::adj_add(var_16, var_21, adj_16, adj_21, adj_22);
    wp::adj_mul(var_18, var_20, adj_18, adj_20, adj_21);
    wp::adj_extract(var_0, var_19, adj_0, adj_19, adj_20);
    wp::adj_extract(var_0, var_17, adj_0, adj_17, adj_18);
    wp::adj_add(var_10, var_15, adj_10, adj_15, adj_16);
    wp::adj_mul(var_12, var_14, adj_12, adj_14, adj_15);
    wp::adj_extract(var_0, var_13, adj_0, adj_13, adj_14);
    wp::adj_extract(var_0, var_11, adj_0, adj_11, adj_12);
    wp::adj_mul(var_7, var_9, adj_7, adj_9, adj_10);
    wp::adj_extract(var_0, var_8, adj_0, adj_8, adj_9);
    wp::adj_extract(var_0, var_6, adj_0, adj_6, adj_7);
    // adj: sys[0] = basis_row0[0] * basis_row0[0] + basis_row0[1] * basis_row0[1] + basis_row0[2] * basis_row0[2]  <L 27>
    // adj: rhs = wp.vec2d()                                                                  <L 26>
    // adj: sys = wp.vec4d()                                                                  <L 25>
    wp::adj_cross(var_0, var_2, adj_0, adj_2, adj_3);
    // adj: basis_row1 = wp.cross(basis_row0, nVec)                                           <L 23>
    wp::adj_cross(var_0, var_1, adj_0, adj_1, adj_2);
    // adj: nVec = wp.cross(basis_row0, basis_row1)                                           <L 22>
    wp::adj_sub(var_t2, var_t0, adj_t2, adj_t0, adj_1);
    // adj: basis_row1 = t2 - t0                                                              <L 21>
    wp::adj_sub(var_t1, var_t0, adj_t1, adj_t0, adj_0);
    // adj: basis_row0 = t1 - t0                                                              <L 20>
    // adj: def point_triangle_distance_type(p: wp.vec3d, t0: wp.vec3d, t1: wp.vec3d, t2: wp.vec3d) -> int:  <L 19>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/point_point_distance.py:4
static CUDA_CALLABLE wp::float64 point_point_distance_0(
    wp::vec_t<3,wp::float64> var_a,
    wp::vec_t<3,wp::float64> var_b)
{
    //---------
    // primal vars
    wp::vec_t<3,wp::float64> var_0;
    wp::float64 var_1;
    //---------
    // forward
    // def point_point_distance(a: wp.vec3d, b: wp.vec3d):                                    <L 5>
    // return wp.length_sq(a - b)                                                             <L 6>
    var_0 = wp::sub(var_a, var_b);
    var_1 = wp::length_sq(var_0);
    return var_1;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/point_point_distance.py:4
static CUDA_CALLABLE void adj_point_point_distance_0(
    wp::vec_t<3,wp::float64> var_a,
    wp::vec_t<3,wp::float64> var_b,
    wp::vec_t<3,wp::float64> & adj_a,
    wp::vec_t<3,wp::float64> & adj_b,
    wp::float64 & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3,wp::float64> var_0;
    wp::float64 var_1;
    //---------
    // dual vars
    wp::vec_t<3,wp::float64> adj_0 = {};
    wp::float64 adj_1 = {};
    //---------
    // forward
    // def point_point_distance(a: wp.vec3d, b: wp.vec3d):                                    <L 5>
    // return wp.length_sq(a - b)                                                             <L 6>
    var_0 = wp::sub(var_a, var_b);
    var_1 = wp::length_sq(var_0);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_1 += adj_ret;
    wp::adj_length_sq(var_0, adj_0, adj_1);
    wp::adj_sub(var_a, var_b, adj_a, adj_b, adj_0);
    // adj: return wp.length_sq(a - b)                                                        <L 6>
    // adj: def point_point_distance(a: wp.vec3d, b: wp.vec3d):                               <L 5>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/point_line_distance.py:287
static CUDA_CALLABLE wp::float64 point_line_distance_0(
    wp::vec_t<3,wp::float64> var_p,
    wp::vec_t<3,wp::float64> var_e0,
    wp::vec_t<3,wp::float64> var_e1)
{
    //---------
    // primal vars
    wp::vec_t<3,wp::float64> var_0;
    wp::vec_t<3,wp::float64> var_1;
    wp::vec_t<3,wp::float64> var_2;
    wp::vec_t<3,wp::float64> var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    wp::float64 var_6;
    //---------
    // forward
    // def point_line_distance(p: wp.vec3d, e0: wp.vec3d, e1: wp.vec3d):                      <L 288>
    // pe0 = e0 - p                                                                           <L 290>
    var_0 = wp::sub(var_e0, var_p);
    // pe1 = e1 - p                                                                           <L 291>
    var_1 = wp::sub(var_e1, var_p);
    // e0e1 = e1 - e0                                                                         <L 292>
    var_2 = wp::sub(var_e1, var_e0);
    // nor = wp.cross(pe0, pe1)                                                               <L 293>
    var_3 = wp::cross(var_0, var_1);
    // return wp.length_sq(nor) / wp.length_sq(e0e1)                                          <L 294>
    var_4 = wp::length_sq(var_3);
    var_5 = wp::length_sq(var_2);
    var_6 = wp::div(var_4, var_5);
    return var_6;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/point_line_distance.py:287
static CUDA_CALLABLE void adj_point_line_distance_0(
    wp::vec_t<3,wp::float64> var_p,
    wp::vec_t<3,wp::float64> var_e0,
    wp::vec_t<3,wp::float64> var_e1,
    wp::vec_t<3,wp::float64> & adj_p,
    wp::vec_t<3,wp::float64> & adj_e0,
    wp::vec_t<3,wp::float64> & adj_e1,
    wp::float64 & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3,wp::float64> var_0;
    wp::vec_t<3,wp::float64> var_1;
    wp::vec_t<3,wp::float64> var_2;
    wp::vec_t<3,wp::float64> var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    wp::float64 var_6;
    //---------
    // dual vars
    wp::vec_t<3,wp::float64> adj_0 = {};
    wp::vec_t<3,wp::float64> adj_1 = {};
    wp::vec_t<3,wp::float64> adj_2 = {};
    wp::vec_t<3,wp::float64> adj_3 = {};
    wp::float64 adj_4 = {};
    wp::float64 adj_5 = {};
    wp::float64 adj_6 = {};
    //---------
    // forward
    // def point_line_distance(p: wp.vec3d, e0: wp.vec3d, e1: wp.vec3d):                      <L 288>
    // pe0 = e0 - p                                                                           <L 290>
    var_0 = wp::sub(var_e0, var_p);
    // pe1 = e1 - p                                                                           <L 291>
    var_1 = wp::sub(var_e1, var_p);
    // e0e1 = e1 - e0                                                                         <L 292>
    var_2 = wp::sub(var_e1, var_e0);
    // nor = wp.cross(pe0, pe1)                                                               <L 293>
    var_3 = wp::cross(var_0, var_1);
    // return wp.length_sq(nor) / wp.length_sq(e0e1)                                          <L 294>
    var_4 = wp::length_sq(var_3);
    var_5 = wp::length_sq(var_2);
    var_6 = wp::div(var_4, var_5);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_6 += adj_ret;
    wp::adj_div(var_4, var_5, var_6, adj_4, adj_5, adj_6);
    wp::adj_length_sq(var_2, adj_2, adj_5);
    wp::adj_length_sq(var_3, adj_3, adj_4);
    // adj: return wp.length_sq(nor) / wp.length_sq(e0e1)                                     <L 294>
    wp::adj_cross(var_0, var_1, adj_0, adj_1, adj_3);
    // adj: nor = wp.cross(pe0, pe1)                                                          <L 293>
    wp::adj_sub(var_e1, var_e0, adj_e1, adj_e0, adj_2);
    // adj: e0e1 = e1 - e0                                                                    <L 292>
    wp::adj_sub(var_e1, var_p, adj_e1, adj_p, adj_1);
    // adj: pe1 = e1 - p                                                                      <L 291>
    wp::adj_sub(var_e0, var_p, adj_e0, adj_p, adj_0);
    // adj: pe0 = e0 - p                                                                      <L 290>
    // adj: def point_line_distance(p: wp.vec3d, e0: wp.vec3d, e1: wp.vec3d):                 <L 288>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/point_plane_distance.py:555
static CUDA_CALLABLE wp::float64 point_plane_distance_0(
    wp::vec_t<3,wp::float64> var_p,
    wp::vec_t<3,wp::float64> var_t0,
    wp::vec_t<3,wp::float64> var_t1,
    wp::vec_t<3,wp::float64> var_t2)
{
    //---------
    // primal vars
    wp::vec_t<3,wp::float64> var_0;
    wp::vec_t<3,wp::float64> var_1;
    wp::vec_t<3,wp::float64> var_2;
    wp::vec_t<3,wp::float64> var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    wp::float64 var_6;
    wp::float64 var_7;
    //---------
    // forward
    // def point_plane_distance(p: wp.vec3d, t0: wp.vec3d, t1: wp.vec3d, t2: wp.vec3d):       <L 556>
    // t0t1 = t1 - t0                                                                         <L 561>
    var_0 = wp::sub(var_t1, var_t0);
    // t0t2 = t2 - t0                                                                         <L 562>
    var_1 = wp::sub(var_t2, var_t0);
    // t0p = p - t0                                                                           <L 563>
    var_2 = wp::sub(var_p, var_t0);
    // b = wp.cross(t0t1, t0t2)                                                               <L 564>
    var_3 = wp::cross(var_0, var_1);
    // aTb = wp.dot(t0p, b)                                                                   <L 565>
    var_4 = wp::dot(var_2, var_3);
    // return aTb * aTb / wp.length_sq(b)                                                     <L 566>
    var_5 = wp::mul(var_4, var_4);
    var_6 = wp::length_sq(var_3);
    var_7 = wp::div(var_5, var_6);
    return var_7;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/point_plane_distance.py:555
static CUDA_CALLABLE void adj_point_plane_distance_0(
    wp::vec_t<3,wp::float64> var_p,
    wp::vec_t<3,wp::float64> var_t0,
    wp::vec_t<3,wp::float64> var_t1,
    wp::vec_t<3,wp::float64> var_t2,
    wp::vec_t<3,wp::float64> & adj_p,
    wp::vec_t<3,wp::float64> & adj_t0,
    wp::vec_t<3,wp::float64> & adj_t1,
    wp::vec_t<3,wp::float64> & adj_t2,
    wp::float64 & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3,wp::float64> var_0;
    wp::vec_t<3,wp::float64> var_1;
    wp::vec_t<3,wp::float64> var_2;
    wp::vec_t<3,wp::float64> var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    wp::float64 var_6;
    wp::float64 var_7;
    //---------
    // dual vars
    wp::vec_t<3,wp::float64> adj_0 = {};
    wp::vec_t<3,wp::float64> adj_1 = {};
    wp::vec_t<3,wp::float64> adj_2 = {};
    wp::vec_t<3,wp::float64> adj_3 = {};
    wp::float64 adj_4 = {};
    wp::float64 adj_5 = {};
    wp::float64 adj_6 = {};
    wp::float64 adj_7 = {};
    //---------
    // forward
    // def point_plane_distance(p: wp.vec3d, t0: wp.vec3d, t1: wp.vec3d, t2: wp.vec3d):       <L 556>
    // t0t1 = t1 - t0                                                                         <L 561>
    var_0 = wp::sub(var_t1, var_t0);
    // t0t2 = t2 - t0                                                                         <L 562>
    var_1 = wp::sub(var_t2, var_t0);
    // t0p = p - t0                                                                           <L 563>
    var_2 = wp::sub(var_p, var_t0);
    // b = wp.cross(t0t1, t0t2)                                                               <L 564>
    var_3 = wp::cross(var_0, var_1);
    // aTb = wp.dot(t0p, b)                                                                   <L 565>
    var_4 = wp::dot(var_2, var_3);
    // return aTb * aTb / wp.length_sq(b)                                                     <L 566>
    var_5 = wp::mul(var_4, var_4);
    var_6 = wp::length_sq(var_3);
    var_7 = wp::div(var_5, var_6);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_7 += adj_ret;
    wp::adj_div(var_5, var_6, var_7, adj_5, adj_6, adj_7);
    wp::adj_length_sq(var_3, adj_3, adj_6);
    wp::adj_mul(var_4, var_4, adj_4, adj_4, adj_5);
    // adj: return aTb * aTb / wp.length_sq(b)                                                <L 566>
    wp::adj_dot(var_2, var_3, adj_2, adj_3, adj_4);
    // adj: aTb = wp.dot(t0p, b)                                                              <L 565>
    wp::adj_cross(var_0, var_1, adj_0, adj_1, adj_3);
    // adj: b = wp.cross(t0t1, t0t2)                                                          <L 564>
    wp::adj_sub(var_p, var_t0, adj_p, adj_t0, adj_2);
    // adj: t0p = p - t0                                                                      <L 563>
    wp::adj_sub(var_t2, var_t0, adj_t2, adj_t0, adj_1);
    // adj: t0t2 = t2 - t0                                                                    <L 562>
    wp::adj_sub(var_t1, var_t0, adj_t1, adj_t0, adj_0);
    // adj: t0t1 = t1 - t0                                                                    <L 561>
    // adj: def point_plane_distance(p: wp.vec3d, t0: wp.vec3d, t1: wp.vec3d, t2: wp.vec3d):  <L 556>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/point_triangle_distance.py:9
static CUDA_CALLABLE wp::float64 point_triangle_distance_0(
    wp::vec_t<3,wp::float64> var_p,
    wp::vec_t<3,wp::float64> var_t0,
    wp::vec_t<3,wp::float64> var_t1,
    wp::vec_t<3,wp::float64> var_t2)
{
    //---------
    // primal vars
    wp::int32 var_0;
    const wp::int32 var_1 = 0;
    bool var_2;
    wp::float64 var_3;
    const wp::int32 var_4 = 1;
    bool var_5;
    wp::float64 var_6;
    const wp::int32 var_7 = 2;
    bool var_8;
    wp::float64 var_9;
    const wp::int32 var_10 = 3;
    bool var_11;
    wp::float64 var_12;
    const wp::int32 var_13 = 4;
    bool var_14;
    wp::float64 var_15;
    const wp::int32 var_16 = 5;
    bool var_17;
    wp::float64 var_18;
    const wp::int32 var_19 = 6;
    bool var_20;
    wp::float64 var_21;
    const wp::float32 var_22 = 1e+20;
    wp::float64 var_23;
    //---------
    // forward
    // def point_triangle_distance(p: wp.vec3d, t0: wp.vec3d, t1: wp.vec3d, t2: wp.vec3d) -> wp.float64:       <L 10>
    // dt = point_triangle_distance_type(p, t0, t1, t2)                                       <L 11>
    var_0 = point_triangle_distance_type_0(var_p, var_t0, var_t1, var_t2);
    // if dt == 0:                                                                            <L 12>
    var_2 = (var_0 == var_1);
    if (var_2) {
        // return point_point_distance(p, t0)                                                 <L 13>
        var_3 = point_point_distance_0(var_p, var_t0);
        return var_3;
    }
    if (!var_2) {
        // elif dt == 1:                                                                      <L 14>
        var_5 = (var_0 == var_4);
        if (var_5) {
            // return point_point_distance(p, t1)                                             <L 15>
            var_6 = point_point_distance_0(var_p, var_t1);
            return var_6;
        }
        if (!var_5) {
            // elif dt == 2:                                                                  <L 16>
            var_8 = (var_0 == var_7);
            if (var_8) {
                // return point_point_distance(p, t2)                                         <L 17>
                var_9 = point_point_distance_0(var_p, var_t2);
                return var_9;
            }
            if (!var_8) {
                // elif dt == 3:                                                              <L 18>
                var_11 = (var_0 == var_10);
                if (var_11) {
                    // return point_line_distance(p, t0, t1)                                  <L 19>
                    var_12 = point_line_distance_0(var_p, var_t0, var_t1);
                    return var_12;
                }
                if (!var_11) {
                    // elif dt == 4:                                                          <L 20>
                    var_14 = (var_0 == var_13);
                    if (var_14) {
                        // return point_line_distance(p, t1, t2)                              <L 21>
                        var_15 = point_line_distance_0(var_p, var_t1, var_t2);
                        return var_15;
                    }
                    if (!var_14) {
                        // elif dt == 5:                                                      <L 22>
                        var_17 = (var_0 == var_16);
                        if (var_17) {
                            // return point_line_distance(p, t2, t0)                          <L 23>
                            var_18 = point_line_distance_0(var_p, var_t2, var_t0);
                            return var_18;
                        }
                        if (!var_17) {
                            // elif dt == 6:                                                  <L 24>
                            var_20 = (var_0 == var_19);
                            if (var_20) {
                                // return point_plane_distance(p, t0, t1, t2)                 <L 25>
                                var_21 = point_plane_distance_0(var_p, var_t0, var_t1, var_t2);
                                return var_21;
                            }
                            if (!var_20) {
                                // return wp.float64(1e20)                                    <L 27>
                                var_23 = wp::float64(var_22);
                                return var_23;
                            }
                        }
                    }
                }
            }
        }
    }
    return {};
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/point_triangle_distance.py:9
static CUDA_CALLABLE void adj_point_triangle_distance_0(
    wp::vec_t<3,wp::float64> var_p,
    wp::vec_t<3,wp::float64> var_t0,
    wp::vec_t<3,wp::float64> var_t1,
    wp::vec_t<3,wp::float64> var_t2,
    wp::vec_t<3,wp::float64> & adj_p,
    wp::vec_t<3,wp::float64> & adj_t0,
    wp::vec_t<3,wp::float64> & adj_t1,
    wp::vec_t<3,wp::float64> & adj_t2,
    wp::float64 & adj_ret)
{
    //---------
    // primal vars
    wp::int32 var_0;
    const wp::int32 var_1 = 0;
    bool var_2;
    wp::float64 var_3;
    const wp::int32 var_4 = 1;
    bool var_5;
    wp::float64 var_6;
    const wp::int32 var_7 = 2;
    bool var_8;
    wp::float64 var_9;
    const wp::int32 var_10 = 3;
    bool var_11;
    wp::float64 var_12;
    const wp::int32 var_13 = 4;
    bool var_14;
    wp::float64 var_15;
    const wp::int32 var_16 = 5;
    bool var_17;
    wp::float64 var_18;
    const wp::int32 var_19 = 6;
    bool var_20;
    wp::float64 var_21;
    const wp::float32 var_22 = 1e+20;
    wp::float64 var_23;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::int32 adj_1 = {};
    bool adj_2 = {};
    wp::float64 adj_3 = {};
    wp::int32 adj_4 = {};
    bool adj_5 = {};
    wp::float64 adj_6 = {};
    wp::int32 adj_7 = {};
    bool adj_8 = {};
    wp::float64 adj_9 = {};
    wp::int32 adj_10 = {};
    bool adj_11 = {};
    wp::float64 adj_12 = {};
    wp::int32 adj_13 = {};
    bool adj_14 = {};
    wp::float64 adj_15 = {};
    wp::int32 adj_16 = {};
    bool adj_17 = {};
    wp::float64 adj_18 = {};
    wp::int32 adj_19 = {};
    bool adj_20 = {};
    wp::float64 adj_21 = {};
    wp::float32 adj_22 = {};
    wp::float64 adj_23 = {};
    //---------
    // forward
    // def point_triangle_distance(p: wp.vec3d, t0: wp.vec3d, t1: wp.vec3d, t2: wp.vec3d) -> wp.float64:       <L 10>
    // dt = point_triangle_distance_type(p, t0, t1, t2)                                       <L 11>
    var_0 = point_triangle_distance_type_0(var_p, var_t0, var_t1, var_t2);
    // if dt == 0:                                                                            <L 12>
    var_2 = (var_0 == var_1);
    if (var_2) {
        // return point_point_distance(p, t0)                                                 <L 13>
        var_3 = point_point_distance_0(var_p, var_t0);
        goto label0;
    }
    if (!var_2) {
        // elif dt == 1:                                                                      <L 14>
        var_5 = (var_0 == var_4);
        if (var_5) {
            // return point_point_distance(p, t1)                                             <L 15>
            var_6 = point_point_distance_0(var_p, var_t1);
            goto label1;
        }
        if (!var_5) {
            // elif dt == 2:                                                                  <L 16>
            var_8 = (var_0 == var_7);
            if (var_8) {
                // return point_point_distance(p, t2)                                         <L 17>
                var_9 = point_point_distance_0(var_p, var_t2);
                goto label2;
            }
            if (!var_8) {
                // elif dt == 3:                                                              <L 18>
                var_11 = (var_0 == var_10);
                if (var_11) {
                    // return point_line_distance(p, t0, t1)                                  <L 19>
                    var_12 = point_line_distance_0(var_p, var_t0, var_t1);
                    goto label3;
                }
                if (!var_11) {
                    // elif dt == 4:                                                          <L 20>
                    var_14 = (var_0 == var_13);
                    if (var_14) {
                        // return point_line_distance(p, t1, t2)                              <L 21>
                        var_15 = point_line_distance_0(var_p, var_t1, var_t2);
                        goto label4;
                    }
                    if (!var_14) {
                        // elif dt == 5:                                                      <L 22>
                        var_17 = (var_0 == var_16);
                        if (var_17) {
                            // return point_line_distance(p, t2, t0)                          <L 23>
                            var_18 = point_line_distance_0(var_p, var_t2, var_t0);
                            goto label5;
                        }
                        if (!var_17) {
                            // elif dt == 6:                                                  <L 24>
                            var_20 = (var_0 == var_19);
                            if (var_20) {
                                // return point_plane_distance(p, t0, t1, t2)                 <L 25>
                                var_21 = point_plane_distance_0(var_p, var_t0, var_t1, var_t2);
                                goto label6;
                            }
                            if (!var_20) {
                                // return wp.float64(1e20)                                    <L 27>
                                var_23 = wp::float64(var_22);
                                goto label7;
                            }
                        }
                    }
                }
            }
        }
    }
    //---------
    // reverse
    if (!var_2) {
        if (!var_5) {
            if (!var_8) {
                if (!var_11) {
                    if (!var_14) {
                        if (!var_17) {
                            if (!var_20) {
                                label7:;
                                adj_23 += adj_ret;
                                wp::adj_float64(var_22, adj_22, adj_23);
                                // adj: return wp.float64(1e20)                               <L 27>
                            }
                            if (var_20) {
                                label6:;
                                adj_21 += adj_ret;
                                adj_point_plane_distance_0(var_p, var_t0, var_t1, var_t2, adj_p, adj_t0, adj_t1, adj_t2, adj_21);
                                // adj: return point_plane_distance(p, t0, t1, t2)            <L 25>
                            }
                            // adj: elif dt == 6:                                             <L 24>
                        }
                        if (var_17) {
                            label5:;
                            adj_18 += adj_ret;
                            adj_point_line_distance_0(var_p, var_t2, var_t0, adj_p, adj_t2, adj_t0, adj_18);
                            // adj: return point_line_distance(p, t2, t0)                     <L 23>
                        }
                        // adj: elif dt == 5:                                                 <L 22>
                    }
                    if (var_14) {
                        label4:;
                        adj_15 += adj_ret;
                        adj_point_line_distance_0(var_p, var_t1, var_t2, adj_p, adj_t1, adj_t2, adj_15);
                        // adj: return point_line_distance(p, t1, t2)                         <L 21>
                    }
                    // adj: elif dt == 4:                                                     <L 20>
                }
                if (var_11) {
                    label3:;
                    adj_12 += adj_ret;
                    adj_point_line_distance_0(var_p, var_t0, var_t1, adj_p, adj_t0, adj_t1, adj_12);
                    // adj: return point_line_distance(p, t0, t1)                             <L 19>
                }
                // adj: elif dt == 3:                                                         <L 18>
            }
            if (var_8) {
                label2:;
                adj_9 += adj_ret;
                adj_point_point_distance_0(var_p, var_t2, adj_p, adj_t2, adj_9);
                // adj: return point_point_distance(p, t2)                                    <L 17>
            }
            // adj: elif dt == 2:                                                             <L 16>
        }
        if (var_5) {
            label1:;
            adj_6 += adj_ret;
            adj_point_point_distance_0(var_p, var_t1, adj_p, adj_t1, adj_6);
            // adj: return point_point_distance(p, t1)                                        <L 15>
        }
        // adj: elif dt == 1:                                                                 <L 14>
    }
    if (var_2) {
        label0:;
        adj_3 += adj_ret;
        adj_point_point_distance_0(var_p, var_t0, adj_p, adj_t0, adj_3);
        // adj: return point_point_distance(p, t0)                                            <L 13>
    }
    // adj: if dt == 0:                                                                       <L 12>
    adj_point_triangle_distance_type_0(var_p, var_t0, var_t1, var_t2, adj_p, adj_t0, adj_t1, adj_t2, adj_0);
    // adj: dt = point_triangle_distance_type(p, t0, t1, t2)                                  <L 11>
    // adj: def point_triangle_distance(p: wp.vec3d, t0: wp.vec3d, t1: wp.vec3d, t2: wp.vec3d) -> wp.float64:  <L 10>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/point_point_distance.py:9
static CUDA_CALLABLE wp::vec_t<6,wp::float64> point_point_distance_gradient_0(
    wp::vec_t<3,wp::float64> var_a,
    wp::vec_t<3,wp::float64> var_b)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 2.0;
    wp::float64 var_1;
    wp::vec_t<3,wp::float64> var_2;
    wp::vec_t<3,wp::float64> var_3;
    const wp::int32 var_4 = 0;
    wp::float64 var_5;
    const wp::int32 var_6 = 1;
    wp::float64 var_7;
    const wp::int32 var_8 = 2;
    wp::float64 var_9;
    const wp::int32 var_10 = 0;
    wp::float64 var_11;
    wp::float64 var_12;
    const wp::int32 var_13 = 1;
    wp::float64 var_14;
    wp::float64 var_15;
    const wp::int32 var_16 = 2;
    wp::float64 var_17;
    wp::float64 var_18;
    wp::vec_t<6,wp::float64> var_19;
    //---------
    // forward
    // def point_point_distance_gradient(a: wp.vec3d, b: wp.vec3d):                           <L 10>
    // g0 = wp.float64(2.0) * (a - b)                                                         <L 11>
    var_1 = wp::float64(var_0);
    var_2 = wp::sub(var_a, var_b);
    var_3 = wp::mul(var_1, var_2);
    // grad = wp.vector(g0[0], g0[1], g0[2], -g0[0], -g0[1], -g0[2])                          <L 12>
    var_5 = wp::extract(var_3, var_4);
    var_7 = wp::extract(var_3, var_6);
    var_9 = wp::extract(var_3, var_8);
    var_11 = wp::extract(var_3, var_10);
    var_12 = wp::neg(var_11);
    var_14 = wp::extract(var_3, var_13);
    var_15 = wp::neg(var_14);
    var_17 = wp::extract(var_3, var_16);
    var_18 = wp::neg(var_17);
    var_19 = wp::vec_t<6,wp::float64>({var_5, var_7, var_9, var_12, var_15, var_18});
    // return grad                                                                            <L 13>
    return var_19;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/point_point_distance.py:9
static CUDA_CALLABLE void adj_point_point_distance_gradient_0(
    wp::vec_t<3,wp::float64> var_a,
    wp::vec_t<3,wp::float64> var_b,
    wp::vec_t<3,wp::float64> & adj_a,
    wp::vec_t<3,wp::float64> & adj_b,
    wp::vec_t<6,wp::float64> & adj_ret)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 2.0;
    wp::float64 var_1;
    wp::vec_t<3,wp::float64> var_2;
    wp::vec_t<3,wp::float64> var_3;
    const wp::int32 var_4 = 0;
    wp::float64 var_5;
    const wp::int32 var_6 = 1;
    wp::float64 var_7;
    const wp::int32 var_8 = 2;
    wp::float64 var_9;
    const wp::int32 var_10 = 0;
    wp::float64 var_11;
    wp::float64 var_12;
    const wp::int32 var_13 = 1;
    wp::float64 var_14;
    wp::float64 var_15;
    const wp::int32 var_16 = 2;
    wp::float64 var_17;
    wp::float64 var_18;
    wp::vec_t<6,wp::float64> var_19;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    wp::float64 adj_1 = {};
    wp::vec_t<3,wp::float64> adj_2 = {};
    wp::vec_t<3,wp::float64> adj_3 = {};
    wp::int32 adj_4 = {};
    wp::float64 adj_5 = {};
    wp::int32 adj_6 = {};
    wp::float64 adj_7 = {};
    wp::int32 adj_8 = {};
    wp::float64 adj_9 = {};
    wp::int32 adj_10 = {};
    wp::float64 adj_11 = {};
    wp::float64 adj_12 = {};
    wp::int32 adj_13 = {};
    wp::float64 adj_14 = {};
    wp::float64 adj_15 = {};
    wp::int32 adj_16 = {};
    wp::float64 adj_17 = {};
    wp::float64 adj_18 = {};
    wp::vec_t<6,wp::float64> adj_19 = {};
    //---------
    // forward
    // def point_point_distance_gradient(a: wp.vec3d, b: wp.vec3d):                           <L 10>
    // g0 = wp.float64(2.0) * (a - b)                                                         <L 11>
    var_1 = wp::float64(var_0);
    var_2 = wp::sub(var_a, var_b);
    var_3 = wp::mul(var_1, var_2);
    // grad = wp.vector(g0[0], g0[1], g0[2], -g0[0], -g0[1], -g0[2])                          <L 12>
    var_5 = wp::extract(var_3, var_4);
    var_7 = wp::extract(var_3, var_6);
    var_9 = wp::extract(var_3, var_8);
    var_11 = wp::extract(var_3, var_10);
    var_12 = wp::neg(var_11);
    var_14 = wp::extract(var_3, var_13);
    var_15 = wp::neg(var_14);
    var_17 = wp::extract(var_3, var_16);
    var_18 = wp::neg(var_17);
    var_19 = wp::vec_t<6,wp::float64>({var_5, var_7, var_9, var_12, var_15, var_18});
    // return grad                                                                            <L 13>
    goto label0;
    //---------
    // reverse
    label0:;
    adj_19 += adj_ret;
    // adj: return grad                                                                       <L 13>
    wp::adj_vec_t({var_5, var_7, var_9, var_12, var_15, var_18}, {&adj_5, &adj_7, &adj_9, &adj_12, &adj_15, &adj_18}, adj_19);
    wp::adj_neg(var_17, adj_17, adj_18);
    wp::adj_extract(var_3, var_16, adj_3, adj_16, adj_17);
    wp::adj_neg(var_14, adj_14, adj_15);
    wp::adj_extract(var_3, var_13, adj_3, adj_13, adj_14);
    wp::adj_neg(var_11, adj_11, adj_12);
    wp::adj_extract(var_3, var_10, adj_3, adj_10, adj_11);
    wp::adj_extract(var_3, var_8, adj_3, adj_8, adj_9);
    wp::adj_extract(var_3, var_6, adj_3, adj_6, adj_7);
    wp::adj_extract(var_3, var_4, adj_3, adj_4, adj_5);
    // adj: grad = wp.vector(g0[0], g0[1], g0[2], -g0[0], -g0[1], -g0[2])                     <L 12>
    wp::adj_mul(var_1, var_2, adj_1, adj_2, adj_3);
    wp::adj_sub(var_a, var_b, adj_a, adj_b, adj_2);
    wp::adj_float64(var_0, adj_0, adj_1);
    // adj: g0 = wp.float64(2.0) * (a - b)                                                    <L 11>
    // adj: def point_point_distance_gradient(a: wp.vec3d, b: wp.vec3d):                      <L 10>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/point_line_distance.py:4
static CUDA_CALLABLE wp::vec_t<9,wp::float64> g_PE3D_0(
    wp::float64 var_v01,
    wp::float64 var_v02,
    wp::float64 var_v03,
    wp::float64 var_v11,
    wp::float64 var_v12,
    wp::float64 var_v13,
    wp::float64 var_v21,
    wp::float64 var_v22,
    wp::float64 var_v23)
{
    //---------
    // primal vars
    wp::float64 var_0;
    wp::float64 var_1;
    wp::float64 var_2;
    wp::float64 var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    wp::float64 var_6;
    wp::float64 var_7;
    wp::float64 var_8;
    wp::float64 var_9;
    wp::float64 var_10;
    wp::float64 var_11;
    wp::float64 var_12;
    wp::float64 var_13;
    wp::float64 var_14;
    wp::float64 var_15;
    wp::float64 var_16;
    wp::float64 var_17;
    const wp::float32 var_18 = 1.0;
    wp::float64 var_19;
    wp::float64 var_20;
    wp::float64 var_21;
    wp::float64 var_22;
    wp::float64 var_23;
    wp::float64 var_24;
    wp::float64 var_25;
    wp::float64 var_26;
    wp::float64 var_27;
    wp::float64 var_28;
    wp::float64 var_29;
    wp::float64 var_30;
    wp::float64 var_31;
    wp::float64 var_32;
    wp::float64 var_33;
    wp::float64 var_34;
    wp::float64 var_35;
    wp::float64 var_36;
    wp::float64 var_37;
    wp::float64 var_38;
    wp::float64 var_39;
    wp::float64 var_40;
    wp::float64 var_41;
    wp::float64 var_42;
    wp::float64 var_43;
    const wp::float32 var_44 = 2.0;
    wp::float64 var_45;
    wp::float64 var_46;
    const wp::float32 var_47 = 2.0;
    wp::float64 var_48;
    wp::float64 var_49;
    wp::float64 var_50;
    wp::float64 var_51;
    wp::float64 var_52;
    wp::float64 var_53;
    const wp::float32 var_54 = 2.0;
    wp::float64 var_55;
    wp::float64 var_56;
    const wp::float32 var_57 = 2.0;
    wp::float64 var_58;
    wp::float64 var_59;
    wp::float64 var_60;
    wp::float64 var_61;
    wp::float64 var_62;
    wp::float64 var_63;
    const wp::float32 var_64 = 2.0;
    wp::float64 var_65;
    wp::float64 var_66;
    const wp::float32 var_67 = 2.0;
    wp::float64 var_68;
    wp::float64 var_69;
    wp::float64 var_70;
    wp::float64 var_71;
    wp::float64 var_72;
    wp::float64 var_73;
    wp::float64 var_74;
    const wp::float32 var_75 = 2.0;
    wp::float64 var_76;
    wp::float64 var_77;
    wp::float64 var_78;
    const wp::float32 var_79 = 2.0;
    wp::float64 var_80;
    wp::float64 var_81;
    wp::float64 var_82;
    wp::float64 var_83;
    wp::float64 var_84;
    wp::float64 var_85;
    const wp::float32 var_86 = 2.0;
    wp::float64 var_87;
    wp::float64 var_88;
    wp::float64 var_89;
    const wp::float32 var_90 = 2.0;
    wp::float64 var_91;
    wp::float64 var_92;
    wp::float64 var_93;
    wp::float64 var_94;
    wp::float64 var_95;
    wp::float64 var_96;
    const wp::float32 var_97 = 2.0;
    wp::float64 var_98;
    wp::float64 var_99;
    wp::float64 var_100;
    const wp::float32 var_101 = 2.0;
    wp::float64 var_102;
    wp::float64 var_103;
    wp::float64 var_104;
    wp::float64 var_105;
    wp::float64 var_106;
    wp::float64 var_107;
    const wp::float32 var_108 = 2.0;
    wp::float64 var_109;
    wp::float64 var_110;
    wp::float64 var_111;
    const wp::float32 var_112 = 2.0;
    wp::float64 var_113;
    wp::float64 var_114;
    wp::float64 var_115;
    wp::float64 var_116;
    wp::float64 var_117;
    wp::float64 var_118;
    wp::float64 var_119;
    const wp::float32 var_120 = 2.0;
    wp::float64 var_121;
    wp::float64 var_122;
    wp::float64 var_123;
    const wp::float32 var_124 = 2.0;
    wp::float64 var_125;
    wp::float64 var_126;
    wp::float64 var_127;
    wp::float64 var_128;
    wp::float64 var_129;
    wp::float64 var_130;
    wp::float64 var_131;
    const wp::float32 var_132 = 2.0;
    wp::float64 var_133;
    wp::float64 var_134;
    wp::float64 var_135;
    const wp::float32 var_136 = 2.0;
    wp::float64 var_137;
    wp::float64 var_138;
    wp::float64 var_139;
    wp::float64 var_140;
    wp::float64 var_141;
    wp::float64 var_142;
    const wp::float32 var_143 = 2.0;
    wp::float64 var_144;
    wp::float64 var_145;
    wp::float64 var_146;
    const wp::float32 var_147 = 2.0;
    wp::float64 var_148;
    wp::float64 var_149;
    wp::float64 var_150;
    wp::float64 var_151;
    wp::float64 var_152;
    wp::float64 var_153;
    const wp::float32 var_154 = 2.0;
    wp::float64 var_155;
    wp::float64 var_156;
    wp::float64 var_157;
    const wp::float32 var_158 = 2.0;
    wp::float64 var_159;
    wp::float64 var_160;
    wp::float64 var_161;
    wp::float64 var_162;
    wp::float64 var_163;
    wp::float64 var_164;
    const wp::float32 var_165 = 2.0;
    wp::float64 var_166;
    wp::float64 var_167;
    wp::float64 var_168;
    const wp::float32 var_169 = 2.0;
    wp::float64 var_170;
    wp::float64 var_171;
    wp::float64 var_172;
    wp::float64 var_173;
    wp::float64 var_174;
    wp::vec_t<9,wp::float64> var_175;
    //---------
    // forward
    // def g_PE3D(                                                                            <L 5>
    // t17 = -v11 + v01                                                                       <L 20>
    var_0 = wp::neg(var_v11);
    var_1 = wp::add(var_0, var_v01);
    // t18 = -v12 + v02                                                                       <L 21>
    var_2 = wp::neg(var_v12);
    var_3 = wp::add(var_2, var_v02);
    // t19 = -v13 + v03                                                                       <L 22>
    var_4 = wp::neg(var_v13);
    var_5 = wp::add(var_4, var_v03);
    // t20 = -v21 + v01                                                                       <L 23>
    var_6 = wp::neg(var_v21);
    var_7 = wp::add(var_6, var_v01);
    // t21 = -v22 + v02                                                                       <L 24>
    var_8 = wp::neg(var_v22);
    var_9 = wp::add(var_8, var_v02);
    // t22 = -v23 + v03                                                                       <L 25>
    var_10 = wp::neg(var_v23);
    var_11 = wp::add(var_10, var_v03);
    // t23 = -v21 + v11                                                                       <L 26>
    var_12 = wp::neg(var_v21);
    var_13 = wp::add(var_12, var_v11);
    // t24 = -v22 + v12                                                                       <L 27>
    var_14 = wp::neg(var_v22);
    var_15 = wp::add(var_14, var_v12);
    // t25 = -v23 + v13                                                                       <L 28>
    var_16 = wp::neg(var_v23);
    var_17 = wp::add(var_16, var_v13);
    // t42 = wp.float64(1.0) / ((t23 * t23 + t24 * t24) + t25 * t25)                          <L 29>
    var_19 = wp::float64(var_18);
    var_20 = wp::mul(var_13, var_13);
    var_21 = wp::mul(var_15, var_15);
    var_22 = wp::add(var_20, var_21);
    var_23 = wp::mul(var_17, var_17);
    var_24 = wp::add(var_22, var_23);
    var_25 = wp::div(var_19, var_24);
    // t44 = t17 * t21 + -(t18 * t20)                                                         <L 30>
    var_26 = wp::mul(var_1, var_9);
    var_27 = wp::mul(var_3, var_7);
    var_28 = wp::neg(var_27);
    var_29 = wp::add(var_26, var_28);
    // t45 = t17 * t22 + -(t19 * t20)                                                         <L 31>
    var_30 = wp::mul(var_1, var_11);
    var_31 = wp::mul(var_5, var_7);
    var_32 = wp::neg(var_31);
    var_33 = wp::add(var_30, var_32);
    // t46 = t18 * t22 + -(t19 * t21)                                                         <L 32>
    var_34 = wp::mul(var_3, var_11);
    var_35 = wp::mul(var_5, var_9);
    var_36 = wp::neg(var_35);
    var_37 = wp::add(var_34, var_36);
    // t43 = t42 * t42                                                                        <L 33>
    var_38 = wp::mul(var_25, var_25);
    // t50 = (t44 * t44 + t45 * t45) + t46 * t46                                              <L 34>
    var_39 = wp::mul(var_29, var_29);
    var_40 = wp::mul(var_33, var_33);
    var_41 = wp::add(var_39, var_40);
    var_42 = wp::mul(var_37, var_37);
    var_43 = wp::add(var_41, var_42);
    // t51 = (v11 * wp.float64(2.0) + -(v21 * wp.float64(2.0))) * t43 * t50                   <L 35>
    var_45 = wp::float64(var_44);
    var_46 = wp::mul(var_v11, var_45);
    var_48 = wp::float64(var_47);
    var_49 = wp::mul(var_v21, var_48);
    var_50 = wp::neg(var_49);
    var_51 = wp::add(var_46, var_50);
    var_52 = wp::mul(var_51, var_38);
    var_53 = wp::mul(var_52, var_43);
    // t52 = (v12 * wp.float64(2.0) + -(v22 * wp.float64(2.0))) * t43 * t50                   <L 36>
    var_55 = wp::float64(var_54);
    var_56 = wp::mul(var_v12, var_55);
    var_58 = wp::float64(var_57);
    var_59 = wp::mul(var_v22, var_58);
    var_60 = wp::neg(var_59);
    var_61 = wp::add(var_56, var_60);
    var_62 = wp::mul(var_61, var_38);
    var_63 = wp::mul(var_62, var_43);
    // t43 = (v13 * wp.float64(2.0) + -(v23 * wp.float64(2.0))) * t43 * t50                   <L 37>
    var_65 = wp::float64(var_64);
    var_66 = wp::mul(var_v13, var_65);
    var_68 = wp::float64(var_67);
    var_69 = wp::mul(var_v23, var_68);
    var_70 = wp::neg(var_69);
    var_71 = wp::add(var_66, var_70);
    var_72 = wp::mul(var_71, var_38);
    var_73 = wp::mul(var_72, var_43);
    // g = wp.vector(                                                                         <L 38>
    // t42 * (t24 * t44 * wp.float64(2.0) + t25 * t45 * wp.float64(2.0)),                     <L 39>
    var_74 = wp::mul(var_15, var_29);
    var_76 = wp::float64(var_75);
    var_77 = wp::mul(var_74, var_76);
    var_78 = wp::mul(var_17, var_33);
    var_80 = wp::float64(var_79);
    var_81 = wp::mul(var_78, var_80);
    var_82 = wp::add(var_77, var_81);
    var_83 = wp::mul(var_25, var_82);
    // -t42 * (t23 * t44 * wp.float64(2.0) - t25 * t46 * wp.float64(2.0)),                    <L 40>
    var_84 = wp::neg(var_25);
    var_85 = wp::mul(var_13, var_29);
    var_87 = wp::float64(var_86);
    var_88 = wp::mul(var_85, var_87);
    var_89 = wp::mul(var_17, var_37);
    var_91 = wp::float64(var_90);
    var_92 = wp::mul(var_89, var_91);
    var_93 = wp::sub(var_88, var_92);
    var_94 = wp::mul(var_84, var_93);
    // -t42 * (t23 * t45 * wp.float64(2.0) + t24 * t46 * wp.float64(2.0)),                    <L 41>
    var_95 = wp::neg(var_25);
    var_96 = wp::mul(var_13, var_33);
    var_98 = wp::float64(var_97);
    var_99 = wp::mul(var_96, var_98);
    var_100 = wp::mul(var_15, var_37);
    var_102 = wp::float64(var_101);
    var_103 = wp::mul(var_100, var_102);
    var_104 = wp::add(var_99, var_103);
    var_105 = wp::mul(var_95, var_104);
    // -t51 - t42 * (t21 * t44 * wp.float64(2.0) + t22 * t45 * wp.float64(2.0)),              <L 42>
    var_106 = wp::neg(var_53);
    var_107 = wp::mul(var_9, var_29);
    var_109 = wp::float64(var_108);
    var_110 = wp::mul(var_107, var_109);
    var_111 = wp::mul(var_11, var_33);
    var_113 = wp::float64(var_112);
    var_114 = wp::mul(var_111, var_113);
    var_115 = wp::add(var_110, var_114);
    var_116 = wp::mul(var_25, var_115);
    var_117 = wp::sub(var_106, var_116);
    // -t52 + t42 * (t20 * t44 * wp.float64(2.0) - t22 * t46 * wp.float64(2.0)),              <L 43>
    var_118 = wp::neg(var_63);
    var_119 = wp::mul(var_7, var_29);
    var_121 = wp::float64(var_120);
    var_122 = wp::mul(var_119, var_121);
    var_123 = wp::mul(var_11, var_37);
    var_125 = wp::float64(var_124);
    var_126 = wp::mul(var_123, var_125);
    var_127 = wp::sub(var_122, var_126);
    var_128 = wp::mul(var_25, var_127);
    var_129 = wp::add(var_118, var_128);
    // -t43 + t42 * (t20 * t45 * wp.float64(2.0) + t21 * t46 * wp.float64(2.0)),              <L 44>
    var_130 = wp::neg(var_73);
    var_131 = wp::mul(var_7, var_33);
    var_133 = wp::float64(var_132);
    var_134 = wp::mul(var_131, var_133);
    var_135 = wp::mul(var_9, var_37);
    var_137 = wp::float64(var_136);
    var_138 = wp::mul(var_135, var_137);
    var_139 = wp::add(var_134, var_138);
    var_140 = wp::mul(var_25, var_139);
    var_141 = wp::add(var_130, var_140);
    // t51 + t42 * (t18 * t44 * wp.float64(2.0) + t19 * t45 * wp.float64(2.0)),               <L 45>
    var_142 = wp::mul(var_3, var_29);
    var_144 = wp::float64(var_143);
    var_145 = wp::mul(var_142, var_144);
    var_146 = wp::mul(var_5, var_33);
    var_148 = wp::float64(var_147);
    var_149 = wp::mul(var_146, var_148);
    var_150 = wp::add(var_145, var_149);
    var_151 = wp::mul(var_25, var_150);
    var_152 = wp::add(var_53, var_151);
    // t52 - t42 * (t17 * t44 * wp.float64(2.0) - t19 * t46 * wp.float64(2.0)),               <L 46>
    var_153 = wp::mul(var_1, var_29);
    var_155 = wp::float64(var_154);
    var_156 = wp::mul(var_153, var_155);
    var_157 = wp::mul(var_5, var_37);
    var_159 = wp::float64(var_158);
    var_160 = wp::mul(var_157, var_159);
    var_161 = wp::sub(var_156, var_160);
    var_162 = wp::mul(var_25, var_161);
    var_163 = wp::sub(var_63, var_162);
    // t43 - t42 * (t17 * t45 * wp.float64(2.0) + t18 * t46 * wp.float64(2.0)),               <L 47>
    var_164 = wp::mul(var_1, var_33);
    var_166 = wp::float64(var_165);
    var_167 = wp::mul(var_164, var_166);
    var_168 = wp::mul(var_3, var_37);
    var_170 = wp::float64(var_169);
    var_171 = wp::mul(var_168, var_170);
    var_172 = wp::add(var_167, var_171);
    var_173 = wp::mul(var_25, var_172);
    var_174 = wp::sub(var_73, var_173);
    var_175 = wp::vec_t<9,wp::float64>({var_83, var_94, var_105, var_117, var_129, var_141, var_152, var_163, var_174});
    // return g                                                                               <L 49>
    return var_175;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/point_line_distance.py:4
static CUDA_CALLABLE void adj_g_PE3D_0(
    wp::float64 var_v01,
    wp::float64 var_v02,
    wp::float64 var_v03,
    wp::float64 var_v11,
    wp::float64 var_v12,
    wp::float64 var_v13,
    wp::float64 var_v21,
    wp::float64 var_v22,
    wp::float64 var_v23,
    wp::float64 & adj_v01,
    wp::float64 & adj_v02,
    wp::float64 & adj_v03,
    wp::float64 & adj_v11,
    wp::float64 & adj_v12,
    wp::float64 & adj_v13,
    wp::float64 & adj_v21,
    wp::float64 & adj_v22,
    wp::float64 & adj_v23,
    wp::vec_t<9,wp::float64> & adj_ret)
{
    //---------
    // primal vars
    wp::float64 var_0;
    wp::float64 var_1;
    wp::float64 var_2;
    wp::float64 var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    wp::float64 var_6;
    wp::float64 var_7;
    wp::float64 var_8;
    wp::float64 var_9;
    wp::float64 var_10;
    wp::float64 var_11;
    wp::float64 var_12;
    wp::float64 var_13;
    wp::float64 var_14;
    wp::float64 var_15;
    wp::float64 var_16;
    wp::float64 var_17;
    const wp::float32 var_18 = 1.0;
    wp::float64 var_19;
    wp::float64 var_20;
    wp::float64 var_21;
    wp::float64 var_22;
    wp::float64 var_23;
    wp::float64 var_24;
    wp::float64 var_25;
    wp::float64 var_26;
    wp::float64 var_27;
    wp::float64 var_28;
    wp::float64 var_29;
    wp::float64 var_30;
    wp::float64 var_31;
    wp::float64 var_32;
    wp::float64 var_33;
    wp::float64 var_34;
    wp::float64 var_35;
    wp::float64 var_36;
    wp::float64 var_37;
    wp::float64 var_38;
    wp::float64 var_39;
    wp::float64 var_40;
    wp::float64 var_41;
    wp::float64 var_42;
    wp::float64 var_43;
    const wp::float32 var_44 = 2.0;
    wp::float64 var_45;
    wp::float64 var_46;
    const wp::float32 var_47 = 2.0;
    wp::float64 var_48;
    wp::float64 var_49;
    wp::float64 var_50;
    wp::float64 var_51;
    wp::float64 var_52;
    wp::float64 var_53;
    const wp::float32 var_54 = 2.0;
    wp::float64 var_55;
    wp::float64 var_56;
    const wp::float32 var_57 = 2.0;
    wp::float64 var_58;
    wp::float64 var_59;
    wp::float64 var_60;
    wp::float64 var_61;
    wp::float64 var_62;
    wp::float64 var_63;
    const wp::float32 var_64 = 2.0;
    wp::float64 var_65;
    wp::float64 var_66;
    const wp::float32 var_67 = 2.0;
    wp::float64 var_68;
    wp::float64 var_69;
    wp::float64 var_70;
    wp::float64 var_71;
    wp::float64 var_72;
    wp::float64 var_73;
    wp::float64 var_74;
    const wp::float32 var_75 = 2.0;
    wp::float64 var_76;
    wp::float64 var_77;
    wp::float64 var_78;
    const wp::float32 var_79 = 2.0;
    wp::float64 var_80;
    wp::float64 var_81;
    wp::float64 var_82;
    wp::float64 var_83;
    wp::float64 var_84;
    wp::float64 var_85;
    const wp::float32 var_86 = 2.0;
    wp::float64 var_87;
    wp::float64 var_88;
    wp::float64 var_89;
    const wp::float32 var_90 = 2.0;
    wp::float64 var_91;
    wp::float64 var_92;
    wp::float64 var_93;
    wp::float64 var_94;
    wp::float64 var_95;
    wp::float64 var_96;
    const wp::float32 var_97 = 2.0;
    wp::float64 var_98;
    wp::float64 var_99;
    wp::float64 var_100;
    const wp::float32 var_101 = 2.0;
    wp::float64 var_102;
    wp::float64 var_103;
    wp::float64 var_104;
    wp::float64 var_105;
    wp::float64 var_106;
    wp::float64 var_107;
    const wp::float32 var_108 = 2.0;
    wp::float64 var_109;
    wp::float64 var_110;
    wp::float64 var_111;
    const wp::float32 var_112 = 2.0;
    wp::float64 var_113;
    wp::float64 var_114;
    wp::float64 var_115;
    wp::float64 var_116;
    wp::float64 var_117;
    wp::float64 var_118;
    wp::float64 var_119;
    const wp::float32 var_120 = 2.0;
    wp::float64 var_121;
    wp::float64 var_122;
    wp::float64 var_123;
    const wp::float32 var_124 = 2.0;
    wp::float64 var_125;
    wp::float64 var_126;
    wp::float64 var_127;
    wp::float64 var_128;
    wp::float64 var_129;
    wp::float64 var_130;
    wp::float64 var_131;
    const wp::float32 var_132 = 2.0;
    wp::float64 var_133;
    wp::float64 var_134;
    wp::float64 var_135;
    const wp::float32 var_136 = 2.0;
    wp::float64 var_137;
    wp::float64 var_138;
    wp::float64 var_139;
    wp::float64 var_140;
    wp::float64 var_141;
    wp::float64 var_142;
    const wp::float32 var_143 = 2.0;
    wp::float64 var_144;
    wp::float64 var_145;
    wp::float64 var_146;
    const wp::float32 var_147 = 2.0;
    wp::float64 var_148;
    wp::float64 var_149;
    wp::float64 var_150;
    wp::float64 var_151;
    wp::float64 var_152;
    wp::float64 var_153;
    const wp::float32 var_154 = 2.0;
    wp::float64 var_155;
    wp::float64 var_156;
    wp::float64 var_157;
    const wp::float32 var_158 = 2.0;
    wp::float64 var_159;
    wp::float64 var_160;
    wp::float64 var_161;
    wp::float64 var_162;
    wp::float64 var_163;
    wp::float64 var_164;
    const wp::float32 var_165 = 2.0;
    wp::float64 var_166;
    wp::float64 var_167;
    wp::float64 var_168;
    const wp::float32 var_169 = 2.0;
    wp::float64 var_170;
    wp::float64 var_171;
    wp::float64 var_172;
    wp::float64 var_173;
    wp::float64 var_174;
    wp::vec_t<9,wp::float64> var_175;
    //---------
    // dual vars
    wp::float64 adj_0 = {};
    wp::float64 adj_1 = {};
    wp::float64 adj_2 = {};
    wp::float64 adj_3 = {};
    wp::float64 adj_4 = {};
    wp::float64 adj_5 = {};
    wp::float64 adj_6 = {};
    wp::float64 adj_7 = {};
    wp::float64 adj_8 = {};
    wp::float64 adj_9 = {};
    wp::float64 adj_10 = {};
    wp::float64 adj_11 = {};
    wp::float64 adj_12 = {};
    wp::float64 adj_13 = {};
    wp::float64 adj_14 = {};
    wp::float64 adj_15 = {};
    wp::float64 adj_16 = {};
    wp::float64 adj_17 = {};
    wp::float32 adj_18 = {};
    wp::float64 adj_19 = {};
    wp::float64 adj_20 = {};
    wp::float64 adj_21 = {};
    wp::float64 adj_22 = {};
    wp::float64 adj_23 = {};
    wp::float64 adj_24 = {};
    wp::float64 adj_25 = {};
    wp::float64 adj_26 = {};
    wp::float64 adj_27 = {};
    wp::float64 adj_28 = {};
    wp::float64 adj_29 = {};
    wp::float64 adj_30 = {};
    wp::float64 adj_31 = {};
    wp::float64 adj_32 = {};
    wp::float64 adj_33 = {};
    wp::float64 adj_34 = {};
    wp::float64 adj_35 = {};
    wp::float64 adj_36 = {};
    wp::float64 adj_37 = {};
    wp::float64 adj_38 = {};
    wp::float64 adj_39 = {};
    wp::float64 adj_40 = {};
    wp::float64 adj_41 = {};
    wp::float64 adj_42 = {};
    wp::float64 adj_43 = {};
    wp::float32 adj_44 = {};
    wp::float64 adj_45 = {};
    wp::float64 adj_46 = {};
    wp::float32 adj_47 = {};
    wp::float64 adj_48 = {};
    wp::float64 adj_49 = {};
    wp::float64 adj_50 = {};
    wp::float64 adj_51 = {};
    wp::float64 adj_52 = {};
    wp::float64 adj_53 = {};
    wp::float32 adj_54 = {};
    wp::float64 adj_55 = {};
    wp::float64 adj_56 = {};
    wp::float32 adj_57 = {};
    wp::float64 adj_58 = {};
    wp::float64 adj_59 = {};
    wp::float64 adj_60 = {};
    wp::float64 adj_61 = {};
    wp::float64 adj_62 = {};
    wp::float64 adj_63 = {};
    wp::float32 adj_64 = {};
    wp::float64 adj_65 = {};
    wp::float64 adj_66 = {};
    wp::float32 adj_67 = {};
    wp::float64 adj_68 = {};
    wp::float64 adj_69 = {};
    wp::float64 adj_70 = {};
    wp::float64 adj_71 = {};
    wp::float64 adj_72 = {};
    wp::float64 adj_73 = {};
    wp::float64 adj_74 = {};
    wp::float32 adj_75 = {};
    wp::float64 adj_76 = {};
    wp::float64 adj_77 = {};
    wp::float64 adj_78 = {};
    wp::float32 adj_79 = {};
    wp::float64 adj_80 = {};
    wp::float64 adj_81 = {};
    wp::float64 adj_82 = {};
    wp::float64 adj_83 = {};
    wp::float64 adj_84 = {};
    wp::float64 adj_85 = {};
    wp::float32 adj_86 = {};
    wp::float64 adj_87 = {};
    wp::float64 adj_88 = {};
    wp::float64 adj_89 = {};
    wp::float32 adj_90 = {};
    wp::float64 adj_91 = {};
    wp::float64 adj_92 = {};
    wp::float64 adj_93 = {};
    wp::float64 adj_94 = {};
    wp::float64 adj_95 = {};
    wp::float64 adj_96 = {};
    wp::float32 adj_97 = {};
    wp::float64 adj_98 = {};
    wp::float64 adj_99 = {};
    wp::float64 adj_100 = {};
    wp::float32 adj_101 = {};
    wp::float64 adj_102 = {};
    wp::float64 adj_103 = {};
    wp::float64 adj_104 = {};
    wp::float64 adj_105 = {};
    wp::float64 adj_106 = {};
    wp::float64 adj_107 = {};
    wp::float32 adj_108 = {};
    wp::float64 adj_109 = {};
    wp::float64 adj_110 = {};
    wp::float64 adj_111 = {};
    wp::float32 adj_112 = {};
    wp::float64 adj_113 = {};
    wp::float64 adj_114 = {};
    wp::float64 adj_115 = {};
    wp::float64 adj_116 = {};
    wp::float64 adj_117 = {};
    wp::float64 adj_118 = {};
    wp::float64 adj_119 = {};
    wp::float32 adj_120 = {};
    wp::float64 adj_121 = {};
    wp::float64 adj_122 = {};
    wp::float64 adj_123 = {};
    wp::float32 adj_124 = {};
    wp::float64 adj_125 = {};
    wp::float64 adj_126 = {};
    wp::float64 adj_127 = {};
    wp::float64 adj_128 = {};
    wp::float64 adj_129 = {};
    wp::float64 adj_130 = {};
    wp::float64 adj_131 = {};
    wp::float32 adj_132 = {};
    wp::float64 adj_133 = {};
    wp::float64 adj_134 = {};
    wp::float64 adj_135 = {};
    wp::float32 adj_136 = {};
    wp::float64 adj_137 = {};
    wp::float64 adj_138 = {};
    wp::float64 adj_139 = {};
    wp::float64 adj_140 = {};
    wp::float64 adj_141 = {};
    wp::float64 adj_142 = {};
    wp::float32 adj_143 = {};
    wp::float64 adj_144 = {};
    wp::float64 adj_145 = {};
    wp::float64 adj_146 = {};
    wp::float32 adj_147 = {};
    wp::float64 adj_148 = {};
    wp::float64 adj_149 = {};
    wp::float64 adj_150 = {};
    wp::float64 adj_151 = {};
    wp::float64 adj_152 = {};
    wp::float64 adj_153 = {};
    wp::float32 adj_154 = {};
    wp::float64 adj_155 = {};
    wp::float64 adj_156 = {};
    wp::float64 adj_157 = {};
    wp::float32 adj_158 = {};
    wp::float64 adj_159 = {};
    wp::float64 adj_160 = {};
    wp::float64 adj_161 = {};
    wp::float64 adj_162 = {};
    wp::float64 adj_163 = {};
    wp::float64 adj_164 = {};
    wp::float32 adj_165 = {};
    wp::float64 adj_166 = {};
    wp::float64 adj_167 = {};
    wp::float64 adj_168 = {};
    wp::float32 adj_169 = {};
    wp::float64 adj_170 = {};
    wp::float64 adj_171 = {};
    wp::float64 adj_172 = {};
    wp::float64 adj_173 = {};
    wp::float64 adj_174 = {};
    wp::vec_t<9,wp::float64> adj_175 = {};
    //---------
    // forward
    // def g_PE3D(                                                                            <L 5>
    // t17 = -v11 + v01                                                                       <L 20>
    var_0 = wp::neg(var_v11);
    var_1 = wp::add(var_0, var_v01);
    // t18 = -v12 + v02                                                                       <L 21>
    var_2 = wp::neg(var_v12);
    var_3 = wp::add(var_2, var_v02);
    // t19 = -v13 + v03                                                                       <L 22>
    var_4 = wp::neg(var_v13);
    var_5 = wp::add(var_4, var_v03);
    // t20 = -v21 + v01                                                                       <L 23>
    var_6 = wp::neg(var_v21);
    var_7 = wp::add(var_6, var_v01);
    // t21 = -v22 + v02                                                                       <L 24>
    var_8 = wp::neg(var_v22);
    var_9 = wp::add(var_8, var_v02);
    // t22 = -v23 + v03                                                                       <L 25>
    var_10 = wp::neg(var_v23);
    var_11 = wp::add(var_10, var_v03);
    // t23 = -v21 + v11                                                                       <L 26>
    var_12 = wp::neg(var_v21);
    var_13 = wp::add(var_12, var_v11);
    // t24 = -v22 + v12                                                                       <L 27>
    var_14 = wp::neg(var_v22);
    var_15 = wp::add(var_14, var_v12);
    // t25 = -v23 + v13                                                                       <L 28>
    var_16 = wp::neg(var_v23);
    var_17 = wp::add(var_16, var_v13);
    // t42 = wp.float64(1.0) / ((t23 * t23 + t24 * t24) + t25 * t25)                          <L 29>
    var_19 = wp::float64(var_18);
    var_20 = wp::mul(var_13, var_13);
    var_21 = wp::mul(var_15, var_15);
    var_22 = wp::add(var_20, var_21);
    var_23 = wp::mul(var_17, var_17);
    var_24 = wp::add(var_22, var_23);
    var_25 = wp::div(var_19, var_24);
    // t44 = t17 * t21 + -(t18 * t20)                                                         <L 30>
    var_26 = wp::mul(var_1, var_9);
    var_27 = wp::mul(var_3, var_7);
    var_28 = wp::neg(var_27);
    var_29 = wp::add(var_26, var_28);
    // t45 = t17 * t22 + -(t19 * t20)                                                         <L 31>
    var_30 = wp::mul(var_1, var_11);
    var_31 = wp::mul(var_5, var_7);
    var_32 = wp::neg(var_31);
    var_33 = wp::add(var_30, var_32);
    // t46 = t18 * t22 + -(t19 * t21)                                                         <L 32>
    var_34 = wp::mul(var_3, var_11);
    var_35 = wp::mul(var_5, var_9);
    var_36 = wp::neg(var_35);
    var_37 = wp::add(var_34, var_36);
    // t43 = t42 * t42                                                                        <L 33>
    var_38 = wp::mul(var_25, var_25);
    // t50 = (t44 * t44 + t45 * t45) + t46 * t46                                              <L 34>
    var_39 = wp::mul(var_29, var_29);
    var_40 = wp::mul(var_33, var_33);
    var_41 = wp::add(var_39, var_40);
    var_42 = wp::mul(var_37, var_37);
    var_43 = wp::add(var_41, var_42);
    // t51 = (v11 * wp.float64(2.0) + -(v21 * wp.float64(2.0))) * t43 * t50                   <L 35>
    var_45 = wp::float64(var_44);
    var_46 = wp::mul(var_v11, var_45);
    var_48 = wp::float64(var_47);
    var_49 = wp::mul(var_v21, var_48);
    var_50 = wp::neg(var_49);
    var_51 = wp::add(var_46, var_50);
    var_52 = wp::mul(var_51, var_38);
    var_53 = wp::mul(var_52, var_43);
    // t52 = (v12 * wp.float64(2.0) + -(v22 * wp.float64(2.0))) * t43 * t50                   <L 36>
    var_55 = wp::float64(var_54);
    var_56 = wp::mul(var_v12, var_55);
    var_58 = wp::float64(var_57);
    var_59 = wp::mul(var_v22, var_58);
    var_60 = wp::neg(var_59);
    var_61 = wp::add(var_56, var_60);
    var_62 = wp::mul(var_61, var_38);
    var_63 = wp::mul(var_62, var_43);
    // t43 = (v13 * wp.float64(2.0) + -(v23 * wp.float64(2.0))) * t43 * t50                   <L 37>
    var_65 = wp::float64(var_64);
    var_66 = wp::mul(var_v13, var_65);
    var_68 = wp::float64(var_67);
    var_69 = wp::mul(var_v23, var_68);
    var_70 = wp::neg(var_69);
    var_71 = wp::add(var_66, var_70);
    var_72 = wp::mul(var_71, var_38);
    var_73 = wp::mul(var_72, var_43);
    // g = wp.vector(                                                                         <L 38>
    // t42 * (t24 * t44 * wp.float64(2.0) + t25 * t45 * wp.float64(2.0)),                     <L 39>
    var_74 = wp::mul(var_15, var_29);
    var_76 = wp::float64(var_75);
    var_77 = wp::mul(var_74, var_76);
    var_78 = wp::mul(var_17, var_33);
    var_80 = wp::float64(var_79);
    var_81 = wp::mul(var_78, var_80);
    var_82 = wp::add(var_77, var_81);
    var_83 = wp::mul(var_25, var_82);
    // -t42 * (t23 * t44 * wp.float64(2.0) - t25 * t46 * wp.float64(2.0)),                    <L 40>
    var_84 = wp::neg(var_25);
    var_85 = wp::mul(var_13, var_29);
    var_87 = wp::float64(var_86);
    var_88 = wp::mul(var_85, var_87);
    var_89 = wp::mul(var_17, var_37);
    var_91 = wp::float64(var_90);
    var_92 = wp::mul(var_89, var_91);
    var_93 = wp::sub(var_88, var_92);
    var_94 = wp::mul(var_84, var_93);
    // -t42 * (t23 * t45 * wp.float64(2.0) + t24 * t46 * wp.float64(2.0)),                    <L 41>
    var_95 = wp::neg(var_25);
    var_96 = wp::mul(var_13, var_33);
    var_98 = wp::float64(var_97);
    var_99 = wp::mul(var_96, var_98);
    var_100 = wp::mul(var_15, var_37);
    var_102 = wp::float64(var_101);
    var_103 = wp::mul(var_100, var_102);
    var_104 = wp::add(var_99, var_103);
    var_105 = wp::mul(var_95, var_104);
    // -t51 - t42 * (t21 * t44 * wp.float64(2.0) + t22 * t45 * wp.float64(2.0)),              <L 42>
    var_106 = wp::neg(var_53);
    var_107 = wp::mul(var_9, var_29);
    var_109 = wp::float64(var_108);
    var_110 = wp::mul(var_107, var_109);
    var_111 = wp::mul(var_11, var_33);
    var_113 = wp::float64(var_112);
    var_114 = wp::mul(var_111, var_113);
    var_115 = wp::add(var_110, var_114);
    var_116 = wp::mul(var_25, var_115);
    var_117 = wp::sub(var_106, var_116);
    // -t52 + t42 * (t20 * t44 * wp.float64(2.0) - t22 * t46 * wp.float64(2.0)),              <L 43>
    var_118 = wp::neg(var_63);
    var_119 = wp::mul(var_7, var_29);
    var_121 = wp::float64(var_120);
    var_122 = wp::mul(var_119, var_121);
    var_123 = wp::mul(var_11, var_37);
    var_125 = wp::float64(var_124);
    var_126 = wp::mul(var_123, var_125);
    var_127 = wp::sub(var_122, var_126);
    var_128 = wp::mul(var_25, var_127);
    var_129 = wp::add(var_118, var_128);
    // -t43 + t42 * (t20 * t45 * wp.float64(2.0) + t21 * t46 * wp.float64(2.0)),              <L 44>
    var_130 = wp::neg(var_73);
    var_131 = wp::mul(var_7, var_33);
    var_133 = wp::float64(var_132);
    var_134 = wp::mul(var_131, var_133);
    var_135 = wp::mul(var_9, var_37);
    var_137 = wp::float64(var_136);
    var_138 = wp::mul(var_135, var_137);
    var_139 = wp::add(var_134, var_138);
    var_140 = wp::mul(var_25, var_139);
    var_141 = wp::add(var_130, var_140);
    // t51 + t42 * (t18 * t44 * wp.float64(2.0) + t19 * t45 * wp.float64(2.0)),               <L 45>
    var_142 = wp::mul(var_3, var_29);
    var_144 = wp::float64(var_143);
    var_145 = wp::mul(var_142, var_144);
    var_146 = wp::mul(var_5, var_33);
    var_148 = wp::float64(var_147);
    var_149 = wp::mul(var_146, var_148);
    var_150 = wp::add(var_145, var_149);
    var_151 = wp::mul(var_25, var_150);
    var_152 = wp::add(var_53, var_151);
    // t52 - t42 * (t17 * t44 * wp.float64(2.0) - t19 * t46 * wp.float64(2.0)),               <L 46>
    var_153 = wp::mul(var_1, var_29);
    var_155 = wp::float64(var_154);
    var_156 = wp::mul(var_153, var_155);
    var_157 = wp::mul(var_5, var_37);
    var_159 = wp::float64(var_158);
    var_160 = wp::mul(var_157, var_159);
    var_161 = wp::sub(var_156, var_160);
    var_162 = wp::mul(var_25, var_161);
    var_163 = wp::sub(var_63, var_162);
    // t43 - t42 * (t17 * t45 * wp.float64(2.0) + t18 * t46 * wp.float64(2.0)),               <L 47>
    var_164 = wp::mul(var_1, var_33);
    var_166 = wp::float64(var_165);
    var_167 = wp::mul(var_164, var_166);
    var_168 = wp::mul(var_3, var_37);
    var_170 = wp::float64(var_169);
    var_171 = wp::mul(var_168, var_170);
    var_172 = wp::add(var_167, var_171);
    var_173 = wp::mul(var_25, var_172);
    var_174 = wp::sub(var_73, var_173);
    var_175 = wp::vec_t<9,wp::float64>({var_83, var_94, var_105, var_117, var_129, var_141, var_152, var_163, var_174});
    // return g                                                                               <L 49>
    goto label0;
    //---------
    // reverse
    label0:;
    adj_175 += adj_ret;
    // adj: return g                                                                          <L 49>
    wp::adj_vec_t({var_83, var_94, var_105, var_117, var_129, var_141, var_152, var_163, var_174}, {&adj_83, &adj_94, &adj_105, &adj_117, &adj_129, &adj_141, &adj_152, &adj_163, &adj_174}, adj_175);
    wp::adj_sub(var_73, var_173, adj_73, adj_173, adj_174);
    wp::adj_mul(var_25, var_172, adj_25, adj_172, adj_173);
    wp::adj_add(var_167, var_171, adj_167, adj_171, adj_172);
    wp::adj_mul(var_168, var_170, adj_168, adj_170, adj_171);
    wp::adj_float64(var_169, adj_169, adj_170);
    wp::adj_mul(var_3, var_37, adj_3, adj_37, adj_168);
    wp::adj_mul(var_164, var_166, adj_164, adj_166, adj_167);
    wp::adj_float64(var_165, adj_165, adj_166);
    wp::adj_mul(var_1, var_33, adj_1, adj_33, adj_164);
    // adj: t43 - t42 * (t17 * t45 * wp.float64(2.0) + t18 * t46 * wp.float64(2.0)),          <L 47>
    wp::adj_sub(var_63, var_162, adj_63, adj_162, adj_163);
    wp::adj_mul(var_25, var_161, adj_25, adj_161, adj_162);
    wp::adj_sub(var_156, var_160, adj_156, adj_160, adj_161);
    wp::adj_mul(var_157, var_159, adj_157, adj_159, adj_160);
    wp::adj_float64(var_158, adj_158, adj_159);
    wp::adj_mul(var_5, var_37, adj_5, adj_37, adj_157);
    wp::adj_mul(var_153, var_155, adj_153, adj_155, adj_156);
    wp::adj_float64(var_154, adj_154, adj_155);
    wp::adj_mul(var_1, var_29, adj_1, adj_29, adj_153);
    // adj: t52 - t42 * (t17 * t44 * wp.float64(2.0) - t19 * t46 * wp.float64(2.0)),          <L 46>
    wp::adj_add(var_53, var_151, adj_53, adj_151, adj_152);
    wp::adj_mul(var_25, var_150, adj_25, adj_150, adj_151);
    wp::adj_add(var_145, var_149, adj_145, adj_149, adj_150);
    wp::adj_mul(var_146, var_148, adj_146, adj_148, adj_149);
    wp::adj_float64(var_147, adj_147, adj_148);
    wp::adj_mul(var_5, var_33, adj_5, adj_33, adj_146);
    wp::adj_mul(var_142, var_144, adj_142, adj_144, adj_145);
    wp::adj_float64(var_143, adj_143, adj_144);
    wp::adj_mul(var_3, var_29, adj_3, adj_29, adj_142);
    // adj: t51 + t42 * (t18 * t44 * wp.float64(2.0) + t19 * t45 * wp.float64(2.0)),          <L 45>
    wp::adj_add(var_130, var_140, adj_130, adj_140, adj_141);
    wp::adj_mul(var_25, var_139, adj_25, adj_139, adj_140);
    wp::adj_add(var_134, var_138, adj_134, adj_138, adj_139);
    wp::adj_mul(var_135, var_137, adj_135, adj_137, adj_138);
    wp::adj_float64(var_136, adj_136, adj_137);
    wp::adj_mul(var_9, var_37, adj_9, adj_37, adj_135);
    wp::adj_mul(var_131, var_133, adj_131, adj_133, adj_134);
    wp::adj_float64(var_132, adj_132, adj_133);
    wp::adj_mul(var_7, var_33, adj_7, adj_33, adj_131);
    wp::adj_neg(var_73, adj_73, adj_130);
    // adj: -t43 + t42 * (t20 * t45 * wp.float64(2.0) + t21 * t46 * wp.float64(2.0)),         <L 44>
    wp::adj_add(var_118, var_128, adj_118, adj_128, adj_129);
    wp::adj_mul(var_25, var_127, adj_25, adj_127, adj_128);
    wp::adj_sub(var_122, var_126, adj_122, adj_126, adj_127);
    wp::adj_mul(var_123, var_125, adj_123, adj_125, adj_126);
    wp::adj_float64(var_124, adj_124, adj_125);
    wp::adj_mul(var_11, var_37, adj_11, adj_37, adj_123);
    wp::adj_mul(var_119, var_121, adj_119, adj_121, adj_122);
    wp::adj_float64(var_120, adj_120, adj_121);
    wp::adj_mul(var_7, var_29, adj_7, adj_29, adj_119);
    wp::adj_neg(var_63, adj_63, adj_118);
    // adj: -t52 + t42 * (t20 * t44 * wp.float64(2.0) - t22 * t46 * wp.float64(2.0)),         <L 43>
    wp::adj_sub(var_106, var_116, adj_106, adj_116, adj_117);
    wp::adj_mul(var_25, var_115, adj_25, adj_115, adj_116);
    wp::adj_add(var_110, var_114, adj_110, adj_114, adj_115);
    wp::adj_mul(var_111, var_113, adj_111, adj_113, adj_114);
    wp::adj_float64(var_112, adj_112, adj_113);
    wp::adj_mul(var_11, var_33, adj_11, adj_33, adj_111);
    wp::adj_mul(var_107, var_109, adj_107, adj_109, adj_110);
    wp::adj_float64(var_108, adj_108, adj_109);
    wp::adj_mul(var_9, var_29, adj_9, adj_29, adj_107);
    wp::adj_neg(var_53, adj_53, adj_106);
    // adj: -t51 - t42 * (t21 * t44 * wp.float64(2.0) + t22 * t45 * wp.float64(2.0)),         <L 42>
    wp::adj_mul(var_95, var_104, adj_95, adj_104, adj_105);
    wp::adj_add(var_99, var_103, adj_99, adj_103, adj_104);
    wp::adj_mul(var_100, var_102, adj_100, adj_102, adj_103);
    wp::adj_float64(var_101, adj_101, adj_102);
    wp::adj_mul(var_15, var_37, adj_15, adj_37, adj_100);
    wp::adj_mul(var_96, var_98, adj_96, adj_98, adj_99);
    wp::adj_float64(var_97, adj_97, adj_98);
    wp::adj_mul(var_13, var_33, adj_13, adj_33, adj_96);
    wp::adj_neg(var_25, adj_25, adj_95);
    // adj: -t42 * (t23 * t45 * wp.float64(2.0) + t24 * t46 * wp.float64(2.0)),               <L 41>
    wp::adj_mul(var_84, var_93, adj_84, adj_93, adj_94);
    wp::adj_sub(var_88, var_92, adj_88, adj_92, adj_93);
    wp::adj_mul(var_89, var_91, adj_89, adj_91, adj_92);
    wp::adj_float64(var_90, adj_90, adj_91);
    wp::adj_mul(var_17, var_37, adj_17, adj_37, adj_89);
    wp::adj_mul(var_85, var_87, adj_85, adj_87, adj_88);
    wp::adj_float64(var_86, adj_86, adj_87);
    wp::adj_mul(var_13, var_29, adj_13, adj_29, adj_85);
    wp::adj_neg(var_25, adj_25, adj_84);
    // adj: -t42 * (t23 * t44 * wp.float64(2.0) - t25 * t46 * wp.float64(2.0)),               <L 40>
    wp::adj_mul(var_25, var_82, adj_25, adj_82, adj_83);
    wp::adj_add(var_77, var_81, adj_77, adj_81, adj_82);
    wp::adj_mul(var_78, var_80, adj_78, adj_80, adj_81);
    wp::adj_float64(var_79, adj_79, adj_80);
    wp::adj_mul(var_17, var_33, adj_17, adj_33, adj_78);
    wp::adj_mul(var_74, var_76, adj_74, adj_76, adj_77);
    wp::adj_float64(var_75, adj_75, adj_76);
    wp::adj_mul(var_15, var_29, adj_15, adj_29, adj_74);
    // adj: t42 * (t24 * t44 * wp.float64(2.0) + t25 * t45 * wp.float64(2.0)),                <L 39>
    // adj: g = wp.vector(                                                                    <L 38>
    wp::adj_mul(var_72, var_43, adj_72, adj_43, adj_73);
    wp::adj_mul(var_71, var_38, adj_71, adj_38, adj_72);
    wp::adj_add(var_66, var_70, adj_66, adj_70, adj_71);
    wp::adj_neg(var_69, adj_69, adj_70);
    wp::adj_mul(var_v23, var_68, adj_v23, adj_68, adj_69);
    wp::adj_float64(var_67, adj_67, adj_68);
    wp::adj_mul(var_v13, var_65, adj_v13, adj_65, adj_66);
    wp::adj_float64(var_64, adj_64, adj_65);
    // adj: t43 = (v13 * wp.float64(2.0) + -(v23 * wp.float64(2.0))) * t43 * t50              <L 37>
    wp::adj_mul(var_62, var_43, adj_62, adj_43, adj_63);
    wp::adj_mul(var_61, var_38, adj_61, adj_38, adj_62);
    wp::adj_add(var_56, var_60, adj_56, adj_60, adj_61);
    wp::adj_neg(var_59, adj_59, adj_60);
    wp::adj_mul(var_v22, var_58, adj_v22, adj_58, adj_59);
    wp::adj_float64(var_57, adj_57, adj_58);
    wp::adj_mul(var_v12, var_55, adj_v12, adj_55, adj_56);
    wp::adj_float64(var_54, adj_54, adj_55);
    // adj: t52 = (v12 * wp.float64(2.0) + -(v22 * wp.float64(2.0))) * t43 * t50              <L 36>
    wp::adj_mul(var_52, var_43, adj_52, adj_43, adj_53);
    wp::adj_mul(var_51, var_38, adj_51, adj_38, adj_52);
    wp::adj_add(var_46, var_50, adj_46, adj_50, adj_51);
    wp::adj_neg(var_49, adj_49, adj_50);
    wp::adj_mul(var_v21, var_48, adj_v21, adj_48, adj_49);
    wp::adj_float64(var_47, adj_47, adj_48);
    wp::adj_mul(var_v11, var_45, adj_v11, adj_45, adj_46);
    wp::adj_float64(var_44, adj_44, adj_45);
    // adj: t51 = (v11 * wp.float64(2.0) + -(v21 * wp.float64(2.0))) * t43 * t50              <L 35>
    wp::adj_add(var_41, var_42, adj_41, adj_42, adj_43);
    wp::adj_mul(var_37, var_37, adj_37, adj_37, adj_42);
    wp::adj_add(var_39, var_40, adj_39, adj_40, adj_41);
    wp::adj_mul(var_33, var_33, adj_33, adj_33, adj_40);
    wp::adj_mul(var_29, var_29, adj_29, adj_29, adj_39);
    // adj: t50 = (t44 * t44 + t45 * t45) + t46 * t46                                         <L 34>
    wp::adj_mul(var_25, var_25, adj_25, adj_25, adj_38);
    // adj: t43 = t42 * t42                                                                   <L 33>
    wp::adj_add(var_34, var_36, adj_34, adj_36, adj_37);
    wp::adj_neg(var_35, adj_35, adj_36);
    wp::adj_mul(var_5, var_9, adj_5, adj_9, adj_35);
    wp::adj_mul(var_3, var_11, adj_3, adj_11, adj_34);
    // adj: t46 = t18 * t22 + -(t19 * t21)                                                    <L 32>
    wp::adj_add(var_30, var_32, adj_30, adj_32, adj_33);
    wp::adj_neg(var_31, adj_31, adj_32);
    wp::adj_mul(var_5, var_7, adj_5, adj_7, adj_31);
    wp::adj_mul(var_1, var_11, adj_1, adj_11, adj_30);
    // adj: t45 = t17 * t22 + -(t19 * t20)                                                    <L 31>
    wp::adj_add(var_26, var_28, adj_26, adj_28, adj_29);
    wp::adj_neg(var_27, adj_27, adj_28);
    wp::adj_mul(var_3, var_7, adj_3, adj_7, adj_27);
    wp::adj_mul(var_1, var_9, adj_1, adj_9, adj_26);
    // adj: t44 = t17 * t21 + -(t18 * t20)                                                    <L 30>
    wp::adj_div(var_19, var_24, var_25, adj_19, adj_24, adj_25);
    wp::adj_add(var_22, var_23, adj_22, adj_23, adj_24);
    wp::adj_mul(var_17, var_17, adj_17, adj_17, adj_23);
    wp::adj_add(var_20, var_21, adj_20, adj_21, adj_22);
    wp::adj_mul(var_15, var_15, adj_15, adj_15, adj_21);
    wp::adj_mul(var_13, var_13, adj_13, adj_13, adj_20);
    wp::adj_float64(var_18, adj_18, adj_19);
    // adj: t42 = wp.float64(1.0) / ((t23 * t23 + t24 * t24) + t25 * t25)                     <L 29>
    wp::adj_add(var_16, var_v13, adj_16, adj_v13, adj_17);
    wp::adj_neg(var_v23, adj_v23, adj_16);
    // adj: t25 = -v23 + v13                                                                  <L 28>
    wp::adj_add(var_14, var_v12, adj_14, adj_v12, adj_15);
    wp::adj_neg(var_v22, adj_v22, adj_14);
    // adj: t24 = -v22 + v12                                                                  <L 27>
    wp::adj_add(var_12, var_v11, adj_12, adj_v11, adj_13);
    wp::adj_neg(var_v21, adj_v21, adj_12);
    // adj: t23 = -v21 + v11                                                                  <L 26>
    wp::adj_add(var_10, var_v03, adj_10, adj_v03, adj_11);
    wp::adj_neg(var_v23, adj_v23, adj_10);
    // adj: t22 = -v23 + v03                                                                  <L 25>
    wp::adj_add(var_8, var_v02, adj_8, adj_v02, adj_9);
    wp::adj_neg(var_v22, adj_v22, adj_8);
    // adj: t21 = -v22 + v02                                                                  <L 24>
    wp::adj_add(var_6, var_v01, adj_6, adj_v01, adj_7);
    wp::adj_neg(var_v21, adj_v21, adj_6);
    // adj: t20 = -v21 + v01                                                                  <L 23>
    wp::adj_add(var_4, var_v03, adj_4, adj_v03, adj_5);
    wp::adj_neg(var_v13, adj_v13, adj_4);
    // adj: t19 = -v13 + v03                                                                  <L 22>
    wp::adj_add(var_2, var_v02, adj_2, adj_v02, adj_3);
    wp::adj_neg(var_v12, adj_v12, adj_2);
    // adj: t18 = -v12 + v02                                                                  <L 21>
    wp::adj_add(var_0, var_v01, adj_0, adj_v01, adj_1);
    wp::adj_neg(var_v11, adj_v11, adj_0);
    // adj: t17 = -v11 + v01                                                                  <L 20>
    // adj: def g_PE3D(                                                                       <L 5>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/point_line_distance.py:297
static CUDA_CALLABLE wp::vec_t<9,wp::float64> point_line_distance_gradient_0(
    wp::vec_t<3,wp::float64> var_p,
    wp::vec_t<3,wp::float64> var_e0,
    wp::vec_t<3,wp::float64> var_e1)
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
    wp::vec_t<9,wp::float64> var_18;
    //---------
    // forward
    // def point_line_distance_gradient(p: wp.vec3d, e0: wp.vec3d, e1: wp.vec3d):             <L 298>
    // return g_PE3D(p[0], p[1], p[2], e0[0], e0[1], e0[2], e1[0], e1[1], e1[2])              <L 299>
    var_1 = wp::extract(var_p, var_0);
    var_3 = wp::extract(var_p, var_2);
    var_5 = wp::extract(var_p, var_4);
    var_7 = wp::extract(var_e0, var_6);
    var_9 = wp::extract(var_e0, var_8);
    var_11 = wp::extract(var_e0, var_10);
    var_13 = wp::extract(var_e1, var_12);
    var_15 = wp::extract(var_e1, var_14);
    var_17 = wp::extract(var_e1, var_16);
    var_18 = g_PE3D_0(var_1, var_3, var_5, var_7, var_9, var_11, var_13, var_15, var_17);
    return var_18;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/point_line_distance.py:297
static CUDA_CALLABLE void adj_point_line_distance_gradient_0(
    wp::vec_t<3,wp::float64> var_p,
    wp::vec_t<3,wp::float64> var_e0,
    wp::vec_t<3,wp::float64> var_e1,
    wp::vec_t<3,wp::float64> & adj_p,
    wp::vec_t<3,wp::float64> & adj_e0,
    wp::vec_t<3,wp::float64> & adj_e1,
    wp::vec_t<9,wp::float64> & adj_ret)
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
    wp::vec_t<9,wp::float64> var_18;
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
    wp::vec_t<9,wp::float64> adj_18 = {};
    //---------
    // forward
    // def point_line_distance_gradient(p: wp.vec3d, e0: wp.vec3d, e1: wp.vec3d):             <L 298>
    // return g_PE3D(p[0], p[1], p[2], e0[0], e0[1], e0[2], e1[0], e1[1], e1[2])              <L 299>
    var_1 = wp::extract(var_p, var_0);
    var_3 = wp::extract(var_p, var_2);
    var_5 = wp::extract(var_p, var_4);
    var_7 = wp::extract(var_e0, var_6);
    var_9 = wp::extract(var_e0, var_8);
    var_11 = wp::extract(var_e0, var_10);
    var_13 = wp::extract(var_e1, var_12);
    var_15 = wp::extract(var_e1, var_14);
    var_17 = wp::extract(var_e1, var_16);
    var_18 = g_PE3D_0(var_1, var_3, var_5, var_7, var_9, var_11, var_13, var_15, var_17);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_18 += adj_ret;
    adj_g_PE3D_0(var_1, var_3, var_5, var_7, var_9, var_11, var_13, var_15, var_17, adj_1, adj_3, adj_5, adj_7, adj_9, adj_11, adj_13, adj_15, adj_17, adj_18);
    wp::adj_extract(var_e1, var_16, adj_e1, adj_16, adj_17);
    wp::adj_extract(var_e1, var_14, adj_e1, adj_14, adj_15);
    wp::adj_extract(var_e1, var_12, adj_e1, adj_12, adj_13);
    wp::adj_extract(var_e0, var_10, adj_e0, adj_10, adj_11);
    wp::adj_extract(var_e0, var_8, adj_e0, adj_8, adj_9);
    wp::adj_extract(var_e0, var_6, adj_e0, adj_6, adj_7);
    wp::adj_extract(var_p, var_4, adj_p, adj_4, adj_5);
    wp::adj_extract(var_p, var_2, adj_p, adj_2, adj_3);
    wp::adj_extract(var_p, var_0, adj_p, adj_0, adj_1);
    // adj: return g_PE3D(p[0], p[1], p[2], e0[0], e0[1], e0[2], e1[0], e1[1], e1[2])         <L 299>
    // adj: def point_line_distance_gradient(p: wp.vec3d, e0: wp.vec3d, e1: wp.vec3d):        <L 298>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/point_plane_distance.py:4
static CUDA_CALLABLE wp::vec_t<12,wp::float64> g_PT_0(
    wp::float64 var_v01,
    wp::float64 var_v02,
    wp::float64 var_v03,
    wp::float64 var_v11,
    wp::float64 var_v12,
    wp::float64 var_v13,
    wp::float64 var_v21,
    wp::float64 var_v22,
    wp::float64 var_v23,
    wp::float64 var_v31,
    wp::float64 var_v32,
    wp::float64 var_v33)
{
    //---------
    // primal vars
    wp::float64 var_0;
    wp::float64 var_1;
    wp::float64 var_2;
    wp::float64 var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    wp::float64 var_6;
    wp::float64 var_7;
    wp::float64 var_8;
    wp::float64 var_9;
    wp::float64 var_10;
    wp::float64 var_11;
    wp::float64 var_12;
    wp::float64 var_13;
    wp::float64 var_14;
    wp::float64 var_15;
    wp::float64 var_16;
    wp::float64 var_17;
    wp::float64 var_18;
    wp::float64 var_19;
    wp::float64 var_20;
    wp::float64 var_21;
    wp::float64 var_22;
    wp::float64 var_23;
    wp::float64 var_24;
    wp::float64 var_25;
    wp::float64 var_26;
    wp::float64 var_27;
    wp::float64 var_28;
    wp::float64 var_29;
    wp::float64 var_30;
    wp::float64 var_31;
    wp::float64 var_32;
    wp::float64 var_33;
    wp::float64 var_34;
    wp::float64 var_35;
    const wp::float32 var_36 = 1.0;
    wp::float64 var_37;
    wp::float64 var_38;
    wp::float64 var_39;
    wp::float64 var_40;
    wp::float64 var_41;
    wp::float64 var_42;
    wp::float64 var_43;
    wp::float64 var_44;
    wp::float64 var_45;
    wp::float64 var_46;
    wp::float64 var_47;
    wp::float64 var_48;
    wp::float64 var_49;
    wp::float64 var_50;
    wp::float64 var_51;
    wp::float64 var_52;
    wp::float64 var_53;
    wp::float64 var_54;
    wp::float64 var_55;
    const wp::float32 var_56 = 2.0;
    wp::float64 var_57;
    wp::float64 var_58;
    wp::float64 var_59;
    wp::float64 var_60;
    const wp::float32 var_61 = 2.0;
    wp::float64 var_62;
    wp::float64 var_63;
    wp::float64 var_64;
    wp::float64 var_65;
    wp::float64 var_66;
    const wp::float32 var_67 = 2.0;
    wp::float64 var_68;
    wp::float64 var_69;
    wp::float64 var_70;
    wp::float64 var_71;
    wp::float64 var_72;
    const wp::float32 var_73 = 2.0;
    wp::float64 var_74;
    wp::float64 var_75;
    wp::float64 var_76;
    const wp::float32 var_77 = 2.0;
    wp::float64 var_78;
    wp::float64 var_79;
    wp::float64 var_80;
    wp::float64 var_81;
    wp::float64 var_82;
    wp::float64 var_83;
    wp::float64 var_84;
    wp::float64 var_85;
    wp::float64 var_86;
    const wp::float32 var_87 = 2.0;
    wp::float64 var_88;
    wp::float64 var_89;
    wp::float64 var_90;
    wp::float64 var_91;
    const wp::float32 var_92 = 2.0;
    wp::float64 var_93;
    wp::float64 var_94;
    wp::float64 var_95;
    const wp::float32 var_96 = 2.0;
    wp::float64 var_97;
    wp::float64 var_98;
    wp::float64 var_99;
    wp::float64 var_100;
    wp::float64 var_101;
    wp::float64 var_102;
    wp::float64 var_103;
    wp::float64 var_104;
    wp::float64 var_105;
    const wp::float32 var_106 = 2.0;
    wp::float64 var_107;
    wp::float64 var_108;
    wp::float64 var_109;
    wp::float64 var_110;
    const wp::float32 var_111 = 2.0;
    wp::float64 var_112;
    wp::float64 var_113;
    wp::float64 var_114;
    const wp::float32 var_115 = 2.0;
    wp::float64 var_116;
    wp::float64 var_117;
    wp::float64 var_118;
    wp::float64 var_119;
    wp::float64 var_120;
    wp::float64 var_121;
    wp::float64 var_122;
    wp::float64 var_123;
    wp::float64 var_124;
    const wp::float32 var_125 = 2.0;
    wp::float64 var_126;
    wp::float64 var_127;
    wp::float64 var_128;
    wp::float64 var_129;
    wp::float64 var_130;
    wp::float64 var_131;
    wp::float64 var_132;
    const wp::float32 var_133 = 2.0;
    wp::float64 var_134;
    wp::float64 var_135;
    wp::float64 var_136;
    const wp::float32 var_137 = 2.0;
    wp::float64 var_138;
    wp::float64 var_139;
    wp::float64 var_140;
    const wp::float32 var_141 = 2.0;
    wp::float64 var_142;
    wp::float64 var_143;
    wp::float64 var_144;
    wp::float64 var_145;
    wp::float64 var_146;
    wp::float64 var_147;
    wp::float64 var_148;
    wp::float64 var_149;
    wp::float64 var_150;
    const wp::float32 var_151 = 2.0;
    wp::float64 var_152;
    wp::float64 var_153;
    wp::float64 var_154;
    wp::float64 var_155;
    const wp::float32 var_156 = 2.0;
    wp::float64 var_157;
    wp::float64 var_158;
    wp::float64 var_159;
    const wp::float32 var_160 = 2.0;
    wp::float64 var_161;
    wp::float64 var_162;
    wp::float64 var_163;
    wp::float64 var_164;
    wp::float64 var_165;
    wp::float64 var_166;
    wp::float64 var_167;
    wp::float64 var_168;
    wp::float64 var_169;
    const wp::float32 var_170 = 2.0;
    wp::float64 var_171;
    wp::float64 var_172;
    wp::float64 var_173;
    const wp::float32 var_174 = 2.0;
    wp::float64 var_175;
    wp::float64 var_176;
    wp::float64 var_177;
    const wp::float32 var_178 = 2.0;
    wp::float64 var_179;
    wp::float64 var_180;
    wp::float64 var_181;
    wp::float64 var_182;
    wp::float64 var_183;
    wp::float64 var_184;
    wp::float64 var_185;
    wp::float64 var_186;
    wp::float64 var_187;
    const wp::float32 var_188 = 2.0;
    wp::float64 var_189;
    wp::float64 var_190;
    wp::float64 var_191;
    wp::float64 var_192;
    const wp::float32 var_193 = 2.0;
    wp::float64 var_194;
    wp::float64 var_195;
    wp::float64 var_196;
    const wp::float32 var_197 = 2.0;
    wp::float64 var_198;
    wp::float64 var_199;
    wp::float64 var_200;
    wp::float64 var_201;
    wp::float64 var_202;
    wp::float64 var_203;
    wp::float64 var_204;
    wp::float64 var_205;
    wp::float64 var_206;
    const wp::float32 var_207 = 2.0;
    wp::float64 var_208;
    wp::float64 var_209;
    wp::float64 var_210;
    const wp::float32 var_211 = 2.0;
    wp::float64 var_212;
    wp::float64 var_213;
    wp::float64 var_214;
    const wp::float32 var_215 = 2.0;
    wp::float64 var_216;
    wp::float64 var_217;
    wp::float64 var_218;
    wp::float64 var_219;
    wp::float64 var_220;
    wp::float64 var_221;
    wp::float64 var_222;
    wp::float64 var_223;
    wp::float64 var_224;
    const wp::float32 var_225 = 2.0;
    wp::float64 var_226;
    wp::float64 var_227;
    wp::float64 var_228;
    wp::float64 var_229;
    const wp::float32 var_230 = 2.0;
    wp::float64 var_231;
    wp::float64 var_232;
    wp::float64 var_233;
    const wp::float32 var_234 = 2.0;
    wp::float64 var_235;
    wp::float64 var_236;
    wp::float64 var_237;
    wp::float64 var_238;
    wp::float64 var_239;
    wp::vec_t<12,wp::float64> var_240;
    //---------
    // forward
    // def g_PT(                                                                              <L 5>
    // t11 = -v11 + v01                                                                       <L 23>
    var_0 = wp::neg(var_v11);
    var_1 = wp::add(var_0, var_v01);
    // t12 = -v12 + v02                                                                       <L 24>
    var_2 = wp::neg(var_v12);
    var_3 = wp::add(var_2, var_v02);
    // t13 = -v13 + v03                                                                       <L 25>
    var_4 = wp::neg(var_v13);
    var_5 = wp::add(var_4, var_v03);
    // t14 = -v21 + v11                                                                       <L 26>
    var_6 = wp::neg(var_v21);
    var_7 = wp::add(var_6, var_v11);
    // t15 = -v22 + v12                                                                       <L 27>
    var_8 = wp::neg(var_v22);
    var_9 = wp::add(var_8, var_v12);
    // t16 = -v23 + v13                                                                       <L 28>
    var_10 = wp::neg(var_v23);
    var_11 = wp::add(var_10, var_v13);
    // t17 = -v31 + v11                                                                       <L 29>
    var_12 = wp::neg(var_v31);
    var_13 = wp::add(var_12, var_v11);
    // t18 = -v32 + v12                                                                       <L 30>
    var_14 = wp::neg(var_v32);
    var_15 = wp::add(var_14, var_v12);
    // t19 = -v33 + v13                                                                       <L 31>
    var_16 = wp::neg(var_v33);
    var_17 = wp::add(var_16, var_v13);
    // t20 = -v31 + v21                                                                       <L 32>
    var_18 = wp::neg(var_v31);
    var_19 = wp::add(var_18, var_v21);
    // t21 = -v32 + v22                                                                       <L 33>
    var_20 = wp::neg(var_v32);
    var_21 = wp::add(var_20, var_v22);
    // t22 = -v33 + v23                                                                       <L 34>
    var_22 = wp::neg(var_v33);
    var_23 = wp::add(var_22, var_v23);
    // t32 = t14 * t18 + -(t15 * t17)                                                         <L 35>
    var_24 = wp::mul(var_7, var_15);
    var_25 = wp::mul(var_9, var_13);
    var_26 = wp::neg(var_25);
    var_27 = wp::add(var_24, var_26);
    // t33 = t14 * t19 + -(t16 * t17)                                                         <L 36>
    var_28 = wp::mul(var_7, var_17);
    var_29 = wp::mul(var_11, var_13);
    var_30 = wp::neg(var_29);
    var_31 = wp::add(var_28, var_30);
    // t34 = t15 * t19 + -(t16 * t18)                                                         <L 37>
    var_32 = wp::mul(var_9, var_17);
    var_33 = wp::mul(var_11, var_15);
    var_34 = wp::neg(var_33);
    var_35 = wp::add(var_32, var_34);
    // t43_1 = wp.float64(1.0) / ((t32 * t32 + t33 * t33) + t34 * t34)                        <L 38>
    var_37 = wp::float64(var_36);
    var_38 = wp::mul(var_27, var_27);
    var_39 = wp::mul(var_31, var_31);
    var_40 = wp::add(var_38, var_39);
    var_41 = wp::mul(var_35, var_35);
    var_42 = wp::add(var_40, var_41);
    var_43 = wp::div(var_37, var_42);
    // t45_1 = (t13 * t32 + t11 * t34) + -(t12 * t33)                                         <L 39>
    var_44 = wp::mul(var_5, var_27);
    var_45 = wp::mul(var_1, var_35);
    var_46 = wp::add(var_44, var_45);
    var_47 = wp::mul(var_3, var_31);
    var_48 = wp::neg(var_47);
    var_49 = wp::add(var_46, var_48);
    // t44 = t43_1 * t43_1                                                                    <L 40>
    var_50 = wp::mul(var_43, var_43);
    // t46 = t45_1 * t45_1                                                                    <L 41>
    var_51 = wp::mul(var_49, var_49);
    // t45 = t45_1 * t43_1                                                                    <L 42>
    var_52 = wp::mul(var_49, var_43);
    // t43 = t44 * t46                                                                        <L 43>
    var_53 = wp::mul(var_50, var_51);
    // g = wp.vector(                                                                         <L 44>
    // t34 * t43_1 * t45_1 * wp.float64(2.0),                                                 <L 45>
    var_54 = wp::mul(var_35, var_43);
    var_55 = wp::mul(var_54, var_49);
    var_57 = wp::float64(var_56);
    var_58 = wp::mul(var_55, var_57);
    // t33 * t43_1 * t45_1 * -wp.float64(2.0),                                                <L 46>
    var_59 = wp::mul(var_31, var_43);
    var_60 = wp::mul(var_59, var_49);
    var_62 = wp::float64(var_61);
    var_63 = wp::neg(var_62);
    var_64 = wp::mul(var_60, var_63);
    // t32 * t43_1 * t45_1 * wp.float64(2.0),                                                 <L 47>
    var_65 = wp::mul(var_27, var_43);
    var_66 = wp::mul(var_65, var_49);
    var_68 = wp::float64(var_67);
    var_69 = wp::mul(var_66, var_68);
    // -t44 * t46 * (t21 * t32 * wp.float64(2.0) + t22 * t33 * wp.float64(2.0)) - t45 * ((t34 + t12 * t22) - t13 * t21) * wp.float64(2.0),       <L 48>
    var_70 = wp::neg(var_50);
    var_71 = wp::mul(var_70, var_51);
    var_72 = wp::mul(var_21, var_27);
    var_74 = wp::float64(var_73);
    var_75 = wp::mul(var_72, var_74);
    var_76 = wp::mul(var_23, var_31);
    var_78 = wp::float64(var_77);
    var_79 = wp::mul(var_76, var_78);
    var_80 = wp::add(var_75, var_79);
    var_81 = wp::mul(var_71, var_80);
    var_82 = wp::mul(var_3, var_23);
    var_83 = wp::add(var_35, var_82);
    var_84 = wp::mul(var_5, var_21);
    var_85 = wp::sub(var_83, var_84);
    var_86 = wp::mul(var_52, var_85);
    var_88 = wp::float64(var_87);
    var_89 = wp::mul(var_86, var_88);
    var_90 = wp::sub(var_81, var_89);
    // t43 * (t20 * t32 * wp.float64(2.0) - t22 * t34 * wp.float64(2.0)) + t45 * ((t33 + t11 * t22) - t13 * t20) * wp.float64(2.0),       <L 49>
    var_91 = wp::mul(var_19, var_27);
    var_93 = wp::float64(var_92);
    var_94 = wp::mul(var_91, var_93);
    var_95 = wp::mul(var_23, var_35);
    var_97 = wp::float64(var_96);
    var_98 = wp::mul(var_95, var_97);
    var_99 = wp::sub(var_94, var_98);
    var_100 = wp::mul(var_53, var_99);
    var_101 = wp::mul(var_1, var_23);
    var_102 = wp::add(var_31, var_101);
    var_103 = wp::mul(var_5, var_19);
    var_104 = wp::sub(var_102, var_103);
    var_105 = wp::mul(var_52, var_104);
    var_107 = wp::float64(var_106);
    var_108 = wp::mul(var_105, var_107);
    var_109 = wp::add(var_100, var_108);
    // t43 * (t20 * t33 * wp.float64(2.0) + t21 * t34 * wp.float64(2.0)) - t45 * ((t32 + t11 * t21) - t12 * t20) * wp.float64(2.0),       <L 50>
    var_110 = wp::mul(var_19, var_31);
    var_112 = wp::float64(var_111);
    var_113 = wp::mul(var_110, var_112);
    var_114 = wp::mul(var_21, var_35);
    var_116 = wp::float64(var_115);
    var_117 = wp::mul(var_114, var_116);
    var_118 = wp::add(var_113, var_117);
    var_119 = wp::mul(var_53, var_118);
    var_120 = wp::mul(var_1, var_21);
    var_121 = wp::add(var_27, var_120);
    var_122 = wp::mul(var_3, var_19);
    var_123 = wp::sub(var_121, var_122);
    var_124 = wp::mul(var_52, var_123);
    var_126 = wp::float64(var_125);
    var_127 = wp::mul(var_124, var_126);
    var_128 = wp::sub(var_119, var_127);
    // t45 * (t12 * t19 - t13 * t18) * wp.float64(2.0) + t43 * (t18 * t32 * wp.float64(2.0) + t19 * t33 * wp.float64(2.0)),       <L 51>
    var_129 = wp::mul(var_3, var_17);
    var_130 = wp::mul(var_5, var_15);
    var_131 = wp::sub(var_129, var_130);
    var_132 = wp::mul(var_52, var_131);
    var_134 = wp::float64(var_133);
    var_135 = wp::mul(var_132, var_134);
    var_136 = wp::mul(var_15, var_27);
    var_138 = wp::float64(var_137);
    var_139 = wp::mul(var_136, var_138);
    var_140 = wp::mul(var_17, var_31);
    var_142 = wp::float64(var_141);
    var_143 = wp::mul(var_140, var_142);
    var_144 = wp::add(var_139, var_143);
    var_145 = wp::mul(var_53, var_144);
    var_146 = wp::add(var_135, var_145);
    // t45 * (t11 * t19 - t13 * t17) * -wp.float64(2.0) - t43 * (t17 * t32 * wp.float64(2.0) - t19 * t34 * wp.float64(2.0)),       <L 52>
    var_147 = wp::mul(var_1, var_17);
    var_148 = wp::mul(var_5, var_13);
    var_149 = wp::sub(var_147, var_148);
    var_150 = wp::mul(var_52, var_149);
    var_152 = wp::float64(var_151);
    var_153 = wp::neg(var_152);
    var_154 = wp::mul(var_150, var_153);
    var_155 = wp::mul(var_13, var_27);
    var_157 = wp::float64(var_156);
    var_158 = wp::mul(var_155, var_157);
    var_159 = wp::mul(var_17, var_35);
    var_161 = wp::float64(var_160);
    var_162 = wp::mul(var_159, var_161);
    var_163 = wp::sub(var_158, var_162);
    var_164 = wp::mul(var_53, var_163);
    var_165 = wp::sub(var_154, var_164);
    // t45 * (t11 * t18 - t12 * t17) * wp.float64(2.0) - t43 * (t17 * t33 * wp.float64(2.0) + t18 * t34 * wp.float64(2.0)),       <L 53>
    var_166 = wp::mul(var_1, var_15);
    var_167 = wp::mul(var_3, var_13);
    var_168 = wp::sub(var_166, var_167);
    var_169 = wp::mul(var_52, var_168);
    var_171 = wp::float64(var_170);
    var_172 = wp::mul(var_169, var_171);
    var_173 = wp::mul(var_13, var_31);
    var_175 = wp::float64(var_174);
    var_176 = wp::mul(var_173, var_175);
    var_177 = wp::mul(var_15, var_35);
    var_179 = wp::float64(var_178);
    var_180 = wp::mul(var_177, var_179);
    var_181 = wp::add(var_176, var_180);
    var_182 = wp::mul(var_53, var_181);
    var_183 = wp::sub(var_172, var_182);
    // t45 * (t12 * t16 - t13 * t15) * -wp.float64(2.0) - t43 * (t15 * t32 * wp.float64(2.0) + t16 * t33 * wp.float64(2.0)),       <L 54>
    var_184 = wp::mul(var_3, var_11);
    var_185 = wp::mul(var_5, var_9);
    var_186 = wp::sub(var_184, var_185);
    var_187 = wp::mul(var_52, var_186);
    var_189 = wp::float64(var_188);
    var_190 = wp::neg(var_189);
    var_191 = wp::mul(var_187, var_190);
    var_192 = wp::mul(var_9, var_27);
    var_194 = wp::float64(var_193);
    var_195 = wp::mul(var_192, var_194);
    var_196 = wp::mul(var_11, var_31);
    var_198 = wp::float64(var_197);
    var_199 = wp::mul(var_196, var_198);
    var_200 = wp::add(var_195, var_199);
    var_201 = wp::mul(var_53, var_200);
    var_202 = wp::sub(var_191, var_201);
    // t45 * (t11 * t16 - t13 * t14) * wp.float64(2.0) + t43 * (t14 * t32 * wp.float64(2.0) - t16 * t34 * wp.float64(2.0)),       <L 55>
    var_203 = wp::mul(var_1, var_11);
    var_204 = wp::mul(var_5, var_7);
    var_205 = wp::sub(var_203, var_204);
    var_206 = wp::mul(var_52, var_205);
    var_208 = wp::float64(var_207);
    var_209 = wp::mul(var_206, var_208);
    var_210 = wp::mul(var_7, var_27);
    var_212 = wp::float64(var_211);
    var_213 = wp::mul(var_210, var_212);
    var_214 = wp::mul(var_11, var_35);
    var_216 = wp::float64(var_215);
    var_217 = wp::mul(var_214, var_216);
    var_218 = wp::sub(var_213, var_217);
    var_219 = wp::mul(var_53, var_218);
    var_220 = wp::add(var_209, var_219);
    // t45 * (t11 * t15 - t12 * t14) * -wp.float64(2.0) + t43 * (t14 * t33 * wp.float64(2.0) + t15 * t34 * wp.float64(2.0)),       <L 56>
    var_221 = wp::mul(var_1, var_9);
    var_222 = wp::mul(var_3, var_7);
    var_223 = wp::sub(var_221, var_222);
    var_224 = wp::mul(var_52, var_223);
    var_226 = wp::float64(var_225);
    var_227 = wp::neg(var_226);
    var_228 = wp::mul(var_224, var_227);
    var_229 = wp::mul(var_7, var_31);
    var_231 = wp::float64(var_230);
    var_232 = wp::mul(var_229, var_231);
    var_233 = wp::mul(var_9, var_35);
    var_235 = wp::float64(var_234);
    var_236 = wp::mul(var_233, var_235);
    var_237 = wp::add(var_232, var_236);
    var_238 = wp::mul(var_53, var_237);
    var_239 = wp::add(var_228, var_238);
    var_240 = wp::vec_t<12,wp::float64>({var_58, var_64, var_69, var_90, var_109, var_128, var_146, var_165, var_183, var_202, var_220, var_239});
    // return g                                                                               <L 58>
    return var_240;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/point_plane_distance.py:4
static CUDA_CALLABLE void adj_g_PT_0(
    wp::float64 var_v01,
    wp::float64 var_v02,
    wp::float64 var_v03,
    wp::float64 var_v11,
    wp::float64 var_v12,
    wp::float64 var_v13,
    wp::float64 var_v21,
    wp::float64 var_v22,
    wp::float64 var_v23,
    wp::float64 var_v31,
    wp::float64 var_v32,
    wp::float64 var_v33,
    wp::float64 & adj_v01,
    wp::float64 & adj_v02,
    wp::float64 & adj_v03,
    wp::float64 & adj_v11,
    wp::float64 & adj_v12,
    wp::float64 & adj_v13,
    wp::float64 & adj_v21,
    wp::float64 & adj_v22,
    wp::float64 & adj_v23,
    wp::float64 & adj_v31,
    wp::float64 & adj_v32,
    wp::float64 & adj_v33,
    wp::vec_t<12,wp::float64> & adj_ret)
{
    //---------
    // primal vars
    wp::float64 var_0;
    wp::float64 var_1;
    wp::float64 var_2;
    wp::float64 var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    wp::float64 var_6;
    wp::float64 var_7;
    wp::float64 var_8;
    wp::float64 var_9;
    wp::float64 var_10;
    wp::float64 var_11;
    wp::float64 var_12;
    wp::float64 var_13;
    wp::float64 var_14;
    wp::float64 var_15;
    wp::float64 var_16;
    wp::float64 var_17;
    wp::float64 var_18;
    wp::float64 var_19;
    wp::float64 var_20;
    wp::float64 var_21;
    wp::float64 var_22;
    wp::float64 var_23;
    wp::float64 var_24;
    wp::float64 var_25;
    wp::float64 var_26;
    wp::float64 var_27;
    wp::float64 var_28;
    wp::float64 var_29;
    wp::float64 var_30;
    wp::float64 var_31;
    wp::float64 var_32;
    wp::float64 var_33;
    wp::float64 var_34;
    wp::float64 var_35;
    const wp::float32 var_36 = 1.0;
    wp::float64 var_37;
    wp::float64 var_38;
    wp::float64 var_39;
    wp::float64 var_40;
    wp::float64 var_41;
    wp::float64 var_42;
    wp::float64 var_43;
    wp::float64 var_44;
    wp::float64 var_45;
    wp::float64 var_46;
    wp::float64 var_47;
    wp::float64 var_48;
    wp::float64 var_49;
    wp::float64 var_50;
    wp::float64 var_51;
    wp::float64 var_52;
    wp::float64 var_53;
    wp::float64 var_54;
    wp::float64 var_55;
    const wp::float32 var_56 = 2.0;
    wp::float64 var_57;
    wp::float64 var_58;
    wp::float64 var_59;
    wp::float64 var_60;
    const wp::float32 var_61 = 2.0;
    wp::float64 var_62;
    wp::float64 var_63;
    wp::float64 var_64;
    wp::float64 var_65;
    wp::float64 var_66;
    const wp::float32 var_67 = 2.0;
    wp::float64 var_68;
    wp::float64 var_69;
    wp::float64 var_70;
    wp::float64 var_71;
    wp::float64 var_72;
    const wp::float32 var_73 = 2.0;
    wp::float64 var_74;
    wp::float64 var_75;
    wp::float64 var_76;
    const wp::float32 var_77 = 2.0;
    wp::float64 var_78;
    wp::float64 var_79;
    wp::float64 var_80;
    wp::float64 var_81;
    wp::float64 var_82;
    wp::float64 var_83;
    wp::float64 var_84;
    wp::float64 var_85;
    wp::float64 var_86;
    const wp::float32 var_87 = 2.0;
    wp::float64 var_88;
    wp::float64 var_89;
    wp::float64 var_90;
    wp::float64 var_91;
    const wp::float32 var_92 = 2.0;
    wp::float64 var_93;
    wp::float64 var_94;
    wp::float64 var_95;
    const wp::float32 var_96 = 2.0;
    wp::float64 var_97;
    wp::float64 var_98;
    wp::float64 var_99;
    wp::float64 var_100;
    wp::float64 var_101;
    wp::float64 var_102;
    wp::float64 var_103;
    wp::float64 var_104;
    wp::float64 var_105;
    const wp::float32 var_106 = 2.0;
    wp::float64 var_107;
    wp::float64 var_108;
    wp::float64 var_109;
    wp::float64 var_110;
    const wp::float32 var_111 = 2.0;
    wp::float64 var_112;
    wp::float64 var_113;
    wp::float64 var_114;
    const wp::float32 var_115 = 2.0;
    wp::float64 var_116;
    wp::float64 var_117;
    wp::float64 var_118;
    wp::float64 var_119;
    wp::float64 var_120;
    wp::float64 var_121;
    wp::float64 var_122;
    wp::float64 var_123;
    wp::float64 var_124;
    const wp::float32 var_125 = 2.0;
    wp::float64 var_126;
    wp::float64 var_127;
    wp::float64 var_128;
    wp::float64 var_129;
    wp::float64 var_130;
    wp::float64 var_131;
    wp::float64 var_132;
    const wp::float32 var_133 = 2.0;
    wp::float64 var_134;
    wp::float64 var_135;
    wp::float64 var_136;
    const wp::float32 var_137 = 2.0;
    wp::float64 var_138;
    wp::float64 var_139;
    wp::float64 var_140;
    const wp::float32 var_141 = 2.0;
    wp::float64 var_142;
    wp::float64 var_143;
    wp::float64 var_144;
    wp::float64 var_145;
    wp::float64 var_146;
    wp::float64 var_147;
    wp::float64 var_148;
    wp::float64 var_149;
    wp::float64 var_150;
    const wp::float32 var_151 = 2.0;
    wp::float64 var_152;
    wp::float64 var_153;
    wp::float64 var_154;
    wp::float64 var_155;
    const wp::float32 var_156 = 2.0;
    wp::float64 var_157;
    wp::float64 var_158;
    wp::float64 var_159;
    const wp::float32 var_160 = 2.0;
    wp::float64 var_161;
    wp::float64 var_162;
    wp::float64 var_163;
    wp::float64 var_164;
    wp::float64 var_165;
    wp::float64 var_166;
    wp::float64 var_167;
    wp::float64 var_168;
    wp::float64 var_169;
    const wp::float32 var_170 = 2.0;
    wp::float64 var_171;
    wp::float64 var_172;
    wp::float64 var_173;
    const wp::float32 var_174 = 2.0;
    wp::float64 var_175;
    wp::float64 var_176;
    wp::float64 var_177;
    const wp::float32 var_178 = 2.0;
    wp::float64 var_179;
    wp::float64 var_180;
    wp::float64 var_181;
    wp::float64 var_182;
    wp::float64 var_183;
    wp::float64 var_184;
    wp::float64 var_185;
    wp::float64 var_186;
    wp::float64 var_187;
    const wp::float32 var_188 = 2.0;
    wp::float64 var_189;
    wp::float64 var_190;
    wp::float64 var_191;
    wp::float64 var_192;
    const wp::float32 var_193 = 2.0;
    wp::float64 var_194;
    wp::float64 var_195;
    wp::float64 var_196;
    const wp::float32 var_197 = 2.0;
    wp::float64 var_198;
    wp::float64 var_199;
    wp::float64 var_200;
    wp::float64 var_201;
    wp::float64 var_202;
    wp::float64 var_203;
    wp::float64 var_204;
    wp::float64 var_205;
    wp::float64 var_206;
    const wp::float32 var_207 = 2.0;
    wp::float64 var_208;
    wp::float64 var_209;
    wp::float64 var_210;
    const wp::float32 var_211 = 2.0;
    wp::float64 var_212;
    wp::float64 var_213;
    wp::float64 var_214;
    const wp::float32 var_215 = 2.0;
    wp::float64 var_216;
    wp::float64 var_217;
    wp::float64 var_218;
    wp::float64 var_219;
    wp::float64 var_220;
    wp::float64 var_221;
    wp::float64 var_222;
    wp::float64 var_223;
    wp::float64 var_224;
    const wp::float32 var_225 = 2.0;
    wp::float64 var_226;
    wp::float64 var_227;
    wp::float64 var_228;
    wp::float64 var_229;
    const wp::float32 var_230 = 2.0;
    wp::float64 var_231;
    wp::float64 var_232;
    wp::float64 var_233;
    const wp::float32 var_234 = 2.0;
    wp::float64 var_235;
    wp::float64 var_236;
    wp::float64 var_237;
    wp::float64 var_238;
    wp::float64 var_239;
    wp::vec_t<12,wp::float64> var_240;
    //---------
    // dual vars
    wp::float64 adj_0 = {};
    wp::float64 adj_1 = {};
    wp::float64 adj_2 = {};
    wp::float64 adj_3 = {};
    wp::float64 adj_4 = {};
    wp::float64 adj_5 = {};
    wp::float64 adj_6 = {};
    wp::float64 adj_7 = {};
    wp::float64 adj_8 = {};
    wp::float64 adj_9 = {};
    wp::float64 adj_10 = {};
    wp::float64 adj_11 = {};
    wp::float64 adj_12 = {};
    wp::float64 adj_13 = {};
    wp::float64 adj_14 = {};
    wp::float64 adj_15 = {};
    wp::float64 adj_16 = {};
    wp::float64 adj_17 = {};
    wp::float64 adj_18 = {};
    wp::float64 adj_19 = {};
    wp::float64 adj_20 = {};
    wp::float64 adj_21 = {};
    wp::float64 adj_22 = {};
    wp::float64 adj_23 = {};
    wp::float64 adj_24 = {};
    wp::float64 adj_25 = {};
    wp::float64 adj_26 = {};
    wp::float64 adj_27 = {};
    wp::float64 adj_28 = {};
    wp::float64 adj_29 = {};
    wp::float64 adj_30 = {};
    wp::float64 adj_31 = {};
    wp::float64 adj_32 = {};
    wp::float64 adj_33 = {};
    wp::float64 adj_34 = {};
    wp::float64 adj_35 = {};
    wp::float32 adj_36 = {};
    wp::float64 adj_37 = {};
    wp::float64 adj_38 = {};
    wp::float64 adj_39 = {};
    wp::float64 adj_40 = {};
    wp::float64 adj_41 = {};
    wp::float64 adj_42 = {};
    wp::float64 adj_43 = {};
    wp::float64 adj_44 = {};
    wp::float64 adj_45 = {};
    wp::float64 adj_46 = {};
    wp::float64 adj_47 = {};
    wp::float64 adj_48 = {};
    wp::float64 adj_49 = {};
    wp::float64 adj_50 = {};
    wp::float64 adj_51 = {};
    wp::float64 adj_52 = {};
    wp::float64 adj_53 = {};
    wp::float64 adj_54 = {};
    wp::float64 adj_55 = {};
    wp::float32 adj_56 = {};
    wp::float64 adj_57 = {};
    wp::float64 adj_58 = {};
    wp::float64 adj_59 = {};
    wp::float64 adj_60 = {};
    wp::float32 adj_61 = {};
    wp::float64 adj_62 = {};
    wp::float64 adj_63 = {};
    wp::float64 adj_64 = {};
    wp::float64 adj_65 = {};
    wp::float64 adj_66 = {};
    wp::float32 adj_67 = {};
    wp::float64 adj_68 = {};
    wp::float64 adj_69 = {};
    wp::float64 adj_70 = {};
    wp::float64 adj_71 = {};
    wp::float64 adj_72 = {};
    wp::float32 adj_73 = {};
    wp::float64 adj_74 = {};
    wp::float64 adj_75 = {};
    wp::float64 adj_76 = {};
    wp::float32 adj_77 = {};
    wp::float64 adj_78 = {};
    wp::float64 adj_79 = {};
    wp::float64 adj_80 = {};
    wp::float64 adj_81 = {};
    wp::float64 adj_82 = {};
    wp::float64 adj_83 = {};
    wp::float64 adj_84 = {};
    wp::float64 adj_85 = {};
    wp::float64 adj_86 = {};
    wp::float32 adj_87 = {};
    wp::float64 adj_88 = {};
    wp::float64 adj_89 = {};
    wp::float64 adj_90 = {};
    wp::float64 adj_91 = {};
    wp::float32 adj_92 = {};
    wp::float64 adj_93 = {};
    wp::float64 adj_94 = {};
    wp::float64 adj_95 = {};
    wp::float32 adj_96 = {};
    wp::float64 adj_97 = {};
    wp::float64 adj_98 = {};
    wp::float64 adj_99 = {};
    wp::float64 adj_100 = {};
    wp::float64 adj_101 = {};
    wp::float64 adj_102 = {};
    wp::float64 adj_103 = {};
    wp::float64 adj_104 = {};
    wp::float64 adj_105 = {};
    wp::float32 adj_106 = {};
    wp::float64 adj_107 = {};
    wp::float64 adj_108 = {};
    wp::float64 adj_109 = {};
    wp::float64 adj_110 = {};
    wp::float32 adj_111 = {};
    wp::float64 adj_112 = {};
    wp::float64 adj_113 = {};
    wp::float64 adj_114 = {};
    wp::float32 adj_115 = {};
    wp::float64 adj_116 = {};
    wp::float64 adj_117 = {};
    wp::float64 adj_118 = {};
    wp::float64 adj_119 = {};
    wp::float64 adj_120 = {};
    wp::float64 adj_121 = {};
    wp::float64 adj_122 = {};
    wp::float64 adj_123 = {};
    wp::float64 adj_124 = {};
    wp::float32 adj_125 = {};
    wp::float64 adj_126 = {};
    wp::float64 adj_127 = {};
    wp::float64 adj_128 = {};
    wp::float64 adj_129 = {};
    wp::float64 adj_130 = {};
    wp::float64 adj_131 = {};
    wp::float64 adj_132 = {};
    wp::float32 adj_133 = {};
    wp::float64 adj_134 = {};
    wp::float64 adj_135 = {};
    wp::float64 adj_136 = {};
    wp::float32 adj_137 = {};
    wp::float64 adj_138 = {};
    wp::float64 adj_139 = {};
    wp::float64 adj_140 = {};
    wp::float32 adj_141 = {};
    wp::float64 adj_142 = {};
    wp::float64 adj_143 = {};
    wp::float64 adj_144 = {};
    wp::float64 adj_145 = {};
    wp::float64 adj_146 = {};
    wp::float64 adj_147 = {};
    wp::float64 adj_148 = {};
    wp::float64 adj_149 = {};
    wp::float64 adj_150 = {};
    wp::float32 adj_151 = {};
    wp::float64 adj_152 = {};
    wp::float64 adj_153 = {};
    wp::float64 adj_154 = {};
    wp::float64 adj_155 = {};
    wp::float32 adj_156 = {};
    wp::float64 adj_157 = {};
    wp::float64 adj_158 = {};
    wp::float64 adj_159 = {};
    wp::float32 adj_160 = {};
    wp::float64 adj_161 = {};
    wp::float64 adj_162 = {};
    wp::float64 adj_163 = {};
    wp::float64 adj_164 = {};
    wp::float64 adj_165 = {};
    wp::float64 adj_166 = {};
    wp::float64 adj_167 = {};
    wp::float64 adj_168 = {};
    wp::float64 adj_169 = {};
    wp::float32 adj_170 = {};
    wp::float64 adj_171 = {};
    wp::float64 adj_172 = {};
    wp::float64 adj_173 = {};
    wp::float32 adj_174 = {};
    wp::float64 adj_175 = {};
    wp::float64 adj_176 = {};
    wp::float64 adj_177 = {};
    wp::float32 adj_178 = {};
    wp::float64 adj_179 = {};
    wp::float64 adj_180 = {};
    wp::float64 adj_181 = {};
    wp::float64 adj_182 = {};
    wp::float64 adj_183 = {};
    wp::float64 adj_184 = {};
    wp::float64 adj_185 = {};
    wp::float64 adj_186 = {};
    wp::float64 adj_187 = {};
    wp::float32 adj_188 = {};
    wp::float64 adj_189 = {};
    wp::float64 adj_190 = {};
    wp::float64 adj_191 = {};
    wp::float64 adj_192 = {};
    wp::float32 adj_193 = {};
    wp::float64 adj_194 = {};
    wp::float64 adj_195 = {};
    wp::float64 adj_196 = {};
    wp::float32 adj_197 = {};
    wp::float64 adj_198 = {};
    wp::float64 adj_199 = {};
    wp::float64 adj_200 = {};
    wp::float64 adj_201 = {};
    wp::float64 adj_202 = {};
    wp::float64 adj_203 = {};
    wp::float64 adj_204 = {};
    wp::float64 adj_205 = {};
    wp::float64 adj_206 = {};
    wp::float32 adj_207 = {};
    wp::float64 adj_208 = {};
    wp::float64 adj_209 = {};
    wp::float64 adj_210 = {};
    wp::float32 adj_211 = {};
    wp::float64 adj_212 = {};
    wp::float64 adj_213 = {};
    wp::float64 adj_214 = {};
    wp::float32 adj_215 = {};
    wp::float64 adj_216 = {};
    wp::float64 adj_217 = {};
    wp::float64 adj_218 = {};
    wp::float64 adj_219 = {};
    wp::float64 adj_220 = {};
    wp::float64 adj_221 = {};
    wp::float64 adj_222 = {};
    wp::float64 adj_223 = {};
    wp::float64 adj_224 = {};
    wp::float32 adj_225 = {};
    wp::float64 adj_226 = {};
    wp::float64 adj_227 = {};
    wp::float64 adj_228 = {};
    wp::float64 adj_229 = {};
    wp::float32 adj_230 = {};
    wp::float64 adj_231 = {};
    wp::float64 adj_232 = {};
    wp::float64 adj_233 = {};
    wp::float32 adj_234 = {};
    wp::float64 adj_235 = {};
    wp::float64 adj_236 = {};
    wp::float64 adj_237 = {};
    wp::float64 adj_238 = {};
    wp::float64 adj_239 = {};
    wp::vec_t<12,wp::float64> adj_240 = {};
    //---------
    // forward
    // def g_PT(                                                                              <L 5>
    // t11 = -v11 + v01                                                                       <L 23>
    var_0 = wp::neg(var_v11);
    var_1 = wp::add(var_0, var_v01);
    // t12 = -v12 + v02                                                                       <L 24>
    var_2 = wp::neg(var_v12);
    var_3 = wp::add(var_2, var_v02);
    // t13 = -v13 + v03                                                                       <L 25>
    var_4 = wp::neg(var_v13);
    var_5 = wp::add(var_4, var_v03);
    // t14 = -v21 + v11                                                                       <L 26>
    var_6 = wp::neg(var_v21);
    var_7 = wp::add(var_6, var_v11);
    // t15 = -v22 + v12                                                                       <L 27>
    var_8 = wp::neg(var_v22);
    var_9 = wp::add(var_8, var_v12);
    // t16 = -v23 + v13                                                                       <L 28>
    var_10 = wp::neg(var_v23);
    var_11 = wp::add(var_10, var_v13);
    // t17 = -v31 + v11                                                                       <L 29>
    var_12 = wp::neg(var_v31);
    var_13 = wp::add(var_12, var_v11);
    // t18 = -v32 + v12                                                                       <L 30>
    var_14 = wp::neg(var_v32);
    var_15 = wp::add(var_14, var_v12);
    // t19 = -v33 + v13                                                                       <L 31>
    var_16 = wp::neg(var_v33);
    var_17 = wp::add(var_16, var_v13);
    // t20 = -v31 + v21                                                                       <L 32>
    var_18 = wp::neg(var_v31);
    var_19 = wp::add(var_18, var_v21);
    // t21 = -v32 + v22                                                                       <L 33>
    var_20 = wp::neg(var_v32);
    var_21 = wp::add(var_20, var_v22);
    // t22 = -v33 + v23                                                                       <L 34>
    var_22 = wp::neg(var_v33);
    var_23 = wp::add(var_22, var_v23);
    // t32 = t14 * t18 + -(t15 * t17)                                                         <L 35>
    var_24 = wp::mul(var_7, var_15);
    var_25 = wp::mul(var_9, var_13);
    var_26 = wp::neg(var_25);
    var_27 = wp::add(var_24, var_26);
    // t33 = t14 * t19 + -(t16 * t17)                                                         <L 36>
    var_28 = wp::mul(var_7, var_17);
    var_29 = wp::mul(var_11, var_13);
    var_30 = wp::neg(var_29);
    var_31 = wp::add(var_28, var_30);
    // t34 = t15 * t19 + -(t16 * t18)                                                         <L 37>
    var_32 = wp::mul(var_9, var_17);
    var_33 = wp::mul(var_11, var_15);
    var_34 = wp::neg(var_33);
    var_35 = wp::add(var_32, var_34);
    // t43_1 = wp.float64(1.0) / ((t32 * t32 + t33 * t33) + t34 * t34)                        <L 38>
    var_37 = wp::float64(var_36);
    var_38 = wp::mul(var_27, var_27);
    var_39 = wp::mul(var_31, var_31);
    var_40 = wp::add(var_38, var_39);
    var_41 = wp::mul(var_35, var_35);
    var_42 = wp::add(var_40, var_41);
    var_43 = wp::div(var_37, var_42);
    // t45_1 = (t13 * t32 + t11 * t34) + -(t12 * t33)                                         <L 39>
    var_44 = wp::mul(var_5, var_27);
    var_45 = wp::mul(var_1, var_35);
    var_46 = wp::add(var_44, var_45);
    var_47 = wp::mul(var_3, var_31);
    var_48 = wp::neg(var_47);
    var_49 = wp::add(var_46, var_48);
    // t44 = t43_1 * t43_1                                                                    <L 40>
    var_50 = wp::mul(var_43, var_43);
    // t46 = t45_1 * t45_1                                                                    <L 41>
    var_51 = wp::mul(var_49, var_49);
    // t45 = t45_1 * t43_1                                                                    <L 42>
    var_52 = wp::mul(var_49, var_43);
    // t43 = t44 * t46                                                                        <L 43>
    var_53 = wp::mul(var_50, var_51);
    // g = wp.vector(                                                                         <L 44>
    // t34 * t43_1 * t45_1 * wp.float64(2.0),                                                 <L 45>
    var_54 = wp::mul(var_35, var_43);
    var_55 = wp::mul(var_54, var_49);
    var_57 = wp::float64(var_56);
    var_58 = wp::mul(var_55, var_57);
    // t33 * t43_1 * t45_1 * -wp.float64(2.0),                                                <L 46>
    var_59 = wp::mul(var_31, var_43);
    var_60 = wp::mul(var_59, var_49);
    var_62 = wp::float64(var_61);
    var_63 = wp::neg(var_62);
    var_64 = wp::mul(var_60, var_63);
    // t32 * t43_1 * t45_1 * wp.float64(2.0),                                                 <L 47>
    var_65 = wp::mul(var_27, var_43);
    var_66 = wp::mul(var_65, var_49);
    var_68 = wp::float64(var_67);
    var_69 = wp::mul(var_66, var_68);
    // -t44 * t46 * (t21 * t32 * wp.float64(2.0) + t22 * t33 * wp.float64(2.0)) - t45 * ((t34 + t12 * t22) - t13 * t21) * wp.float64(2.0),       <L 48>
    var_70 = wp::neg(var_50);
    var_71 = wp::mul(var_70, var_51);
    var_72 = wp::mul(var_21, var_27);
    var_74 = wp::float64(var_73);
    var_75 = wp::mul(var_72, var_74);
    var_76 = wp::mul(var_23, var_31);
    var_78 = wp::float64(var_77);
    var_79 = wp::mul(var_76, var_78);
    var_80 = wp::add(var_75, var_79);
    var_81 = wp::mul(var_71, var_80);
    var_82 = wp::mul(var_3, var_23);
    var_83 = wp::add(var_35, var_82);
    var_84 = wp::mul(var_5, var_21);
    var_85 = wp::sub(var_83, var_84);
    var_86 = wp::mul(var_52, var_85);
    var_88 = wp::float64(var_87);
    var_89 = wp::mul(var_86, var_88);
    var_90 = wp::sub(var_81, var_89);
    // t43 * (t20 * t32 * wp.float64(2.0) - t22 * t34 * wp.float64(2.0)) + t45 * ((t33 + t11 * t22) - t13 * t20) * wp.float64(2.0),       <L 49>
    var_91 = wp::mul(var_19, var_27);
    var_93 = wp::float64(var_92);
    var_94 = wp::mul(var_91, var_93);
    var_95 = wp::mul(var_23, var_35);
    var_97 = wp::float64(var_96);
    var_98 = wp::mul(var_95, var_97);
    var_99 = wp::sub(var_94, var_98);
    var_100 = wp::mul(var_53, var_99);
    var_101 = wp::mul(var_1, var_23);
    var_102 = wp::add(var_31, var_101);
    var_103 = wp::mul(var_5, var_19);
    var_104 = wp::sub(var_102, var_103);
    var_105 = wp::mul(var_52, var_104);
    var_107 = wp::float64(var_106);
    var_108 = wp::mul(var_105, var_107);
    var_109 = wp::add(var_100, var_108);
    // t43 * (t20 * t33 * wp.float64(2.0) + t21 * t34 * wp.float64(2.0)) - t45 * ((t32 + t11 * t21) - t12 * t20) * wp.float64(2.0),       <L 50>
    var_110 = wp::mul(var_19, var_31);
    var_112 = wp::float64(var_111);
    var_113 = wp::mul(var_110, var_112);
    var_114 = wp::mul(var_21, var_35);
    var_116 = wp::float64(var_115);
    var_117 = wp::mul(var_114, var_116);
    var_118 = wp::add(var_113, var_117);
    var_119 = wp::mul(var_53, var_118);
    var_120 = wp::mul(var_1, var_21);
    var_121 = wp::add(var_27, var_120);
    var_122 = wp::mul(var_3, var_19);
    var_123 = wp::sub(var_121, var_122);
    var_124 = wp::mul(var_52, var_123);
    var_126 = wp::float64(var_125);
    var_127 = wp::mul(var_124, var_126);
    var_128 = wp::sub(var_119, var_127);
    // t45 * (t12 * t19 - t13 * t18) * wp.float64(2.0) + t43 * (t18 * t32 * wp.float64(2.0) + t19 * t33 * wp.float64(2.0)),       <L 51>
    var_129 = wp::mul(var_3, var_17);
    var_130 = wp::mul(var_5, var_15);
    var_131 = wp::sub(var_129, var_130);
    var_132 = wp::mul(var_52, var_131);
    var_134 = wp::float64(var_133);
    var_135 = wp::mul(var_132, var_134);
    var_136 = wp::mul(var_15, var_27);
    var_138 = wp::float64(var_137);
    var_139 = wp::mul(var_136, var_138);
    var_140 = wp::mul(var_17, var_31);
    var_142 = wp::float64(var_141);
    var_143 = wp::mul(var_140, var_142);
    var_144 = wp::add(var_139, var_143);
    var_145 = wp::mul(var_53, var_144);
    var_146 = wp::add(var_135, var_145);
    // t45 * (t11 * t19 - t13 * t17) * -wp.float64(2.0) - t43 * (t17 * t32 * wp.float64(2.0) - t19 * t34 * wp.float64(2.0)),       <L 52>
    var_147 = wp::mul(var_1, var_17);
    var_148 = wp::mul(var_5, var_13);
    var_149 = wp::sub(var_147, var_148);
    var_150 = wp::mul(var_52, var_149);
    var_152 = wp::float64(var_151);
    var_153 = wp::neg(var_152);
    var_154 = wp::mul(var_150, var_153);
    var_155 = wp::mul(var_13, var_27);
    var_157 = wp::float64(var_156);
    var_158 = wp::mul(var_155, var_157);
    var_159 = wp::mul(var_17, var_35);
    var_161 = wp::float64(var_160);
    var_162 = wp::mul(var_159, var_161);
    var_163 = wp::sub(var_158, var_162);
    var_164 = wp::mul(var_53, var_163);
    var_165 = wp::sub(var_154, var_164);
    // t45 * (t11 * t18 - t12 * t17) * wp.float64(2.0) - t43 * (t17 * t33 * wp.float64(2.0) + t18 * t34 * wp.float64(2.0)),       <L 53>
    var_166 = wp::mul(var_1, var_15);
    var_167 = wp::mul(var_3, var_13);
    var_168 = wp::sub(var_166, var_167);
    var_169 = wp::mul(var_52, var_168);
    var_171 = wp::float64(var_170);
    var_172 = wp::mul(var_169, var_171);
    var_173 = wp::mul(var_13, var_31);
    var_175 = wp::float64(var_174);
    var_176 = wp::mul(var_173, var_175);
    var_177 = wp::mul(var_15, var_35);
    var_179 = wp::float64(var_178);
    var_180 = wp::mul(var_177, var_179);
    var_181 = wp::add(var_176, var_180);
    var_182 = wp::mul(var_53, var_181);
    var_183 = wp::sub(var_172, var_182);
    // t45 * (t12 * t16 - t13 * t15) * -wp.float64(2.0) - t43 * (t15 * t32 * wp.float64(2.0) + t16 * t33 * wp.float64(2.0)),       <L 54>
    var_184 = wp::mul(var_3, var_11);
    var_185 = wp::mul(var_5, var_9);
    var_186 = wp::sub(var_184, var_185);
    var_187 = wp::mul(var_52, var_186);
    var_189 = wp::float64(var_188);
    var_190 = wp::neg(var_189);
    var_191 = wp::mul(var_187, var_190);
    var_192 = wp::mul(var_9, var_27);
    var_194 = wp::float64(var_193);
    var_195 = wp::mul(var_192, var_194);
    var_196 = wp::mul(var_11, var_31);
    var_198 = wp::float64(var_197);
    var_199 = wp::mul(var_196, var_198);
    var_200 = wp::add(var_195, var_199);
    var_201 = wp::mul(var_53, var_200);
    var_202 = wp::sub(var_191, var_201);
    // t45 * (t11 * t16 - t13 * t14) * wp.float64(2.0) + t43 * (t14 * t32 * wp.float64(2.0) - t16 * t34 * wp.float64(2.0)),       <L 55>
    var_203 = wp::mul(var_1, var_11);
    var_204 = wp::mul(var_5, var_7);
    var_205 = wp::sub(var_203, var_204);
    var_206 = wp::mul(var_52, var_205);
    var_208 = wp::float64(var_207);
    var_209 = wp::mul(var_206, var_208);
    var_210 = wp::mul(var_7, var_27);
    var_212 = wp::float64(var_211);
    var_213 = wp::mul(var_210, var_212);
    var_214 = wp::mul(var_11, var_35);
    var_216 = wp::float64(var_215);
    var_217 = wp::mul(var_214, var_216);
    var_218 = wp::sub(var_213, var_217);
    var_219 = wp::mul(var_53, var_218);
    var_220 = wp::add(var_209, var_219);
    // t45 * (t11 * t15 - t12 * t14) * -wp.float64(2.0) + t43 * (t14 * t33 * wp.float64(2.0) + t15 * t34 * wp.float64(2.0)),       <L 56>
    var_221 = wp::mul(var_1, var_9);
    var_222 = wp::mul(var_3, var_7);
    var_223 = wp::sub(var_221, var_222);
    var_224 = wp::mul(var_52, var_223);
    var_226 = wp::float64(var_225);
    var_227 = wp::neg(var_226);
    var_228 = wp::mul(var_224, var_227);
    var_229 = wp::mul(var_7, var_31);
    var_231 = wp::float64(var_230);
    var_232 = wp::mul(var_229, var_231);
    var_233 = wp::mul(var_9, var_35);
    var_235 = wp::float64(var_234);
    var_236 = wp::mul(var_233, var_235);
    var_237 = wp::add(var_232, var_236);
    var_238 = wp::mul(var_53, var_237);
    var_239 = wp::add(var_228, var_238);
    var_240 = wp::vec_t<12,wp::float64>({var_58, var_64, var_69, var_90, var_109, var_128, var_146, var_165, var_183, var_202, var_220, var_239});
    // return g                                                                               <L 58>
    goto label0;
    //---------
    // reverse
    label0:;
    adj_240 += adj_ret;
    // adj: return g                                                                          <L 58>
    wp::adj_vec_t({var_58, var_64, var_69, var_90, var_109, var_128, var_146, var_165, var_183, var_202, var_220, var_239}, {&adj_58, &adj_64, &adj_69, &adj_90, &adj_109, &adj_128, &adj_146, &adj_165, &adj_183, &adj_202, &adj_220, &adj_239}, adj_240);
    wp::adj_add(var_228, var_238, adj_228, adj_238, adj_239);
    wp::adj_mul(var_53, var_237, adj_53, adj_237, adj_238);
    wp::adj_add(var_232, var_236, adj_232, adj_236, adj_237);
    wp::adj_mul(var_233, var_235, adj_233, adj_235, adj_236);
    wp::adj_float64(var_234, adj_234, adj_235);
    wp::adj_mul(var_9, var_35, adj_9, adj_35, adj_233);
    wp::adj_mul(var_229, var_231, adj_229, adj_231, adj_232);
    wp::adj_float64(var_230, adj_230, adj_231);
    wp::adj_mul(var_7, var_31, adj_7, adj_31, adj_229);
    wp::adj_mul(var_224, var_227, adj_224, adj_227, adj_228);
    wp::adj_neg(var_226, adj_226, adj_227);
    wp::adj_float64(var_225, adj_225, adj_226);
    wp::adj_mul(var_52, var_223, adj_52, adj_223, adj_224);
    wp::adj_sub(var_221, var_222, adj_221, adj_222, adj_223);
    wp::adj_mul(var_3, var_7, adj_3, adj_7, adj_222);
    wp::adj_mul(var_1, var_9, adj_1, adj_9, adj_221);
    // adj: t45 * (t11 * t15 - t12 * t14) * -wp.float64(2.0) + t43 * (t14 * t33 * wp.float64(2.0) + t15 * t34 * wp.float64(2.0)),  <L 56>
    wp::adj_add(var_209, var_219, adj_209, adj_219, adj_220);
    wp::adj_mul(var_53, var_218, adj_53, adj_218, adj_219);
    wp::adj_sub(var_213, var_217, adj_213, adj_217, adj_218);
    wp::adj_mul(var_214, var_216, adj_214, adj_216, adj_217);
    wp::adj_float64(var_215, adj_215, adj_216);
    wp::adj_mul(var_11, var_35, adj_11, adj_35, adj_214);
    wp::adj_mul(var_210, var_212, adj_210, adj_212, adj_213);
    wp::adj_float64(var_211, adj_211, adj_212);
    wp::adj_mul(var_7, var_27, adj_7, adj_27, adj_210);
    wp::adj_mul(var_206, var_208, adj_206, adj_208, adj_209);
    wp::adj_float64(var_207, adj_207, adj_208);
    wp::adj_mul(var_52, var_205, adj_52, adj_205, adj_206);
    wp::adj_sub(var_203, var_204, adj_203, adj_204, adj_205);
    wp::adj_mul(var_5, var_7, adj_5, adj_7, adj_204);
    wp::adj_mul(var_1, var_11, adj_1, adj_11, adj_203);
    // adj: t45 * (t11 * t16 - t13 * t14) * wp.float64(2.0) + t43 * (t14 * t32 * wp.float64(2.0) - t16 * t34 * wp.float64(2.0)),  <L 55>
    wp::adj_sub(var_191, var_201, adj_191, adj_201, adj_202);
    wp::adj_mul(var_53, var_200, adj_53, adj_200, adj_201);
    wp::adj_add(var_195, var_199, adj_195, adj_199, adj_200);
    wp::adj_mul(var_196, var_198, adj_196, adj_198, adj_199);
    wp::adj_float64(var_197, adj_197, adj_198);
    wp::adj_mul(var_11, var_31, adj_11, adj_31, adj_196);
    wp::adj_mul(var_192, var_194, adj_192, adj_194, adj_195);
    wp::adj_float64(var_193, adj_193, adj_194);
    wp::adj_mul(var_9, var_27, adj_9, adj_27, adj_192);
    wp::adj_mul(var_187, var_190, adj_187, adj_190, adj_191);
    wp::adj_neg(var_189, adj_189, adj_190);
    wp::adj_float64(var_188, adj_188, adj_189);
    wp::adj_mul(var_52, var_186, adj_52, adj_186, adj_187);
    wp::adj_sub(var_184, var_185, adj_184, adj_185, adj_186);
    wp::adj_mul(var_5, var_9, adj_5, adj_9, adj_185);
    wp::adj_mul(var_3, var_11, adj_3, adj_11, adj_184);
    // adj: t45 * (t12 * t16 - t13 * t15) * -wp.float64(2.0) - t43 * (t15 * t32 * wp.float64(2.0) + t16 * t33 * wp.float64(2.0)),  <L 54>
    wp::adj_sub(var_172, var_182, adj_172, adj_182, adj_183);
    wp::adj_mul(var_53, var_181, adj_53, adj_181, adj_182);
    wp::adj_add(var_176, var_180, adj_176, adj_180, adj_181);
    wp::adj_mul(var_177, var_179, adj_177, adj_179, adj_180);
    wp::adj_float64(var_178, adj_178, adj_179);
    wp::adj_mul(var_15, var_35, adj_15, adj_35, adj_177);
    wp::adj_mul(var_173, var_175, adj_173, adj_175, adj_176);
    wp::adj_float64(var_174, adj_174, adj_175);
    wp::adj_mul(var_13, var_31, adj_13, adj_31, adj_173);
    wp::adj_mul(var_169, var_171, adj_169, adj_171, adj_172);
    wp::adj_float64(var_170, adj_170, adj_171);
    wp::adj_mul(var_52, var_168, adj_52, adj_168, adj_169);
    wp::adj_sub(var_166, var_167, adj_166, adj_167, adj_168);
    wp::adj_mul(var_3, var_13, adj_3, adj_13, adj_167);
    wp::adj_mul(var_1, var_15, adj_1, adj_15, adj_166);
    // adj: t45 * (t11 * t18 - t12 * t17) * wp.float64(2.0) - t43 * (t17 * t33 * wp.float64(2.0) + t18 * t34 * wp.float64(2.0)),  <L 53>
    wp::adj_sub(var_154, var_164, adj_154, adj_164, adj_165);
    wp::adj_mul(var_53, var_163, adj_53, adj_163, adj_164);
    wp::adj_sub(var_158, var_162, adj_158, adj_162, adj_163);
    wp::adj_mul(var_159, var_161, adj_159, adj_161, adj_162);
    wp::adj_float64(var_160, adj_160, adj_161);
    wp::adj_mul(var_17, var_35, adj_17, adj_35, adj_159);
    wp::adj_mul(var_155, var_157, adj_155, adj_157, adj_158);
    wp::adj_float64(var_156, adj_156, adj_157);
    wp::adj_mul(var_13, var_27, adj_13, adj_27, adj_155);
    wp::adj_mul(var_150, var_153, adj_150, adj_153, adj_154);
    wp::adj_neg(var_152, adj_152, adj_153);
    wp::adj_float64(var_151, adj_151, adj_152);
    wp::adj_mul(var_52, var_149, adj_52, adj_149, adj_150);
    wp::adj_sub(var_147, var_148, adj_147, adj_148, adj_149);
    wp::adj_mul(var_5, var_13, adj_5, adj_13, adj_148);
    wp::adj_mul(var_1, var_17, adj_1, adj_17, adj_147);
    // adj: t45 * (t11 * t19 - t13 * t17) * -wp.float64(2.0) - t43 * (t17 * t32 * wp.float64(2.0) - t19 * t34 * wp.float64(2.0)),  <L 52>
    wp::adj_add(var_135, var_145, adj_135, adj_145, adj_146);
    wp::adj_mul(var_53, var_144, adj_53, adj_144, adj_145);
    wp::adj_add(var_139, var_143, adj_139, adj_143, adj_144);
    wp::adj_mul(var_140, var_142, adj_140, adj_142, adj_143);
    wp::adj_float64(var_141, adj_141, adj_142);
    wp::adj_mul(var_17, var_31, adj_17, adj_31, adj_140);
    wp::adj_mul(var_136, var_138, adj_136, adj_138, adj_139);
    wp::adj_float64(var_137, adj_137, adj_138);
    wp::adj_mul(var_15, var_27, adj_15, adj_27, adj_136);
    wp::adj_mul(var_132, var_134, adj_132, adj_134, adj_135);
    wp::adj_float64(var_133, adj_133, adj_134);
    wp::adj_mul(var_52, var_131, adj_52, adj_131, adj_132);
    wp::adj_sub(var_129, var_130, adj_129, adj_130, adj_131);
    wp::adj_mul(var_5, var_15, adj_5, adj_15, adj_130);
    wp::adj_mul(var_3, var_17, adj_3, adj_17, adj_129);
    // adj: t45 * (t12 * t19 - t13 * t18) * wp.float64(2.0) + t43 * (t18 * t32 * wp.float64(2.0) + t19 * t33 * wp.float64(2.0)),  <L 51>
    wp::adj_sub(var_119, var_127, adj_119, adj_127, adj_128);
    wp::adj_mul(var_124, var_126, adj_124, adj_126, adj_127);
    wp::adj_float64(var_125, adj_125, adj_126);
    wp::adj_mul(var_52, var_123, adj_52, adj_123, adj_124);
    wp::adj_sub(var_121, var_122, adj_121, adj_122, adj_123);
    wp::adj_mul(var_3, var_19, adj_3, adj_19, adj_122);
    wp::adj_add(var_27, var_120, adj_27, adj_120, adj_121);
    wp::adj_mul(var_1, var_21, adj_1, adj_21, adj_120);
    wp::adj_mul(var_53, var_118, adj_53, adj_118, adj_119);
    wp::adj_add(var_113, var_117, adj_113, adj_117, adj_118);
    wp::adj_mul(var_114, var_116, adj_114, adj_116, adj_117);
    wp::adj_float64(var_115, adj_115, adj_116);
    wp::adj_mul(var_21, var_35, adj_21, adj_35, adj_114);
    wp::adj_mul(var_110, var_112, adj_110, adj_112, adj_113);
    wp::adj_float64(var_111, adj_111, adj_112);
    wp::adj_mul(var_19, var_31, adj_19, adj_31, adj_110);
    // adj: t43 * (t20 * t33 * wp.float64(2.0) + t21 * t34 * wp.float64(2.0)) - t45 * ((t32 + t11 * t21) - t12 * t20) * wp.float64(2.0),  <L 50>
    wp::adj_add(var_100, var_108, adj_100, adj_108, adj_109);
    wp::adj_mul(var_105, var_107, adj_105, adj_107, adj_108);
    wp::adj_float64(var_106, adj_106, adj_107);
    wp::adj_mul(var_52, var_104, adj_52, adj_104, adj_105);
    wp::adj_sub(var_102, var_103, adj_102, adj_103, adj_104);
    wp::adj_mul(var_5, var_19, adj_5, adj_19, adj_103);
    wp::adj_add(var_31, var_101, adj_31, adj_101, adj_102);
    wp::adj_mul(var_1, var_23, adj_1, adj_23, adj_101);
    wp::adj_mul(var_53, var_99, adj_53, adj_99, adj_100);
    wp::adj_sub(var_94, var_98, adj_94, adj_98, adj_99);
    wp::adj_mul(var_95, var_97, adj_95, adj_97, adj_98);
    wp::adj_float64(var_96, adj_96, adj_97);
    wp::adj_mul(var_23, var_35, adj_23, adj_35, adj_95);
    wp::adj_mul(var_91, var_93, adj_91, adj_93, adj_94);
    wp::adj_float64(var_92, adj_92, adj_93);
    wp::adj_mul(var_19, var_27, adj_19, adj_27, adj_91);
    // adj: t43 * (t20 * t32 * wp.float64(2.0) - t22 * t34 * wp.float64(2.0)) + t45 * ((t33 + t11 * t22) - t13 * t20) * wp.float64(2.0),  <L 49>
    wp::adj_sub(var_81, var_89, adj_81, adj_89, adj_90);
    wp::adj_mul(var_86, var_88, adj_86, adj_88, adj_89);
    wp::adj_float64(var_87, adj_87, adj_88);
    wp::adj_mul(var_52, var_85, adj_52, adj_85, adj_86);
    wp::adj_sub(var_83, var_84, adj_83, adj_84, adj_85);
    wp::adj_mul(var_5, var_21, adj_5, adj_21, adj_84);
    wp::adj_add(var_35, var_82, adj_35, adj_82, adj_83);
    wp::adj_mul(var_3, var_23, adj_3, adj_23, adj_82);
    wp::adj_mul(var_71, var_80, adj_71, adj_80, adj_81);
    wp::adj_add(var_75, var_79, adj_75, adj_79, adj_80);
    wp::adj_mul(var_76, var_78, adj_76, adj_78, adj_79);
    wp::adj_float64(var_77, adj_77, adj_78);
    wp::adj_mul(var_23, var_31, adj_23, adj_31, adj_76);
    wp::adj_mul(var_72, var_74, adj_72, adj_74, adj_75);
    wp::adj_float64(var_73, adj_73, adj_74);
    wp::adj_mul(var_21, var_27, adj_21, adj_27, adj_72);
    wp::adj_mul(var_70, var_51, adj_70, adj_51, adj_71);
    wp::adj_neg(var_50, adj_50, adj_70);
    // adj: -t44 * t46 * (t21 * t32 * wp.float64(2.0) + t22 * t33 * wp.float64(2.0)) - t45 * ((t34 + t12 * t22) - t13 * t21) * wp.float64(2.0),  <L 48>
    wp::adj_mul(var_66, var_68, adj_66, adj_68, adj_69);
    wp::adj_float64(var_67, adj_67, adj_68);
    wp::adj_mul(var_65, var_49, adj_65, adj_49, adj_66);
    wp::adj_mul(var_27, var_43, adj_27, adj_43, adj_65);
    // adj: t32 * t43_1 * t45_1 * wp.float64(2.0),                                            <L 47>
    wp::adj_mul(var_60, var_63, adj_60, adj_63, adj_64);
    wp::adj_neg(var_62, adj_62, adj_63);
    wp::adj_float64(var_61, adj_61, adj_62);
    wp::adj_mul(var_59, var_49, adj_59, adj_49, adj_60);
    wp::adj_mul(var_31, var_43, adj_31, adj_43, adj_59);
    // adj: t33 * t43_1 * t45_1 * -wp.float64(2.0),                                           <L 46>
    wp::adj_mul(var_55, var_57, adj_55, adj_57, adj_58);
    wp::adj_float64(var_56, adj_56, adj_57);
    wp::adj_mul(var_54, var_49, adj_54, adj_49, adj_55);
    wp::adj_mul(var_35, var_43, adj_35, adj_43, adj_54);
    // adj: t34 * t43_1 * t45_1 * wp.float64(2.0),                                            <L 45>
    // adj: g = wp.vector(                                                                    <L 44>
    wp::adj_mul(var_50, var_51, adj_50, adj_51, adj_53);
    // adj: t43 = t44 * t46                                                                   <L 43>
    wp::adj_mul(var_49, var_43, adj_49, adj_43, adj_52);
    // adj: t45 = t45_1 * t43_1                                                               <L 42>
    wp::adj_mul(var_49, var_49, adj_49, adj_49, adj_51);
    // adj: t46 = t45_1 * t45_1                                                               <L 41>
    wp::adj_mul(var_43, var_43, adj_43, adj_43, adj_50);
    // adj: t44 = t43_1 * t43_1                                                               <L 40>
    wp::adj_add(var_46, var_48, adj_46, adj_48, adj_49);
    wp::adj_neg(var_47, adj_47, adj_48);
    wp::adj_mul(var_3, var_31, adj_3, adj_31, adj_47);
    wp::adj_add(var_44, var_45, adj_44, adj_45, adj_46);
    wp::adj_mul(var_1, var_35, adj_1, adj_35, adj_45);
    wp::adj_mul(var_5, var_27, adj_5, adj_27, adj_44);
    // adj: t45_1 = (t13 * t32 + t11 * t34) + -(t12 * t33)                                    <L 39>
    wp::adj_div(var_37, var_42, var_43, adj_37, adj_42, adj_43);
    wp::adj_add(var_40, var_41, adj_40, adj_41, adj_42);
    wp::adj_mul(var_35, var_35, adj_35, adj_35, adj_41);
    wp::adj_add(var_38, var_39, adj_38, adj_39, adj_40);
    wp::adj_mul(var_31, var_31, adj_31, adj_31, adj_39);
    wp::adj_mul(var_27, var_27, adj_27, adj_27, adj_38);
    wp::adj_float64(var_36, adj_36, adj_37);
    // adj: t43_1 = wp.float64(1.0) / ((t32 * t32 + t33 * t33) + t34 * t34)                   <L 38>
    wp::adj_add(var_32, var_34, adj_32, adj_34, adj_35);
    wp::adj_neg(var_33, adj_33, adj_34);
    wp::adj_mul(var_11, var_15, adj_11, adj_15, adj_33);
    wp::adj_mul(var_9, var_17, adj_9, adj_17, adj_32);
    // adj: t34 = t15 * t19 + -(t16 * t18)                                                    <L 37>
    wp::adj_add(var_28, var_30, adj_28, adj_30, adj_31);
    wp::adj_neg(var_29, adj_29, adj_30);
    wp::adj_mul(var_11, var_13, adj_11, adj_13, adj_29);
    wp::adj_mul(var_7, var_17, adj_7, adj_17, adj_28);
    // adj: t33 = t14 * t19 + -(t16 * t17)                                                    <L 36>
    wp::adj_add(var_24, var_26, adj_24, adj_26, adj_27);
    wp::adj_neg(var_25, adj_25, adj_26);
    wp::adj_mul(var_9, var_13, adj_9, adj_13, adj_25);
    wp::adj_mul(var_7, var_15, adj_7, adj_15, adj_24);
    // adj: t32 = t14 * t18 + -(t15 * t17)                                                    <L 35>
    wp::adj_add(var_22, var_v23, adj_22, adj_v23, adj_23);
    wp::adj_neg(var_v33, adj_v33, adj_22);
    // adj: t22 = -v33 + v23                                                                  <L 34>
    wp::adj_add(var_20, var_v22, adj_20, adj_v22, adj_21);
    wp::adj_neg(var_v32, adj_v32, adj_20);
    // adj: t21 = -v32 + v22                                                                  <L 33>
    wp::adj_add(var_18, var_v21, adj_18, adj_v21, adj_19);
    wp::adj_neg(var_v31, adj_v31, adj_18);
    // adj: t20 = -v31 + v21                                                                  <L 32>
    wp::adj_add(var_16, var_v13, adj_16, adj_v13, adj_17);
    wp::adj_neg(var_v33, adj_v33, adj_16);
    // adj: t19 = -v33 + v13                                                                  <L 31>
    wp::adj_add(var_14, var_v12, adj_14, adj_v12, adj_15);
    wp::adj_neg(var_v32, adj_v32, adj_14);
    // adj: t18 = -v32 + v12                                                                  <L 30>
    wp::adj_add(var_12, var_v11, adj_12, adj_v11, adj_13);
    wp::adj_neg(var_v31, adj_v31, adj_12);
    // adj: t17 = -v31 + v11                                                                  <L 29>
    wp::adj_add(var_10, var_v13, adj_10, adj_v13, adj_11);
    wp::adj_neg(var_v23, adj_v23, adj_10);
    // adj: t16 = -v23 + v13                                                                  <L 28>
    wp::adj_add(var_8, var_v12, adj_8, adj_v12, adj_9);
    wp::adj_neg(var_v22, adj_v22, adj_8);
    // adj: t15 = -v22 + v12                                                                  <L 27>
    wp::adj_add(var_6, var_v11, adj_6, adj_v11, adj_7);
    wp::adj_neg(var_v21, adj_v21, adj_6);
    // adj: t14 = -v21 + v11                                                                  <L 26>
    wp::adj_add(var_4, var_v03, adj_4, adj_v03, adj_5);
    wp::adj_neg(var_v13, adj_v13, adj_4);
    // adj: t13 = -v13 + v03                                                                  <L 25>
    wp::adj_add(var_2, var_v02, adj_2, adj_v02, adj_3);
    wp::adj_neg(var_v12, adj_v12, adj_2);
    // adj: t12 = -v12 + v02                                                                  <L 24>
    wp::adj_add(var_0, var_v01, adj_0, adj_v01, adj_1);
    wp::adj_neg(var_v11, adj_v11, adj_0);
    // adj: t11 = -v11 + v01                                                                  <L 23>
    // adj: def g_PT(                                                                         <L 5>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/point_plane_distance.py:569
static CUDA_CALLABLE wp::vec_t<12,wp::float64> point_plane_distance_gradient_0(
    wp::vec_t<3,wp::float64> var_p,
    wp::vec_t<3,wp::float64> var_t0,
    wp::vec_t<3,wp::float64> var_t1,
    wp::vec_t<3,wp::float64> var_t2)
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
    // def point_plane_distance_gradient(p: wp.vec3d, t0: wp.vec3d, t1: wp.vec3d, t2: wp.vec3d):       <L 570>
    // return g_PT(p[0], p[1], p[2], t0[0], t0[1], t0[2], t1[0], t1[1], t1[2], t2[0], t2[1], t2[2])       <L 571>
    var_1 = wp::extract(var_p, var_0);
    var_3 = wp::extract(var_p, var_2);
    var_5 = wp::extract(var_p, var_4);
    var_7 = wp::extract(var_t0, var_6);
    var_9 = wp::extract(var_t0, var_8);
    var_11 = wp::extract(var_t0, var_10);
    var_13 = wp::extract(var_t1, var_12);
    var_15 = wp::extract(var_t1, var_14);
    var_17 = wp::extract(var_t1, var_16);
    var_19 = wp::extract(var_t2, var_18);
    var_21 = wp::extract(var_t2, var_20);
    var_23 = wp::extract(var_t2, var_22);
    var_24 = g_PT_0(var_1, var_3, var_5, var_7, var_9, var_11, var_13, var_15, var_17, var_19, var_21, var_23);
    return var_24;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/point_plane_distance.py:569
static CUDA_CALLABLE void adj_point_plane_distance_gradient_0(
    wp::vec_t<3,wp::float64> var_p,
    wp::vec_t<3,wp::float64> var_t0,
    wp::vec_t<3,wp::float64> var_t1,
    wp::vec_t<3,wp::float64> var_t2,
    wp::vec_t<3,wp::float64> & adj_p,
    wp::vec_t<3,wp::float64> & adj_t0,
    wp::vec_t<3,wp::float64> & adj_t1,
    wp::vec_t<3,wp::float64> & adj_t2,
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
    // def point_plane_distance_gradient(p: wp.vec3d, t0: wp.vec3d, t1: wp.vec3d, t2: wp.vec3d):       <L 570>
    // return g_PT(p[0], p[1], p[2], t0[0], t0[1], t0[2], t1[0], t1[1], t1[2], t2[0], t2[1], t2[2])       <L 571>
    var_1 = wp::extract(var_p, var_0);
    var_3 = wp::extract(var_p, var_2);
    var_5 = wp::extract(var_p, var_4);
    var_7 = wp::extract(var_t0, var_6);
    var_9 = wp::extract(var_t0, var_8);
    var_11 = wp::extract(var_t0, var_10);
    var_13 = wp::extract(var_t1, var_12);
    var_15 = wp::extract(var_t1, var_14);
    var_17 = wp::extract(var_t1, var_16);
    var_19 = wp::extract(var_t2, var_18);
    var_21 = wp::extract(var_t2, var_20);
    var_23 = wp::extract(var_t2, var_22);
    var_24 = g_PT_0(var_1, var_3, var_5, var_7, var_9, var_11, var_13, var_15, var_17, var_19, var_21, var_23);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_24 += adj_ret;
    adj_g_PT_0(var_1, var_3, var_5, var_7, var_9, var_11, var_13, var_15, var_17, var_19, var_21, var_23, adj_1, adj_3, adj_5, adj_7, adj_9, adj_11, adj_13, adj_15, adj_17, adj_19, adj_21, adj_23, adj_24);
    wp::adj_extract(var_t2, var_22, adj_t2, adj_22, adj_23);
    wp::adj_extract(var_t2, var_20, adj_t2, adj_20, adj_21);
    wp::adj_extract(var_t2, var_18, adj_t2, adj_18, adj_19);
    wp::adj_extract(var_t1, var_16, adj_t1, adj_16, adj_17);
    wp::adj_extract(var_t1, var_14, adj_t1, adj_14, adj_15);
    wp::adj_extract(var_t1, var_12, adj_t1, adj_12, adj_13);
    wp::adj_extract(var_t0, var_10, adj_t0, adj_10, adj_11);
    wp::adj_extract(var_t0, var_8, adj_t0, adj_8, adj_9);
    wp::adj_extract(var_t0, var_6, adj_t0, adj_6, adj_7);
    wp::adj_extract(var_p, var_4, adj_p, adj_4, adj_5);
    wp::adj_extract(var_p, var_2, adj_p, adj_2, adj_3);
    wp::adj_extract(var_p, var_0, adj_p, adj_0, adj_1);
    // adj: return g_PT(p[0], p[1], p[2], t0[0], t0[1], t0[2], t1[0], t1[1], t1[2], t2[0], t2[1], t2[2])  <L 571>
    // adj: def point_plane_distance_gradient(p: wp.vec3d, t0: wp.vec3d, t1: wp.vec3d, t2: wp.vec3d):  <L 570>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/point_triangle_distance.py:30
static CUDA_CALLABLE wp::vec_t<12,wp::float64> point_triangle_distance_gradient_0(
    wp::vec_t<3,wp::float64> var_p,
    wp::vec_t<3,wp::float64> var_t0,
    wp::vec_t<3,wp::float64> var_t1,
    wp::vec_t<3,wp::float64> var_t2)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 12;
    wp::vec_t<12,wp::float64> var_1;
    const wp::float32 var_2 = 0.0;
    wp::float64 var_3;
    wp::int32 var_4;
    const wp::int32 var_5 = 0;
    bool var_6;
    wp::vec_t<6,wp::float64> var_7;
    const wp::int32 var_8 = 0;
    wp::float64 var_9;
    const wp::int32 var_10 = 1;
    wp::float64 var_11;
    const wp::int32 var_12 = 2;
    wp::float64 var_13;
    const wp::int32 var_14 = 3;
    wp::float64 var_15;
    const wp::int32 var_16 = 4;
    wp::float64 var_17;
    const wp::int32 var_18 = 5;
    wp::float64 var_19;
    wp::vec_t<12,wp::float64> var_20;
    wp::vec_t<12,wp::float64> var_21;
    const wp::int32 var_22 = 1;
    bool var_23;
    wp::vec_t<6,wp::float64> var_24;
    const wp::int32 var_25 = 0;
    wp::float64 var_26;
    const wp::int32 var_27 = 1;
    wp::float64 var_28;
    const wp::int32 var_29 = 2;
    wp::float64 var_30;
    const wp::int32 var_31 = 3;
    wp::float64 var_32;
    const wp::int32 var_33 = 4;
    wp::float64 var_34;
    const wp::int32 var_35 = 5;
    wp::float64 var_36;
    wp::vec_t<12,wp::float64> var_37;
    wp::vec_t<12,wp::float64> var_38;
    const wp::int32 var_39 = 2;
    bool var_40;
    wp::vec_t<6,wp::float64> var_41;
    const wp::int32 var_42 = 0;
    wp::float64 var_43;
    const wp::int32 var_44 = 1;
    wp::float64 var_45;
    const wp::int32 var_46 = 2;
    wp::float64 var_47;
    const wp::int32 var_48 = 3;
    wp::float64 var_49;
    const wp::int32 var_50 = 4;
    wp::float64 var_51;
    const wp::int32 var_52 = 5;
    wp::float64 var_53;
    wp::vec_t<12,wp::float64> var_54;
    wp::vec_t<12,wp::float64> var_55;
    const wp::int32 var_56 = 3;
    bool var_57;
    wp::vec_t<9,wp::float64> var_58;
    const wp::int32 var_59 = 0;
    wp::float64 var_60;
    const wp::int32 var_61 = 1;
    wp::float64 var_62;
    const wp::int32 var_63 = 2;
    wp::float64 var_64;
    const wp::int32 var_65 = 3;
    wp::float64 var_66;
    const wp::int32 var_67 = 4;
    wp::float64 var_68;
    const wp::int32 var_69 = 5;
    wp::float64 var_70;
    const wp::int32 var_71 = 6;
    wp::float64 var_72;
    const wp::int32 var_73 = 7;
    wp::float64 var_74;
    const wp::int32 var_75 = 8;
    wp::float64 var_76;
    wp::vec_t<12,wp::float64> var_77;
    wp::vec_t<12,wp::float64> var_78;
    const wp::int32 var_79 = 4;
    bool var_80;
    wp::vec_t<9,wp::float64> var_81;
    const wp::int32 var_82 = 0;
    wp::float64 var_83;
    const wp::int32 var_84 = 1;
    wp::float64 var_85;
    const wp::int32 var_86 = 2;
    wp::float64 var_87;
    const wp::int32 var_88 = 3;
    wp::float64 var_89;
    const wp::int32 var_90 = 4;
    wp::float64 var_91;
    const wp::int32 var_92 = 5;
    wp::float64 var_93;
    const wp::int32 var_94 = 6;
    wp::float64 var_95;
    const wp::int32 var_96 = 7;
    wp::float64 var_97;
    const wp::int32 var_98 = 8;
    wp::float64 var_99;
    wp::vec_t<12,wp::float64> var_100;
    wp::vec_t<12,wp::float64> var_101;
    const wp::int32 var_102 = 5;
    bool var_103;
    wp::vec_t<9,wp::float64> var_104;
    const wp::int32 var_105 = 0;
    wp::float64 var_106;
    const wp::int32 var_107 = 1;
    wp::float64 var_108;
    const wp::int32 var_109 = 2;
    wp::float64 var_110;
    const wp::int32 var_111 = 3;
    wp::float64 var_112;
    const wp::int32 var_113 = 4;
    wp::float64 var_114;
    const wp::int32 var_115 = 5;
    wp::float64 var_116;
    const wp::int32 var_117 = 6;
    wp::float64 var_118;
    const wp::int32 var_119 = 7;
    wp::float64 var_120;
    const wp::int32 var_121 = 8;
    wp::float64 var_122;
    wp::vec_t<12,wp::float64> var_123;
    wp::vec_t<12,wp::float64> var_124;
    const wp::int32 var_125 = 6;
    bool var_126;
    wp::vec_t<12,wp::float64> var_127;
    wp::vec_t<12,wp::float64> var_128;
    wp::vec_t<12,wp::float64> var_129;
    wp::vec_t<12,wp::float64> var_130;
    wp::vec_t<12,wp::float64> var_131;
    wp::vec_t<12,wp::float64> var_132;
    wp::vec_t<12,wp::float64> var_133;
    wp::vec_t<12,wp::float64> var_134;
    //---------
    // forward
    // def point_triangle_distance_gradient(p: wp.vec3d, t0: wp.vec3d, t1: wp.vec3d, t2: wp.vec3d):       <L 31>
    // grad = wp.vec(length=12, dtype=wp.float64)                                             <L 32>
    var_1 = wp::vec_t<12,wp::float64>();
    // _0 = wp.float64(0.0)                                                                   <L 33>
    var_3 = wp::float64(var_2);
    // dt = point_triangle_distance_type(p, t0, t1, t2)                                       <L 34>
    var_4 = point_triangle_distance_type_0(var_p, var_t0, var_t1, var_t2);
    // if dt == 0:                                                                            <L 35>
    var_6 = (var_4 == var_5);
    if (var_6) {
        // grad_compact0 = point_point_distance_gradient(p, t0)                               <L 36>
        var_7 = point_point_distance_gradient_0(var_p, var_t0);
        // grad = wp.vector(                                                                  <L 37>
        // grad_compact0[0], grad_compact0[1], grad_compact0[2], grad_compact0[3], grad_compact0[4], grad_compact0[5], _0, _0, _0, _0, _0, _0       <L 38>
        var_9 = wp::extract(var_7, var_8);
        var_11 = wp::extract(var_7, var_10);
        var_13 = wp::extract(var_7, var_12);
        var_15 = wp::extract(var_7, var_14);
        var_17 = wp::extract(var_7, var_16);
        var_19 = wp::extract(var_7, var_18);
        var_20 = wp::vec_t<12,wp::float64>({var_9, var_11, var_13, var_15, var_17, var_19, var_3, var_3, var_3, var_3, var_3, var_3});
    }
    var_21 = wp::where(var_6, var_20, var_1);
    if (!var_6) {
        // elif dt == 1:                                                                      <L 40>
        var_23 = (var_4 == var_22);
        if (var_23) {
            // grad_compact1 = point_point_distance_gradient(p, t1)                           <L 41>
            var_24 = point_point_distance_gradient_0(var_p, var_t1);
            // grad = wp.vector(                                                              <L 42>
            // grad_compact1[0], grad_compact1[1], grad_compact1[2], _0, _0, _0, grad_compact1[3], grad_compact1[4], grad_compact1[5], _0, _0, _0       <L 43>
            var_26 = wp::extract(var_24, var_25);
            var_28 = wp::extract(var_24, var_27);
            var_30 = wp::extract(var_24, var_29);
            var_32 = wp::extract(var_24, var_31);
            var_34 = wp::extract(var_24, var_33);
            var_36 = wp::extract(var_24, var_35);
            var_37 = wp::vec_t<12,wp::float64>({var_26, var_28, var_30, var_3, var_3, var_3, var_32, var_34, var_36, var_3, var_3, var_3});
        }
        var_38 = wp::where(var_23, var_37, var_21);
        if (!var_23) {
            // elif dt == 2:                                                                  <L 45>
            var_40 = (var_4 == var_39);
            if (var_40) {
                // grad_compact2 = point_point_distance_gradient(p, t2)                       <L 46>
                var_41 = point_point_distance_gradient_0(var_p, var_t2);
                // grad = wp.vector(                                                          <L 47>
                // grad_compact2[0], grad_compact2[1], grad_compact2[2], _0, _0, _0, _0, _0, _0, grad_compact2[3], grad_compact2[4], grad_compact2[5]       <L 48>
                var_43 = wp::extract(var_41, var_42);
                var_45 = wp::extract(var_41, var_44);
                var_47 = wp::extract(var_41, var_46);
                var_49 = wp::extract(var_41, var_48);
                var_51 = wp::extract(var_41, var_50);
                var_53 = wp::extract(var_41, var_52);
                var_54 = wp::vec_t<12,wp::float64>({var_43, var_45, var_47, var_3, var_3, var_3, var_3, var_3, var_3, var_49, var_51, var_53});
            }
            var_55 = wp::where(var_40, var_54, var_38);
            if (!var_40) {
                // elif dt == 3:                                                              <L 50>
                var_57 = (var_4 == var_56);
                if (var_57) {
                    // grad_compact3 = point_line_distance_gradient(p, t0, t1)                <L 51>
                    var_58 = point_line_distance_gradient_0(var_p, var_t0, var_t1);
                    // grad = wp.vector(                                                      <L 52>
                    // grad_compact3[0],                                                      <L 53>
                    var_60 = wp::extract(var_58, var_59);
                    // grad_compact3[1],                                                      <L 54>
                    var_62 = wp::extract(var_58, var_61);
                    // grad_compact3[2],                                                      <L 55>
                    var_64 = wp::extract(var_58, var_63);
                    // grad_compact3[3],                                                      <L 56>
                    var_66 = wp::extract(var_58, var_65);
                    // grad_compact3[4],                                                      <L 57>
                    var_68 = wp::extract(var_58, var_67);
                    // grad_compact3[5],                                                      <L 58>
                    var_70 = wp::extract(var_58, var_69);
                    // grad_compact3[6],                                                      <L 59>
                    var_72 = wp::extract(var_58, var_71);
                    // grad_compact3[7],                                                      <L 60>
                    var_74 = wp::extract(var_58, var_73);
                    // grad_compact3[8],                                                      <L 61>
                    var_76 = wp::extract(var_58, var_75);
                    // _0,                                                                    <L 62>
                    // _0,                                                                    <L 63>
                    // _0,                                                                    <L 64>
                    var_77 = wp::vec_t<12,wp::float64>({var_60, var_62, var_64, var_66, var_68, var_70, var_72, var_74, var_76, var_3, var_3, var_3});
                }
                var_78 = wp::where(var_57, var_77, var_55);
                if (!var_57) {
                    // elif dt == 4:                                                          <L 66>
                    var_80 = (var_4 == var_79);
                    if (var_80) {
                        // grad_compact4 = point_line_distance_gradient(p, t1, t2)            <L 67>
                        var_81 = point_line_distance_gradient_0(var_p, var_t1, var_t2);
                        // grad = wp.vector(                                                  <L 68>
                        // grad_compact4[0],                                                  <L 69>
                        var_83 = wp::extract(var_81, var_82);
                        // grad_compact4[1],                                                  <L 70>
                        var_85 = wp::extract(var_81, var_84);
                        // grad_compact4[2],                                                  <L 71>
                        var_87 = wp::extract(var_81, var_86);
                        // _0,                                                                <L 72>
                        // _0,                                                                <L 73>
                        // _0,                                                                <L 74>
                        // grad_compact4[3],                                                  <L 75>
                        var_89 = wp::extract(var_81, var_88);
                        // grad_compact4[4],                                                  <L 76>
                        var_91 = wp::extract(var_81, var_90);
                        // grad_compact4[5],                                                  <L 77>
                        var_93 = wp::extract(var_81, var_92);
                        // grad_compact4[6],                                                  <L 78>
                        var_95 = wp::extract(var_81, var_94);
                        // grad_compact4[7],                                                  <L 79>
                        var_97 = wp::extract(var_81, var_96);
                        // grad_compact4[8],                                                  <L 80>
                        var_99 = wp::extract(var_81, var_98);
                        var_100 = wp::vec_t<12,wp::float64>({var_83, var_85, var_87, var_3, var_3, var_3, var_89, var_91, var_93, var_95, var_97, var_99});
                    }
                    var_101 = wp::where(var_80, var_100, var_78);
                    if (!var_80) {
                        // elif dt == 5:                                                      <L 82>
                        var_103 = (var_4 == var_102);
                        if (var_103) {
                            // grad_compact5 = point_line_distance_gradient(p, t0, t2)        <L 83>
                            var_104 = point_line_distance_gradient_0(var_p, var_t0, var_t2);
                            // grad = wp.vector(                                              <L 84>
                            // grad_compact5[0],                                              <L 85>
                            var_106 = wp::extract(var_104, var_105);
                            // grad_compact5[1],                                              <L 86>
                            var_108 = wp::extract(var_104, var_107);
                            // grad_compact5[2],                                              <L 87>
                            var_110 = wp::extract(var_104, var_109);
                            // grad_compact5[3],                                              <L 88>
                            var_112 = wp::extract(var_104, var_111);
                            // grad_compact5[4],                                              <L 89>
                            var_114 = wp::extract(var_104, var_113);
                            // grad_compact5[5],                                              <L 90>
                            var_116 = wp::extract(var_104, var_115);
                            // _0,                                                            <L 91>
                            // _0,                                                            <L 92>
                            // _0,                                                            <L 93>
                            // grad_compact5[6],                                              <L 94>
                            var_118 = wp::extract(var_104, var_117);
                            // grad_compact5[7],                                              <L 95>
                            var_120 = wp::extract(var_104, var_119);
                            // grad_compact5[8],                                              <L 96>
                            var_122 = wp::extract(var_104, var_121);
                            var_123 = wp::vec_t<12,wp::float64>({var_106, var_108, var_110, var_112, var_114, var_116, var_3, var_3, var_3, var_118, var_120, var_122});
                        }
                        var_124 = wp::where(var_103, var_123, var_101);
                        if (!var_103) {
                            // elif dt == 6:                                                  <L 98>
                            var_126 = (var_4 == var_125);
                            if (var_126) {
                                // grad = point_plane_distance_gradient(p, t0, t1, t2)        <L 99>
                                var_127 = point_plane_distance_gradient_0(var_p, var_t0, var_t1, var_t2);
                            }
                            var_128 = wp::where(var_126, var_127, var_124);
                        }
                        var_129 = wp::where(var_103, var_124, var_128);
                    }
                    var_130 = wp::where(var_80, var_101, var_129);
                }
                var_131 = wp::where(var_57, var_78, var_130);
            }
            var_132 = wp::where(var_40, var_55, var_131);
        }
        var_133 = wp::where(var_23, var_38, var_132);
    }
    var_134 = wp::where(var_6, var_21, var_133);
    // return grad                                                                            <L 101>
    return var_134;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/point_triangle_distance.py:30
static CUDA_CALLABLE void adj_point_triangle_distance_gradient_0(
    wp::vec_t<3,wp::float64> var_p,
    wp::vec_t<3,wp::float64> var_t0,
    wp::vec_t<3,wp::float64> var_t1,
    wp::vec_t<3,wp::float64> var_t2,
    wp::vec_t<3,wp::float64> & adj_p,
    wp::vec_t<3,wp::float64> & adj_t0,
    wp::vec_t<3,wp::float64> & adj_t1,
    wp::vec_t<3,wp::float64> & adj_t2,
    wp::vec_t<12,wp::float64> & adj_ret)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 12;
    wp::vec_t<12,wp::float64> var_1;
    const wp::float32 var_2 = 0.0;
    wp::float64 var_3;
    wp::int32 var_4;
    const wp::int32 var_5 = 0;
    bool var_6;
    wp::vec_t<6,wp::float64> var_7;
    const wp::int32 var_8 = 0;
    wp::float64 var_9;
    const wp::int32 var_10 = 1;
    wp::float64 var_11;
    const wp::int32 var_12 = 2;
    wp::float64 var_13;
    const wp::int32 var_14 = 3;
    wp::float64 var_15;
    const wp::int32 var_16 = 4;
    wp::float64 var_17;
    const wp::int32 var_18 = 5;
    wp::float64 var_19;
    wp::vec_t<12,wp::float64> var_20;
    wp::vec_t<12,wp::float64> var_21;
    const wp::int32 var_22 = 1;
    bool var_23;
    wp::vec_t<6,wp::float64> var_24;
    const wp::int32 var_25 = 0;
    wp::float64 var_26;
    const wp::int32 var_27 = 1;
    wp::float64 var_28;
    const wp::int32 var_29 = 2;
    wp::float64 var_30;
    const wp::int32 var_31 = 3;
    wp::float64 var_32;
    const wp::int32 var_33 = 4;
    wp::float64 var_34;
    const wp::int32 var_35 = 5;
    wp::float64 var_36;
    wp::vec_t<12,wp::float64> var_37;
    wp::vec_t<12,wp::float64> var_38;
    const wp::int32 var_39 = 2;
    bool var_40;
    wp::vec_t<6,wp::float64> var_41;
    const wp::int32 var_42 = 0;
    wp::float64 var_43;
    const wp::int32 var_44 = 1;
    wp::float64 var_45;
    const wp::int32 var_46 = 2;
    wp::float64 var_47;
    const wp::int32 var_48 = 3;
    wp::float64 var_49;
    const wp::int32 var_50 = 4;
    wp::float64 var_51;
    const wp::int32 var_52 = 5;
    wp::float64 var_53;
    wp::vec_t<12,wp::float64> var_54;
    wp::vec_t<12,wp::float64> var_55;
    const wp::int32 var_56 = 3;
    bool var_57;
    wp::vec_t<9,wp::float64> var_58;
    const wp::int32 var_59 = 0;
    wp::float64 var_60;
    const wp::int32 var_61 = 1;
    wp::float64 var_62;
    const wp::int32 var_63 = 2;
    wp::float64 var_64;
    const wp::int32 var_65 = 3;
    wp::float64 var_66;
    const wp::int32 var_67 = 4;
    wp::float64 var_68;
    const wp::int32 var_69 = 5;
    wp::float64 var_70;
    const wp::int32 var_71 = 6;
    wp::float64 var_72;
    const wp::int32 var_73 = 7;
    wp::float64 var_74;
    const wp::int32 var_75 = 8;
    wp::float64 var_76;
    wp::vec_t<12,wp::float64> var_77;
    wp::vec_t<12,wp::float64> var_78;
    const wp::int32 var_79 = 4;
    bool var_80;
    wp::vec_t<9,wp::float64> var_81;
    const wp::int32 var_82 = 0;
    wp::float64 var_83;
    const wp::int32 var_84 = 1;
    wp::float64 var_85;
    const wp::int32 var_86 = 2;
    wp::float64 var_87;
    const wp::int32 var_88 = 3;
    wp::float64 var_89;
    const wp::int32 var_90 = 4;
    wp::float64 var_91;
    const wp::int32 var_92 = 5;
    wp::float64 var_93;
    const wp::int32 var_94 = 6;
    wp::float64 var_95;
    const wp::int32 var_96 = 7;
    wp::float64 var_97;
    const wp::int32 var_98 = 8;
    wp::float64 var_99;
    wp::vec_t<12,wp::float64> var_100;
    wp::vec_t<12,wp::float64> var_101;
    const wp::int32 var_102 = 5;
    bool var_103;
    wp::vec_t<9,wp::float64> var_104;
    const wp::int32 var_105 = 0;
    wp::float64 var_106;
    const wp::int32 var_107 = 1;
    wp::float64 var_108;
    const wp::int32 var_109 = 2;
    wp::float64 var_110;
    const wp::int32 var_111 = 3;
    wp::float64 var_112;
    const wp::int32 var_113 = 4;
    wp::float64 var_114;
    const wp::int32 var_115 = 5;
    wp::float64 var_116;
    const wp::int32 var_117 = 6;
    wp::float64 var_118;
    const wp::int32 var_119 = 7;
    wp::float64 var_120;
    const wp::int32 var_121 = 8;
    wp::float64 var_122;
    wp::vec_t<12,wp::float64> var_123;
    wp::vec_t<12,wp::float64> var_124;
    const wp::int32 var_125 = 6;
    bool var_126;
    wp::vec_t<12,wp::float64> var_127;
    wp::vec_t<12,wp::float64> var_128;
    wp::vec_t<12,wp::float64> var_129;
    wp::vec_t<12,wp::float64> var_130;
    wp::vec_t<12,wp::float64> var_131;
    wp::vec_t<12,wp::float64> var_132;
    wp::vec_t<12,wp::float64> var_133;
    wp::vec_t<12,wp::float64> var_134;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::vec_t<12,wp::float64> adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float64 adj_3 = {};
    wp::int32 adj_4 = {};
    wp::int32 adj_5 = {};
    bool adj_6 = {};
    wp::vec_t<6,wp::float64> adj_7 = {};
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
    wp::vec_t<12,wp::float64> adj_20 = {};
    wp::vec_t<12,wp::float64> adj_21 = {};
    wp::int32 adj_22 = {};
    bool adj_23 = {};
    wp::vec_t<6,wp::float64> adj_24 = {};
    wp::int32 adj_25 = {};
    wp::float64 adj_26 = {};
    wp::int32 adj_27 = {};
    wp::float64 adj_28 = {};
    wp::int32 adj_29 = {};
    wp::float64 adj_30 = {};
    wp::int32 adj_31 = {};
    wp::float64 adj_32 = {};
    wp::int32 adj_33 = {};
    wp::float64 adj_34 = {};
    wp::int32 adj_35 = {};
    wp::float64 adj_36 = {};
    wp::vec_t<12,wp::float64> adj_37 = {};
    wp::vec_t<12,wp::float64> adj_38 = {};
    wp::int32 adj_39 = {};
    bool adj_40 = {};
    wp::vec_t<6,wp::float64> adj_41 = {};
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
    wp::vec_t<12,wp::float64> adj_54 = {};
    wp::vec_t<12,wp::float64> adj_55 = {};
    wp::int32 adj_56 = {};
    bool adj_57 = {};
    wp::vec_t<9,wp::float64> adj_58 = {};
    wp::int32 adj_59 = {};
    wp::float64 adj_60 = {};
    wp::int32 adj_61 = {};
    wp::float64 adj_62 = {};
    wp::int32 adj_63 = {};
    wp::float64 adj_64 = {};
    wp::int32 adj_65 = {};
    wp::float64 adj_66 = {};
    wp::int32 adj_67 = {};
    wp::float64 adj_68 = {};
    wp::int32 adj_69 = {};
    wp::float64 adj_70 = {};
    wp::int32 adj_71 = {};
    wp::float64 adj_72 = {};
    wp::int32 adj_73 = {};
    wp::float64 adj_74 = {};
    wp::int32 adj_75 = {};
    wp::float64 adj_76 = {};
    wp::vec_t<12,wp::float64> adj_77 = {};
    wp::vec_t<12,wp::float64> adj_78 = {};
    wp::int32 adj_79 = {};
    bool adj_80 = {};
    wp::vec_t<9,wp::float64> adj_81 = {};
    wp::int32 adj_82 = {};
    wp::float64 adj_83 = {};
    wp::int32 adj_84 = {};
    wp::float64 adj_85 = {};
    wp::int32 adj_86 = {};
    wp::float64 adj_87 = {};
    wp::int32 adj_88 = {};
    wp::float64 adj_89 = {};
    wp::int32 adj_90 = {};
    wp::float64 adj_91 = {};
    wp::int32 adj_92 = {};
    wp::float64 adj_93 = {};
    wp::int32 adj_94 = {};
    wp::float64 adj_95 = {};
    wp::int32 adj_96 = {};
    wp::float64 adj_97 = {};
    wp::int32 adj_98 = {};
    wp::float64 adj_99 = {};
    wp::vec_t<12,wp::float64> adj_100 = {};
    wp::vec_t<12,wp::float64> adj_101 = {};
    wp::int32 adj_102 = {};
    bool adj_103 = {};
    wp::vec_t<9,wp::float64> adj_104 = {};
    wp::int32 adj_105 = {};
    wp::float64 adj_106 = {};
    wp::int32 adj_107 = {};
    wp::float64 adj_108 = {};
    wp::int32 adj_109 = {};
    wp::float64 adj_110 = {};
    wp::int32 adj_111 = {};
    wp::float64 adj_112 = {};
    wp::int32 adj_113 = {};
    wp::float64 adj_114 = {};
    wp::int32 adj_115 = {};
    wp::float64 adj_116 = {};
    wp::int32 adj_117 = {};
    wp::float64 adj_118 = {};
    wp::int32 adj_119 = {};
    wp::float64 adj_120 = {};
    wp::int32 adj_121 = {};
    wp::float64 adj_122 = {};
    wp::vec_t<12,wp::float64> adj_123 = {};
    wp::vec_t<12,wp::float64> adj_124 = {};
    wp::int32 adj_125 = {};
    bool adj_126 = {};
    wp::vec_t<12,wp::float64> adj_127 = {};
    wp::vec_t<12,wp::float64> adj_128 = {};
    wp::vec_t<12,wp::float64> adj_129 = {};
    wp::vec_t<12,wp::float64> adj_130 = {};
    wp::vec_t<12,wp::float64> adj_131 = {};
    wp::vec_t<12,wp::float64> adj_132 = {};
    wp::vec_t<12,wp::float64> adj_133 = {};
    wp::vec_t<12,wp::float64> adj_134 = {};
    //---------
    // forward
    // def point_triangle_distance_gradient(p: wp.vec3d, t0: wp.vec3d, t1: wp.vec3d, t2: wp.vec3d):       <L 31>
    // grad = wp.vec(length=12, dtype=wp.float64)                                             <L 32>
    var_1 = wp::vec_t<12,wp::float64>();
    // _0 = wp.float64(0.0)                                                                   <L 33>
    var_3 = wp::float64(var_2);
    // dt = point_triangle_distance_type(p, t0, t1, t2)                                       <L 34>
    var_4 = point_triangle_distance_type_0(var_p, var_t0, var_t1, var_t2);
    // if dt == 0:                                                                            <L 35>
    var_6 = (var_4 == var_5);
    if (var_6) {
        // grad_compact0 = point_point_distance_gradient(p, t0)                               <L 36>
        var_7 = point_point_distance_gradient_0(var_p, var_t0);
        // grad = wp.vector(                                                                  <L 37>
        // grad_compact0[0], grad_compact0[1], grad_compact0[2], grad_compact0[3], grad_compact0[4], grad_compact0[5], _0, _0, _0, _0, _0, _0       <L 38>
        var_9 = wp::extract(var_7, var_8);
        var_11 = wp::extract(var_7, var_10);
        var_13 = wp::extract(var_7, var_12);
        var_15 = wp::extract(var_7, var_14);
        var_17 = wp::extract(var_7, var_16);
        var_19 = wp::extract(var_7, var_18);
        var_20 = wp::vec_t<12,wp::float64>({var_9, var_11, var_13, var_15, var_17, var_19, var_3, var_3, var_3, var_3, var_3, var_3});
    }
    var_21 = wp::where(var_6, var_20, var_1);
    if (!var_6) {
        // elif dt == 1:                                                                      <L 40>
        var_23 = (var_4 == var_22);
        if (var_23) {
            // grad_compact1 = point_point_distance_gradient(p, t1)                           <L 41>
            var_24 = point_point_distance_gradient_0(var_p, var_t1);
            // grad = wp.vector(                                                              <L 42>
            // grad_compact1[0], grad_compact1[1], grad_compact1[2], _0, _0, _0, grad_compact1[3], grad_compact1[4], grad_compact1[5], _0, _0, _0       <L 43>
            var_26 = wp::extract(var_24, var_25);
            var_28 = wp::extract(var_24, var_27);
            var_30 = wp::extract(var_24, var_29);
            var_32 = wp::extract(var_24, var_31);
            var_34 = wp::extract(var_24, var_33);
            var_36 = wp::extract(var_24, var_35);
            var_37 = wp::vec_t<12,wp::float64>({var_26, var_28, var_30, var_3, var_3, var_3, var_32, var_34, var_36, var_3, var_3, var_3});
        }
        var_38 = wp::where(var_23, var_37, var_21);
        if (!var_23) {
            // elif dt == 2:                                                                  <L 45>
            var_40 = (var_4 == var_39);
            if (var_40) {
                // grad_compact2 = point_point_distance_gradient(p, t2)                       <L 46>
                var_41 = point_point_distance_gradient_0(var_p, var_t2);
                // grad = wp.vector(                                                          <L 47>
                // grad_compact2[0], grad_compact2[1], grad_compact2[2], _0, _0, _0, _0, _0, _0, grad_compact2[3], grad_compact2[4], grad_compact2[5]       <L 48>
                var_43 = wp::extract(var_41, var_42);
                var_45 = wp::extract(var_41, var_44);
                var_47 = wp::extract(var_41, var_46);
                var_49 = wp::extract(var_41, var_48);
                var_51 = wp::extract(var_41, var_50);
                var_53 = wp::extract(var_41, var_52);
                var_54 = wp::vec_t<12,wp::float64>({var_43, var_45, var_47, var_3, var_3, var_3, var_3, var_3, var_3, var_49, var_51, var_53});
            }
            var_55 = wp::where(var_40, var_54, var_38);
            if (!var_40) {
                // elif dt == 3:                                                              <L 50>
                var_57 = (var_4 == var_56);
                if (var_57) {
                    // grad_compact3 = point_line_distance_gradient(p, t0, t1)                <L 51>
                    var_58 = point_line_distance_gradient_0(var_p, var_t0, var_t1);
                    // grad = wp.vector(                                                      <L 52>
                    // grad_compact3[0],                                                      <L 53>
                    var_60 = wp::extract(var_58, var_59);
                    // grad_compact3[1],                                                      <L 54>
                    var_62 = wp::extract(var_58, var_61);
                    // grad_compact3[2],                                                      <L 55>
                    var_64 = wp::extract(var_58, var_63);
                    // grad_compact3[3],                                                      <L 56>
                    var_66 = wp::extract(var_58, var_65);
                    // grad_compact3[4],                                                      <L 57>
                    var_68 = wp::extract(var_58, var_67);
                    // grad_compact3[5],                                                      <L 58>
                    var_70 = wp::extract(var_58, var_69);
                    // grad_compact3[6],                                                      <L 59>
                    var_72 = wp::extract(var_58, var_71);
                    // grad_compact3[7],                                                      <L 60>
                    var_74 = wp::extract(var_58, var_73);
                    // grad_compact3[8],                                                      <L 61>
                    var_76 = wp::extract(var_58, var_75);
                    // _0,                                                                    <L 62>
                    // _0,                                                                    <L 63>
                    // _0,                                                                    <L 64>
                    var_77 = wp::vec_t<12,wp::float64>({var_60, var_62, var_64, var_66, var_68, var_70, var_72, var_74, var_76, var_3, var_3, var_3});
                }
                var_78 = wp::where(var_57, var_77, var_55);
                if (!var_57) {
                    // elif dt == 4:                                                          <L 66>
                    var_80 = (var_4 == var_79);
                    if (var_80) {
                        // grad_compact4 = point_line_distance_gradient(p, t1, t2)            <L 67>
                        var_81 = point_line_distance_gradient_0(var_p, var_t1, var_t2);
                        // grad = wp.vector(                                                  <L 68>
                        // grad_compact4[0],                                                  <L 69>
                        var_83 = wp::extract(var_81, var_82);
                        // grad_compact4[1],                                                  <L 70>
                        var_85 = wp::extract(var_81, var_84);
                        // grad_compact4[2],                                                  <L 71>
                        var_87 = wp::extract(var_81, var_86);
                        // _0,                                                                <L 72>
                        // _0,                                                                <L 73>
                        // _0,                                                                <L 74>
                        // grad_compact4[3],                                                  <L 75>
                        var_89 = wp::extract(var_81, var_88);
                        // grad_compact4[4],                                                  <L 76>
                        var_91 = wp::extract(var_81, var_90);
                        // grad_compact4[5],                                                  <L 77>
                        var_93 = wp::extract(var_81, var_92);
                        // grad_compact4[6],                                                  <L 78>
                        var_95 = wp::extract(var_81, var_94);
                        // grad_compact4[7],                                                  <L 79>
                        var_97 = wp::extract(var_81, var_96);
                        // grad_compact4[8],                                                  <L 80>
                        var_99 = wp::extract(var_81, var_98);
                        var_100 = wp::vec_t<12,wp::float64>({var_83, var_85, var_87, var_3, var_3, var_3, var_89, var_91, var_93, var_95, var_97, var_99});
                    }
                    var_101 = wp::where(var_80, var_100, var_78);
                    if (!var_80) {
                        // elif dt == 5:                                                      <L 82>
                        var_103 = (var_4 == var_102);
                        if (var_103) {
                            // grad_compact5 = point_line_distance_gradient(p, t0, t2)        <L 83>
                            var_104 = point_line_distance_gradient_0(var_p, var_t0, var_t2);
                            // grad = wp.vector(                                              <L 84>
                            // grad_compact5[0],                                              <L 85>
                            var_106 = wp::extract(var_104, var_105);
                            // grad_compact5[1],                                              <L 86>
                            var_108 = wp::extract(var_104, var_107);
                            // grad_compact5[2],                                              <L 87>
                            var_110 = wp::extract(var_104, var_109);
                            // grad_compact5[3],                                              <L 88>
                            var_112 = wp::extract(var_104, var_111);
                            // grad_compact5[4],                                              <L 89>
                            var_114 = wp::extract(var_104, var_113);
                            // grad_compact5[5],                                              <L 90>
                            var_116 = wp::extract(var_104, var_115);
                            // _0,                                                            <L 91>
                            // _0,                                                            <L 92>
                            // _0,                                                            <L 93>
                            // grad_compact5[6],                                              <L 94>
                            var_118 = wp::extract(var_104, var_117);
                            // grad_compact5[7],                                              <L 95>
                            var_120 = wp::extract(var_104, var_119);
                            // grad_compact5[8],                                              <L 96>
                            var_122 = wp::extract(var_104, var_121);
                            var_123 = wp::vec_t<12,wp::float64>({var_106, var_108, var_110, var_112, var_114, var_116, var_3, var_3, var_3, var_118, var_120, var_122});
                        }
                        var_124 = wp::where(var_103, var_123, var_101);
                        if (!var_103) {
                            // elif dt == 6:                                                  <L 98>
                            var_126 = (var_4 == var_125);
                            if (var_126) {
                                // grad = point_plane_distance_gradient(p, t0, t1, t2)        <L 99>
                                var_127 = point_plane_distance_gradient_0(var_p, var_t0, var_t1, var_t2);
                            }
                            var_128 = wp::where(var_126, var_127, var_124);
                        }
                        var_129 = wp::where(var_103, var_124, var_128);
                    }
                    var_130 = wp::where(var_80, var_101, var_129);
                }
                var_131 = wp::where(var_57, var_78, var_130);
            }
            var_132 = wp::where(var_40, var_55, var_131);
        }
        var_133 = wp::where(var_23, var_38, var_132);
    }
    var_134 = wp::where(var_6, var_21, var_133);
    // return grad                                                                            <L 101>
    goto label0;
    //---------
    // reverse
    label0:;
    adj_134 += adj_ret;
    // adj: return grad                                                                       <L 101>
    wp::adj_where(var_6, var_21, var_133, adj_6, adj_21, adj_133, adj_134);
    if (!var_6) {
        wp::adj_where(var_23, var_38, var_132, adj_23, adj_38, adj_132, adj_133);
        if (!var_23) {
            wp::adj_where(var_40, var_55, var_131, adj_40, adj_55, adj_131, adj_132);
            if (!var_40) {
                wp::adj_where(var_57, var_78, var_130, adj_57, adj_78, adj_130, adj_131);
                if (!var_57) {
                    wp::adj_where(var_80, var_101, var_129, adj_80, adj_101, adj_129, adj_130);
                    if (!var_80) {
                        wp::adj_where(var_103, var_124, var_128, adj_103, adj_124, adj_128, adj_129);
                        if (!var_103) {
                            wp::adj_where(var_126, var_127, var_124, adj_126, adj_127, adj_124, adj_128);
                            if (var_126) {
                                adj_point_plane_distance_gradient_0(var_p, var_t0, var_t1, var_t2, adj_p, adj_t0, adj_t1, adj_t2, adj_127);
                                // adj: grad = point_plane_distance_gradient(p, t0, t1, t2)   <L 99>
                            }
                            // adj: elif dt == 6:                                             <L 98>
                        }
                        wp::adj_where(var_103, var_123, var_101, adj_103, adj_123, adj_101, adj_124);
                        if (var_103) {
                            wp::adj_vec_t({var_106, var_108, var_110, var_112, var_114, var_116, var_3, var_3, var_3, var_118, var_120, var_122}, {&adj_106, &adj_108, &adj_110, &adj_112, &adj_114, &adj_116, &adj_3, &adj_3, &adj_3, &adj_118, &adj_120, &adj_122}, adj_123);
                            wp::adj_extract(var_104, var_121, adj_104, adj_121, adj_122);
                            // adj: grad_compact5[8],                                         <L 96>
                            wp::adj_extract(var_104, var_119, adj_104, adj_119, adj_120);
                            // adj: grad_compact5[7],                                         <L 95>
                            wp::adj_extract(var_104, var_117, adj_104, adj_117, adj_118);
                            // adj: grad_compact5[6],                                         <L 94>
                            // adj: _0,                                                       <L 93>
                            // adj: _0,                                                       <L 92>
                            // adj: _0,                                                       <L 91>
                            wp::adj_extract(var_104, var_115, adj_104, adj_115, adj_116);
                            // adj: grad_compact5[5],                                         <L 90>
                            wp::adj_extract(var_104, var_113, adj_104, adj_113, adj_114);
                            // adj: grad_compact5[4],                                         <L 89>
                            wp::adj_extract(var_104, var_111, adj_104, adj_111, adj_112);
                            // adj: grad_compact5[3],                                         <L 88>
                            wp::adj_extract(var_104, var_109, adj_104, adj_109, adj_110);
                            // adj: grad_compact5[2],                                         <L 87>
                            wp::adj_extract(var_104, var_107, adj_104, adj_107, adj_108);
                            // adj: grad_compact5[1],                                         <L 86>
                            wp::adj_extract(var_104, var_105, adj_104, adj_105, adj_106);
                            // adj: grad_compact5[0],                                         <L 85>
                            // adj: grad = wp.vector(                                         <L 84>
                            adj_point_line_distance_gradient_0(var_p, var_t0, var_t2, adj_p, adj_t0, adj_t2, adj_104);
                            // adj: grad_compact5 = point_line_distance_gradient(p, t0, t2)   <L 83>
                        }
                        // adj: elif dt == 5:                                                 <L 82>
                    }
                    wp::adj_where(var_80, var_100, var_78, adj_80, adj_100, adj_78, adj_101);
                    if (var_80) {
                        wp::adj_vec_t({var_83, var_85, var_87, var_3, var_3, var_3, var_89, var_91, var_93, var_95, var_97, var_99}, {&adj_83, &adj_85, &adj_87, &adj_3, &adj_3, &adj_3, &adj_89, &adj_91, &adj_93, &adj_95, &adj_97, &adj_99}, adj_100);
                        wp::adj_extract(var_81, var_98, adj_81, adj_98, adj_99);
                        // adj: grad_compact4[8],                                             <L 80>
                        wp::adj_extract(var_81, var_96, adj_81, adj_96, adj_97);
                        // adj: grad_compact4[7],                                             <L 79>
                        wp::adj_extract(var_81, var_94, adj_81, adj_94, adj_95);
                        // adj: grad_compact4[6],                                             <L 78>
                        wp::adj_extract(var_81, var_92, adj_81, adj_92, adj_93);
                        // adj: grad_compact4[5],                                             <L 77>
                        wp::adj_extract(var_81, var_90, adj_81, adj_90, adj_91);
                        // adj: grad_compact4[4],                                             <L 76>
                        wp::adj_extract(var_81, var_88, adj_81, adj_88, adj_89);
                        // adj: grad_compact4[3],                                             <L 75>
                        // adj: _0,                                                           <L 74>
                        // adj: _0,                                                           <L 73>
                        // adj: _0,                                                           <L 72>
                        wp::adj_extract(var_81, var_86, adj_81, adj_86, adj_87);
                        // adj: grad_compact4[2],                                             <L 71>
                        wp::adj_extract(var_81, var_84, adj_81, adj_84, adj_85);
                        // adj: grad_compact4[1],                                             <L 70>
                        wp::adj_extract(var_81, var_82, adj_81, adj_82, adj_83);
                        // adj: grad_compact4[0],                                             <L 69>
                        // adj: grad = wp.vector(                                             <L 68>
                        adj_point_line_distance_gradient_0(var_p, var_t1, var_t2, adj_p, adj_t1, adj_t2, adj_81);
                        // adj: grad_compact4 = point_line_distance_gradient(p, t1, t2)       <L 67>
                    }
                    // adj: elif dt == 4:                                                     <L 66>
                }
                wp::adj_where(var_57, var_77, var_55, adj_57, adj_77, adj_55, adj_78);
                if (var_57) {
                    wp::adj_vec_t({var_60, var_62, var_64, var_66, var_68, var_70, var_72, var_74, var_76, var_3, var_3, var_3}, {&adj_60, &adj_62, &adj_64, &adj_66, &adj_68, &adj_70, &adj_72, &adj_74, &adj_76, &adj_3, &adj_3, &adj_3}, adj_77);
                    // adj: _0,                                                               <L 64>
                    // adj: _0,                                                               <L 63>
                    // adj: _0,                                                               <L 62>
                    wp::adj_extract(var_58, var_75, adj_58, adj_75, adj_76);
                    // adj: grad_compact3[8],                                                 <L 61>
                    wp::adj_extract(var_58, var_73, adj_58, adj_73, adj_74);
                    // adj: grad_compact3[7],                                                 <L 60>
                    wp::adj_extract(var_58, var_71, adj_58, adj_71, adj_72);
                    // adj: grad_compact3[6],                                                 <L 59>
                    wp::adj_extract(var_58, var_69, adj_58, adj_69, adj_70);
                    // adj: grad_compact3[5],                                                 <L 58>
                    wp::adj_extract(var_58, var_67, adj_58, adj_67, adj_68);
                    // adj: grad_compact3[4],                                                 <L 57>
                    wp::adj_extract(var_58, var_65, adj_58, adj_65, adj_66);
                    // adj: grad_compact3[3],                                                 <L 56>
                    wp::adj_extract(var_58, var_63, adj_58, adj_63, adj_64);
                    // adj: grad_compact3[2],                                                 <L 55>
                    wp::adj_extract(var_58, var_61, adj_58, adj_61, adj_62);
                    // adj: grad_compact3[1],                                                 <L 54>
                    wp::adj_extract(var_58, var_59, adj_58, adj_59, adj_60);
                    // adj: grad_compact3[0],                                                 <L 53>
                    // adj: grad = wp.vector(                                                 <L 52>
                    adj_point_line_distance_gradient_0(var_p, var_t0, var_t1, adj_p, adj_t0, adj_t1, adj_58);
                    // adj: grad_compact3 = point_line_distance_gradient(p, t0, t1)           <L 51>
                }
                // adj: elif dt == 3:                                                         <L 50>
            }
            wp::adj_where(var_40, var_54, var_38, adj_40, adj_54, adj_38, adj_55);
            if (var_40) {
                wp::adj_vec_t({var_43, var_45, var_47, var_3, var_3, var_3, var_3, var_3, var_3, var_49, var_51, var_53}, {&adj_43, &adj_45, &adj_47, &adj_3, &adj_3, &adj_3, &adj_3, &adj_3, &adj_3, &adj_49, &adj_51, &adj_53}, adj_54);
                wp::adj_extract(var_41, var_52, adj_41, adj_52, adj_53);
                wp::adj_extract(var_41, var_50, adj_41, adj_50, adj_51);
                wp::adj_extract(var_41, var_48, adj_41, adj_48, adj_49);
                wp::adj_extract(var_41, var_46, adj_41, adj_46, adj_47);
                wp::adj_extract(var_41, var_44, adj_41, adj_44, adj_45);
                wp::adj_extract(var_41, var_42, adj_41, adj_42, adj_43);
                // adj: grad_compact2[0], grad_compact2[1], grad_compact2[2], _0, _0, _0, _0, _0, _0, grad_compact2[3], grad_compact2[4], grad_compact2[5]  <L 48>
                // adj: grad = wp.vector(                                                     <L 47>
                adj_point_point_distance_gradient_0(var_p, var_t2, adj_p, adj_t2, adj_41);
                // adj: grad_compact2 = point_point_distance_gradient(p, t2)                  <L 46>
            }
            // adj: elif dt == 2:                                                             <L 45>
        }
        wp::adj_where(var_23, var_37, var_21, adj_23, adj_37, adj_21, adj_38);
        if (var_23) {
            wp::adj_vec_t({var_26, var_28, var_30, var_3, var_3, var_3, var_32, var_34, var_36, var_3, var_3, var_3}, {&adj_26, &adj_28, &adj_30, &adj_3, &adj_3, &adj_3, &adj_32, &adj_34, &adj_36, &adj_3, &adj_3, &adj_3}, adj_37);
            wp::adj_extract(var_24, var_35, adj_24, adj_35, adj_36);
            wp::adj_extract(var_24, var_33, adj_24, adj_33, adj_34);
            wp::adj_extract(var_24, var_31, adj_24, adj_31, adj_32);
            wp::adj_extract(var_24, var_29, adj_24, adj_29, adj_30);
            wp::adj_extract(var_24, var_27, adj_24, adj_27, adj_28);
            wp::adj_extract(var_24, var_25, adj_24, adj_25, adj_26);
            // adj: grad_compact1[0], grad_compact1[1], grad_compact1[2], _0, _0, _0, grad_compact1[3], grad_compact1[4], grad_compact1[5], _0, _0, _0  <L 43>
            // adj: grad = wp.vector(                                                         <L 42>
            adj_point_point_distance_gradient_0(var_p, var_t1, adj_p, adj_t1, adj_24);
            // adj: grad_compact1 = point_point_distance_gradient(p, t1)                      <L 41>
        }
        // adj: elif dt == 1:                                                                 <L 40>
    }
    wp::adj_where(var_6, var_20, var_1, adj_6, adj_20, adj_1, adj_21);
    if (var_6) {
        wp::adj_vec_t({var_9, var_11, var_13, var_15, var_17, var_19, var_3, var_3, var_3, var_3, var_3, var_3}, {&adj_9, &adj_11, &adj_13, &adj_15, &adj_17, &adj_19, &adj_3, &adj_3, &adj_3, &adj_3, &adj_3, &adj_3}, adj_20);
        wp::adj_extract(var_7, var_18, adj_7, adj_18, adj_19);
        wp::adj_extract(var_7, var_16, adj_7, adj_16, adj_17);
        wp::adj_extract(var_7, var_14, adj_7, adj_14, adj_15);
        wp::adj_extract(var_7, var_12, adj_7, adj_12, adj_13);
        wp::adj_extract(var_7, var_10, adj_7, adj_10, adj_11);
        wp::adj_extract(var_7, var_8, adj_7, adj_8, adj_9);
        // adj: grad_compact0[0], grad_compact0[1], grad_compact0[2], grad_compact0[3], grad_compact0[4], grad_compact0[5], _0, _0, _0, _0, _0, _0  <L 38>
        // adj: grad = wp.vector(                                                             <L 37>
        adj_point_point_distance_gradient_0(var_p, var_t0, adj_p, adj_t0, adj_7);
        // adj: grad_compact0 = point_point_distance_gradient(p, t0)                          <L 36>
    }
    // adj: if dt == 0:                                                                       <L 35>
    adj_point_triangle_distance_type_0(var_p, var_t0, var_t1, var_t2, adj_p, adj_t0, adj_t1, adj_t2, adj_4);
    // adj: dt = point_triangle_distance_type(p, t0, t1, t2)                                  <L 34>
    wp::adj_float64(var_2, adj_2, adj_3);
    // adj: _0 = wp.float64(0.0)                                                              <L 33>
    // adj: grad = wp.vec(length=12, dtype=wp.float64)                                        <L 32>
    // adj: def point_triangle_distance_gradient(p: wp.vec3d, t0: wp.vec3d, t1: wp.vec3d, t2: wp.vec3d):  <L 31>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/utils/wp_math.py:152
static CUDA_CALLABLE wp::float64 safe_divide_0(
    wp::float64 var_num,
    wp::float64 var_denom)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    wp::float64 var_1;
    bool var_2;
    wp::float64 var_3;
    const wp::float32 var_4 = 0.0;
    wp::float64 var_5;
    wp::float64 var_6;
    //---------
    // forward
    // def safe_divide(num: wp.float64, denom: wp.float64):                                   <L 153>
    // return wp.select(denom == wp.float64(0.0), num / denom, wp.float64(0.0))               <L 154>
    var_1 = wp::float64(var_0);
    var_2 = (var_denom == var_1);
    var_3 = wp::div(var_num, var_denom);
    var_5 = wp::float64(var_4);
    var_6 = wp::select(var_2, var_3, var_5);
    return var_6;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/utils/wp_math.py:152
static CUDA_CALLABLE void adj_safe_divide_0(
    wp::float64 var_num,
    wp::float64 var_denom,
    wp::float64 & adj_num,
    wp::float64 & adj_denom,
    wp::float64 & adj_ret)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    wp::float64 var_1;
    bool var_2;
    wp::float64 var_3;
    const wp::float32 var_4 = 0.0;
    wp::float64 var_5;
    wp::float64 var_6;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    wp::float64 adj_1 = {};
    bool adj_2 = {};
    wp::float64 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::float64 adj_5 = {};
    wp::float64 adj_6 = {};
    //---------
    // forward
    // def safe_divide(num: wp.float64, denom: wp.float64):                                   <L 153>
    // return wp.select(denom == wp.float64(0.0), num / denom, wp.float64(0.0))               <L 154>
    var_1 = wp::float64(var_0);
    var_2 = (var_denom == var_1);
    var_3 = wp::div(var_num, var_denom);
    var_5 = wp::float64(var_4);
    var_6 = wp::select(var_2, var_3, var_5);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_6 += adj_ret;
    wp::adj_select(var_2, var_3, var_5, adj_2, adj_3, adj_5, adj_6);
    wp::adj_float64(var_4, adj_4, adj_5);
    wp::adj_div(var_num, var_denom, var_3, adj_num, adj_denom, adj_3);
    wp::adj_float64(var_0, adj_0, adj_1);
    // adj: return wp.select(denom == wp.float64(0.0), num / denom, wp.float64(0.0))          <L 154>
    // adj: def safe_divide(num: wp.float64, denom: wp.float64):                              <L 153>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/utils/wp_math.py:161
static CUDA_CALLABLE wp::float64 get_combined_coulomb_friction_0(
    wp::float64 var_mu1,
    wp::float64 var_mu2)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 2.0;
    wp::float64 var_1;
    wp::float64 var_2;
    wp::float64 var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    //---------
    // forward
    // def get_combined_coulomb_friction(mu1: wp.float64, mu2: wp.float64):                   <L 162>
    // return safe_divide(wp.float64(2.0) * mu1 * mu2, mu1 + mu2)                             <L 163>
    var_1 = wp::float64(var_0);
    var_2 = wp::mul(var_1, var_mu1);
    var_3 = wp::mul(var_2, var_mu2);
    var_4 = wp::add(var_mu1, var_mu2);
    var_5 = safe_divide_0(var_3, var_4);
    return var_5;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/utils/wp_math.py:161
static CUDA_CALLABLE void adj_get_combined_coulomb_friction_0(
    wp::float64 var_mu1,
    wp::float64 var_mu2,
    wp::float64 & adj_mu1,
    wp::float64 & adj_mu2,
    wp::float64 & adj_ret)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 2.0;
    wp::float64 var_1;
    wp::float64 var_2;
    wp::float64 var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    wp::float64 adj_1 = {};
    wp::float64 adj_2 = {};
    wp::float64 adj_3 = {};
    wp::float64 adj_4 = {};
    wp::float64 adj_5 = {};
    //---------
    // forward
    // def get_combined_coulomb_friction(mu1: wp.float64, mu2: wp.float64):                   <L 162>
    // return safe_divide(wp.float64(2.0) * mu1 * mu2, mu1 + mu2)                             <L 163>
    var_1 = wp::float64(var_0);
    var_2 = wp::mul(var_1, var_mu1);
    var_3 = wp::mul(var_2, var_mu2);
    var_4 = wp::add(var_mu1, var_mu2);
    var_5 = safe_divide_0(var_3, var_4);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_5 += adj_ret;
    adj_safe_divide_0(var_3, var_4, adj_3, adj_4, adj_5);
    wp::adj_add(var_mu1, var_mu2, adj_mu1, adj_mu2, adj_4);
    wp::adj_mul(var_2, var_mu2, adj_2, adj_mu2, adj_3);
    wp::adj_mul(var_1, var_mu1, adj_1, adj_mu1, adj_2);
    wp::adj_float64(var_0, adj_0, adj_1);
    // adj: return safe_divide(wp.float64(2.0) * mu1 * mu2, mu1 + mu2)                        <L 163>
    // adj: def get_combined_coulomb_friction(mu1: wp.float64, mu2: wp.float64):              <L 162>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/distance_type.py:78
static CUDA_CALLABLE wp::int32 edge_edge_distance_type_0(
    wp::vec_t<3,wp::float64> var_ea0,
    wp::vec_t<3,wp::float64> var_ea1,
    wp::vec_t<3,wp::float64> var_eb0,
    wp::vec_t<3,wp::float64> var_eb1)
{
    //---------
    // primal vars
    wp::vec_t<3,wp::float64> var_0;
    wp::vec_t<3,wp::float64> var_1;
    wp::vec_t<3,wp::float64> var_2;
    wp::float64 var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    wp::float64 var_6;
    wp::float64 var_7;
    wp::float64 var_8;
    wp::float64 var_9;
    wp::float64 var_10;
    wp::float64 var_11;
    const wp::int32 var_12 = 8;
    const wp::float32 var_13 = 0.0;
    wp::float64 var_14;
    const wp::float32 var_15 = 0.0;
    wp::float64 var_16;
    wp::float64 var_17;
    wp::float64 var_18;
    wp::float64 var_19;
    const wp::float32 var_20 = 0.0;
    wp::float64 var_21;
    bool var_22;
    wp::float64 var_23;
    wp::float64 var_24;
    const wp::int32 var_25 = 2;
    wp::float64 var_26;
    wp::int32 var_27;
    wp::float64 var_28;
    bool var_29;
    wp::float64 var_30;
    wp::float64 var_31;
    const wp::int32 var_32 = 5;
    wp::float64 var_33;
    wp::int32 var_34;
    wp::float64 var_35;
    wp::float64 var_36;
    wp::float64 var_37;
    wp::float64 var_38;
    wp::vec_t<3,wp::float64> var_39;
    const wp::float32 var_40 = 0.0;
    wp::float64 var_41;
    bool var_42;
    bool var_43;
    wp::float64 var_44;
    const wp::float32 var_45 = 0.0;
    wp::float64 var_46;
    bool var_47;
    wp::float64 var_48;
    const wp::float32 var_49 = 1e-20;
    wp::float64 var_50;
    wp::float64 var_51;
    wp::float64 var_52;
    bool var_53;
    bool var_54;
    bool var_55;
    const wp::float32 var_56 = 2.0;
    wp::float64 var_57;
    wp::float64 var_58;
    bool var_59;
    wp::float64 var_60;
    wp::float64 var_61;
    const wp::int32 var_62 = 2;
    wp::float64 var_63;
    wp::int32 var_64;
    wp::float64 var_65;
    wp::float64 var_66;
    wp::float64 var_67;
    const wp::int32 var_68 = 5;
    wp::float64 var_69;
    wp::int32 var_70;
    wp::float64 var_71;
    wp::float64 var_72;
    wp::int32 var_73;
    wp::float64 var_74;
    wp::float64 var_75;
    wp::int32 var_76;
    wp::float64 var_77;
    wp::float64 var_78;
    wp::int32 var_79;
    wp::float64 var_80;
    const wp::float32 var_81 = 0.0;
    wp::float64 var_82;
    bool var_83;
    wp::float64 var_84;
    const wp::float32 var_85 = 0.0;
    wp::float64 var_86;
    bool var_87;
    const wp::int32 var_88 = 0;
    wp::float64 var_89;
    bool var_90;
    const wp::int32 var_91 = 3;
    const wp::int32 var_92 = 6;
    bool var_93;
    wp::float64 var_94;
    wp::float64 var_95;
    const wp::float32 var_96 = 0.0;
    wp::float64 var_97;
    bool var_98;
    const wp::int32 var_99 = 1;
    wp::float64 var_100;
    wp::float64 var_101;
    bool var_102;
    const wp::int32 var_103 = 4;
    const wp::int32 var_104 = 7;
    //---------
    // forward
    // def edge_edge_distance_type(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d) -> int:       <L 79>
    // u = ea1 - ea0                                                                          <L 80>
    var_0 = wp::sub(var_ea1, var_ea0);
    // v = eb1 - eb0                                                                          <L 81>
    var_1 = wp::sub(var_eb1, var_eb0);
    // w = ea0 - eb0                                                                          <L 82>
    var_2 = wp::sub(var_ea0, var_eb0);
    // a = wp.length_sq(u)  # always >= 0                                                     <L 84>
    var_3 = wp::length_sq(var_0);
    // b = wp.dot(u, v)                                                                       <L 85>
    var_4 = wp::dot(var_0, var_1);
    // c = wp.length_sq(v)  # always >= 0                                                     <L 86>
    var_5 = wp::length_sq(var_1);
    // d = wp.dot(u, w)                                                                       <L 87>
    var_6 = wp::dot(var_0, var_2);
    // e = wp.dot(v, w)                                                                       <L 88>
    var_7 = wp::dot(var_1, var_2);
    // D = a * c - b * b  # always >= 0                                                       <L 89>
    var_8 = wp::mul(var_3, var_5);
    var_9 = wp::mul(var_4, var_4);
    var_10 = wp::sub(var_8, var_9);
    // tD = D  # tc = tN / tD, default tD = D >= 0                                            <L 90>
    var_11 = wp::copy(var_10);
    // defaultCase = 8                                                                        <L 91>
    // sN, tN = wp.float64(0.0), wp.float64(0.0)                                              <L 92>
    var_14 = wp::float64(var_13);
    var_16 = wp::float64(var_15);
    // sN = b * e - c * d                                                                     <L 95>
    var_17 = wp::mul(var_4, var_7);
    var_18 = wp::mul(var_5, var_6);
    var_19 = wp::sub(var_17, var_18);
    // if sN <= wp.float64(0.0):  # sc < 0 => the s=0 edge is visible                         <L 96>
    var_21 = wp::float64(var_20);
    var_22 = (var_19 <= var_21);
    if (var_22) {
        // tN = e                                                                             <L 97>
        var_23 = wp::copy(var_7);
        // tD = c                                                                             <L 98>
        var_24 = wp::copy(var_5);
        // defaultCase = 2                                                                    <L 99>
    }
    var_26 = wp::where(var_22, var_24, var_11);
    var_27 = wp::where(var_22, var_25, var_12);
    var_28 = wp::where(var_22, var_23, var_16);
    if (!var_22) {
        // elif sN >= D:  # sc > 1  => the s=1 edge is visible                                <L 100>
        var_29 = (var_19 >= var_10);
        if (var_29) {
            // tN = e + b                                                                     <L 101>
            var_30 = wp::add(var_7, var_4);
            // tD = c                                                                         <L 102>
            var_31 = wp::copy(var_5);
            // defaultCase = 5                                                                <L 103>
        }
        var_33 = wp::where(var_29, var_31, var_26);
        var_34 = wp::where(var_29, var_32, var_27);
        var_35 = wp::where(var_29, var_30, var_28);
        if (!var_29) {
            // tN = a * e - b * d                                                             <L 105>
            var_36 = wp::mul(var_3, var_7);
            var_37 = wp::mul(var_4, var_6);
            var_38 = wp::sub(var_36, var_37);
            // uv = wp.cross(u, v)                                                            <L 106>
            var_39 = wp::cross(var_0, var_1);
            // if tN > wp.float64(0.0) and tN < tD and (wp.dot(uv, w) == wp.float64(0.0) or wp.length_sq(uv) < wp.float64(1e-20) * a * c):       <L 108>
            var_41 = wp::float64(var_40);
            var_42 = (var_38 > var_41);
            var_43 = (var_38 < var_33);
            var_44 = wp::dot(var_39, var_2);
            var_46 = wp::float64(var_45);
            var_47 = (var_44 == var_46);
            var_48 = wp::length_sq(var_39);
            var_50 = wp::float64(var_49);
            var_51 = wp::mul(var_50, var_3);
            var_52 = wp::mul(var_51, var_5);
            var_53 = (var_48 < var_52);
            var_54 = var_47 || var_53;
            var_55 = var_42 && var_43 && var_54;
            if (var_55) {
                // if sN < D / wp.float64(2.0):                                               <L 113>
                var_57 = wp::float64(var_56);
                var_58 = wp::div(var_10, var_57);
                var_59 = (var_19 < var_58);
                if (var_59) {
                    // tN = e                                                                 <L 114>
                    var_60 = wp::copy(var_7);
                    // tD = c                                                                 <L 115>
                    var_61 = wp::copy(var_5);
                    // defaultCase = 2                                                        <L 116>
                }
                var_63 = wp::where(var_59, var_61, var_33);
                var_64 = wp::where(var_59, var_62, var_34);
                var_65 = wp::where(var_59, var_60, var_38);
                if (!var_59) {
                    // tN = e + b                                                             <L 118>
                    var_66 = wp::add(var_7, var_4);
                    // tD = c                                                                 <L 119>
                    var_67 = wp::copy(var_5);
                    // defaultCase = 5                                                        <L 120>
                }
                var_69 = wp::where(var_59, var_63, var_67);
                var_70 = wp::where(var_59, var_64, var_68);
                var_71 = wp::where(var_59, var_65, var_66);
            }
            var_72 = wp::where(var_55, var_69, var_33);
            var_73 = wp::where(var_55, var_70, var_34);
            var_74 = wp::where(var_55, var_71, var_38);
        }
        var_75 = wp::where(var_29, var_33, var_72);
        var_76 = wp::where(var_29, var_34, var_73);
        var_77 = wp::where(var_29, var_35, var_74);
    }
    var_78 = wp::where(var_22, var_26, var_75);
    var_79 = wp::where(var_22, var_27, var_76);
    var_80 = wp::where(var_22, var_28, var_77);
    // if tN <= wp.float64(0.0):  # tc < 0 => the t=0 edge is visible                         <L 123>
    var_82 = wp::float64(var_81);
    var_83 = (var_80 <= var_82);
    if (var_83) {
        // if -d <= wp.float64(0.0):                                                          <L 125>
        var_84 = wp::neg(var_6);
        var_86 = wp::float64(var_85);
        var_87 = (var_84 <= var_86);
        if (var_87) {
            // return 0                                                                       <L 126>
            return var_88;
        }
        if (!var_87) {
            // elif -d >= a:                                                                  <L 127>
            var_89 = wp::neg(var_6);
            var_90 = (var_89 >= var_3);
            if (var_90) {
                // return 3                                                                   <L 128>
                return var_91;
            }
            if (!var_90) {
                // return 6                                                                   <L 130>
                return var_92;
            }
        }
    }
    if (!var_83) {
        // elif tN >= tD:  # tc > 1  => the t=1 edge is visible                               <L 131>
        var_93 = (var_80 >= var_78);
        if (var_93) {
            // if (-d + b) <= wp.float64(0.0):                                                <L 133>
            var_94 = wp::neg(var_6);
            var_95 = wp::add(var_94, var_4);
            var_97 = wp::float64(var_96);
            var_98 = (var_95 <= var_97);
            if (var_98) {
                // return 1                                                                   <L 134>
                return var_99;
            }
            if (!var_98) {
                // elif (-d + b) >= a:                                                        <L 135>
                var_100 = wp::neg(var_6);
                var_101 = wp::add(var_100, var_4);
                var_102 = (var_101 >= var_3);
                if (var_102) {
                    // return 4                                                               <L 136>
                    return var_103;
                }
                if (!var_102) {
                    // return 7                                                               <L 138>
                    return var_104;
                }
            }
        }
    }
    // return defaultCase                                                                     <L 140>
    return var_79;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/distance_type.py:78
static CUDA_CALLABLE void adj_edge_edge_distance_type_0(
    wp::vec_t<3,wp::float64> var_ea0,
    wp::vec_t<3,wp::float64> var_ea1,
    wp::vec_t<3,wp::float64> var_eb0,
    wp::vec_t<3,wp::float64> var_eb1,
    wp::vec_t<3,wp::float64> & adj_ea0,
    wp::vec_t<3,wp::float64> & adj_ea1,
    wp::vec_t<3,wp::float64> & adj_eb0,
    wp::vec_t<3,wp::float64> & adj_eb1,
    wp::int32 & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3,wp::float64> var_0;
    wp::vec_t<3,wp::float64> var_1;
    wp::vec_t<3,wp::float64> var_2;
    wp::float64 var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    wp::float64 var_6;
    wp::float64 var_7;
    wp::float64 var_8;
    wp::float64 var_9;
    wp::float64 var_10;
    wp::float64 var_11;
    const wp::int32 var_12 = 8;
    const wp::float32 var_13 = 0.0;
    wp::float64 var_14;
    const wp::float32 var_15 = 0.0;
    wp::float64 var_16;
    wp::float64 var_17;
    wp::float64 var_18;
    wp::float64 var_19;
    const wp::float32 var_20 = 0.0;
    wp::float64 var_21;
    bool var_22;
    wp::float64 var_23;
    wp::float64 var_24;
    const wp::int32 var_25 = 2;
    wp::float64 var_26;
    wp::int32 var_27;
    wp::float64 var_28;
    bool var_29;
    wp::float64 var_30;
    wp::float64 var_31;
    const wp::int32 var_32 = 5;
    wp::float64 var_33;
    wp::int32 var_34;
    wp::float64 var_35;
    wp::float64 var_36;
    wp::float64 var_37;
    wp::float64 var_38;
    wp::vec_t<3,wp::float64> var_39;
    const wp::float32 var_40 = 0.0;
    wp::float64 var_41;
    bool var_42;
    bool var_43;
    wp::float64 var_44;
    const wp::float32 var_45 = 0.0;
    wp::float64 var_46;
    bool var_47;
    wp::float64 var_48;
    const wp::float32 var_49 = 1e-20;
    wp::float64 var_50;
    wp::float64 var_51;
    wp::float64 var_52;
    bool var_53;
    bool var_54;
    bool var_55;
    const wp::float32 var_56 = 2.0;
    wp::float64 var_57;
    wp::float64 var_58;
    bool var_59;
    wp::float64 var_60;
    wp::float64 var_61;
    const wp::int32 var_62 = 2;
    wp::float64 var_63;
    wp::int32 var_64;
    wp::float64 var_65;
    wp::float64 var_66;
    wp::float64 var_67;
    const wp::int32 var_68 = 5;
    wp::float64 var_69;
    wp::int32 var_70;
    wp::float64 var_71;
    wp::float64 var_72;
    wp::int32 var_73;
    wp::float64 var_74;
    wp::float64 var_75;
    wp::int32 var_76;
    wp::float64 var_77;
    wp::float64 var_78;
    wp::int32 var_79;
    wp::float64 var_80;
    const wp::float32 var_81 = 0.0;
    wp::float64 var_82;
    bool var_83;
    wp::float64 var_84;
    const wp::float32 var_85 = 0.0;
    wp::float64 var_86;
    bool var_87;
    const wp::int32 var_88 = 0;
    wp::float64 var_89;
    bool var_90;
    const wp::int32 var_91 = 3;
    const wp::int32 var_92 = 6;
    bool var_93;
    wp::float64 var_94;
    wp::float64 var_95;
    const wp::float32 var_96 = 0.0;
    wp::float64 var_97;
    bool var_98;
    const wp::int32 var_99 = 1;
    wp::float64 var_100;
    wp::float64 var_101;
    bool var_102;
    const wp::int32 var_103 = 4;
    const wp::int32 var_104 = 7;
    //---------
    // dual vars
    wp::vec_t<3,wp::float64> adj_0 = {};
    wp::vec_t<3,wp::float64> adj_1 = {};
    wp::vec_t<3,wp::float64> adj_2 = {};
    wp::float64 adj_3 = {};
    wp::float64 adj_4 = {};
    wp::float64 adj_5 = {};
    wp::float64 adj_6 = {};
    wp::float64 adj_7 = {};
    wp::float64 adj_8 = {};
    wp::float64 adj_9 = {};
    wp::float64 adj_10 = {};
    wp::float64 adj_11 = {};
    wp::int32 adj_12 = {};
    wp::float32 adj_13 = {};
    wp::float64 adj_14 = {};
    wp::float32 adj_15 = {};
    wp::float64 adj_16 = {};
    wp::float64 adj_17 = {};
    wp::float64 adj_18 = {};
    wp::float64 adj_19 = {};
    wp::float32 adj_20 = {};
    wp::float64 adj_21 = {};
    bool adj_22 = {};
    wp::float64 adj_23 = {};
    wp::float64 adj_24 = {};
    wp::int32 adj_25 = {};
    wp::float64 adj_26 = {};
    wp::int32 adj_27 = {};
    wp::float64 adj_28 = {};
    bool adj_29 = {};
    wp::float64 adj_30 = {};
    wp::float64 adj_31 = {};
    wp::int32 adj_32 = {};
    wp::float64 adj_33 = {};
    wp::int32 adj_34 = {};
    wp::float64 adj_35 = {};
    wp::float64 adj_36 = {};
    wp::float64 adj_37 = {};
    wp::float64 adj_38 = {};
    wp::vec_t<3,wp::float64> adj_39 = {};
    wp::float32 adj_40 = {};
    wp::float64 adj_41 = {};
    bool adj_42 = {};
    bool adj_43 = {};
    wp::float64 adj_44 = {};
    wp::float32 adj_45 = {};
    wp::float64 adj_46 = {};
    bool adj_47 = {};
    wp::float64 adj_48 = {};
    wp::float32 adj_49 = {};
    wp::float64 adj_50 = {};
    wp::float64 adj_51 = {};
    wp::float64 adj_52 = {};
    bool adj_53 = {};
    bool adj_54 = {};
    bool adj_55 = {};
    wp::float32 adj_56 = {};
    wp::float64 adj_57 = {};
    wp::float64 adj_58 = {};
    bool adj_59 = {};
    wp::float64 adj_60 = {};
    wp::float64 adj_61 = {};
    wp::int32 adj_62 = {};
    wp::float64 adj_63 = {};
    wp::int32 adj_64 = {};
    wp::float64 adj_65 = {};
    wp::float64 adj_66 = {};
    wp::float64 adj_67 = {};
    wp::int32 adj_68 = {};
    wp::float64 adj_69 = {};
    wp::int32 adj_70 = {};
    wp::float64 adj_71 = {};
    wp::float64 adj_72 = {};
    wp::int32 adj_73 = {};
    wp::float64 adj_74 = {};
    wp::float64 adj_75 = {};
    wp::int32 adj_76 = {};
    wp::float64 adj_77 = {};
    wp::float64 adj_78 = {};
    wp::int32 adj_79 = {};
    wp::float64 adj_80 = {};
    wp::float32 adj_81 = {};
    wp::float64 adj_82 = {};
    bool adj_83 = {};
    wp::float64 adj_84 = {};
    wp::float32 adj_85 = {};
    wp::float64 adj_86 = {};
    bool adj_87 = {};
    wp::int32 adj_88 = {};
    wp::float64 adj_89 = {};
    bool adj_90 = {};
    wp::int32 adj_91 = {};
    wp::int32 adj_92 = {};
    bool adj_93 = {};
    wp::float64 adj_94 = {};
    wp::float64 adj_95 = {};
    wp::float32 adj_96 = {};
    wp::float64 adj_97 = {};
    bool adj_98 = {};
    wp::int32 adj_99 = {};
    wp::float64 adj_100 = {};
    wp::float64 adj_101 = {};
    bool adj_102 = {};
    wp::int32 adj_103 = {};
    wp::int32 adj_104 = {};
    //---------
    // forward
    // def edge_edge_distance_type(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d) -> int:       <L 79>
    // u = ea1 - ea0                                                                          <L 80>
    var_0 = wp::sub(var_ea1, var_ea0);
    // v = eb1 - eb0                                                                          <L 81>
    var_1 = wp::sub(var_eb1, var_eb0);
    // w = ea0 - eb0                                                                          <L 82>
    var_2 = wp::sub(var_ea0, var_eb0);
    // a = wp.length_sq(u)  # always >= 0                                                     <L 84>
    var_3 = wp::length_sq(var_0);
    // b = wp.dot(u, v)                                                                       <L 85>
    var_4 = wp::dot(var_0, var_1);
    // c = wp.length_sq(v)  # always >= 0                                                     <L 86>
    var_5 = wp::length_sq(var_1);
    // d = wp.dot(u, w)                                                                       <L 87>
    var_6 = wp::dot(var_0, var_2);
    // e = wp.dot(v, w)                                                                       <L 88>
    var_7 = wp::dot(var_1, var_2);
    // D = a * c - b * b  # always >= 0                                                       <L 89>
    var_8 = wp::mul(var_3, var_5);
    var_9 = wp::mul(var_4, var_4);
    var_10 = wp::sub(var_8, var_9);
    // tD = D  # tc = tN / tD, default tD = D >= 0                                            <L 90>
    var_11 = wp::copy(var_10);
    // defaultCase = 8                                                                        <L 91>
    // sN, tN = wp.float64(0.0), wp.float64(0.0)                                              <L 92>
    var_14 = wp::float64(var_13);
    var_16 = wp::float64(var_15);
    // sN = b * e - c * d                                                                     <L 95>
    var_17 = wp::mul(var_4, var_7);
    var_18 = wp::mul(var_5, var_6);
    var_19 = wp::sub(var_17, var_18);
    // if sN <= wp.float64(0.0):  # sc < 0 => the s=0 edge is visible                         <L 96>
    var_21 = wp::float64(var_20);
    var_22 = (var_19 <= var_21);
    if (var_22) {
        // tN = e                                                                             <L 97>
        var_23 = wp::copy(var_7);
        // tD = c                                                                             <L 98>
        var_24 = wp::copy(var_5);
        // defaultCase = 2                                                                    <L 99>
    }
    var_26 = wp::where(var_22, var_24, var_11);
    var_27 = wp::where(var_22, var_25, var_12);
    var_28 = wp::where(var_22, var_23, var_16);
    if (!var_22) {
        // elif sN >= D:  # sc > 1  => the s=1 edge is visible                                <L 100>
        var_29 = (var_19 >= var_10);
        if (var_29) {
            // tN = e + b                                                                     <L 101>
            var_30 = wp::add(var_7, var_4);
            // tD = c                                                                         <L 102>
            var_31 = wp::copy(var_5);
            // defaultCase = 5                                                                <L 103>
        }
        var_33 = wp::where(var_29, var_31, var_26);
        var_34 = wp::where(var_29, var_32, var_27);
        var_35 = wp::where(var_29, var_30, var_28);
        if (!var_29) {
            // tN = a * e - b * d                                                             <L 105>
            var_36 = wp::mul(var_3, var_7);
            var_37 = wp::mul(var_4, var_6);
            var_38 = wp::sub(var_36, var_37);
            // uv = wp.cross(u, v)                                                            <L 106>
            var_39 = wp::cross(var_0, var_1);
            // if tN > wp.float64(0.0) and tN < tD and (wp.dot(uv, w) == wp.float64(0.0) or wp.length_sq(uv) < wp.float64(1e-20) * a * c):       <L 108>
            var_41 = wp::float64(var_40);
            var_42 = (var_38 > var_41);
            var_43 = (var_38 < var_33);
            var_44 = wp::dot(var_39, var_2);
            var_46 = wp::float64(var_45);
            var_47 = (var_44 == var_46);
            var_48 = wp::length_sq(var_39);
            var_50 = wp::float64(var_49);
            var_51 = wp::mul(var_50, var_3);
            var_52 = wp::mul(var_51, var_5);
            var_53 = (var_48 < var_52);
            var_54 = var_47 || var_53;
            var_55 = var_42 && var_43 && var_54;
            if (var_55) {
                // if sN < D / wp.float64(2.0):                                               <L 113>
                var_57 = wp::float64(var_56);
                var_58 = wp::div(var_10, var_57);
                var_59 = (var_19 < var_58);
                if (var_59) {
                    // tN = e                                                                 <L 114>
                    var_60 = wp::copy(var_7);
                    // tD = c                                                                 <L 115>
                    var_61 = wp::copy(var_5);
                    // defaultCase = 2                                                        <L 116>
                }
                var_63 = wp::where(var_59, var_61, var_33);
                var_64 = wp::where(var_59, var_62, var_34);
                var_65 = wp::where(var_59, var_60, var_38);
                if (!var_59) {
                    // tN = e + b                                                             <L 118>
                    var_66 = wp::add(var_7, var_4);
                    // tD = c                                                                 <L 119>
                    var_67 = wp::copy(var_5);
                    // defaultCase = 5                                                        <L 120>
                }
                var_69 = wp::where(var_59, var_63, var_67);
                var_70 = wp::where(var_59, var_64, var_68);
                var_71 = wp::where(var_59, var_65, var_66);
            }
            var_72 = wp::where(var_55, var_69, var_33);
            var_73 = wp::where(var_55, var_70, var_34);
            var_74 = wp::where(var_55, var_71, var_38);
        }
        var_75 = wp::where(var_29, var_33, var_72);
        var_76 = wp::where(var_29, var_34, var_73);
        var_77 = wp::where(var_29, var_35, var_74);
    }
    var_78 = wp::where(var_22, var_26, var_75);
    var_79 = wp::where(var_22, var_27, var_76);
    var_80 = wp::where(var_22, var_28, var_77);
    // if tN <= wp.float64(0.0):  # tc < 0 => the t=0 edge is visible                         <L 123>
    var_82 = wp::float64(var_81);
    var_83 = (var_80 <= var_82);
    if (var_83) {
        // if -d <= wp.float64(0.0):                                                          <L 125>
        var_84 = wp::neg(var_6);
        var_86 = wp::float64(var_85);
        var_87 = (var_84 <= var_86);
        if (var_87) {
            // return 0                                                                       <L 126>
            goto label0;
        }
        if (!var_87) {
            // elif -d >= a:                                                                  <L 127>
            var_89 = wp::neg(var_6);
            var_90 = (var_89 >= var_3);
            if (var_90) {
                // return 3                                                                   <L 128>
                goto label1;
            }
            if (!var_90) {
                // return 6                                                                   <L 130>
                goto label2;
            }
        }
    }
    if (!var_83) {
        // elif tN >= tD:  # tc > 1  => the t=1 edge is visible                               <L 131>
        var_93 = (var_80 >= var_78);
        if (var_93) {
            // if (-d + b) <= wp.float64(0.0):                                                <L 133>
            var_94 = wp::neg(var_6);
            var_95 = wp::add(var_94, var_4);
            var_97 = wp::float64(var_96);
            var_98 = (var_95 <= var_97);
            if (var_98) {
                // return 1                                                                   <L 134>
                goto label3;
            }
            if (!var_98) {
                // elif (-d + b) >= a:                                                        <L 135>
                var_100 = wp::neg(var_6);
                var_101 = wp::add(var_100, var_4);
                var_102 = (var_101 >= var_3);
                if (var_102) {
                    // return 4                                                               <L 136>
                    goto label4;
                }
                if (!var_102) {
                    // return 7                                                               <L 138>
                    goto label5;
                }
            }
        }
    }
    // return defaultCase                                                                     <L 140>
    goto label6;
    //---------
    // reverse
    label6:;
    adj_79 += adj_ret;
    // adj: return defaultCase                                                                <L 140>
    if (!var_83) {
        if (var_93) {
            if (!var_98) {
                if (!var_102) {
                    label5:;
                    adj_104 += adj_ret;
                    // adj: return 7                                                          <L 138>
                }
                if (var_102) {
                    label4:;
                    adj_103 += adj_ret;
                    // adj: return 4                                                          <L 136>
                }
                wp::adj_add(var_100, var_4, adj_100, adj_4, adj_101);
                wp::adj_neg(var_6, adj_6, adj_100);
                // adj: elif (-d + b) >= a:                                                   <L 135>
            }
            if (var_98) {
                label3:;
                adj_99 += adj_ret;
                // adj: return 1                                                              <L 134>
            }
            wp::adj_float64(var_96, adj_96, adj_97);
            wp::adj_add(var_94, var_4, adj_94, adj_4, adj_95);
            wp::adj_neg(var_6, adj_6, adj_94);
            // adj: if (-d + b) <= wp.float64(0.0):                                           <L 133>
        }
        // adj: elif tN >= tD:  # tc > 1  => the t=1 edge is visible                          <L 131>
    }
    if (var_83) {
        if (!var_87) {
            if (!var_90) {
                label2:;
                adj_92 += adj_ret;
                // adj: return 6                                                              <L 130>
            }
            if (var_90) {
                label1:;
                adj_91 += adj_ret;
                // adj: return 3                                                              <L 128>
            }
            wp::adj_neg(var_6, adj_6, adj_89);
            // adj: elif -d >= a:                                                             <L 127>
        }
        if (var_87) {
            label0:;
            adj_88 += adj_ret;
            // adj: return 0                                                                  <L 126>
        }
        wp::adj_float64(var_85, adj_85, adj_86);
        wp::adj_neg(var_6, adj_6, adj_84);
        // adj: if -d <= wp.float64(0.0):                                                     <L 125>
    }
    wp::adj_float64(var_81, adj_81, adj_82);
    // adj: if tN <= wp.float64(0.0):  # tc < 0 => the t=0 edge is visible                    <L 123>
    wp::adj_where(var_22, var_28, var_77, adj_22, adj_28, adj_77, adj_80);
    wp::adj_where(var_22, var_27, var_76, adj_22, adj_27, adj_76, adj_79);
    wp::adj_where(var_22, var_26, var_75, adj_22, adj_26, adj_75, adj_78);
    if (!var_22) {
        wp::adj_where(var_29, var_35, var_74, adj_29, adj_35, adj_74, adj_77);
        wp::adj_where(var_29, var_34, var_73, adj_29, adj_34, adj_73, adj_76);
        wp::adj_where(var_29, var_33, var_72, adj_29, adj_33, adj_72, adj_75);
        if (!var_29) {
            wp::adj_where(var_55, var_71, var_38, adj_55, adj_71, adj_38, adj_74);
            wp::adj_where(var_55, var_70, var_34, adj_55, adj_70, adj_34, adj_73);
            wp::adj_where(var_55, var_69, var_33, adj_55, adj_69, adj_33, adj_72);
            if (var_55) {
                wp::adj_where(var_59, var_65, var_66, adj_59, adj_65, adj_66, adj_71);
                wp::adj_where(var_59, var_64, var_68, adj_59, adj_64, adj_68, adj_70);
                wp::adj_where(var_59, var_63, var_67, adj_59, adj_63, adj_67, adj_69);
                if (!var_59) {
                    // adj: defaultCase = 5                                                   <L 120>
                    wp::adj_copy(var_5, adj_5, adj_67);
                    // adj: tD = c                                                            <L 119>
                    wp::adj_add(var_7, var_4, adj_7, adj_4, adj_66);
                    // adj: tN = e + b                                                        <L 118>
                }
                wp::adj_where(var_59, var_60, var_38, adj_59, adj_60, adj_38, adj_65);
                wp::adj_where(var_59, var_62, var_34, adj_59, adj_62, adj_34, adj_64);
                wp::adj_where(var_59, var_61, var_33, adj_59, adj_61, adj_33, adj_63);
                if (var_59) {
                    // adj: defaultCase = 2                                                   <L 116>
                    wp::adj_copy(var_5, adj_5, adj_61);
                    // adj: tD = c                                                            <L 115>
                    wp::adj_copy(var_7, adj_7, adj_60);
                    // adj: tN = e                                                            <L 114>
                }
                wp::adj_div(var_10, var_57, var_58, adj_10, adj_57, adj_58);
                wp::adj_float64(var_56, adj_56, adj_57);
                // adj: if sN < D / wp.float64(2.0):                                          <L 113>
            }
            wp::adj_mul(var_51, var_5, adj_51, adj_5, adj_52);
            wp::adj_mul(var_50, var_3, adj_50, adj_3, adj_51);
            wp::adj_float64(var_49, adj_49, adj_50);
            wp::adj_length_sq(var_39, adj_39, adj_48);
            wp::adj_float64(var_45, adj_45, adj_46);
            wp::adj_dot(var_39, var_2, adj_39, adj_2, adj_44);
            wp::adj_float64(var_40, adj_40, adj_41);
            // adj: if tN > wp.float64(0.0) and tN < tD and (wp.dot(uv, w) == wp.float64(0.0) or wp.length_sq(uv) < wp.float64(1e-20) * a * c):  <L 108>
            wp::adj_cross(var_0, var_1, adj_0, adj_1, adj_39);
            // adj: uv = wp.cross(u, v)                                                       <L 106>
            wp::adj_sub(var_36, var_37, adj_36, adj_37, adj_38);
            wp::adj_mul(var_4, var_6, adj_4, adj_6, adj_37);
            wp::adj_mul(var_3, var_7, adj_3, adj_7, adj_36);
            // adj: tN = a * e - b * d                                                        <L 105>
        }
        wp::adj_where(var_29, var_30, var_28, adj_29, adj_30, adj_28, adj_35);
        wp::adj_where(var_29, var_32, var_27, adj_29, adj_32, adj_27, adj_34);
        wp::adj_where(var_29, var_31, var_26, adj_29, adj_31, adj_26, adj_33);
        if (var_29) {
            // adj: defaultCase = 5                                                           <L 103>
            wp::adj_copy(var_5, adj_5, adj_31);
            // adj: tD = c                                                                    <L 102>
            wp::adj_add(var_7, var_4, adj_7, adj_4, adj_30);
            // adj: tN = e + b                                                                <L 101>
        }
        // adj: elif sN >= D:  # sc > 1  => the s=1 edge is visible                           <L 100>
    }
    wp::adj_where(var_22, var_23, var_16, adj_22, adj_23, adj_16, adj_28);
    wp::adj_where(var_22, var_25, var_12, adj_22, adj_25, adj_12, adj_27);
    wp::adj_where(var_22, var_24, var_11, adj_22, adj_24, adj_11, adj_26);
    if (var_22) {
        // adj: defaultCase = 2                                                               <L 99>
        wp::adj_copy(var_5, adj_5, adj_24);
        // adj: tD = c                                                                        <L 98>
        wp::adj_copy(var_7, adj_7, adj_23);
        // adj: tN = e                                                                        <L 97>
    }
    wp::adj_float64(var_20, adj_20, adj_21);
    // adj: if sN <= wp.float64(0.0):  # sc < 0 => the s=0 edge is visible                    <L 96>
    wp::adj_sub(var_17, var_18, adj_17, adj_18, adj_19);
    wp::adj_mul(var_5, var_6, adj_5, adj_6, adj_18);
    wp::adj_mul(var_4, var_7, adj_4, adj_7, adj_17);
    // adj: sN = b * e - c * d                                                                <L 95>
    wp::adj_float64(var_15, adj_15, adj_16);
    wp::adj_float64(var_13, adj_13, adj_14);
    // adj: sN, tN = wp.float64(0.0), wp.float64(0.0)                                         <L 92>
    // adj: defaultCase = 8                                                                   <L 91>
    wp::adj_copy(var_10, adj_10, adj_11);
    // adj: tD = D  # tc = tN / tD, default tD = D >= 0                                       <L 90>
    wp::adj_sub(var_8, var_9, adj_8, adj_9, adj_10);
    wp::adj_mul(var_4, var_4, adj_4, adj_4, adj_9);
    wp::adj_mul(var_3, var_5, adj_3, adj_5, adj_8);
    // adj: D = a * c - b * b  # always >= 0                                                  <L 89>
    wp::adj_dot(var_1, var_2, adj_1, adj_2, adj_7);
    // adj: e = wp.dot(v, w)                                                                  <L 88>
    wp::adj_dot(var_0, var_2, adj_0, adj_2, adj_6);
    // adj: d = wp.dot(u, w)                                                                  <L 87>
    wp::adj_length_sq(var_1, adj_1, adj_5);
    // adj: c = wp.length_sq(v)  # always >= 0                                                <L 86>
    wp::adj_dot(var_0, var_1, adj_0, adj_1, adj_4);
    // adj: b = wp.dot(u, v)                                                                  <L 85>
    wp::adj_length_sq(var_0, adj_0, adj_3);
    // adj: a = wp.length_sq(u)  # always >= 0                                                <L 84>
    wp::adj_sub(var_ea0, var_eb0, adj_ea0, adj_eb0, adj_2);
    // adj: w = ea0 - eb0                                                                     <L 82>
    wp::adj_sub(var_eb1, var_eb0, adj_eb1, adj_eb0, adj_1);
    // adj: v = eb1 - eb0                                                                     <L 81>
    wp::adj_sub(var_ea1, var_ea0, adj_ea1, adj_ea0, adj_0);
    // adj: u = ea1 - ea0                                                                     <L 80>
    // adj: def edge_edge_distance_type(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d) -> int:  <L 79>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/line_line_distance.py:507
static CUDA_CALLABLE wp::float64 line_line_distance_0(
    wp::vec_t<3,wp::float64> var_ea0,
    wp::vec_t<3,wp::float64> var_ea1,
    wp::vec_t<3,wp::float64> var_eb0,
    wp::vec_t<3,wp::float64> var_eb1)
{
    //---------
    // primal vars
    wp::vec_t<3,wp::float64> var_0;
    wp::vec_t<3,wp::float64> var_1;
    wp::vec_t<3,wp::float64> var_2;
    wp::vec_t<3,wp::float64> var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    wp::float64 var_6;
    wp::float64 var_7;
    //---------
    // forward
    // def line_line_distance(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):       <L 508>
    // da = ea1 - ea0                                                                         <L 509>
    var_0 = wp::sub(var_ea1, var_ea0);
    // db = eb1 - eb0                                                                         <L 510>
    var_1 = wp::sub(var_eb1, var_eb0);
    // d = eb0 - ea0                                                                          <L 511>
    var_2 = wp::sub(var_eb0, var_ea0);
    // b = wp.cross(da, db)                                                                   <L 513>
    var_3 = wp::cross(var_0, var_1);
    // aTb = wp.dot(d, b)                                                                     <L 514>
    var_4 = wp::dot(var_2, var_3);
    // return aTb * aTb / wp.length_sq(b)                                                     <L 515>
    var_5 = wp::mul(var_4, var_4);
    var_6 = wp::length_sq(var_3);
    var_7 = wp::div(var_5, var_6);
    return var_7;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/line_line_distance.py:507
static CUDA_CALLABLE void adj_line_line_distance_0(
    wp::vec_t<3,wp::float64> var_ea0,
    wp::vec_t<3,wp::float64> var_ea1,
    wp::vec_t<3,wp::float64> var_eb0,
    wp::vec_t<3,wp::float64> var_eb1,
    wp::vec_t<3,wp::float64> & adj_ea0,
    wp::vec_t<3,wp::float64> & adj_ea1,
    wp::vec_t<3,wp::float64> & adj_eb0,
    wp::vec_t<3,wp::float64> & adj_eb1,
    wp::float64 & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3,wp::float64> var_0;
    wp::vec_t<3,wp::float64> var_1;
    wp::vec_t<3,wp::float64> var_2;
    wp::vec_t<3,wp::float64> var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    wp::float64 var_6;
    wp::float64 var_7;
    //---------
    // dual vars
    wp::vec_t<3,wp::float64> adj_0 = {};
    wp::vec_t<3,wp::float64> adj_1 = {};
    wp::vec_t<3,wp::float64> adj_2 = {};
    wp::vec_t<3,wp::float64> adj_3 = {};
    wp::float64 adj_4 = {};
    wp::float64 adj_5 = {};
    wp::float64 adj_6 = {};
    wp::float64 adj_7 = {};
    //---------
    // forward
    // def line_line_distance(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):       <L 508>
    // da = ea1 - ea0                                                                         <L 509>
    var_0 = wp::sub(var_ea1, var_ea0);
    // db = eb1 - eb0                                                                         <L 510>
    var_1 = wp::sub(var_eb1, var_eb0);
    // d = eb0 - ea0                                                                          <L 511>
    var_2 = wp::sub(var_eb0, var_ea0);
    // b = wp.cross(da, db)                                                                   <L 513>
    var_3 = wp::cross(var_0, var_1);
    // aTb = wp.dot(d, b)                                                                     <L 514>
    var_4 = wp::dot(var_2, var_3);
    // return aTb * aTb / wp.length_sq(b)                                                     <L 515>
    var_5 = wp::mul(var_4, var_4);
    var_6 = wp::length_sq(var_3);
    var_7 = wp::div(var_5, var_6);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_7 += adj_ret;
    wp::adj_div(var_5, var_6, var_7, adj_5, adj_6, adj_7);
    wp::adj_length_sq(var_3, adj_3, adj_6);
    wp::adj_mul(var_4, var_4, adj_4, adj_4, adj_5);
    // adj: return aTb * aTb / wp.length_sq(b)                                                <L 515>
    wp::adj_dot(var_2, var_3, adj_2, adj_3, adj_4);
    // adj: aTb = wp.dot(d, b)                                                                <L 514>
    wp::adj_cross(var_0, var_1, adj_0, adj_1, adj_3);
    // adj: b = wp.cross(da, db)                                                              <L 513>
    wp::adj_sub(var_eb0, var_ea0, adj_eb0, adj_ea0, adj_2);
    // adj: d = eb0 - ea0                                                                     <L 511>
    wp::adj_sub(var_eb1, var_eb0, adj_eb1, adj_eb0, adj_1);
    // adj: db = eb1 - eb0                                                                    <L 510>
    wp::adj_sub(var_ea1, var_ea0, adj_ea1, adj_ea0, adj_0);
    // adj: da = ea1 - ea0                                                                    <L 509>
    // adj: def line_line_distance(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):  <L 508>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/edge_edge_distance.py:9
static CUDA_CALLABLE wp::float64 edge_edge_distance_0(
    wp::vec_t<3,wp::float64> var_ea0,
    wp::vec_t<3,wp::float64> var_ea1,
    wp::vec_t<3,wp::float64> var_eb0,
    wp::vec_t<3,wp::float64> var_eb1)
{
    //---------
    // primal vars
    wp::int32 var_0;
    const wp::int32 var_1 = 0;
    bool var_2;
    wp::float64 var_3;
    const wp::int32 var_4 = 1;
    bool var_5;
    wp::float64 var_6;
    const wp::int32 var_7 = 2;
    bool var_8;
    wp::float64 var_9;
    const wp::int32 var_10 = 3;
    bool var_11;
    wp::float64 var_12;
    const wp::int32 var_13 = 4;
    bool var_14;
    wp::float64 var_15;
    const wp::int32 var_16 = 5;
    bool var_17;
    wp::float64 var_18;
    const wp::int32 var_19 = 6;
    bool var_20;
    wp::float64 var_21;
    const wp::int32 var_22 = 7;
    bool var_23;
    wp::float64 var_24;
    const wp::int32 var_25 = 8;
    bool var_26;
    wp::float64 var_27;
    const wp::float32 var_28 = 1e+20;
    wp::float64 var_29;
    //---------
    // forward
    // def edge_edge_distance(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):       <L 10>
    // dt = edge_edge_distance_type(ea0, ea1, eb0, eb1)                                       <L 11>
    var_0 = edge_edge_distance_type_0(var_ea0, var_ea1, var_eb0, var_eb1);
    // if dt == 0:                                                                            <L 12>
    var_2 = (var_0 == var_1);
    if (var_2) {
        // return point_point_distance(ea0, eb0)                                              <L 13>
        var_3 = point_point_distance_0(var_ea0, var_eb0);
        return var_3;
    }
    if (!var_2) {
        // elif dt == 1:                                                                      <L 14>
        var_5 = (var_0 == var_4);
        if (var_5) {
            // return point_point_distance(ea0, eb1)                                          <L 15>
            var_6 = point_point_distance_0(var_ea0, var_eb1);
            return var_6;
        }
        if (!var_5) {
            // elif dt == 2:                                                                  <L 16>
            var_8 = (var_0 == var_7);
            if (var_8) {
                // return point_line_distance(ea0, eb0, eb1)                                  <L 17>
                var_9 = point_line_distance_0(var_ea0, var_eb0, var_eb1);
                return var_9;
            }
            if (!var_8) {
                // elif dt == 3:                                                              <L 18>
                var_11 = (var_0 == var_10);
                if (var_11) {
                    // return point_point_distance(ea1, eb0)                                  <L 19>
                    var_12 = point_point_distance_0(var_ea1, var_eb0);
                    return var_12;
                }
                if (!var_11) {
                    // elif dt == 4:                                                          <L 20>
                    var_14 = (var_0 == var_13);
                    if (var_14) {
                        // return point_point_distance(ea1, eb1)                              <L 21>
                        var_15 = point_point_distance_0(var_ea1, var_eb1);
                        return var_15;
                    }
                    if (!var_14) {
                        // elif dt == 5:                                                      <L 22>
                        var_17 = (var_0 == var_16);
                        if (var_17) {
                            // return point_line_distance(ea1, eb0, eb1)                      <L 23>
                            var_18 = point_line_distance_0(var_ea1, var_eb0, var_eb1);
                            return var_18;
                        }
                        if (!var_17) {
                            // elif dt == 6:                                                  <L 24>
                            var_20 = (var_0 == var_19);
                            if (var_20) {
                                // return point_line_distance(eb0, ea0, ea1)                  <L 25>
                                var_21 = point_line_distance_0(var_eb0, var_ea0, var_ea1);
                                return var_21;
                            }
                            if (!var_20) {
                                // elif dt == 7:                                              <L 26>
                                var_23 = (var_0 == var_22);
                                if (var_23) {
                                    // return point_line_distance(eb1, ea0, ea1)              <L 27>
                                    var_24 = point_line_distance_0(var_eb1, var_ea0, var_ea1);
                                    return var_24;
                                }
                                if (!var_23) {
                                    // elif dt == 8:                                          <L 28>
                                    var_26 = (var_0 == var_25);
                                    if (var_26) {
                                        // return line_line_distance(ea0, ea1, eb0, eb1)       <L 29>
                                        var_27 = line_line_distance_0(var_ea0, var_ea1, var_eb0, var_eb1);
                                        return var_27;
                                    }
                                    if (!var_26) {
                                        // return wp.float64(1e20)                            <L 31>
                                        var_29 = wp::float64(var_28);
                                        return var_29;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return {};
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/edge_edge_distance.py:9
static CUDA_CALLABLE void adj_edge_edge_distance_0(
    wp::vec_t<3,wp::float64> var_ea0,
    wp::vec_t<3,wp::float64> var_ea1,
    wp::vec_t<3,wp::float64> var_eb0,
    wp::vec_t<3,wp::float64> var_eb1,
    wp::vec_t<3,wp::float64> & adj_ea0,
    wp::vec_t<3,wp::float64> & adj_ea1,
    wp::vec_t<3,wp::float64> & adj_eb0,
    wp::vec_t<3,wp::float64> & adj_eb1,
    wp::float64 & adj_ret)
{
    //---------
    // primal vars
    wp::int32 var_0;
    const wp::int32 var_1 = 0;
    bool var_2;
    wp::float64 var_3;
    const wp::int32 var_4 = 1;
    bool var_5;
    wp::float64 var_6;
    const wp::int32 var_7 = 2;
    bool var_8;
    wp::float64 var_9;
    const wp::int32 var_10 = 3;
    bool var_11;
    wp::float64 var_12;
    const wp::int32 var_13 = 4;
    bool var_14;
    wp::float64 var_15;
    const wp::int32 var_16 = 5;
    bool var_17;
    wp::float64 var_18;
    const wp::int32 var_19 = 6;
    bool var_20;
    wp::float64 var_21;
    const wp::int32 var_22 = 7;
    bool var_23;
    wp::float64 var_24;
    const wp::int32 var_25 = 8;
    bool var_26;
    wp::float64 var_27;
    const wp::float32 var_28 = 1e+20;
    wp::float64 var_29;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::int32 adj_1 = {};
    bool adj_2 = {};
    wp::float64 adj_3 = {};
    wp::int32 adj_4 = {};
    bool adj_5 = {};
    wp::float64 adj_6 = {};
    wp::int32 adj_7 = {};
    bool adj_8 = {};
    wp::float64 adj_9 = {};
    wp::int32 adj_10 = {};
    bool adj_11 = {};
    wp::float64 adj_12 = {};
    wp::int32 adj_13 = {};
    bool adj_14 = {};
    wp::float64 adj_15 = {};
    wp::int32 adj_16 = {};
    bool adj_17 = {};
    wp::float64 adj_18 = {};
    wp::int32 adj_19 = {};
    bool adj_20 = {};
    wp::float64 adj_21 = {};
    wp::int32 adj_22 = {};
    bool adj_23 = {};
    wp::float64 adj_24 = {};
    wp::int32 adj_25 = {};
    bool adj_26 = {};
    wp::float64 adj_27 = {};
    wp::float32 adj_28 = {};
    wp::float64 adj_29 = {};
    //---------
    // forward
    // def edge_edge_distance(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):       <L 10>
    // dt = edge_edge_distance_type(ea0, ea1, eb0, eb1)                                       <L 11>
    var_0 = edge_edge_distance_type_0(var_ea0, var_ea1, var_eb0, var_eb1);
    // if dt == 0:                                                                            <L 12>
    var_2 = (var_0 == var_1);
    if (var_2) {
        // return point_point_distance(ea0, eb0)                                              <L 13>
        var_3 = point_point_distance_0(var_ea0, var_eb0);
        goto label0;
    }
    if (!var_2) {
        // elif dt == 1:                                                                      <L 14>
        var_5 = (var_0 == var_4);
        if (var_5) {
            // return point_point_distance(ea0, eb1)                                          <L 15>
            var_6 = point_point_distance_0(var_ea0, var_eb1);
            goto label1;
        }
        if (!var_5) {
            // elif dt == 2:                                                                  <L 16>
            var_8 = (var_0 == var_7);
            if (var_8) {
                // return point_line_distance(ea0, eb0, eb1)                                  <L 17>
                var_9 = point_line_distance_0(var_ea0, var_eb0, var_eb1);
                goto label2;
            }
            if (!var_8) {
                // elif dt == 3:                                                              <L 18>
                var_11 = (var_0 == var_10);
                if (var_11) {
                    // return point_point_distance(ea1, eb0)                                  <L 19>
                    var_12 = point_point_distance_0(var_ea1, var_eb0);
                    goto label3;
                }
                if (!var_11) {
                    // elif dt == 4:                                                          <L 20>
                    var_14 = (var_0 == var_13);
                    if (var_14) {
                        // return point_point_distance(ea1, eb1)                              <L 21>
                        var_15 = point_point_distance_0(var_ea1, var_eb1);
                        goto label4;
                    }
                    if (!var_14) {
                        // elif dt == 5:                                                      <L 22>
                        var_17 = (var_0 == var_16);
                        if (var_17) {
                            // return point_line_distance(ea1, eb0, eb1)                      <L 23>
                            var_18 = point_line_distance_0(var_ea1, var_eb0, var_eb1);
                            goto label5;
                        }
                        if (!var_17) {
                            // elif dt == 6:                                                  <L 24>
                            var_20 = (var_0 == var_19);
                            if (var_20) {
                                // return point_line_distance(eb0, ea0, ea1)                  <L 25>
                                var_21 = point_line_distance_0(var_eb0, var_ea0, var_ea1);
                                goto label6;
                            }
                            if (!var_20) {
                                // elif dt == 7:                                              <L 26>
                                var_23 = (var_0 == var_22);
                                if (var_23) {
                                    // return point_line_distance(eb1, ea0, ea1)              <L 27>
                                    var_24 = point_line_distance_0(var_eb1, var_ea0, var_ea1);
                                    goto label7;
                                }
                                if (!var_23) {
                                    // elif dt == 8:                                          <L 28>
                                    var_26 = (var_0 == var_25);
                                    if (var_26) {
                                        // return line_line_distance(ea0, ea1, eb0, eb1)       <L 29>
                                        var_27 = line_line_distance_0(var_ea0, var_ea1, var_eb0, var_eb1);
                                        goto label8;
                                    }
                                    if (!var_26) {
                                        // return wp.float64(1e20)                            <L 31>
                                        var_29 = wp::float64(var_28);
                                        goto label9;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    //---------
    // reverse
    if (!var_2) {
        if (!var_5) {
            if (!var_8) {
                if (!var_11) {
                    if (!var_14) {
                        if (!var_17) {
                            if (!var_20) {
                                if (!var_23) {
                                    if (!var_26) {
                                        label9:;
                                        adj_29 += adj_ret;
                                        wp::adj_float64(var_28, adj_28, adj_29);
                                        // adj: return wp.float64(1e20)                       <L 31>
                                    }
                                    if (var_26) {
                                        label8:;
                                        adj_27 += adj_ret;
                                        adj_line_line_distance_0(var_ea0, var_ea1, var_eb0, var_eb1, adj_ea0, adj_ea1, adj_eb0, adj_eb1, adj_27);
                                        // adj: return line_line_distance(ea0, ea1, eb0, eb1)  <L 29>
                                    }
                                    // adj: elif dt == 8:                                     <L 28>
                                }
                                if (var_23) {
                                    label7:;
                                    adj_24 += adj_ret;
                                    adj_point_line_distance_0(var_eb1, var_ea0, var_ea1, adj_eb1, adj_ea0, adj_ea1, adj_24);
                                    // adj: return point_line_distance(eb1, ea0, ea1)         <L 27>
                                }
                                // adj: elif dt == 7:                                         <L 26>
                            }
                            if (var_20) {
                                label6:;
                                adj_21 += adj_ret;
                                adj_point_line_distance_0(var_eb0, var_ea0, var_ea1, adj_eb0, adj_ea0, adj_ea1, adj_21);
                                // adj: return point_line_distance(eb0, ea0, ea1)             <L 25>
                            }
                            // adj: elif dt == 6:                                             <L 24>
                        }
                        if (var_17) {
                            label5:;
                            adj_18 += adj_ret;
                            adj_point_line_distance_0(var_ea1, var_eb0, var_eb1, adj_ea1, adj_eb0, adj_eb1, adj_18);
                            // adj: return point_line_distance(ea1, eb0, eb1)                 <L 23>
                        }
                        // adj: elif dt == 5:                                                 <L 22>
                    }
                    if (var_14) {
                        label4:;
                        adj_15 += adj_ret;
                        adj_point_point_distance_0(var_ea1, var_eb1, adj_ea1, adj_eb1, adj_15);
                        // adj: return point_point_distance(ea1, eb1)                         <L 21>
                    }
                    // adj: elif dt == 4:                                                     <L 20>
                }
                if (var_11) {
                    label3:;
                    adj_12 += adj_ret;
                    adj_point_point_distance_0(var_ea1, var_eb0, adj_ea1, adj_eb0, adj_12);
                    // adj: return point_point_distance(ea1, eb0)                             <L 19>
                }
                // adj: elif dt == 3:                                                         <L 18>
            }
            if (var_8) {
                label2:;
                adj_9 += adj_ret;
                adj_point_line_distance_0(var_ea0, var_eb0, var_eb1, adj_ea0, adj_eb0, adj_eb1, adj_9);
                // adj: return point_line_distance(ea0, eb0, eb1)                             <L 17>
            }
            // adj: elif dt == 2:                                                             <L 16>
        }
        if (var_5) {
            label1:;
            adj_6 += adj_ret;
            adj_point_point_distance_0(var_ea0, var_eb1, adj_ea0, adj_eb1, adj_6);
            // adj: return point_point_distance(ea0, eb1)                                     <L 15>
        }
        // adj: elif dt == 1:                                                                 <L 14>
    }
    if (var_2) {
        label0:;
        adj_3 += adj_ret;
        adj_point_point_distance_0(var_ea0, var_eb0, adj_ea0, adj_eb0, adj_3);
        // adj: return point_point_distance(ea0, eb0)                                         <L 13>
    }
    // adj: if dt == 0:                                                                       <L 12>
    adj_edge_edge_distance_type_0(var_ea0, var_ea1, var_eb0, var_eb1, adj_ea0, adj_ea1, adj_eb0, adj_eb1, adj_0);
    // adj: dt = edge_edge_distance_type(ea0, ea1, eb0, eb1)                                  <L 11>
    // adj: def edge_edge_distance(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):  <L 10>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/edge_edge_mullifier.py:358
static CUDA_CALLABLE wp::float64 edge_edge_mollifier_threshold_0(
    wp::vec_t<3,wp::float64> var_ea0_rest,
    wp::vec_t<3,wp::float64> var_ea1_rest,
    wp::vec_t<3,wp::float64> var_eb0_rest,
    wp::vec_t<3,wp::float64> var_eb1_rest)
{
    //---------
    // primal vars
    wp::vec_t<3,wp::float64> var_0;
    wp::vec_t<3,wp::float64> var_1;
    const wp::float32 var_2 = 0.001;
    wp::float64 var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    wp::float64 var_6;
    wp::float64 var_7;
    //---------
    // forward
    // def edge_edge_mollifier_threshold(ea0_rest: wp.vec3d, ea1_rest: wp.vec3d, eb0_rest: wp.vec3d, eb1_rest: wp.vec3d):       <L 359>
    // da = ea1_rest - ea0_rest                                                               <L 360>
    var_0 = wp::sub(var_ea1_rest, var_ea0_rest);
    // db = eb1_rest - eb0_rest                                                               <L 361>
    var_1 = wp::sub(var_eb1_rest, var_eb0_rest);
    // return wp.float64(1e-3) * wp.length_sq(da) * wp.length_sq(db)                          <L 362>
    var_3 = wp::float64(var_2);
    var_4 = wp::length_sq(var_0);
    var_5 = wp::mul(var_3, var_4);
    var_6 = wp::length_sq(var_1);
    var_7 = wp::mul(var_5, var_6);
    return var_7;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/edge_edge_mullifier.py:358
static CUDA_CALLABLE void adj_edge_edge_mollifier_threshold_0(
    wp::vec_t<3,wp::float64> var_ea0_rest,
    wp::vec_t<3,wp::float64> var_ea1_rest,
    wp::vec_t<3,wp::float64> var_eb0_rest,
    wp::vec_t<3,wp::float64> var_eb1_rest,
    wp::vec_t<3,wp::float64> & adj_ea0_rest,
    wp::vec_t<3,wp::float64> & adj_ea1_rest,
    wp::vec_t<3,wp::float64> & adj_eb0_rest,
    wp::vec_t<3,wp::float64> & adj_eb1_rest,
    wp::float64 & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3,wp::float64> var_0;
    wp::vec_t<3,wp::float64> var_1;
    const wp::float32 var_2 = 0.001;
    wp::float64 var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    wp::float64 var_6;
    wp::float64 var_7;
    //---------
    // dual vars
    wp::vec_t<3,wp::float64> adj_0 = {};
    wp::vec_t<3,wp::float64> adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float64 adj_3 = {};
    wp::float64 adj_4 = {};
    wp::float64 adj_5 = {};
    wp::float64 adj_6 = {};
    wp::float64 adj_7 = {};
    //---------
    // forward
    // def edge_edge_mollifier_threshold(ea0_rest: wp.vec3d, ea1_rest: wp.vec3d, eb0_rest: wp.vec3d, eb1_rest: wp.vec3d):       <L 359>
    // da = ea1_rest - ea0_rest                                                               <L 360>
    var_0 = wp::sub(var_ea1_rest, var_ea0_rest);
    // db = eb1_rest - eb0_rest                                                               <L 361>
    var_1 = wp::sub(var_eb1_rest, var_eb0_rest);
    // return wp.float64(1e-3) * wp.length_sq(da) * wp.length_sq(db)                          <L 362>
    var_3 = wp::float64(var_2);
    var_4 = wp::length_sq(var_0);
    var_5 = wp::mul(var_3, var_4);
    var_6 = wp::length_sq(var_1);
    var_7 = wp::mul(var_5, var_6);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_7 += adj_ret;
    wp::adj_mul(var_5, var_6, adj_5, adj_6, adj_7);
    wp::adj_length_sq(var_1, adj_1, adj_6);
    wp::adj_mul(var_3, var_4, adj_3, adj_4, adj_5);
    wp::adj_length_sq(var_0, adj_0, adj_4);
    wp::adj_float64(var_2, adj_2, adj_3);
    // adj: return wp.float64(1e-3) * wp.length_sq(da) * wp.length_sq(db)                     <L 362>
    wp::adj_sub(var_eb1_rest, var_eb0_rest, adj_eb1_rest, adj_eb0_rest, adj_1);
    // adj: db = eb1_rest - eb0_rest                                                          <L 361>
    wp::adj_sub(var_ea1_rest, var_ea0_rest, adj_ea1_rest, adj_ea0_rest, adj_0);
    // adj: da = ea1_rest - ea0_rest                                                          <L 360>
    // adj: def edge_edge_mollifier_threshold(ea0_rest: wp.vec3d, ea1_rest: wp.vec3d, eb0_rest: wp.vec3d, eb1_rest: wp.vec3d):  <L 359>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/edge_edge_mullifier.py:314
static CUDA_CALLABLE wp::float64 edge_edge_cross_norm2_0(
    wp::vec_t<3,wp::float64> var_ea0,
    wp::vec_t<3,wp::float64> var_ea1,
    wp::vec_t<3,wp::float64> var_eb0,
    wp::vec_t<3,wp::float64> var_eb1)
{
    //---------
    // primal vars
    wp::vec_t<3,wp::float64> var_0;
    wp::vec_t<3,wp::float64> var_1;
    wp::vec_t<3,wp::float64> var_2;
    wp::float64 var_3;
    //---------
    // forward
    // def edge_edge_cross_norm2(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):       <L 315>
    // da = ea1 - ea0                                                                         <L 316>
    var_0 = wp::sub(var_ea1, var_ea0);
    // db = eb1 - eb0                                                                         <L 317>
    var_1 = wp::sub(var_eb1, var_eb0);
    // b = wp.cross(da, db)                                                                   <L 318>
    var_2 = wp::cross(var_0, var_1);
    // return wp.length_sq(b)                                                                 <L 319>
    var_3 = wp::length_sq(var_2);
    return var_3;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/edge_edge_mullifier.py:314
static CUDA_CALLABLE void adj_edge_edge_cross_norm2_0(
    wp::vec_t<3,wp::float64> var_ea0,
    wp::vec_t<3,wp::float64> var_ea1,
    wp::vec_t<3,wp::float64> var_eb0,
    wp::vec_t<3,wp::float64> var_eb1,
    wp::vec_t<3,wp::float64> & adj_ea0,
    wp::vec_t<3,wp::float64> & adj_ea1,
    wp::vec_t<3,wp::float64> & adj_eb0,
    wp::vec_t<3,wp::float64> & adj_eb1,
    wp::float64 & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3,wp::float64> var_0;
    wp::vec_t<3,wp::float64> var_1;
    wp::vec_t<3,wp::float64> var_2;
    wp::float64 var_3;
    //---------
    // dual vars
    wp::vec_t<3,wp::float64> adj_0 = {};
    wp::vec_t<3,wp::float64> adj_1 = {};
    wp::vec_t<3,wp::float64> adj_2 = {};
    wp::float64 adj_3 = {};
    //---------
    // forward
    // def edge_edge_cross_norm2(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):       <L 315>
    // da = ea1 - ea0                                                                         <L 316>
    var_0 = wp::sub(var_ea1, var_ea0);
    // db = eb1 - eb0                                                                         <L 317>
    var_1 = wp::sub(var_eb1, var_eb0);
    // b = wp.cross(da, db)                                                                   <L 318>
    var_2 = wp::cross(var_0, var_1);
    // return wp.length_sq(b)                                                                 <L 319>
    var_3 = wp::length_sq(var_2);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_3 += adj_ret;
    wp::adj_length_sq(var_2, adj_2, adj_3);
    // adj: return wp.length_sq(b)                                                            <L 319>
    wp::adj_cross(var_0, var_1, adj_0, adj_1, adj_2);
    // adj: b = wp.cross(da, db)                                                              <L 318>
    wp::adj_sub(var_eb1, var_eb0, adj_eb1, adj_eb0, adj_1);
    // adj: db = eb1 - eb0                                                                    <L 317>
    wp::adj_sub(var_ea1, var_ea0, adj_ea1, adj_ea0, adj_0);
    // adj: da = ea1 - ea0                                                                    <L 316>
    // adj: def edge_edge_cross_norm2(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):  <L 315>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/edge_edge_mullifier.py:297
static CUDA_CALLABLE wp::float64 EEM_0(
    wp::float64 var_input,
    wp::float64 var_eps_x)
{
    //---------
    // primal vars
    wp::float64 var_0;
    wp::float64 var_1;
    const wp::float32 var_2 = 2.0;
    wp::float64 var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    //---------
    // forward
    // def EEM(input: wp.float64, eps_x: wp.float64):                                         <L 298>
    // input_div_eps_x = input / eps_x                                                        <L 299>
    var_0 = wp::div(var_input, var_eps_x);
    // return (-input_div_eps_x + wp.float64(2.0)) * input_div_eps_x                          <L 300>
    var_1 = wp::neg(var_0);
    var_3 = wp::float64(var_2);
    var_4 = wp::add(var_1, var_3);
    var_5 = wp::mul(var_4, var_0);
    return var_5;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/edge_edge_mullifier.py:297
static CUDA_CALLABLE void adj_EEM_0(
    wp::float64 var_input,
    wp::float64 var_eps_x,
    wp::float64 & adj_input,
    wp::float64 & adj_eps_x,
    wp::float64 & adj_ret)
{
    //---------
    // primal vars
    wp::float64 var_0;
    wp::float64 var_1;
    const wp::float32 var_2 = 2.0;
    wp::float64 var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    //---------
    // dual vars
    wp::float64 adj_0 = {};
    wp::float64 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float64 adj_3 = {};
    wp::float64 adj_4 = {};
    wp::float64 adj_5 = {};
    //---------
    // forward
    // def EEM(input: wp.float64, eps_x: wp.float64):                                         <L 298>
    // input_div_eps_x = input / eps_x                                                        <L 299>
    var_0 = wp::div(var_input, var_eps_x);
    // return (-input_div_eps_x + wp.float64(2.0)) * input_div_eps_x                          <L 300>
    var_1 = wp::neg(var_0);
    var_3 = wp::float64(var_2);
    var_4 = wp::add(var_1, var_3);
    var_5 = wp::mul(var_4, var_0);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_5 += adj_ret;
    wp::adj_mul(var_4, var_0, adj_4, adj_0, adj_5);
    wp::adj_add(var_1, var_3, adj_1, adj_3, adj_4);
    wp::adj_float64(var_2, adj_2, adj_3);
    wp::adj_neg(var_0, adj_0, adj_1);
    // adj: return (-input_div_eps_x + wp.float64(2.0)) * input_div_eps_x                     <L 300>
    wp::adj_div(var_input, var_eps_x, var_0, adj_input, adj_eps_x, adj_0);
    // adj: input_div_eps_x = input / eps_x                                                   <L 299>
    // adj: def EEM(input: wp.float64, eps_x: wp.float64):                                    <L 298>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/edge_edge_mullifier.py:365
static CUDA_CALLABLE wp::float64 edge_edge_mollifier_0(
    wp::vec_t<3,wp::float64> var_ea0,
    wp::vec_t<3,wp::float64> var_ea1,
    wp::vec_t<3,wp::float64> var_eb0,
    wp::vec_t<3,wp::float64> var_eb1,
    wp::float64 var_eps_x)
{
    //---------
    // primal vars
    wp::float64 var_0;
    bool var_1;
    wp::float64 var_2;
    const wp::float32 var_3 = 1.0;
    wp::float64 var_4;
    //---------
    // forward
    // def edge_edge_mollifier(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d, eps_x: wp.float64):       <L 366>
    // EECrossSqNorm = edge_edge_cross_norm2(ea0, ea1, eb0, eb1)                              <L 367>
    var_0 = edge_edge_cross_norm2_0(var_ea0, var_ea1, var_eb0, var_eb1);
    // if EECrossSqNorm < eps_x:                                                              <L 368>
    var_1 = (var_0 < var_eps_x);
    if (var_1) {
        // return EEM(EECrossSqNorm, eps_x)                                                   <L 369>
        var_2 = EEM_0(var_0, var_eps_x);
        return var_2;
    }
    if (!var_1) {
        // return wp.float64(1.0)                                                             <L 371>
        var_4 = wp::float64(var_3);
        return var_4;
    }
    return {};
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/edge_edge_mullifier.py:365
static CUDA_CALLABLE void adj_edge_edge_mollifier_0(
    wp::vec_t<3,wp::float64> var_ea0,
    wp::vec_t<3,wp::float64> var_ea1,
    wp::vec_t<3,wp::float64> var_eb0,
    wp::vec_t<3,wp::float64> var_eb1,
    wp::float64 var_eps_x,
    wp::vec_t<3,wp::float64> & adj_ea0,
    wp::vec_t<3,wp::float64> & adj_ea1,
    wp::vec_t<3,wp::float64> & adj_eb0,
    wp::vec_t<3,wp::float64> & adj_eb1,
    wp::float64 & adj_eps_x,
    wp::float64 & adj_ret)
{
    //---------
    // primal vars
    wp::float64 var_0;
    bool var_1;
    wp::float64 var_2;
    const wp::float32 var_3 = 1.0;
    wp::float64 var_4;
    //---------
    // dual vars
    wp::float64 adj_0 = {};
    bool adj_1 = {};
    wp::float64 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float64 adj_4 = {};
    //---------
    // forward
    // def edge_edge_mollifier(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d, eps_x: wp.float64):       <L 366>
    // EECrossSqNorm = edge_edge_cross_norm2(ea0, ea1, eb0, eb1)                              <L 367>
    var_0 = edge_edge_cross_norm2_0(var_ea0, var_ea1, var_eb0, var_eb1);
    // if EECrossSqNorm < eps_x:                                                              <L 368>
    var_1 = (var_0 < var_eps_x);
    if (var_1) {
        // return EEM(EECrossSqNorm, eps_x)                                                   <L 369>
        var_2 = EEM_0(var_0, var_eps_x);
        goto label0;
    }
    if (!var_1) {
        // return wp.float64(1.0)                                                             <L 371>
        var_4 = wp::float64(var_3);
        goto label1;
    }
    //---------
    // reverse
    if (!var_1) {
        label1:;
        adj_4 += adj_ret;
        wp::adj_float64(var_3, adj_3, adj_4);
        // adj: return wp.float64(1.0)                                                        <L 371>
    }
    if (var_1) {
        label0:;
        adj_2 += adj_ret;
        adj_EEM_0(var_0, var_eps_x, adj_0, adj_eps_x, adj_2);
        // adj: return EEM(EECrossSqNorm, eps_x)                                              <L 369>
    }
    // adj: if EECrossSqNorm < eps_x:                                                         <L 368>
    adj_edge_edge_cross_norm2_0(var_ea0, var_ea1, var_eb0, var_eb1, adj_ea0, adj_ea1, adj_eb0, adj_eb1, adj_0);
    // adj: EECrossSqNorm = edge_edge_cross_norm2(ea0, ea1, eb0, eb1)                         <L 367>
    // adj: def edge_edge_mollifier(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d, eps_x: wp.float64):  <L 366>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/line_line_distance.py:4
static CUDA_CALLABLE wp::vec_t<12,wp::float64> g_EE_0(
    wp::float64 var_v01,
    wp::float64 var_v02,
    wp::float64 var_v03,
    wp::float64 var_v11,
    wp::float64 var_v12,
    wp::float64 var_v13,
    wp::float64 var_v21,
    wp::float64 var_v22,
    wp::float64 var_v23,
    wp::float64 var_v31,
    wp::float64 var_v32,
    wp::float64 var_v33)
{
    //---------
    // primal vars
    wp::float64 var_0;
    wp::float64 var_1;
    wp::float64 var_2;
    wp::float64 var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    wp::float64 var_6;
    wp::float64 var_7;
    wp::float64 var_8;
    wp::float64 var_9;
    wp::float64 var_10;
    wp::float64 var_11;
    wp::float64 var_12;
    wp::float64 var_13;
    wp::float64 var_14;
    wp::float64 var_15;
    wp::float64 var_16;
    wp::float64 var_17;
    wp::float64 var_18;
    wp::float64 var_19;
    wp::float64 var_20;
    wp::float64 var_21;
    wp::float64 var_22;
    wp::float64 var_23;
    wp::float64 var_24;
    wp::float64 var_25;
    wp::float64 var_26;
    wp::float64 var_27;
    wp::float64 var_28;
    wp::float64 var_29;
    wp::float64 var_30;
    wp::float64 var_31;
    wp::float64 var_32;
    wp::float64 var_33;
    wp::float64 var_34;
    wp::float64 var_35;
    const wp::float32 var_36 = 1.0;
    wp::float64 var_37;
    wp::float64 var_38;
    wp::float64 var_39;
    wp::float64 var_40;
    wp::float64 var_41;
    wp::float64 var_42;
    wp::float64 var_43;
    wp::float64 var_44;
    wp::float64 var_45;
    wp::float64 var_46;
    wp::float64 var_47;
    wp::float64 var_48;
    wp::float64 var_49;
    wp::float64 var_50;
    wp::float64 var_51;
    wp::float64 var_52;
    const wp::float32 var_53 = 2.0;
    wp::float64 var_54;
    wp::float64 var_55;
    wp::float64 var_56;
    const wp::float32 var_57 = 2.0;
    wp::float64 var_58;
    wp::float64 var_59;
    wp::float64 var_60;
    wp::float64 var_61;
    wp::float64 var_62;
    wp::float64 var_63;
    const wp::float32 var_64 = 2.0;
    wp::float64 var_65;
    wp::float64 var_66;
    wp::float64 var_67;
    const wp::float32 var_68 = 2.0;
    wp::float64 var_69;
    wp::float64 var_70;
    wp::float64 var_71;
    wp::float64 var_72;
    wp::float64 var_73;
    wp::float64 var_74;
    const wp::float32 var_75 = 2.0;
    wp::float64 var_76;
    wp::float64 var_77;
    wp::float64 var_78;
    const wp::float32 var_79 = 2.0;
    wp::float64 var_80;
    wp::float64 var_81;
    wp::float64 var_82;
    wp::float64 var_83;
    wp::float64 var_84;
    wp::float64 var_85;
    const wp::float32 var_86 = 2.0;
    wp::float64 var_87;
    wp::float64 var_88;
    wp::float64 var_89;
    const wp::float32 var_90 = 2.0;
    wp::float64 var_91;
    wp::float64 var_92;
    wp::float64 var_93;
    wp::float64 var_94;
    wp::float64 var_95;
    wp::float64 var_96;
    const wp::float32 var_97 = 2.0;
    wp::float64 var_98;
    wp::float64 var_99;
    wp::float64 var_100;
    const wp::float32 var_101 = 2.0;
    wp::float64 var_102;
    wp::float64 var_103;
    wp::float64 var_104;
    wp::float64 var_105;
    wp::float64 var_106;
    wp::float64 var_107;
    wp::float64 var_108;
    const wp::float32 var_109 = 2.0;
    wp::float64 var_110;
    wp::float64 var_111;
    wp::float64 var_112;
    const wp::float32 var_113 = 2.0;
    wp::float64 var_114;
    wp::float64 var_115;
    wp::float64 var_116;
    wp::float64 var_117;
    wp::float64 var_118;
    wp::float64 var_119;
    wp::float64 var_120;
    wp::float64 var_121;
    wp::float64 var_122;
    wp::float64 var_123;
    wp::float64 var_124;
    wp::float64 var_125;
    wp::float64 var_126;
    wp::float64 var_127;
    wp::float64 var_128;
    wp::float64 var_129;
    wp::float64 var_130;
    wp::float64 var_131;
    wp::float64 var_132;
    wp::float64 var_133;
    wp::float64 var_134;
    wp::float64 var_135;
    wp::float64 var_136;
    wp::float64 var_137;
    const wp::float32 var_138 = 2.0;
    wp::float64 var_139;
    wp::float64 var_140;
    wp::float64 var_141;
    wp::float64 var_142;
    wp::float64 var_143;
    wp::float64 var_144;
    wp::float64 var_145;
    const wp::float32 var_146 = 2.0;
    wp::float64 var_147;
    wp::float64 var_148;
    wp::float64 var_149;
    wp::float64 var_150;
    wp::float64 var_151;
    wp::float64 var_152;
    wp::float64 var_153;
    const wp::float32 var_154 = 2.0;
    wp::float64 var_155;
    wp::float64 var_156;
    wp::float64 var_157;
    wp::float64 var_158;
    wp::float64 var_159;
    const wp::float32 var_160 = 2.0;
    wp::float64 var_161;
    wp::float64 var_162;
    wp::float64 var_163;
    wp::float64 var_164;
    wp::float64 var_165;
    wp::float64 var_166;
    const wp::float32 var_167 = 2.0;
    wp::float64 var_168;
    wp::float64 var_169;
    wp::float64 var_170;
    wp::float64 var_171;
    wp::float64 var_172;
    wp::float64 var_173;
    const wp::float32 var_174 = 2.0;
    wp::float64 var_175;
    wp::float64 var_176;
    wp::float64 var_177;
    wp::float64 var_178;
    wp::float64 var_179;
    const wp::float32 var_180 = 2.0;
    wp::float64 var_181;
    wp::float64 var_182;
    wp::float64 var_183;
    wp::float64 var_184;
    wp::float64 var_185;
    wp::float64 var_186;
    const wp::float32 var_187 = 2.0;
    wp::float64 var_188;
    wp::float64 var_189;
    wp::float64 var_190;
    wp::float64 var_191;
    wp::float64 var_192;
    wp::float64 var_193;
    const wp::float32 var_194 = 2.0;
    wp::float64 var_195;
    wp::float64 var_196;
    wp::float64 var_197;
    wp::float64 var_198;
    wp::float64 var_199;
    const wp::float32 var_200 = 2.0;
    wp::float64 var_201;
    wp::float64 var_202;
    wp::float64 var_203;
    wp::float64 var_204;
    const wp::float32 var_205 = 2.0;
    wp::float64 var_206;
    wp::float64 var_207;
    wp::float64 var_208;
    wp::float64 var_209;
    const wp::float32 var_210 = 2.0;
    wp::float64 var_211;
    wp::float64 var_212;
    wp::float64 var_213;
    wp::vec_t<12,wp::float64> var_214;
    //---------
    // forward
    // def g_EE(                                                                              <L 5>
    // t11 = -v11 + v01                                                                       <L 23>
    var_0 = wp::neg(var_v11);
    var_1 = wp::add(var_0, var_v01);
    // t12 = -v12 + v02                                                                       <L 24>
    var_2 = wp::neg(var_v12);
    var_3 = wp::add(var_2, var_v02);
    // t13 = -v13 + v03                                                                       <L 25>
    var_4 = wp::neg(var_v13);
    var_5 = wp::add(var_4, var_v03);
    // t14 = -v21 + v01                                                                       <L 26>
    var_6 = wp::neg(var_v21);
    var_7 = wp::add(var_6, var_v01);
    // t15 = -v22 + v02                                                                       <L 27>
    var_8 = wp::neg(var_v22);
    var_9 = wp::add(var_8, var_v02);
    // t16 = -v23 + v03                                                                       <L 28>
    var_10 = wp::neg(var_v23);
    var_11 = wp::add(var_10, var_v03);
    // t17 = -v31 + v21                                                                       <L 29>
    var_12 = wp::neg(var_v31);
    var_13 = wp::add(var_12, var_v21);
    // t18 = -v32 + v22                                                                       <L 30>
    var_14 = wp::neg(var_v32);
    var_15 = wp::add(var_14, var_v22);
    // t19 = -v33 + v23                                                                       <L 31>
    var_16 = wp::neg(var_v33);
    var_17 = wp::add(var_16, var_v23);
    // t32 = t14 * t18                                                                        <L 32>
    var_18 = wp::mul(var_7, var_15);
    // t33 = t15 * t17                                                                        <L 33>
    var_19 = wp::mul(var_9, var_13);
    // t34 = t14 * t19                                                                        <L 34>
    var_20 = wp::mul(var_7, var_17);
    // t35 = t16 * t17                                                                        <L 35>
    var_21 = wp::mul(var_11, var_13);
    // t36 = t15 * t19                                                                        <L 36>
    var_22 = wp::mul(var_9, var_17);
    // t37 = t16 * t18                                                                        <L 37>
    var_23 = wp::mul(var_11, var_15);
    // t44 = t11 * t18 + -(t12 * t17)                                                         <L 38>
    var_24 = wp::mul(var_1, var_15);
    var_25 = wp::mul(var_3, var_13);
    var_26 = wp::neg(var_25);
    var_27 = wp::add(var_24, var_26);
    // t45 = t11 * t19 + -(t13 * t17)                                                         <L 39>
    var_28 = wp::mul(var_1, var_17);
    var_29 = wp::mul(var_5, var_13);
    var_30 = wp::neg(var_29);
    var_31 = wp::add(var_28, var_30);
    // t46 = t12 * t19 + -(t13 * t18)                                                         <L 40>
    var_32 = wp::mul(var_3, var_17);
    var_33 = wp::mul(var_5, var_15);
    var_34 = wp::neg(var_33);
    var_35 = wp::add(var_32, var_34);
    // t75 = wp.float64(1.0) / ((t44 * t44 + t45 * t45) + t46 * t46)                          <L 41>
    var_37 = wp::float64(var_36);
    var_38 = wp::mul(var_27, var_27);
    var_39 = wp::mul(var_31, var_31);
    var_40 = wp::add(var_38, var_39);
    var_41 = wp::mul(var_35, var_35);
    var_42 = wp::add(var_40, var_41);
    var_43 = wp::div(var_37, var_42);
    // t77 = (t16 * t44 + t14 * t46) + -(t15 * t45)                                           <L 42>
    var_44 = wp::mul(var_11, var_27);
    var_45 = wp::mul(var_7, var_35);
    var_46 = wp::add(var_44, var_45);
    var_47 = wp::mul(var_9, var_31);
    var_48 = wp::neg(var_47);
    var_49 = wp::add(var_46, var_48);
    // t76 = t75 * t75                                                                        <L 43>
    var_50 = wp::mul(var_43, var_43);
    // t78 = t77 * t77                                                                        <L 44>
    var_51 = wp::mul(var_49, var_49);
    // t79 = (t12 * t44 * wp.float64(2.0) + t13 * t45 * wp.float64(2.0)) * t76 * t78          <L 45>
    var_52 = wp::mul(var_3, var_27);
    var_54 = wp::float64(var_53);
    var_55 = wp::mul(var_52, var_54);
    var_56 = wp::mul(var_5, var_31);
    var_58 = wp::float64(var_57);
    var_59 = wp::mul(var_56, var_58);
    var_60 = wp::add(var_55, var_59);
    var_61 = wp::mul(var_60, var_50);
    var_62 = wp::mul(var_61, var_51);
    // t80 = (t11 * t45 * wp.float64(2.0) + t12 * t46 * wp.float64(2.0)) * t76 * t78          <L 46>
    var_63 = wp::mul(var_1, var_31);
    var_65 = wp::float64(var_64);
    var_66 = wp::mul(var_63, var_65);
    var_67 = wp::mul(var_3, var_35);
    var_69 = wp::float64(var_68);
    var_70 = wp::mul(var_67, var_69);
    var_71 = wp::add(var_66, var_70);
    var_72 = wp::mul(var_71, var_50);
    var_73 = wp::mul(var_72, var_51);
    // t81 = (t18 * t44 * wp.float64(2.0) + t19 * t45 * wp.float64(2.0)) * t76 * t78          <L 47>
    var_74 = wp::mul(var_15, var_27);
    var_76 = wp::float64(var_75);
    var_77 = wp::mul(var_74, var_76);
    var_78 = wp::mul(var_17, var_31);
    var_80 = wp::float64(var_79);
    var_81 = wp::mul(var_78, var_80);
    var_82 = wp::add(var_77, var_81);
    var_83 = wp::mul(var_82, var_50);
    var_84 = wp::mul(var_83, var_51);
    // t18 = (t17 * t45 * wp.float64(2.0) + t18 * t46 * wp.float64(2.0)) * t76 * t78          <L 48>
    var_85 = wp::mul(var_13, var_31);
    var_87 = wp::float64(var_86);
    var_88 = wp::mul(var_85, var_87);
    var_89 = wp::mul(var_15, var_35);
    var_91 = wp::float64(var_90);
    var_92 = wp::mul(var_89, var_91);
    var_93 = wp::add(var_88, var_92);
    var_94 = wp::mul(var_93, var_50);
    var_95 = wp::mul(var_94, var_51);
    // t83 = (t11 * t44 * wp.float64(2.0) + -(t13 * t46 * wp.float64(2.0))) * t76 * t78       <L 49>
    var_96 = wp::mul(var_1, var_27);
    var_98 = wp::float64(var_97);
    var_99 = wp::mul(var_96, var_98);
    var_100 = wp::mul(var_5, var_35);
    var_102 = wp::float64(var_101);
    var_103 = wp::mul(var_100, var_102);
    var_104 = wp::neg(var_103);
    var_105 = wp::add(var_99, var_104);
    var_106 = wp::mul(var_105, var_50);
    var_107 = wp::mul(var_106, var_51);
    // t19 = (t17 * t44 * wp.float64(2.0) + -(t19 * t46 * wp.float64(2.0))) * t76 * t78       <L 50>
    var_108 = wp::mul(var_13, var_27);
    var_110 = wp::float64(var_109);
    var_111 = wp::mul(var_108, var_110);
    var_112 = wp::mul(var_17, var_35);
    var_114 = wp::float64(var_113);
    var_115 = wp::mul(var_112, var_114);
    var_116 = wp::neg(var_115);
    var_117 = wp::add(var_111, var_116);
    var_118 = wp::mul(var_117, var_50);
    var_119 = wp::mul(var_118, var_51);
    // t76 = t75 * t77                                                                        <L 51>
    var_120 = wp::mul(var_43, var_49);
    // t17_2 = t12 * t16 + -(t13 * t15)                                                       <L 52>
    var_121 = wp::mul(var_3, var_11);
    var_122 = wp::mul(var_5, var_9);
    var_123 = wp::neg(var_122);
    var_124 = wp::add(var_121, var_123);
    // t18_2 = t11 * t16 + -(t13 * t14)                                                       <L 53>
    var_125 = wp::mul(var_1, var_11);
    var_126 = wp::mul(var_5, var_7);
    var_127 = wp::neg(var_126);
    var_128 = wp::add(var_125, var_127);
    // t19_2 = t11 * t15 + -(t12 * t14)                                                       <L 54>
    var_129 = wp::mul(var_1, var_9);
    var_130 = wp::mul(var_3, var_7);
    var_131 = wp::neg(var_130);
    var_132 = wp::add(var_129, var_131);
    // g = wp.vector(                                                                         <L 55>
    // -t81 + t76 * ((-t36 + t37) + t46) * wp.float64(2.0),                                   <L 56>
    var_133 = wp::neg(var_84);
    var_134 = wp::neg(var_22);
    var_135 = wp::add(var_134, var_23);
    var_136 = wp::add(var_135, var_35);
    var_137 = wp::mul(var_120, var_136);
    var_139 = wp::float64(var_138);
    var_140 = wp::mul(var_137, var_139);
    var_141 = wp::add(var_133, var_140);
    // t19 - t76 * ((-t34 + t35) + t45) * wp.float64(2.0),                                    <L 57>
    var_142 = wp::neg(var_20);
    var_143 = wp::add(var_142, var_21);
    var_144 = wp::add(var_143, var_31);
    var_145 = wp::mul(var_120, var_144);
    var_147 = wp::float64(var_146);
    var_148 = wp::mul(var_145, var_147);
    var_149 = wp::sub(var_119, var_148);
    // t18 + t76 * ((-t32 + t33) + t44) * wp.float64(2.0),                                    <L 58>
    var_150 = wp::neg(var_18);
    var_151 = wp::add(var_150, var_19);
    var_152 = wp::add(var_151, var_27);
    var_153 = wp::mul(var_120, var_152);
    var_155 = wp::float64(var_154);
    var_156 = wp::mul(var_153, var_155);
    var_157 = wp::add(var_95, var_156);
    // t81 + t76 * (t36 - t37) * wp.float64(2.0),                                             <L 59>
    var_158 = wp::sub(var_22, var_23);
    var_159 = wp::mul(var_120, var_158);
    var_161 = wp::float64(var_160);
    var_162 = wp::mul(var_159, var_161);
    var_163 = wp::add(var_84, var_162);
    // -t19 - t76 * (t34 - t35) * wp.float64(2.0),                                            <L 60>
    var_164 = wp::neg(var_119);
    var_165 = wp::sub(var_20, var_21);
    var_166 = wp::mul(var_120, var_165);
    var_168 = wp::float64(var_167);
    var_169 = wp::mul(var_166, var_168);
    var_170 = wp::sub(var_164, var_169);
    // -t18 + t76 * (t32 - t33) * wp.float64(2.0),                                            <L 61>
    var_171 = wp::neg(var_95);
    var_172 = wp::sub(var_18, var_19);
    var_173 = wp::mul(var_120, var_172);
    var_175 = wp::float64(var_174);
    var_176 = wp::mul(var_173, var_175);
    var_177 = wp::add(var_171, var_176);
    // t79 - t76 * (t17_2 + t46) * wp.float64(2.0),                                           <L 62>
    var_178 = wp::add(var_124, var_35);
    var_179 = wp::mul(var_120, var_178);
    var_181 = wp::float64(var_180);
    var_182 = wp::mul(var_179, var_181);
    var_183 = wp::sub(var_62, var_182);
    // -t83 + t76 * (t18_2 + t45) * wp.float64(2.0),                                          <L 63>
    var_184 = wp::neg(var_107);
    var_185 = wp::add(var_128, var_31);
    var_186 = wp::mul(var_120, var_185);
    var_188 = wp::float64(var_187);
    var_189 = wp::mul(var_186, var_188);
    var_190 = wp::add(var_184, var_189);
    // -t80 - t76 * (t19_2 + t44) * wp.float64(2.0),                                          <L 64>
    var_191 = wp::neg(var_73);
    var_192 = wp::add(var_132, var_27);
    var_193 = wp::mul(var_120, var_192);
    var_195 = wp::float64(var_194);
    var_196 = wp::mul(var_193, var_195);
    var_197 = wp::sub(var_191, var_196);
    // -t79 + t76 * t17_2 * wp.float64(2.0),                                                  <L 65>
    var_198 = wp::neg(var_62);
    var_199 = wp::mul(var_120, var_124);
    var_201 = wp::float64(var_200);
    var_202 = wp::mul(var_199, var_201);
    var_203 = wp::add(var_198, var_202);
    // t83 - t76 * t18_2 * wp.float64(2.0),                                                   <L 66>
    var_204 = wp::mul(var_120, var_128);
    var_206 = wp::float64(var_205);
    var_207 = wp::mul(var_204, var_206);
    var_208 = wp::sub(var_107, var_207);
    // t80 + t76 * t19_2 * wp.float64(2.0),                                                   <L 67>
    var_209 = wp::mul(var_120, var_132);
    var_211 = wp::float64(var_210);
    var_212 = wp::mul(var_209, var_211);
    var_213 = wp::add(var_73, var_212);
    var_214 = wp::vec_t<12,wp::float64>({var_141, var_149, var_157, var_163, var_170, var_177, var_183, var_190, var_197, var_203, var_208, var_213});
    // return g                                                                               <L 69>
    return var_214;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/line_line_distance.py:4
static CUDA_CALLABLE void adj_g_EE_0(
    wp::float64 var_v01,
    wp::float64 var_v02,
    wp::float64 var_v03,
    wp::float64 var_v11,
    wp::float64 var_v12,
    wp::float64 var_v13,
    wp::float64 var_v21,
    wp::float64 var_v22,
    wp::float64 var_v23,
    wp::float64 var_v31,
    wp::float64 var_v32,
    wp::float64 var_v33,
    wp::float64 & adj_v01,
    wp::float64 & adj_v02,
    wp::float64 & adj_v03,
    wp::float64 & adj_v11,
    wp::float64 & adj_v12,
    wp::float64 & adj_v13,
    wp::float64 & adj_v21,
    wp::float64 & adj_v22,
    wp::float64 & adj_v23,
    wp::float64 & adj_v31,
    wp::float64 & adj_v32,
    wp::float64 & adj_v33,
    wp::vec_t<12,wp::float64> & adj_ret)
{
    //---------
    // primal vars
    wp::float64 var_0;
    wp::float64 var_1;
    wp::float64 var_2;
    wp::float64 var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    wp::float64 var_6;
    wp::float64 var_7;
    wp::float64 var_8;
    wp::float64 var_9;
    wp::float64 var_10;
    wp::float64 var_11;
    wp::float64 var_12;
    wp::float64 var_13;
    wp::float64 var_14;
    wp::float64 var_15;
    wp::float64 var_16;
    wp::float64 var_17;
    wp::float64 var_18;
    wp::float64 var_19;
    wp::float64 var_20;
    wp::float64 var_21;
    wp::float64 var_22;
    wp::float64 var_23;
    wp::float64 var_24;
    wp::float64 var_25;
    wp::float64 var_26;
    wp::float64 var_27;
    wp::float64 var_28;
    wp::float64 var_29;
    wp::float64 var_30;
    wp::float64 var_31;
    wp::float64 var_32;
    wp::float64 var_33;
    wp::float64 var_34;
    wp::float64 var_35;
    const wp::float32 var_36 = 1.0;
    wp::float64 var_37;
    wp::float64 var_38;
    wp::float64 var_39;
    wp::float64 var_40;
    wp::float64 var_41;
    wp::float64 var_42;
    wp::float64 var_43;
    wp::float64 var_44;
    wp::float64 var_45;
    wp::float64 var_46;
    wp::float64 var_47;
    wp::float64 var_48;
    wp::float64 var_49;
    wp::float64 var_50;
    wp::float64 var_51;
    wp::float64 var_52;
    const wp::float32 var_53 = 2.0;
    wp::float64 var_54;
    wp::float64 var_55;
    wp::float64 var_56;
    const wp::float32 var_57 = 2.0;
    wp::float64 var_58;
    wp::float64 var_59;
    wp::float64 var_60;
    wp::float64 var_61;
    wp::float64 var_62;
    wp::float64 var_63;
    const wp::float32 var_64 = 2.0;
    wp::float64 var_65;
    wp::float64 var_66;
    wp::float64 var_67;
    const wp::float32 var_68 = 2.0;
    wp::float64 var_69;
    wp::float64 var_70;
    wp::float64 var_71;
    wp::float64 var_72;
    wp::float64 var_73;
    wp::float64 var_74;
    const wp::float32 var_75 = 2.0;
    wp::float64 var_76;
    wp::float64 var_77;
    wp::float64 var_78;
    const wp::float32 var_79 = 2.0;
    wp::float64 var_80;
    wp::float64 var_81;
    wp::float64 var_82;
    wp::float64 var_83;
    wp::float64 var_84;
    wp::float64 var_85;
    const wp::float32 var_86 = 2.0;
    wp::float64 var_87;
    wp::float64 var_88;
    wp::float64 var_89;
    const wp::float32 var_90 = 2.0;
    wp::float64 var_91;
    wp::float64 var_92;
    wp::float64 var_93;
    wp::float64 var_94;
    wp::float64 var_95;
    wp::float64 var_96;
    const wp::float32 var_97 = 2.0;
    wp::float64 var_98;
    wp::float64 var_99;
    wp::float64 var_100;
    const wp::float32 var_101 = 2.0;
    wp::float64 var_102;
    wp::float64 var_103;
    wp::float64 var_104;
    wp::float64 var_105;
    wp::float64 var_106;
    wp::float64 var_107;
    wp::float64 var_108;
    const wp::float32 var_109 = 2.0;
    wp::float64 var_110;
    wp::float64 var_111;
    wp::float64 var_112;
    const wp::float32 var_113 = 2.0;
    wp::float64 var_114;
    wp::float64 var_115;
    wp::float64 var_116;
    wp::float64 var_117;
    wp::float64 var_118;
    wp::float64 var_119;
    wp::float64 var_120;
    wp::float64 var_121;
    wp::float64 var_122;
    wp::float64 var_123;
    wp::float64 var_124;
    wp::float64 var_125;
    wp::float64 var_126;
    wp::float64 var_127;
    wp::float64 var_128;
    wp::float64 var_129;
    wp::float64 var_130;
    wp::float64 var_131;
    wp::float64 var_132;
    wp::float64 var_133;
    wp::float64 var_134;
    wp::float64 var_135;
    wp::float64 var_136;
    wp::float64 var_137;
    const wp::float32 var_138 = 2.0;
    wp::float64 var_139;
    wp::float64 var_140;
    wp::float64 var_141;
    wp::float64 var_142;
    wp::float64 var_143;
    wp::float64 var_144;
    wp::float64 var_145;
    const wp::float32 var_146 = 2.0;
    wp::float64 var_147;
    wp::float64 var_148;
    wp::float64 var_149;
    wp::float64 var_150;
    wp::float64 var_151;
    wp::float64 var_152;
    wp::float64 var_153;
    const wp::float32 var_154 = 2.0;
    wp::float64 var_155;
    wp::float64 var_156;
    wp::float64 var_157;
    wp::float64 var_158;
    wp::float64 var_159;
    const wp::float32 var_160 = 2.0;
    wp::float64 var_161;
    wp::float64 var_162;
    wp::float64 var_163;
    wp::float64 var_164;
    wp::float64 var_165;
    wp::float64 var_166;
    const wp::float32 var_167 = 2.0;
    wp::float64 var_168;
    wp::float64 var_169;
    wp::float64 var_170;
    wp::float64 var_171;
    wp::float64 var_172;
    wp::float64 var_173;
    const wp::float32 var_174 = 2.0;
    wp::float64 var_175;
    wp::float64 var_176;
    wp::float64 var_177;
    wp::float64 var_178;
    wp::float64 var_179;
    const wp::float32 var_180 = 2.0;
    wp::float64 var_181;
    wp::float64 var_182;
    wp::float64 var_183;
    wp::float64 var_184;
    wp::float64 var_185;
    wp::float64 var_186;
    const wp::float32 var_187 = 2.0;
    wp::float64 var_188;
    wp::float64 var_189;
    wp::float64 var_190;
    wp::float64 var_191;
    wp::float64 var_192;
    wp::float64 var_193;
    const wp::float32 var_194 = 2.0;
    wp::float64 var_195;
    wp::float64 var_196;
    wp::float64 var_197;
    wp::float64 var_198;
    wp::float64 var_199;
    const wp::float32 var_200 = 2.0;
    wp::float64 var_201;
    wp::float64 var_202;
    wp::float64 var_203;
    wp::float64 var_204;
    const wp::float32 var_205 = 2.0;
    wp::float64 var_206;
    wp::float64 var_207;
    wp::float64 var_208;
    wp::float64 var_209;
    const wp::float32 var_210 = 2.0;
    wp::float64 var_211;
    wp::float64 var_212;
    wp::float64 var_213;
    wp::vec_t<12,wp::float64> var_214;
    //---------
    // dual vars
    wp::float64 adj_0 = {};
    wp::float64 adj_1 = {};
    wp::float64 adj_2 = {};
    wp::float64 adj_3 = {};
    wp::float64 adj_4 = {};
    wp::float64 adj_5 = {};
    wp::float64 adj_6 = {};
    wp::float64 adj_7 = {};
    wp::float64 adj_8 = {};
    wp::float64 adj_9 = {};
    wp::float64 adj_10 = {};
    wp::float64 adj_11 = {};
    wp::float64 adj_12 = {};
    wp::float64 adj_13 = {};
    wp::float64 adj_14 = {};
    wp::float64 adj_15 = {};
    wp::float64 adj_16 = {};
    wp::float64 adj_17 = {};
    wp::float64 adj_18 = {};
    wp::float64 adj_19 = {};
    wp::float64 adj_20 = {};
    wp::float64 adj_21 = {};
    wp::float64 adj_22 = {};
    wp::float64 adj_23 = {};
    wp::float64 adj_24 = {};
    wp::float64 adj_25 = {};
    wp::float64 adj_26 = {};
    wp::float64 adj_27 = {};
    wp::float64 adj_28 = {};
    wp::float64 adj_29 = {};
    wp::float64 adj_30 = {};
    wp::float64 adj_31 = {};
    wp::float64 adj_32 = {};
    wp::float64 adj_33 = {};
    wp::float64 adj_34 = {};
    wp::float64 adj_35 = {};
    wp::float32 adj_36 = {};
    wp::float64 adj_37 = {};
    wp::float64 adj_38 = {};
    wp::float64 adj_39 = {};
    wp::float64 adj_40 = {};
    wp::float64 adj_41 = {};
    wp::float64 adj_42 = {};
    wp::float64 adj_43 = {};
    wp::float64 adj_44 = {};
    wp::float64 adj_45 = {};
    wp::float64 adj_46 = {};
    wp::float64 adj_47 = {};
    wp::float64 adj_48 = {};
    wp::float64 adj_49 = {};
    wp::float64 adj_50 = {};
    wp::float64 adj_51 = {};
    wp::float64 adj_52 = {};
    wp::float32 adj_53 = {};
    wp::float64 adj_54 = {};
    wp::float64 adj_55 = {};
    wp::float64 adj_56 = {};
    wp::float32 adj_57 = {};
    wp::float64 adj_58 = {};
    wp::float64 adj_59 = {};
    wp::float64 adj_60 = {};
    wp::float64 adj_61 = {};
    wp::float64 adj_62 = {};
    wp::float64 adj_63 = {};
    wp::float32 adj_64 = {};
    wp::float64 adj_65 = {};
    wp::float64 adj_66 = {};
    wp::float64 adj_67 = {};
    wp::float32 adj_68 = {};
    wp::float64 adj_69 = {};
    wp::float64 adj_70 = {};
    wp::float64 adj_71 = {};
    wp::float64 adj_72 = {};
    wp::float64 adj_73 = {};
    wp::float64 adj_74 = {};
    wp::float32 adj_75 = {};
    wp::float64 adj_76 = {};
    wp::float64 adj_77 = {};
    wp::float64 adj_78 = {};
    wp::float32 adj_79 = {};
    wp::float64 adj_80 = {};
    wp::float64 adj_81 = {};
    wp::float64 adj_82 = {};
    wp::float64 adj_83 = {};
    wp::float64 adj_84 = {};
    wp::float64 adj_85 = {};
    wp::float32 adj_86 = {};
    wp::float64 adj_87 = {};
    wp::float64 adj_88 = {};
    wp::float64 adj_89 = {};
    wp::float32 adj_90 = {};
    wp::float64 adj_91 = {};
    wp::float64 adj_92 = {};
    wp::float64 adj_93 = {};
    wp::float64 adj_94 = {};
    wp::float64 adj_95 = {};
    wp::float64 adj_96 = {};
    wp::float32 adj_97 = {};
    wp::float64 adj_98 = {};
    wp::float64 adj_99 = {};
    wp::float64 adj_100 = {};
    wp::float32 adj_101 = {};
    wp::float64 adj_102 = {};
    wp::float64 adj_103 = {};
    wp::float64 adj_104 = {};
    wp::float64 adj_105 = {};
    wp::float64 adj_106 = {};
    wp::float64 adj_107 = {};
    wp::float64 adj_108 = {};
    wp::float32 adj_109 = {};
    wp::float64 adj_110 = {};
    wp::float64 adj_111 = {};
    wp::float64 adj_112 = {};
    wp::float32 adj_113 = {};
    wp::float64 adj_114 = {};
    wp::float64 adj_115 = {};
    wp::float64 adj_116 = {};
    wp::float64 adj_117 = {};
    wp::float64 adj_118 = {};
    wp::float64 adj_119 = {};
    wp::float64 adj_120 = {};
    wp::float64 adj_121 = {};
    wp::float64 adj_122 = {};
    wp::float64 adj_123 = {};
    wp::float64 adj_124 = {};
    wp::float64 adj_125 = {};
    wp::float64 adj_126 = {};
    wp::float64 adj_127 = {};
    wp::float64 adj_128 = {};
    wp::float64 adj_129 = {};
    wp::float64 adj_130 = {};
    wp::float64 adj_131 = {};
    wp::float64 adj_132 = {};
    wp::float64 adj_133 = {};
    wp::float64 adj_134 = {};
    wp::float64 adj_135 = {};
    wp::float64 adj_136 = {};
    wp::float64 adj_137 = {};
    wp::float32 adj_138 = {};
    wp::float64 adj_139 = {};
    wp::float64 adj_140 = {};
    wp::float64 adj_141 = {};
    wp::float64 adj_142 = {};
    wp::float64 adj_143 = {};
    wp::float64 adj_144 = {};
    wp::float64 adj_145 = {};
    wp::float32 adj_146 = {};
    wp::float64 adj_147 = {};
    wp::float64 adj_148 = {};
    wp::float64 adj_149 = {};
    wp::float64 adj_150 = {};
    wp::float64 adj_151 = {};
    wp::float64 adj_152 = {};
    wp::float64 adj_153 = {};
    wp::float32 adj_154 = {};
    wp::float64 adj_155 = {};
    wp::float64 adj_156 = {};
    wp::float64 adj_157 = {};
    wp::float64 adj_158 = {};
    wp::float64 adj_159 = {};
    wp::float32 adj_160 = {};
    wp::float64 adj_161 = {};
    wp::float64 adj_162 = {};
    wp::float64 adj_163 = {};
    wp::float64 adj_164 = {};
    wp::float64 adj_165 = {};
    wp::float64 adj_166 = {};
    wp::float32 adj_167 = {};
    wp::float64 adj_168 = {};
    wp::float64 adj_169 = {};
    wp::float64 adj_170 = {};
    wp::float64 adj_171 = {};
    wp::float64 adj_172 = {};
    wp::float64 adj_173 = {};
    wp::float32 adj_174 = {};
    wp::float64 adj_175 = {};
    wp::float64 adj_176 = {};
    wp::float64 adj_177 = {};
    wp::float64 adj_178 = {};
    wp::float64 adj_179 = {};
    wp::float32 adj_180 = {};
    wp::float64 adj_181 = {};
    wp::float64 adj_182 = {};
    wp::float64 adj_183 = {};
    wp::float64 adj_184 = {};
    wp::float64 adj_185 = {};
    wp::float64 adj_186 = {};
    wp::float32 adj_187 = {};
    wp::float64 adj_188 = {};
    wp::float64 adj_189 = {};
    wp::float64 adj_190 = {};
    wp::float64 adj_191 = {};
    wp::float64 adj_192 = {};
    wp::float64 adj_193 = {};
    wp::float32 adj_194 = {};
    wp::float64 adj_195 = {};
    wp::float64 adj_196 = {};
    wp::float64 adj_197 = {};
    wp::float64 adj_198 = {};
    wp::float64 adj_199 = {};
    wp::float32 adj_200 = {};
    wp::float64 adj_201 = {};
    wp::float64 adj_202 = {};
    wp::float64 adj_203 = {};
    wp::float64 adj_204 = {};
    wp::float32 adj_205 = {};
    wp::float64 adj_206 = {};
    wp::float64 adj_207 = {};
    wp::float64 adj_208 = {};
    wp::float64 adj_209 = {};
    wp::float32 adj_210 = {};
    wp::float64 adj_211 = {};
    wp::float64 adj_212 = {};
    wp::float64 adj_213 = {};
    wp::vec_t<12,wp::float64> adj_214 = {};
    //---------
    // forward
    // def g_EE(                                                                              <L 5>
    // t11 = -v11 + v01                                                                       <L 23>
    var_0 = wp::neg(var_v11);
    var_1 = wp::add(var_0, var_v01);
    // t12 = -v12 + v02                                                                       <L 24>
    var_2 = wp::neg(var_v12);
    var_3 = wp::add(var_2, var_v02);
    // t13 = -v13 + v03                                                                       <L 25>
    var_4 = wp::neg(var_v13);
    var_5 = wp::add(var_4, var_v03);
    // t14 = -v21 + v01                                                                       <L 26>
    var_6 = wp::neg(var_v21);
    var_7 = wp::add(var_6, var_v01);
    // t15 = -v22 + v02                                                                       <L 27>
    var_8 = wp::neg(var_v22);
    var_9 = wp::add(var_8, var_v02);
    // t16 = -v23 + v03                                                                       <L 28>
    var_10 = wp::neg(var_v23);
    var_11 = wp::add(var_10, var_v03);
    // t17 = -v31 + v21                                                                       <L 29>
    var_12 = wp::neg(var_v31);
    var_13 = wp::add(var_12, var_v21);
    // t18 = -v32 + v22                                                                       <L 30>
    var_14 = wp::neg(var_v32);
    var_15 = wp::add(var_14, var_v22);
    // t19 = -v33 + v23                                                                       <L 31>
    var_16 = wp::neg(var_v33);
    var_17 = wp::add(var_16, var_v23);
    // t32 = t14 * t18                                                                        <L 32>
    var_18 = wp::mul(var_7, var_15);
    // t33 = t15 * t17                                                                        <L 33>
    var_19 = wp::mul(var_9, var_13);
    // t34 = t14 * t19                                                                        <L 34>
    var_20 = wp::mul(var_7, var_17);
    // t35 = t16 * t17                                                                        <L 35>
    var_21 = wp::mul(var_11, var_13);
    // t36 = t15 * t19                                                                        <L 36>
    var_22 = wp::mul(var_9, var_17);
    // t37 = t16 * t18                                                                        <L 37>
    var_23 = wp::mul(var_11, var_15);
    // t44 = t11 * t18 + -(t12 * t17)                                                         <L 38>
    var_24 = wp::mul(var_1, var_15);
    var_25 = wp::mul(var_3, var_13);
    var_26 = wp::neg(var_25);
    var_27 = wp::add(var_24, var_26);
    // t45 = t11 * t19 + -(t13 * t17)                                                         <L 39>
    var_28 = wp::mul(var_1, var_17);
    var_29 = wp::mul(var_5, var_13);
    var_30 = wp::neg(var_29);
    var_31 = wp::add(var_28, var_30);
    // t46 = t12 * t19 + -(t13 * t18)                                                         <L 40>
    var_32 = wp::mul(var_3, var_17);
    var_33 = wp::mul(var_5, var_15);
    var_34 = wp::neg(var_33);
    var_35 = wp::add(var_32, var_34);
    // t75 = wp.float64(1.0) / ((t44 * t44 + t45 * t45) + t46 * t46)                          <L 41>
    var_37 = wp::float64(var_36);
    var_38 = wp::mul(var_27, var_27);
    var_39 = wp::mul(var_31, var_31);
    var_40 = wp::add(var_38, var_39);
    var_41 = wp::mul(var_35, var_35);
    var_42 = wp::add(var_40, var_41);
    var_43 = wp::div(var_37, var_42);
    // t77 = (t16 * t44 + t14 * t46) + -(t15 * t45)                                           <L 42>
    var_44 = wp::mul(var_11, var_27);
    var_45 = wp::mul(var_7, var_35);
    var_46 = wp::add(var_44, var_45);
    var_47 = wp::mul(var_9, var_31);
    var_48 = wp::neg(var_47);
    var_49 = wp::add(var_46, var_48);
    // t76 = t75 * t75                                                                        <L 43>
    var_50 = wp::mul(var_43, var_43);
    // t78 = t77 * t77                                                                        <L 44>
    var_51 = wp::mul(var_49, var_49);
    // t79 = (t12 * t44 * wp.float64(2.0) + t13 * t45 * wp.float64(2.0)) * t76 * t78          <L 45>
    var_52 = wp::mul(var_3, var_27);
    var_54 = wp::float64(var_53);
    var_55 = wp::mul(var_52, var_54);
    var_56 = wp::mul(var_5, var_31);
    var_58 = wp::float64(var_57);
    var_59 = wp::mul(var_56, var_58);
    var_60 = wp::add(var_55, var_59);
    var_61 = wp::mul(var_60, var_50);
    var_62 = wp::mul(var_61, var_51);
    // t80 = (t11 * t45 * wp.float64(2.0) + t12 * t46 * wp.float64(2.0)) * t76 * t78          <L 46>
    var_63 = wp::mul(var_1, var_31);
    var_65 = wp::float64(var_64);
    var_66 = wp::mul(var_63, var_65);
    var_67 = wp::mul(var_3, var_35);
    var_69 = wp::float64(var_68);
    var_70 = wp::mul(var_67, var_69);
    var_71 = wp::add(var_66, var_70);
    var_72 = wp::mul(var_71, var_50);
    var_73 = wp::mul(var_72, var_51);
    // t81 = (t18 * t44 * wp.float64(2.0) + t19 * t45 * wp.float64(2.0)) * t76 * t78          <L 47>
    var_74 = wp::mul(var_15, var_27);
    var_76 = wp::float64(var_75);
    var_77 = wp::mul(var_74, var_76);
    var_78 = wp::mul(var_17, var_31);
    var_80 = wp::float64(var_79);
    var_81 = wp::mul(var_78, var_80);
    var_82 = wp::add(var_77, var_81);
    var_83 = wp::mul(var_82, var_50);
    var_84 = wp::mul(var_83, var_51);
    // t18 = (t17 * t45 * wp.float64(2.0) + t18 * t46 * wp.float64(2.0)) * t76 * t78          <L 48>
    var_85 = wp::mul(var_13, var_31);
    var_87 = wp::float64(var_86);
    var_88 = wp::mul(var_85, var_87);
    var_89 = wp::mul(var_15, var_35);
    var_91 = wp::float64(var_90);
    var_92 = wp::mul(var_89, var_91);
    var_93 = wp::add(var_88, var_92);
    var_94 = wp::mul(var_93, var_50);
    var_95 = wp::mul(var_94, var_51);
    // t83 = (t11 * t44 * wp.float64(2.0) + -(t13 * t46 * wp.float64(2.0))) * t76 * t78       <L 49>
    var_96 = wp::mul(var_1, var_27);
    var_98 = wp::float64(var_97);
    var_99 = wp::mul(var_96, var_98);
    var_100 = wp::mul(var_5, var_35);
    var_102 = wp::float64(var_101);
    var_103 = wp::mul(var_100, var_102);
    var_104 = wp::neg(var_103);
    var_105 = wp::add(var_99, var_104);
    var_106 = wp::mul(var_105, var_50);
    var_107 = wp::mul(var_106, var_51);
    // t19 = (t17 * t44 * wp.float64(2.0) + -(t19 * t46 * wp.float64(2.0))) * t76 * t78       <L 50>
    var_108 = wp::mul(var_13, var_27);
    var_110 = wp::float64(var_109);
    var_111 = wp::mul(var_108, var_110);
    var_112 = wp::mul(var_17, var_35);
    var_114 = wp::float64(var_113);
    var_115 = wp::mul(var_112, var_114);
    var_116 = wp::neg(var_115);
    var_117 = wp::add(var_111, var_116);
    var_118 = wp::mul(var_117, var_50);
    var_119 = wp::mul(var_118, var_51);
    // t76 = t75 * t77                                                                        <L 51>
    var_120 = wp::mul(var_43, var_49);
    // t17_2 = t12 * t16 + -(t13 * t15)                                                       <L 52>
    var_121 = wp::mul(var_3, var_11);
    var_122 = wp::mul(var_5, var_9);
    var_123 = wp::neg(var_122);
    var_124 = wp::add(var_121, var_123);
    // t18_2 = t11 * t16 + -(t13 * t14)                                                       <L 53>
    var_125 = wp::mul(var_1, var_11);
    var_126 = wp::mul(var_5, var_7);
    var_127 = wp::neg(var_126);
    var_128 = wp::add(var_125, var_127);
    // t19_2 = t11 * t15 + -(t12 * t14)                                                       <L 54>
    var_129 = wp::mul(var_1, var_9);
    var_130 = wp::mul(var_3, var_7);
    var_131 = wp::neg(var_130);
    var_132 = wp::add(var_129, var_131);
    // g = wp.vector(                                                                         <L 55>
    // -t81 + t76 * ((-t36 + t37) + t46) * wp.float64(2.0),                                   <L 56>
    var_133 = wp::neg(var_84);
    var_134 = wp::neg(var_22);
    var_135 = wp::add(var_134, var_23);
    var_136 = wp::add(var_135, var_35);
    var_137 = wp::mul(var_120, var_136);
    var_139 = wp::float64(var_138);
    var_140 = wp::mul(var_137, var_139);
    var_141 = wp::add(var_133, var_140);
    // t19 - t76 * ((-t34 + t35) + t45) * wp.float64(2.0),                                    <L 57>
    var_142 = wp::neg(var_20);
    var_143 = wp::add(var_142, var_21);
    var_144 = wp::add(var_143, var_31);
    var_145 = wp::mul(var_120, var_144);
    var_147 = wp::float64(var_146);
    var_148 = wp::mul(var_145, var_147);
    var_149 = wp::sub(var_119, var_148);
    // t18 + t76 * ((-t32 + t33) + t44) * wp.float64(2.0),                                    <L 58>
    var_150 = wp::neg(var_18);
    var_151 = wp::add(var_150, var_19);
    var_152 = wp::add(var_151, var_27);
    var_153 = wp::mul(var_120, var_152);
    var_155 = wp::float64(var_154);
    var_156 = wp::mul(var_153, var_155);
    var_157 = wp::add(var_95, var_156);
    // t81 + t76 * (t36 - t37) * wp.float64(2.0),                                             <L 59>
    var_158 = wp::sub(var_22, var_23);
    var_159 = wp::mul(var_120, var_158);
    var_161 = wp::float64(var_160);
    var_162 = wp::mul(var_159, var_161);
    var_163 = wp::add(var_84, var_162);
    // -t19 - t76 * (t34 - t35) * wp.float64(2.0),                                            <L 60>
    var_164 = wp::neg(var_119);
    var_165 = wp::sub(var_20, var_21);
    var_166 = wp::mul(var_120, var_165);
    var_168 = wp::float64(var_167);
    var_169 = wp::mul(var_166, var_168);
    var_170 = wp::sub(var_164, var_169);
    // -t18 + t76 * (t32 - t33) * wp.float64(2.0),                                            <L 61>
    var_171 = wp::neg(var_95);
    var_172 = wp::sub(var_18, var_19);
    var_173 = wp::mul(var_120, var_172);
    var_175 = wp::float64(var_174);
    var_176 = wp::mul(var_173, var_175);
    var_177 = wp::add(var_171, var_176);
    // t79 - t76 * (t17_2 + t46) * wp.float64(2.0),                                           <L 62>
    var_178 = wp::add(var_124, var_35);
    var_179 = wp::mul(var_120, var_178);
    var_181 = wp::float64(var_180);
    var_182 = wp::mul(var_179, var_181);
    var_183 = wp::sub(var_62, var_182);
    // -t83 + t76 * (t18_2 + t45) * wp.float64(2.0),                                          <L 63>
    var_184 = wp::neg(var_107);
    var_185 = wp::add(var_128, var_31);
    var_186 = wp::mul(var_120, var_185);
    var_188 = wp::float64(var_187);
    var_189 = wp::mul(var_186, var_188);
    var_190 = wp::add(var_184, var_189);
    // -t80 - t76 * (t19_2 + t44) * wp.float64(2.0),                                          <L 64>
    var_191 = wp::neg(var_73);
    var_192 = wp::add(var_132, var_27);
    var_193 = wp::mul(var_120, var_192);
    var_195 = wp::float64(var_194);
    var_196 = wp::mul(var_193, var_195);
    var_197 = wp::sub(var_191, var_196);
    // -t79 + t76 * t17_2 * wp.float64(2.0),                                                  <L 65>
    var_198 = wp::neg(var_62);
    var_199 = wp::mul(var_120, var_124);
    var_201 = wp::float64(var_200);
    var_202 = wp::mul(var_199, var_201);
    var_203 = wp::add(var_198, var_202);
    // t83 - t76 * t18_2 * wp.float64(2.0),                                                   <L 66>
    var_204 = wp::mul(var_120, var_128);
    var_206 = wp::float64(var_205);
    var_207 = wp::mul(var_204, var_206);
    var_208 = wp::sub(var_107, var_207);
    // t80 + t76 * t19_2 * wp.float64(2.0),                                                   <L 67>
    var_209 = wp::mul(var_120, var_132);
    var_211 = wp::float64(var_210);
    var_212 = wp::mul(var_209, var_211);
    var_213 = wp::add(var_73, var_212);
    var_214 = wp::vec_t<12,wp::float64>({var_141, var_149, var_157, var_163, var_170, var_177, var_183, var_190, var_197, var_203, var_208, var_213});
    // return g                                                                               <L 69>
    goto label0;
    //---------
    // reverse
    label0:;
    adj_214 += adj_ret;
    // adj: return g                                                                          <L 69>
    wp::adj_vec_t({var_141, var_149, var_157, var_163, var_170, var_177, var_183, var_190, var_197, var_203, var_208, var_213}, {&adj_141, &adj_149, &adj_157, &adj_163, &adj_170, &adj_177, &adj_183, &adj_190, &adj_197, &adj_203, &adj_208, &adj_213}, adj_214);
    wp::adj_add(var_73, var_212, adj_73, adj_212, adj_213);
    wp::adj_mul(var_209, var_211, adj_209, adj_211, adj_212);
    wp::adj_float64(var_210, adj_210, adj_211);
    wp::adj_mul(var_120, var_132, adj_120, adj_132, adj_209);
    // adj: t80 + t76 * t19_2 * wp.float64(2.0),                                              <L 67>
    wp::adj_sub(var_107, var_207, adj_107, adj_207, adj_208);
    wp::adj_mul(var_204, var_206, adj_204, adj_206, adj_207);
    wp::adj_float64(var_205, adj_205, adj_206);
    wp::adj_mul(var_120, var_128, adj_120, adj_128, adj_204);
    // adj: t83 - t76 * t18_2 * wp.float64(2.0),                                              <L 66>
    wp::adj_add(var_198, var_202, adj_198, adj_202, adj_203);
    wp::adj_mul(var_199, var_201, adj_199, adj_201, adj_202);
    wp::adj_float64(var_200, adj_200, adj_201);
    wp::adj_mul(var_120, var_124, adj_120, adj_124, adj_199);
    wp::adj_neg(var_62, adj_62, adj_198);
    // adj: -t79 + t76 * t17_2 * wp.float64(2.0),                                             <L 65>
    wp::adj_sub(var_191, var_196, adj_191, adj_196, adj_197);
    wp::adj_mul(var_193, var_195, adj_193, adj_195, adj_196);
    wp::adj_float64(var_194, adj_194, adj_195);
    wp::adj_mul(var_120, var_192, adj_120, adj_192, adj_193);
    wp::adj_add(var_132, var_27, adj_132, adj_27, adj_192);
    wp::adj_neg(var_73, adj_73, adj_191);
    // adj: -t80 - t76 * (t19_2 + t44) * wp.float64(2.0),                                     <L 64>
    wp::adj_add(var_184, var_189, adj_184, adj_189, adj_190);
    wp::adj_mul(var_186, var_188, adj_186, adj_188, adj_189);
    wp::adj_float64(var_187, adj_187, adj_188);
    wp::adj_mul(var_120, var_185, adj_120, adj_185, adj_186);
    wp::adj_add(var_128, var_31, adj_128, adj_31, adj_185);
    wp::adj_neg(var_107, adj_107, adj_184);
    // adj: -t83 + t76 * (t18_2 + t45) * wp.float64(2.0),                                     <L 63>
    wp::adj_sub(var_62, var_182, adj_62, adj_182, adj_183);
    wp::adj_mul(var_179, var_181, adj_179, adj_181, adj_182);
    wp::adj_float64(var_180, adj_180, adj_181);
    wp::adj_mul(var_120, var_178, adj_120, adj_178, adj_179);
    wp::adj_add(var_124, var_35, adj_124, adj_35, adj_178);
    // adj: t79 - t76 * (t17_2 + t46) * wp.float64(2.0),                                      <L 62>
    wp::adj_add(var_171, var_176, adj_171, adj_176, adj_177);
    wp::adj_mul(var_173, var_175, adj_173, adj_175, adj_176);
    wp::adj_float64(var_174, adj_174, adj_175);
    wp::adj_mul(var_120, var_172, adj_120, adj_172, adj_173);
    wp::adj_sub(var_18, var_19, adj_18, adj_19, adj_172);
    wp::adj_neg(var_95, adj_95, adj_171);
    // adj: -t18 + t76 * (t32 - t33) * wp.float64(2.0),                                       <L 61>
    wp::adj_sub(var_164, var_169, adj_164, adj_169, adj_170);
    wp::adj_mul(var_166, var_168, adj_166, adj_168, adj_169);
    wp::adj_float64(var_167, adj_167, adj_168);
    wp::adj_mul(var_120, var_165, adj_120, adj_165, adj_166);
    wp::adj_sub(var_20, var_21, adj_20, adj_21, adj_165);
    wp::adj_neg(var_119, adj_119, adj_164);
    // adj: -t19 - t76 * (t34 - t35) * wp.float64(2.0),                                       <L 60>
    wp::adj_add(var_84, var_162, adj_84, adj_162, adj_163);
    wp::adj_mul(var_159, var_161, adj_159, adj_161, adj_162);
    wp::adj_float64(var_160, adj_160, adj_161);
    wp::adj_mul(var_120, var_158, adj_120, adj_158, adj_159);
    wp::adj_sub(var_22, var_23, adj_22, adj_23, adj_158);
    // adj: t81 + t76 * (t36 - t37) * wp.float64(2.0),                                        <L 59>
    wp::adj_add(var_95, var_156, adj_95, adj_156, adj_157);
    wp::adj_mul(var_153, var_155, adj_153, adj_155, adj_156);
    wp::adj_float64(var_154, adj_154, adj_155);
    wp::adj_mul(var_120, var_152, adj_120, adj_152, adj_153);
    wp::adj_add(var_151, var_27, adj_151, adj_27, adj_152);
    wp::adj_add(var_150, var_19, adj_150, adj_19, adj_151);
    wp::adj_neg(var_18, adj_18, adj_150);
    // adj: t18 + t76 * ((-t32 + t33) + t44) * wp.float64(2.0),                               <L 58>
    wp::adj_sub(var_119, var_148, adj_119, adj_148, adj_149);
    wp::adj_mul(var_145, var_147, adj_145, adj_147, adj_148);
    wp::adj_float64(var_146, adj_146, adj_147);
    wp::adj_mul(var_120, var_144, adj_120, adj_144, adj_145);
    wp::adj_add(var_143, var_31, adj_143, adj_31, adj_144);
    wp::adj_add(var_142, var_21, adj_142, adj_21, adj_143);
    wp::adj_neg(var_20, adj_20, adj_142);
    // adj: t19 - t76 * ((-t34 + t35) + t45) * wp.float64(2.0),                               <L 57>
    wp::adj_add(var_133, var_140, adj_133, adj_140, adj_141);
    wp::adj_mul(var_137, var_139, adj_137, adj_139, adj_140);
    wp::adj_float64(var_138, adj_138, adj_139);
    wp::adj_mul(var_120, var_136, adj_120, adj_136, adj_137);
    wp::adj_add(var_135, var_35, adj_135, adj_35, adj_136);
    wp::adj_add(var_134, var_23, adj_134, adj_23, adj_135);
    wp::adj_neg(var_22, adj_22, adj_134);
    wp::adj_neg(var_84, adj_84, adj_133);
    // adj: -t81 + t76 * ((-t36 + t37) + t46) * wp.float64(2.0),                              <L 56>
    // adj: g = wp.vector(                                                                    <L 55>
    wp::adj_add(var_129, var_131, adj_129, adj_131, adj_132);
    wp::adj_neg(var_130, adj_130, adj_131);
    wp::adj_mul(var_3, var_7, adj_3, adj_7, adj_130);
    wp::adj_mul(var_1, var_9, adj_1, adj_9, adj_129);
    // adj: t19_2 = t11 * t15 + -(t12 * t14)                                                  <L 54>
    wp::adj_add(var_125, var_127, adj_125, adj_127, adj_128);
    wp::adj_neg(var_126, adj_126, adj_127);
    wp::adj_mul(var_5, var_7, adj_5, adj_7, adj_126);
    wp::adj_mul(var_1, var_11, adj_1, adj_11, adj_125);
    // adj: t18_2 = t11 * t16 + -(t13 * t14)                                                  <L 53>
    wp::adj_add(var_121, var_123, adj_121, adj_123, adj_124);
    wp::adj_neg(var_122, adj_122, adj_123);
    wp::adj_mul(var_5, var_9, adj_5, adj_9, adj_122);
    wp::adj_mul(var_3, var_11, adj_3, adj_11, adj_121);
    // adj: t17_2 = t12 * t16 + -(t13 * t15)                                                  <L 52>
    wp::adj_mul(var_43, var_49, adj_43, adj_49, adj_120);
    // adj: t76 = t75 * t77                                                                   <L 51>
    wp::adj_mul(var_118, var_51, adj_118, adj_51, adj_119);
    wp::adj_mul(var_117, var_50, adj_117, adj_50, adj_118);
    wp::adj_add(var_111, var_116, adj_111, adj_116, adj_117);
    wp::adj_neg(var_115, adj_115, adj_116);
    wp::adj_mul(var_112, var_114, adj_112, adj_114, adj_115);
    wp::adj_float64(var_113, adj_113, adj_114);
    wp::adj_mul(var_17, var_35, adj_17, adj_35, adj_112);
    wp::adj_mul(var_108, var_110, adj_108, adj_110, adj_111);
    wp::adj_float64(var_109, adj_109, adj_110);
    wp::adj_mul(var_13, var_27, adj_13, adj_27, adj_108);
    // adj: t19 = (t17 * t44 * wp.float64(2.0) + -(t19 * t46 * wp.float64(2.0))) * t76 * t78  <L 50>
    wp::adj_mul(var_106, var_51, adj_106, adj_51, adj_107);
    wp::adj_mul(var_105, var_50, adj_105, adj_50, adj_106);
    wp::adj_add(var_99, var_104, adj_99, adj_104, adj_105);
    wp::adj_neg(var_103, adj_103, adj_104);
    wp::adj_mul(var_100, var_102, adj_100, adj_102, adj_103);
    wp::adj_float64(var_101, adj_101, adj_102);
    wp::adj_mul(var_5, var_35, adj_5, adj_35, adj_100);
    wp::adj_mul(var_96, var_98, adj_96, adj_98, adj_99);
    wp::adj_float64(var_97, adj_97, adj_98);
    wp::adj_mul(var_1, var_27, adj_1, adj_27, adj_96);
    // adj: t83 = (t11 * t44 * wp.float64(2.0) + -(t13 * t46 * wp.float64(2.0))) * t76 * t78  <L 49>
    wp::adj_mul(var_94, var_51, adj_94, adj_51, adj_95);
    wp::adj_mul(var_93, var_50, adj_93, adj_50, adj_94);
    wp::adj_add(var_88, var_92, adj_88, adj_92, adj_93);
    wp::adj_mul(var_89, var_91, adj_89, adj_91, adj_92);
    wp::adj_float64(var_90, adj_90, adj_91);
    wp::adj_mul(var_15, var_35, adj_15, adj_35, adj_89);
    wp::adj_mul(var_85, var_87, adj_85, adj_87, adj_88);
    wp::adj_float64(var_86, adj_86, adj_87);
    wp::adj_mul(var_13, var_31, adj_13, adj_31, adj_85);
    // adj: t18 = (t17 * t45 * wp.float64(2.0) + t18 * t46 * wp.float64(2.0)) * t76 * t78     <L 48>
    wp::adj_mul(var_83, var_51, adj_83, adj_51, adj_84);
    wp::adj_mul(var_82, var_50, adj_82, adj_50, adj_83);
    wp::adj_add(var_77, var_81, adj_77, adj_81, adj_82);
    wp::adj_mul(var_78, var_80, adj_78, adj_80, adj_81);
    wp::adj_float64(var_79, adj_79, adj_80);
    wp::adj_mul(var_17, var_31, adj_17, adj_31, adj_78);
    wp::adj_mul(var_74, var_76, adj_74, adj_76, adj_77);
    wp::adj_float64(var_75, adj_75, adj_76);
    wp::adj_mul(var_15, var_27, adj_15, adj_27, adj_74);
    // adj: t81 = (t18 * t44 * wp.float64(2.0) + t19 * t45 * wp.float64(2.0)) * t76 * t78     <L 47>
    wp::adj_mul(var_72, var_51, adj_72, adj_51, adj_73);
    wp::adj_mul(var_71, var_50, adj_71, adj_50, adj_72);
    wp::adj_add(var_66, var_70, adj_66, adj_70, adj_71);
    wp::adj_mul(var_67, var_69, adj_67, adj_69, adj_70);
    wp::adj_float64(var_68, adj_68, adj_69);
    wp::adj_mul(var_3, var_35, adj_3, adj_35, adj_67);
    wp::adj_mul(var_63, var_65, adj_63, adj_65, adj_66);
    wp::adj_float64(var_64, adj_64, adj_65);
    wp::adj_mul(var_1, var_31, adj_1, adj_31, adj_63);
    // adj: t80 = (t11 * t45 * wp.float64(2.0) + t12 * t46 * wp.float64(2.0)) * t76 * t78     <L 46>
    wp::adj_mul(var_61, var_51, adj_61, adj_51, adj_62);
    wp::adj_mul(var_60, var_50, adj_60, adj_50, adj_61);
    wp::adj_add(var_55, var_59, adj_55, adj_59, adj_60);
    wp::adj_mul(var_56, var_58, adj_56, adj_58, adj_59);
    wp::adj_float64(var_57, adj_57, adj_58);
    wp::adj_mul(var_5, var_31, adj_5, adj_31, adj_56);
    wp::adj_mul(var_52, var_54, adj_52, adj_54, adj_55);
    wp::adj_float64(var_53, adj_53, adj_54);
    wp::adj_mul(var_3, var_27, adj_3, adj_27, adj_52);
    // adj: t79 = (t12 * t44 * wp.float64(2.0) + t13 * t45 * wp.float64(2.0)) * t76 * t78     <L 45>
    wp::adj_mul(var_49, var_49, adj_49, adj_49, adj_51);
    // adj: t78 = t77 * t77                                                                   <L 44>
    wp::adj_mul(var_43, var_43, adj_43, adj_43, adj_50);
    // adj: t76 = t75 * t75                                                                   <L 43>
    wp::adj_add(var_46, var_48, adj_46, adj_48, adj_49);
    wp::adj_neg(var_47, adj_47, adj_48);
    wp::adj_mul(var_9, var_31, adj_9, adj_31, adj_47);
    wp::adj_add(var_44, var_45, adj_44, adj_45, adj_46);
    wp::adj_mul(var_7, var_35, adj_7, adj_35, adj_45);
    wp::adj_mul(var_11, var_27, adj_11, adj_27, adj_44);
    // adj: t77 = (t16 * t44 + t14 * t46) + -(t15 * t45)                                      <L 42>
    wp::adj_div(var_37, var_42, var_43, adj_37, adj_42, adj_43);
    wp::adj_add(var_40, var_41, adj_40, adj_41, adj_42);
    wp::adj_mul(var_35, var_35, adj_35, adj_35, adj_41);
    wp::adj_add(var_38, var_39, adj_38, adj_39, adj_40);
    wp::adj_mul(var_31, var_31, adj_31, adj_31, adj_39);
    wp::adj_mul(var_27, var_27, adj_27, adj_27, adj_38);
    wp::adj_float64(var_36, adj_36, adj_37);
    // adj: t75 = wp.float64(1.0) / ((t44 * t44 + t45 * t45) + t46 * t46)                     <L 41>
    wp::adj_add(var_32, var_34, adj_32, adj_34, adj_35);
    wp::adj_neg(var_33, adj_33, adj_34);
    wp::adj_mul(var_5, var_15, adj_5, adj_15, adj_33);
    wp::adj_mul(var_3, var_17, adj_3, adj_17, adj_32);
    // adj: t46 = t12 * t19 + -(t13 * t18)                                                    <L 40>
    wp::adj_add(var_28, var_30, adj_28, adj_30, adj_31);
    wp::adj_neg(var_29, adj_29, adj_30);
    wp::adj_mul(var_5, var_13, adj_5, adj_13, adj_29);
    wp::adj_mul(var_1, var_17, adj_1, adj_17, adj_28);
    // adj: t45 = t11 * t19 + -(t13 * t17)                                                    <L 39>
    wp::adj_add(var_24, var_26, adj_24, adj_26, adj_27);
    wp::adj_neg(var_25, adj_25, adj_26);
    wp::adj_mul(var_3, var_13, adj_3, adj_13, adj_25);
    wp::adj_mul(var_1, var_15, adj_1, adj_15, adj_24);
    // adj: t44 = t11 * t18 + -(t12 * t17)                                                    <L 38>
    wp::adj_mul(var_11, var_15, adj_11, adj_15, adj_23);
    // adj: t37 = t16 * t18                                                                   <L 37>
    wp::adj_mul(var_9, var_17, adj_9, adj_17, adj_22);
    // adj: t36 = t15 * t19                                                                   <L 36>
    wp::adj_mul(var_11, var_13, adj_11, adj_13, adj_21);
    // adj: t35 = t16 * t17                                                                   <L 35>
    wp::adj_mul(var_7, var_17, adj_7, adj_17, adj_20);
    // adj: t34 = t14 * t19                                                                   <L 34>
    wp::adj_mul(var_9, var_13, adj_9, adj_13, adj_19);
    // adj: t33 = t15 * t17                                                                   <L 33>
    wp::adj_mul(var_7, var_15, adj_7, adj_15, adj_18);
    // adj: t32 = t14 * t18                                                                   <L 32>
    wp::adj_add(var_16, var_v23, adj_16, adj_v23, adj_17);
    wp::adj_neg(var_v33, adj_v33, adj_16);
    // adj: t19 = -v33 + v23                                                                  <L 31>
    wp::adj_add(var_14, var_v22, adj_14, adj_v22, adj_15);
    wp::adj_neg(var_v32, adj_v32, adj_14);
    // adj: t18 = -v32 + v22                                                                  <L 30>
    wp::adj_add(var_12, var_v21, adj_12, adj_v21, adj_13);
    wp::adj_neg(var_v31, adj_v31, adj_12);
    // adj: t17 = -v31 + v21                                                                  <L 29>
    wp::adj_add(var_10, var_v03, adj_10, adj_v03, adj_11);
    wp::adj_neg(var_v23, adj_v23, adj_10);
    // adj: t16 = -v23 + v03                                                                  <L 28>
    wp::adj_add(var_8, var_v02, adj_8, adj_v02, adj_9);
    wp::adj_neg(var_v22, adj_v22, adj_8);
    // adj: t15 = -v22 + v02                                                                  <L 27>
    wp::adj_add(var_6, var_v01, adj_6, adj_v01, adj_7);
    wp::adj_neg(var_v21, adj_v21, adj_6);
    // adj: t14 = -v21 + v01                                                                  <L 26>
    wp::adj_add(var_4, var_v03, adj_4, adj_v03, adj_5);
    wp::adj_neg(var_v13, adj_v13, adj_4);
    // adj: t13 = -v13 + v03                                                                  <L 25>
    wp::adj_add(var_2, var_v02, adj_2, adj_v02, adj_3);
    wp::adj_neg(var_v12, adj_v12, adj_2);
    // adj: t12 = -v12 + v02                                                                  <L 24>
    wp::adj_add(var_0, var_v01, adj_0, adj_v01, adj_1);
    wp::adj_neg(var_v11, adj_v11, adj_0);
    // adj: t11 = -v11 + v01                                                                  <L 23>
    // adj: def g_EE(                                                                         <L 5>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/line_line_distance.py:518
static CUDA_CALLABLE wp::vec_t<12,wp::float64> line_line_distance_gradient_0(
    wp::vec_t<3,wp::float64> var_ea0,
    wp::vec_t<3,wp::float64> var_ea1,
    wp::vec_t<3,wp::float64> var_eb0,
    wp::vec_t<3,wp::float64> var_eb1)
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
    // def line_line_distance_gradient(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):       <L 519>
    // return g_EE(                                                                           <L 520>
    // ea0[0],                                                                                <L 521>
    var_1 = wp::extract(var_ea0, var_0);
    // ea0[1],                                                                                <L 522>
    var_3 = wp::extract(var_ea0, var_2);
    // ea0[2],                                                                                <L 523>
    var_5 = wp::extract(var_ea0, var_4);
    // ea1[0],                                                                                <L 524>
    var_7 = wp::extract(var_ea1, var_6);
    // ea1[1],                                                                                <L 525>
    var_9 = wp::extract(var_ea1, var_8);
    // ea1[2],                                                                                <L 526>
    var_11 = wp::extract(var_ea1, var_10);
    // eb0[0],                                                                                <L 527>
    var_13 = wp::extract(var_eb0, var_12);
    // eb0[1],                                                                                <L 528>
    var_15 = wp::extract(var_eb0, var_14);
    // eb0[2],                                                                                <L 529>
    var_17 = wp::extract(var_eb0, var_16);
    // eb1[0],                                                                                <L 530>
    var_19 = wp::extract(var_eb1, var_18);
    // eb1[1],                                                                                <L 531>
    var_21 = wp::extract(var_eb1, var_20);
    // eb1[2],                                                                                <L 532>
    var_23 = wp::extract(var_eb1, var_22);
    var_24 = g_EE_0(var_1, var_3, var_5, var_7, var_9, var_11, var_13, var_15, var_17, var_19, var_21, var_23);
    return var_24;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/line_line_distance.py:518
static CUDA_CALLABLE void adj_line_line_distance_gradient_0(
    wp::vec_t<3,wp::float64> var_ea0,
    wp::vec_t<3,wp::float64> var_ea1,
    wp::vec_t<3,wp::float64> var_eb0,
    wp::vec_t<3,wp::float64> var_eb1,
    wp::vec_t<3,wp::float64> & adj_ea0,
    wp::vec_t<3,wp::float64> & adj_ea1,
    wp::vec_t<3,wp::float64> & adj_eb0,
    wp::vec_t<3,wp::float64> & adj_eb1,
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
    // def line_line_distance_gradient(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):       <L 519>
    // return g_EE(                                                                           <L 520>
    // ea0[0],                                                                                <L 521>
    var_1 = wp::extract(var_ea0, var_0);
    // ea0[1],                                                                                <L 522>
    var_3 = wp::extract(var_ea0, var_2);
    // ea0[2],                                                                                <L 523>
    var_5 = wp::extract(var_ea0, var_4);
    // ea1[0],                                                                                <L 524>
    var_7 = wp::extract(var_ea1, var_6);
    // ea1[1],                                                                                <L 525>
    var_9 = wp::extract(var_ea1, var_8);
    // ea1[2],                                                                                <L 526>
    var_11 = wp::extract(var_ea1, var_10);
    // eb0[0],                                                                                <L 527>
    var_13 = wp::extract(var_eb0, var_12);
    // eb0[1],                                                                                <L 528>
    var_15 = wp::extract(var_eb0, var_14);
    // eb0[2],                                                                                <L 529>
    var_17 = wp::extract(var_eb0, var_16);
    // eb1[0],                                                                                <L 530>
    var_19 = wp::extract(var_eb1, var_18);
    // eb1[1],                                                                                <L 531>
    var_21 = wp::extract(var_eb1, var_20);
    // eb1[2],                                                                                <L 532>
    var_23 = wp::extract(var_eb1, var_22);
    var_24 = g_EE_0(var_1, var_3, var_5, var_7, var_9, var_11, var_13, var_15, var_17, var_19, var_21, var_23);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_24 += adj_ret;
    adj_g_EE_0(var_1, var_3, var_5, var_7, var_9, var_11, var_13, var_15, var_17, var_19, var_21, var_23, adj_1, adj_3, adj_5, adj_7, adj_9, adj_11, adj_13, adj_15, adj_17, adj_19, adj_21, adj_23, adj_24);
    wp::adj_extract(var_eb1, var_22, adj_eb1, adj_22, adj_23);
    // adj: eb1[2],                                                                           <L 532>
    wp::adj_extract(var_eb1, var_20, adj_eb1, adj_20, adj_21);
    // adj: eb1[1],                                                                           <L 531>
    wp::adj_extract(var_eb1, var_18, adj_eb1, adj_18, adj_19);
    // adj: eb1[0],                                                                           <L 530>
    wp::adj_extract(var_eb0, var_16, adj_eb0, adj_16, adj_17);
    // adj: eb0[2],                                                                           <L 529>
    wp::adj_extract(var_eb0, var_14, adj_eb0, adj_14, adj_15);
    // adj: eb0[1],                                                                           <L 528>
    wp::adj_extract(var_eb0, var_12, adj_eb0, adj_12, adj_13);
    // adj: eb0[0],                                                                           <L 527>
    wp::adj_extract(var_ea1, var_10, adj_ea1, adj_10, adj_11);
    // adj: ea1[2],                                                                           <L 526>
    wp::adj_extract(var_ea1, var_8, adj_ea1, adj_8, adj_9);
    // adj: ea1[1],                                                                           <L 525>
    wp::adj_extract(var_ea1, var_6, adj_ea1, adj_6, adj_7);
    // adj: ea1[0],                                                                           <L 524>
    wp::adj_extract(var_ea0, var_4, adj_ea0, adj_4, adj_5);
    // adj: ea0[2],                                                                           <L 523>
    wp::adj_extract(var_ea0, var_2, adj_ea0, adj_2, adj_3);
    // adj: ea0[1],                                                                           <L 522>
    wp::adj_extract(var_ea0, var_0, adj_ea0, adj_0, adj_1);
    // adj: ea0[0],                                                                           <L 521>
    // adj: return g_EE(                                                                      <L 520>
    // adj: def line_line_distance_gradient(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):  <L 519>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/edge_edge_distance.py:34
static CUDA_CALLABLE wp::vec_t<12,wp::float64> edge_edge_distance_gradient_0(
    wp::vec_t<3,wp::float64> var_ea0,
    wp::vec_t<3,wp::float64> var_ea1,
    wp::vec_t<3,wp::float64> var_eb0,
    wp::vec_t<3,wp::float64> var_eb1)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 12;
    wp::vec_t<12,wp::float64> var_1;
    wp::int32 var_2;
    const wp::float32 var_3 = 0.0;
    wp::float64 var_4;
    const wp::int32 var_5 = 0;
    bool var_6;
    wp::vec_t<6,wp::float64> var_7;
    const wp::int32 var_8 = 0;
    wp::float64 var_9;
    const wp::int32 var_10 = 1;
    wp::float64 var_11;
    const wp::int32 var_12 = 2;
    wp::float64 var_13;
    const wp::int32 var_14 = 3;
    wp::float64 var_15;
    const wp::int32 var_16 = 4;
    wp::float64 var_17;
    const wp::int32 var_18 = 5;
    wp::float64 var_19;
    wp::vec_t<12,wp::float64> var_20;
    wp::vec_t<12,wp::float64> var_21;
    const wp::int32 var_22 = 1;
    bool var_23;
    wp::vec_t<6,wp::float64> var_24;
    const wp::int32 var_25 = 0;
    wp::float64 var_26;
    const wp::int32 var_27 = 1;
    wp::float64 var_28;
    const wp::int32 var_29 = 2;
    wp::float64 var_30;
    const wp::int32 var_31 = 3;
    wp::float64 var_32;
    const wp::int32 var_33 = 4;
    wp::float64 var_34;
    const wp::int32 var_35 = 5;
    wp::float64 var_36;
    wp::vec_t<12,wp::float64> var_37;
    wp::vec_t<12,wp::float64> var_38;
    const wp::int32 var_39 = 2;
    bool var_40;
    wp::vec_t<9,wp::float64> var_41;
    const wp::int32 var_42 = 0;
    wp::float64 var_43;
    const wp::int32 var_44 = 1;
    wp::float64 var_45;
    const wp::int32 var_46 = 2;
    wp::float64 var_47;
    const wp::int32 var_48 = 3;
    wp::float64 var_49;
    const wp::int32 var_50 = 4;
    wp::float64 var_51;
    const wp::int32 var_52 = 5;
    wp::float64 var_53;
    const wp::int32 var_54 = 6;
    wp::float64 var_55;
    const wp::int32 var_56 = 7;
    wp::float64 var_57;
    const wp::int32 var_58 = 8;
    wp::float64 var_59;
    wp::vec_t<12,wp::float64> var_60;
    wp::vec_t<12,wp::float64> var_61;
    const wp::int32 var_62 = 3;
    bool var_63;
    wp::vec_t<6,wp::float64> var_64;
    const wp::int32 var_65 = 0;
    wp::float64 var_66;
    const wp::int32 var_67 = 1;
    wp::float64 var_68;
    const wp::int32 var_69 = 2;
    wp::float64 var_70;
    const wp::int32 var_71 = 3;
    wp::float64 var_72;
    const wp::int32 var_73 = 4;
    wp::float64 var_74;
    const wp::int32 var_75 = 5;
    wp::float64 var_76;
    wp::vec_t<12,wp::float64> var_77;
    wp::vec_t<12,wp::float64> var_78;
    const wp::int32 var_79 = 4;
    bool var_80;
    wp::vec_t<6,wp::float64> var_81;
    const wp::int32 var_82 = 0;
    wp::float64 var_83;
    const wp::int32 var_84 = 1;
    wp::float64 var_85;
    const wp::int32 var_86 = 2;
    wp::float64 var_87;
    const wp::int32 var_88 = 3;
    wp::float64 var_89;
    const wp::int32 var_90 = 4;
    wp::float64 var_91;
    const wp::int32 var_92 = 5;
    wp::float64 var_93;
    wp::vec_t<12,wp::float64> var_94;
    wp::vec_t<12,wp::float64> var_95;
    const wp::int32 var_96 = 5;
    bool var_97;
    wp::vec_t<9,wp::float64> var_98;
    const wp::int32 var_99 = 0;
    wp::float64 var_100;
    const wp::int32 var_101 = 1;
    wp::float64 var_102;
    const wp::int32 var_103 = 2;
    wp::float64 var_104;
    const wp::int32 var_105 = 3;
    wp::float64 var_106;
    const wp::int32 var_107 = 4;
    wp::float64 var_108;
    const wp::int32 var_109 = 5;
    wp::float64 var_110;
    const wp::int32 var_111 = 6;
    wp::float64 var_112;
    const wp::int32 var_113 = 7;
    wp::float64 var_114;
    const wp::int32 var_115 = 8;
    wp::float64 var_116;
    wp::vec_t<12,wp::float64> var_117;
    wp::vec_t<12,wp::float64> var_118;
    const wp::int32 var_119 = 6;
    bool var_120;
    wp::vec_t<9,wp::float64> var_121;
    const wp::int32 var_122 = 3;
    wp::float64 var_123;
    const wp::int32 var_124 = 4;
    wp::float64 var_125;
    const wp::int32 var_126 = 5;
    wp::float64 var_127;
    const wp::int32 var_128 = 6;
    wp::float64 var_129;
    const wp::int32 var_130 = 7;
    wp::float64 var_131;
    const wp::int32 var_132 = 8;
    wp::float64 var_133;
    const wp::int32 var_134 = 0;
    wp::float64 var_135;
    const wp::int32 var_136 = 1;
    wp::float64 var_137;
    const wp::int32 var_138 = 2;
    wp::float64 var_139;
    wp::vec_t<12,wp::float64> var_140;
    wp::vec_t<12,wp::float64> var_141;
    const wp::int32 var_142 = 7;
    bool var_143;
    wp::vec_t<9,wp::float64> var_144;
    const wp::int32 var_145 = 3;
    wp::float64 var_146;
    const wp::int32 var_147 = 4;
    wp::float64 var_148;
    const wp::int32 var_149 = 5;
    wp::float64 var_150;
    const wp::int32 var_151 = 6;
    wp::float64 var_152;
    const wp::int32 var_153 = 7;
    wp::float64 var_154;
    const wp::int32 var_155 = 8;
    wp::float64 var_156;
    const wp::int32 var_157 = 0;
    wp::float64 var_158;
    const wp::int32 var_159 = 1;
    wp::float64 var_160;
    const wp::int32 var_161 = 2;
    wp::float64 var_162;
    wp::vec_t<12,wp::float64> var_163;
    wp::vec_t<12,wp::float64> var_164;
    const wp::int32 var_165 = 8;
    bool var_166;
    wp::vec_t<12,wp::float64> var_167;
    wp::vec_t<12,wp::float64> var_168;
    wp::vec_t<12,wp::float64> var_169;
    wp::vec_t<12,wp::float64> var_170;
    wp::vec_t<12,wp::float64> var_171;
    wp::vec_t<12,wp::float64> var_172;
    wp::vec_t<12,wp::float64> var_173;
    wp::vec_t<12,wp::float64> var_174;
    wp::vec_t<12,wp::float64> var_175;
    wp::vec_t<12,wp::float64> var_176;
    //---------
    // forward
    // def edge_edge_distance_gradient(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):       <L 35>
    // grad = wp.vec(length=12, dtype=wp.float64)                                             <L 36>
    var_1 = wp::vec_t<12,wp::float64>();
    // dt = edge_edge_distance_type(ea0, ea1, eb0, eb1)                                       <L 37>
    var_2 = edge_edge_distance_type_0(var_ea0, var_ea1, var_eb0, var_eb1);
    // _0 = wp.float64(0.0)                                                                   <L 38>
    var_4 = wp::float64(var_3);
    // if dt == 0:                                                                            <L 39>
    var_6 = (var_2 == var_5);
    if (var_6) {
        // grad_compact0 = point_point_distance_gradient(ea0, eb0)                            <L 40>
        var_7 = point_point_distance_gradient_0(var_ea0, var_eb0);
        // grad = wp.vector(                                                                  <L 41>
        // grad_compact0[0], grad_compact0[1], grad_compact0[2], _0, _0, _0, grad_compact0[3], grad_compact0[4], grad_compact0[5], _0, _0, _0       <L 42>
        var_9 = wp::extract(var_7, var_8);
        var_11 = wp::extract(var_7, var_10);
        var_13 = wp::extract(var_7, var_12);
        var_15 = wp::extract(var_7, var_14);
        var_17 = wp::extract(var_7, var_16);
        var_19 = wp::extract(var_7, var_18);
        var_20 = wp::vec_t<12,wp::float64>({var_9, var_11, var_13, var_4, var_4, var_4, var_15, var_17, var_19, var_4, var_4, var_4});
    }
    var_21 = wp::where(var_6, var_20, var_1);
    if (!var_6) {
        // elif dt == 1:                                                                      <L 44>
        var_23 = (var_2 == var_22);
        if (var_23) {
            // grad_compact1 = point_point_distance_gradient(ea0, eb1)                        <L 45>
            var_24 = point_point_distance_gradient_0(var_ea0, var_eb1);
            // grad = wp.vector(                                                              <L 46>
            // grad_compact1[0], grad_compact1[1], grad_compact1[2], _0, _0, _0, _0, _0, _0, grad_compact1[3], grad_compact1[4], grad_compact1[5]       <L 47>
            var_26 = wp::extract(var_24, var_25);
            var_28 = wp::extract(var_24, var_27);
            var_30 = wp::extract(var_24, var_29);
            var_32 = wp::extract(var_24, var_31);
            var_34 = wp::extract(var_24, var_33);
            var_36 = wp::extract(var_24, var_35);
            var_37 = wp::vec_t<12,wp::float64>({var_26, var_28, var_30, var_4, var_4, var_4, var_4, var_4, var_4, var_32, var_34, var_36});
        }
        var_38 = wp::where(var_23, var_37, var_21);
        if (!var_23) {
            // elif dt == 2:                                                                  <L 49>
            var_40 = (var_2 == var_39);
            if (var_40) {
                // grad_compact2 = point_line_distance_gradient(ea0, eb0, eb1)                <L 50>
                var_41 = point_line_distance_gradient_0(var_ea0, var_eb0, var_eb1);
                // grad = wp.vector(                                                          <L 51>
                // grad_compact2[0],                                                          <L 52>
                var_43 = wp::extract(var_41, var_42);
                // grad_compact2[1],                                                          <L 53>
                var_45 = wp::extract(var_41, var_44);
                // grad_compact2[2],                                                          <L 54>
                var_47 = wp::extract(var_41, var_46);
                // _0,                                                                        <L 55>
                // _0,                                                                        <L 56>
                // _0,                                                                        <L 57>
                // grad_compact2[3],                                                          <L 58>
                var_49 = wp::extract(var_41, var_48);
                // grad_compact2[4],                                                          <L 59>
                var_51 = wp::extract(var_41, var_50);
                // grad_compact2[5],                                                          <L 60>
                var_53 = wp::extract(var_41, var_52);
                // grad_compact2[6],                                                          <L 61>
                var_55 = wp::extract(var_41, var_54);
                // grad_compact2[7],                                                          <L 62>
                var_57 = wp::extract(var_41, var_56);
                // grad_compact2[8],                                                          <L 63>
                var_59 = wp::extract(var_41, var_58);
                var_60 = wp::vec_t<12,wp::float64>({var_43, var_45, var_47, var_4, var_4, var_4, var_49, var_51, var_53, var_55, var_57, var_59});
            }
            var_61 = wp::where(var_40, var_60, var_38);
            if (!var_40) {
                // elif dt == 3:                                                              <L 65>
                var_63 = (var_2 == var_62);
                if (var_63) {
                    // grad_compact3 = point_point_distance_gradient(ea1, eb0)                <L 66>
                    var_64 = point_point_distance_gradient_0(var_ea1, var_eb0);
                    // grad = wp.vector(                                                      <L 67>
                    // _0, _0, _0, grad_compact3[0], grad_compact3[1], grad_compact3[2], grad_compact3[3], grad_compact3[4], grad_compact3[5], _0, _0, _0       <L 68>
                    var_66 = wp::extract(var_64, var_65);
                    var_68 = wp::extract(var_64, var_67);
                    var_70 = wp::extract(var_64, var_69);
                    var_72 = wp::extract(var_64, var_71);
                    var_74 = wp::extract(var_64, var_73);
                    var_76 = wp::extract(var_64, var_75);
                    var_77 = wp::vec_t<12,wp::float64>({var_4, var_4, var_4, var_66, var_68, var_70, var_72, var_74, var_76, var_4, var_4, var_4});
                }
                var_78 = wp::where(var_63, var_77, var_61);
                if (!var_63) {
                    // elif dt == 4:                                                          <L 70>
                    var_80 = (var_2 == var_79);
                    if (var_80) {
                        // grad_compact4 = point_point_distance_gradient(ea1, eb1)            <L 71>
                        var_81 = point_point_distance_gradient_0(var_ea1, var_eb1);
                        // grad = wp.vector(                                                  <L 72>
                        // _0, _0, _0, grad_compact4[0], grad_compact4[1], grad_compact4[2], _0, _0, _0, grad_compact4[3], grad_compact4[4], grad_compact4[5]       <L 73>
                        var_83 = wp::extract(var_81, var_82);
                        var_85 = wp::extract(var_81, var_84);
                        var_87 = wp::extract(var_81, var_86);
                        var_89 = wp::extract(var_81, var_88);
                        var_91 = wp::extract(var_81, var_90);
                        var_93 = wp::extract(var_81, var_92);
                        var_94 = wp::vec_t<12,wp::float64>({var_4, var_4, var_4, var_83, var_85, var_87, var_4, var_4, var_4, var_89, var_91, var_93});
                    }
                    var_95 = wp::where(var_80, var_94, var_78);
                    if (!var_80) {
                        // elif dt == 5:                                                      <L 75>
                        var_97 = (var_2 == var_96);
                        if (var_97) {
                            // grad_compact5 = point_line_distance_gradient(ea1, eb0, eb1)       <L 76>
                            var_98 = point_line_distance_gradient_0(var_ea1, var_eb0, var_eb1);
                            // grad = wp.vector(                                              <L 77>
                            // _0,                                                            <L 78>
                            // _0,                                                            <L 79>
                            // _0,                                                            <L 80>
                            // grad_compact5[0],                                              <L 81>
                            var_100 = wp::extract(var_98, var_99);
                            // grad_compact5[1],                                              <L 82>
                            var_102 = wp::extract(var_98, var_101);
                            // grad_compact5[2],                                              <L 83>
                            var_104 = wp::extract(var_98, var_103);
                            // grad_compact5[3],                                              <L 84>
                            var_106 = wp::extract(var_98, var_105);
                            // grad_compact5[4],                                              <L 85>
                            var_108 = wp::extract(var_98, var_107);
                            // grad_compact5[5],                                              <L 86>
                            var_110 = wp::extract(var_98, var_109);
                            // grad_compact5[6],                                              <L 87>
                            var_112 = wp::extract(var_98, var_111);
                            // grad_compact5[7],                                              <L 88>
                            var_114 = wp::extract(var_98, var_113);
                            // grad_compact5[8],                                              <L 89>
                            var_116 = wp::extract(var_98, var_115);
                            var_117 = wp::vec_t<12,wp::float64>({var_4, var_4, var_4, var_100, var_102, var_104, var_106, var_108, var_110, var_112, var_114, var_116});
                        }
                        var_118 = wp::where(var_97, var_117, var_95);
                        if (!var_97) {
                            // elif dt == 6:                                                  <L 91>
                            var_120 = (var_2 == var_119);
                            if (var_120) {
                                // grad_compact6 = point_line_distance_gradient(eb0, ea0, ea1)       <L 92>
                                var_121 = point_line_distance_gradient_0(var_eb0, var_ea0, var_ea1);
                                // grad = wp.vector(                                          <L 93>
                                // grad_compact6[3],                                          <L 94>
                                var_123 = wp::extract(var_121, var_122);
                                // grad_compact6[4],                                          <L 95>
                                var_125 = wp::extract(var_121, var_124);
                                // grad_compact6[5],                                          <L 96>
                                var_127 = wp::extract(var_121, var_126);
                                // grad_compact6[6],                                          <L 97>
                                var_129 = wp::extract(var_121, var_128);
                                // grad_compact6[7],                                          <L 98>
                                var_131 = wp::extract(var_121, var_130);
                                // grad_compact6[8],                                          <L 99>
                                var_133 = wp::extract(var_121, var_132);
                                // grad_compact6[0],                                          <L 100>
                                var_135 = wp::extract(var_121, var_134);
                                // grad_compact6[1],                                          <L 101>
                                var_137 = wp::extract(var_121, var_136);
                                // grad_compact6[2],                                          <L 102>
                                var_139 = wp::extract(var_121, var_138);
                                // _0,                                                        <L 103>
                                // _0,                                                        <L 104>
                                // _0,                                                        <L 105>
                                var_140 = wp::vec_t<12,wp::float64>({var_123, var_125, var_127, var_129, var_131, var_133, var_135, var_137, var_139, var_4, var_4, var_4});
                            }
                            var_141 = wp::where(var_120, var_140, var_118);
                            if (!var_120) {
                                // elif dt == 7:                                              <L 107>
                                var_143 = (var_2 == var_142);
                                if (var_143) {
                                    // grad_compact7 = point_line_distance_gradient(eb1, ea0, ea1)       <L 108>
                                    var_144 = point_line_distance_gradient_0(var_eb1, var_ea0, var_ea1);
                                    // grad = wp.vector(                                      <L 109>
                                    // grad_compact7[3],                                      <L 110>
                                    var_146 = wp::extract(var_144, var_145);
                                    // grad_compact7[4],                                      <L 111>
                                    var_148 = wp::extract(var_144, var_147);
                                    // grad_compact7[5],                                      <L 112>
                                    var_150 = wp::extract(var_144, var_149);
                                    // grad_compact7[6],                                      <L 113>
                                    var_152 = wp::extract(var_144, var_151);
                                    // grad_compact7[7],                                      <L 114>
                                    var_154 = wp::extract(var_144, var_153);
                                    // grad_compact7[8],                                      <L 115>
                                    var_156 = wp::extract(var_144, var_155);
                                    // _0,                                                    <L 116>
                                    // _0,                                                    <L 117>
                                    // _0,                                                    <L 118>
                                    // grad_compact7[0],                                      <L 119>
                                    var_158 = wp::extract(var_144, var_157);
                                    // grad_compact7[1],                                      <L 120>
                                    var_160 = wp::extract(var_144, var_159);
                                    // grad_compact7[2],                                      <L 121>
                                    var_162 = wp::extract(var_144, var_161);
                                    var_163 = wp::vec_t<12,wp::float64>({var_146, var_148, var_150, var_152, var_154, var_156, var_4, var_4, var_4, var_158, var_160, var_162});
                                }
                                var_164 = wp::where(var_143, var_163, var_141);
                                if (!var_143) {
                                    // elif dt == 8:                                          <L 123>
                                    var_166 = (var_2 == var_165);
                                    if (var_166) {
                                        // grad = line_line_distance_gradient(ea0, ea1, eb0, eb1)       <L 124>
                                        var_167 = line_line_distance_gradient_0(var_ea0, var_ea1, var_eb0, var_eb1);
                                    }
                                    var_168 = wp::where(var_166, var_167, var_164);
                                }
                                var_169 = wp::where(var_143, var_164, var_168);
                            }
                            var_170 = wp::where(var_120, var_141, var_169);
                        }
                        var_171 = wp::where(var_97, var_118, var_170);
                    }
                    var_172 = wp::where(var_80, var_95, var_171);
                }
                var_173 = wp::where(var_63, var_78, var_172);
            }
            var_174 = wp::where(var_40, var_61, var_173);
        }
        var_175 = wp::where(var_23, var_38, var_174);
    }
    var_176 = wp::where(var_6, var_21, var_175);
    // return grad                                                                            <L 126>
    return var_176;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/edge_edge_distance.py:34
static CUDA_CALLABLE void adj_edge_edge_distance_gradient_0(
    wp::vec_t<3,wp::float64> var_ea0,
    wp::vec_t<3,wp::float64> var_ea1,
    wp::vec_t<3,wp::float64> var_eb0,
    wp::vec_t<3,wp::float64> var_eb1,
    wp::vec_t<3,wp::float64> & adj_ea0,
    wp::vec_t<3,wp::float64> & adj_ea1,
    wp::vec_t<3,wp::float64> & adj_eb0,
    wp::vec_t<3,wp::float64> & adj_eb1,
    wp::vec_t<12,wp::float64> & adj_ret)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 12;
    wp::vec_t<12,wp::float64> var_1;
    wp::int32 var_2;
    const wp::float32 var_3 = 0.0;
    wp::float64 var_4;
    const wp::int32 var_5 = 0;
    bool var_6;
    wp::vec_t<6,wp::float64> var_7;
    const wp::int32 var_8 = 0;
    wp::float64 var_9;
    const wp::int32 var_10 = 1;
    wp::float64 var_11;
    const wp::int32 var_12 = 2;
    wp::float64 var_13;
    const wp::int32 var_14 = 3;
    wp::float64 var_15;
    const wp::int32 var_16 = 4;
    wp::float64 var_17;
    const wp::int32 var_18 = 5;
    wp::float64 var_19;
    wp::vec_t<12,wp::float64> var_20;
    wp::vec_t<12,wp::float64> var_21;
    const wp::int32 var_22 = 1;
    bool var_23;
    wp::vec_t<6,wp::float64> var_24;
    const wp::int32 var_25 = 0;
    wp::float64 var_26;
    const wp::int32 var_27 = 1;
    wp::float64 var_28;
    const wp::int32 var_29 = 2;
    wp::float64 var_30;
    const wp::int32 var_31 = 3;
    wp::float64 var_32;
    const wp::int32 var_33 = 4;
    wp::float64 var_34;
    const wp::int32 var_35 = 5;
    wp::float64 var_36;
    wp::vec_t<12,wp::float64> var_37;
    wp::vec_t<12,wp::float64> var_38;
    const wp::int32 var_39 = 2;
    bool var_40;
    wp::vec_t<9,wp::float64> var_41;
    const wp::int32 var_42 = 0;
    wp::float64 var_43;
    const wp::int32 var_44 = 1;
    wp::float64 var_45;
    const wp::int32 var_46 = 2;
    wp::float64 var_47;
    const wp::int32 var_48 = 3;
    wp::float64 var_49;
    const wp::int32 var_50 = 4;
    wp::float64 var_51;
    const wp::int32 var_52 = 5;
    wp::float64 var_53;
    const wp::int32 var_54 = 6;
    wp::float64 var_55;
    const wp::int32 var_56 = 7;
    wp::float64 var_57;
    const wp::int32 var_58 = 8;
    wp::float64 var_59;
    wp::vec_t<12,wp::float64> var_60;
    wp::vec_t<12,wp::float64> var_61;
    const wp::int32 var_62 = 3;
    bool var_63;
    wp::vec_t<6,wp::float64> var_64;
    const wp::int32 var_65 = 0;
    wp::float64 var_66;
    const wp::int32 var_67 = 1;
    wp::float64 var_68;
    const wp::int32 var_69 = 2;
    wp::float64 var_70;
    const wp::int32 var_71 = 3;
    wp::float64 var_72;
    const wp::int32 var_73 = 4;
    wp::float64 var_74;
    const wp::int32 var_75 = 5;
    wp::float64 var_76;
    wp::vec_t<12,wp::float64> var_77;
    wp::vec_t<12,wp::float64> var_78;
    const wp::int32 var_79 = 4;
    bool var_80;
    wp::vec_t<6,wp::float64> var_81;
    const wp::int32 var_82 = 0;
    wp::float64 var_83;
    const wp::int32 var_84 = 1;
    wp::float64 var_85;
    const wp::int32 var_86 = 2;
    wp::float64 var_87;
    const wp::int32 var_88 = 3;
    wp::float64 var_89;
    const wp::int32 var_90 = 4;
    wp::float64 var_91;
    const wp::int32 var_92 = 5;
    wp::float64 var_93;
    wp::vec_t<12,wp::float64> var_94;
    wp::vec_t<12,wp::float64> var_95;
    const wp::int32 var_96 = 5;
    bool var_97;
    wp::vec_t<9,wp::float64> var_98;
    const wp::int32 var_99 = 0;
    wp::float64 var_100;
    const wp::int32 var_101 = 1;
    wp::float64 var_102;
    const wp::int32 var_103 = 2;
    wp::float64 var_104;
    const wp::int32 var_105 = 3;
    wp::float64 var_106;
    const wp::int32 var_107 = 4;
    wp::float64 var_108;
    const wp::int32 var_109 = 5;
    wp::float64 var_110;
    const wp::int32 var_111 = 6;
    wp::float64 var_112;
    const wp::int32 var_113 = 7;
    wp::float64 var_114;
    const wp::int32 var_115 = 8;
    wp::float64 var_116;
    wp::vec_t<12,wp::float64> var_117;
    wp::vec_t<12,wp::float64> var_118;
    const wp::int32 var_119 = 6;
    bool var_120;
    wp::vec_t<9,wp::float64> var_121;
    const wp::int32 var_122 = 3;
    wp::float64 var_123;
    const wp::int32 var_124 = 4;
    wp::float64 var_125;
    const wp::int32 var_126 = 5;
    wp::float64 var_127;
    const wp::int32 var_128 = 6;
    wp::float64 var_129;
    const wp::int32 var_130 = 7;
    wp::float64 var_131;
    const wp::int32 var_132 = 8;
    wp::float64 var_133;
    const wp::int32 var_134 = 0;
    wp::float64 var_135;
    const wp::int32 var_136 = 1;
    wp::float64 var_137;
    const wp::int32 var_138 = 2;
    wp::float64 var_139;
    wp::vec_t<12,wp::float64> var_140;
    wp::vec_t<12,wp::float64> var_141;
    const wp::int32 var_142 = 7;
    bool var_143;
    wp::vec_t<9,wp::float64> var_144;
    const wp::int32 var_145 = 3;
    wp::float64 var_146;
    const wp::int32 var_147 = 4;
    wp::float64 var_148;
    const wp::int32 var_149 = 5;
    wp::float64 var_150;
    const wp::int32 var_151 = 6;
    wp::float64 var_152;
    const wp::int32 var_153 = 7;
    wp::float64 var_154;
    const wp::int32 var_155 = 8;
    wp::float64 var_156;
    const wp::int32 var_157 = 0;
    wp::float64 var_158;
    const wp::int32 var_159 = 1;
    wp::float64 var_160;
    const wp::int32 var_161 = 2;
    wp::float64 var_162;
    wp::vec_t<12,wp::float64> var_163;
    wp::vec_t<12,wp::float64> var_164;
    const wp::int32 var_165 = 8;
    bool var_166;
    wp::vec_t<12,wp::float64> var_167;
    wp::vec_t<12,wp::float64> var_168;
    wp::vec_t<12,wp::float64> var_169;
    wp::vec_t<12,wp::float64> var_170;
    wp::vec_t<12,wp::float64> var_171;
    wp::vec_t<12,wp::float64> var_172;
    wp::vec_t<12,wp::float64> var_173;
    wp::vec_t<12,wp::float64> var_174;
    wp::vec_t<12,wp::float64> var_175;
    wp::vec_t<12,wp::float64> var_176;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::vec_t<12,wp::float64> adj_1 = {};
    wp::int32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float64 adj_4 = {};
    wp::int32 adj_5 = {};
    bool adj_6 = {};
    wp::vec_t<6,wp::float64> adj_7 = {};
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
    wp::vec_t<12,wp::float64> adj_20 = {};
    wp::vec_t<12,wp::float64> adj_21 = {};
    wp::int32 adj_22 = {};
    bool adj_23 = {};
    wp::vec_t<6,wp::float64> adj_24 = {};
    wp::int32 adj_25 = {};
    wp::float64 adj_26 = {};
    wp::int32 adj_27 = {};
    wp::float64 adj_28 = {};
    wp::int32 adj_29 = {};
    wp::float64 adj_30 = {};
    wp::int32 adj_31 = {};
    wp::float64 adj_32 = {};
    wp::int32 adj_33 = {};
    wp::float64 adj_34 = {};
    wp::int32 adj_35 = {};
    wp::float64 adj_36 = {};
    wp::vec_t<12,wp::float64> adj_37 = {};
    wp::vec_t<12,wp::float64> adj_38 = {};
    wp::int32 adj_39 = {};
    bool adj_40 = {};
    wp::vec_t<9,wp::float64> adj_41 = {};
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
    wp::vec_t<12,wp::float64> adj_60 = {};
    wp::vec_t<12,wp::float64> adj_61 = {};
    wp::int32 adj_62 = {};
    bool adj_63 = {};
    wp::vec_t<6,wp::float64> adj_64 = {};
    wp::int32 adj_65 = {};
    wp::float64 adj_66 = {};
    wp::int32 adj_67 = {};
    wp::float64 adj_68 = {};
    wp::int32 adj_69 = {};
    wp::float64 adj_70 = {};
    wp::int32 adj_71 = {};
    wp::float64 adj_72 = {};
    wp::int32 adj_73 = {};
    wp::float64 adj_74 = {};
    wp::int32 adj_75 = {};
    wp::float64 adj_76 = {};
    wp::vec_t<12,wp::float64> adj_77 = {};
    wp::vec_t<12,wp::float64> adj_78 = {};
    wp::int32 adj_79 = {};
    bool adj_80 = {};
    wp::vec_t<6,wp::float64> adj_81 = {};
    wp::int32 adj_82 = {};
    wp::float64 adj_83 = {};
    wp::int32 adj_84 = {};
    wp::float64 adj_85 = {};
    wp::int32 adj_86 = {};
    wp::float64 adj_87 = {};
    wp::int32 adj_88 = {};
    wp::float64 adj_89 = {};
    wp::int32 adj_90 = {};
    wp::float64 adj_91 = {};
    wp::int32 adj_92 = {};
    wp::float64 adj_93 = {};
    wp::vec_t<12,wp::float64> adj_94 = {};
    wp::vec_t<12,wp::float64> adj_95 = {};
    wp::int32 adj_96 = {};
    bool adj_97 = {};
    wp::vec_t<9,wp::float64> adj_98 = {};
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
    wp::int32 adj_109 = {};
    wp::float64 adj_110 = {};
    wp::int32 adj_111 = {};
    wp::float64 adj_112 = {};
    wp::int32 adj_113 = {};
    wp::float64 adj_114 = {};
    wp::int32 adj_115 = {};
    wp::float64 adj_116 = {};
    wp::vec_t<12,wp::float64> adj_117 = {};
    wp::vec_t<12,wp::float64> adj_118 = {};
    wp::int32 adj_119 = {};
    bool adj_120 = {};
    wp::vec_t<9,wp::float64> adj_121 = {};
    wp::int32 adj_122 = {};
    wp::float64 adj_123 = {};
    wp::int32 adj_124 = {};
    wp::float64 adj_125 = {};
    wp::int32 adj_126 = {};
    wp::float64 adj_127 = {};
    wp::int32 adj_128 = {};
    wp::float64 adj_129 = {};
    wp::int32 adj_130 = {};
    wp::float64 adj_131 = {};
    wp::int32 adj_132 = {};
    wp::float64 adj_133 = {};
    wp::int32 adj_134 = {};
    wp::float64 adj_135 = {};
    wp::int32 adj_136 = {};
    wp::float64 adj_137 = {};
    wp::int32 adj_138 = {};
    wp::float64 adj_139 = {};
    wp::vec_t<12,wp::float64> adj_140 = {};
    wp::vec_t<12,wp::float64> adj_141 = {};
    wp::int32 adj_142 = {};
    bool adj_143 = {};
    wp::vec_t<9,wp::float64> adj_144 = {};
    wp::int32 adj_145 = {};
    wp::float64 adj_146 = {};
    wp::int32 adj_147 = {};
    wp::float64 adj_148 = {};
    wp::int32 adj_149 = {};
    wp::float64 adj_150 = {};
    wp::int32 adj_151 = {};
    wp::float64 adj_152 = {};
    wp::int32 adj_153 = {};
    wp::float64 adj_154 = {};
    wp::int32 adj_155 = {};
    wp::float64 adj_156 = {};
    wp::int32 adj_157 = {};
    wp::float64 adj_158 = {};
    wp::int32 adj_159 = {};
    wp::float64 adj_160 = {};
    wp::int32 adj_161 = {};
    wp::float64 adj_162 = {};
    wp::vec_t<12,wp::float64> adj_163 = {};
    wp::vec_t<12,wp::float64> adj_164 = {};
    wp::int32 adj_165 = {};
    bool adj_166 = {};
    wp::vec_t<12,wp::float64> adj_167 = {};
    wp::vec_t<12,wp::float64> adj_168 = {};
    wp::vec_t<12,wp::float64> adj_169 = {};
    wp::vec_t<12,wp::float64> adj_170 = {};
    wp::vec_t<12,wp::float64> adj_171 = {};
    wp::vec_t<12,wp::float64> adj_172 = {};
    wp::vec_t<12,wp::float64> adj_173 = {};
    wp::vec_t<12,wp::float64> adj_174 = {};
    wp::vec_t<12,wp::float64> adj_175 = {};
    wp::vec_t<12,wp::float64> adj_176 = {};
    //---------
    // forward
    // def edge_edge_distance_gradient(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):       <L 35>
    // grad = wp.vec(length=12, dtype=wp.float64)                                             <L 36>
    var_1 = wp::vec_t<12,wp::float64>();
    // dt = edge_edge_distance_type(ea0, ea1, eb0, eb1)                                       <L 37>
    var_2 = edge_edge_distance_type_0(var_ea0, var_ea1, var_eb0, var_eb1);
    // _0 = wp.float64(0.0)                                                                   <L 38>
    var_4 = wp::float64(var_3);
    // if dt == 0:                                                                            <L 39>
    var_6 = (var_2 == var_5);
    if (var_6) {
        // grad_compact0 = point_point_distance_gradient(ea0, eb0)                            <L 40>
        var_7 = point_point_distance_gradient_0(var_ea0, var_eb0);
        // grad = wp.vector(                                                                  <L 41>
        // grad_compact0[0], grad_compact0[1], grad_compact0[2], _0, _0, _0, grad_compact0[3], grad_compact0[4], grad_compact0[5], _0, _0, _0       <L 42>
        var_9 = wp::extract(var_7, var_8);
        var_11 = wp::extract(var_7, var_10);
        var_13 = wp::extract(var_7, var_12);
        var_15 = wp::extract(var_7, var_14);
        var_17 = wp::extract(var_7, var_16);
        var_19 = wp::extract(var_7, var_18);
        var_20 = wp::vec_t<12,wp::float64>({var_9, var_11, var_13, var_4, var_4, var_4, var_15, var_17, var_19, var_4, var_4, var_4});
    }
    var_21 = wp::where(var_6, var_20, var_1);
    if (!var_6) {
        // elif dt == 1:                                                                      <L 44>
        var_23 = (var_2 == var_22);
        if (var_23) {
            // grad_compact1 = point_point_distance_gradient(ea0, eb1)                        <L 45>
            var_24 = point_point_distance_gradient_0(var_ea0, var_eb1);
            // grad = wp.vector(                                                              <L 46>
            // grad_compact1[0], grad_compact1[1], grad_compact1[2], _0, _0, _0, _0, _0, _0, grad_compact1[3], grad_compact1[4], grad_compact1[5]       <L 47>
            var_26 = wp::extract(var_24, var_25);
            var_28 = wp::extract(var_24, var_27);
            var_30 = wp::extract(var_24, var_29);
            var_32 = wp::extract(var_24, var_31);
            var_34 = wp::extract(var_24, var_33);
            var_36 = wp::extract(var_24, var_35);
            var_37 = wp::vec_t<12,wp::float64>({var_26, var_28, var_30, var_4, var_4, var_4, var_4, var_4, var_4, var_32, var_34, var_36});
        }
        var_38 = wp::where(var_23, var_37, var_21);
        if (!var_23) {
            // elif dt == 2:                                                                  <L 49>
            var_40 = (var_2 == var_39);
            if (var_40) {
                // grad_compact2 = point_line_distance_gradient(ea0, eb0, eb1)                <L 50>
                var_41 = point_line_distance_gradient_0(var_ea0, var_eb0, var_eb1);
                // grad = wp.vector(                                                          <L 51>
                // grad_compact2[0],                                                          <L 52>
                var_43 = wp::extract(var_41, var_42);
                // grad_compact2[1],                                                          <L 53>
                var_45 = wp::extract(var_41, var_44);
                // grad_compact2[2],                                                          <L 54>
                var_47 = wp::extract(var_41, var_46);
                // _0,                                                                        <L 55>
                // _0,                                                                        <L 56>
                // _0,                                                                        <L 57>
                // grad_compact2[3],                                                          <L 58>
                var_49 = wp::extract(var_41, var_48);
                // grad_compact2[4],                                                          <L 59>
                var_51 = wp::extract(var_41, var_50);
                // grad_compact2[5],                                                          <L 60>
                var_53 = wp::extract(var_41, var_52);
                // grad_compact2[6],                                                          <L 61>
                var_55 = wp::extract(var_41, var_54);
                // grad_compact2[7],                                                          <L 62>
                var_57 = wp::extract(var_41, var_56);
                // grad_compact2[8],                                                          <L 63>
                var_59 = wp::extract(var_41, var_58);
                var_60 = wp::vec_t<12,wp::float64>({var_43, var_45, var_47, var_4, var_4, var_4, var_49, var_51, var_53, var_55, var_57, var_59});
            }
            var_61 = wp::where(var_40, var_60, var_38);
            if (!var_40) {
                // elif dt == 3:                                                              <L 65>
                var_63 = (var_2 == var_62);
                if (var_63) {
                    // grad_compact3 = point_point_distance_gradient(ea1, eb0)                <L 66>
                    var_64 = point_point_distance_gradient_0(var_ea1, var_eb0);
                    // grad = wp.vector(                                                      <L 67>
                    // _0, _0, _0, grad_compact3[0], grad_compact3[1], grad_compact3[2], grad_compact3[3], grad_compact3[4], grad_compact3[5], _0, _0, _0       <L 68>
                    var_66 = wp::extract(var_64, var_65);
                    var_68 = wp::extract(var_64, var_67);
                    var_70 = wp::extract(var_64, var_69);
                    var_72 = wp::extract(var_64, var_71);
                    var_74 = wp::extract(var_64, var_73);
                    var_76 = wp::extract(var_64, var_75);
                    var_77 = wp::vec_t<12,wp::float64>({var_4, var_4, var_4, var_66, var_68, var_70, var_72, var_74, var_76, var_4, var_4, var_4});
                }
                var_78 = wp::where(var_63, var_77, var_61);
                if (!var_63) {
                    // elif dt == 4:                                                          <L 70>
                    var_80 = (var_2 == var_79);
                    if (var_80) {
                        // grad_compact4 = point_point_distance_gradient(ea1, eb1)            <L 71>
                        var_81 = point_point_distance_gradient_0(var_ea1, var_eb1);
                        // grad = wp.vector(                                                  <L 72>
                        // _0, _0, _0, grad_compact4[0], grad_compact4[1], grad_compact4[2], _0, _0, _0, grad_compact4[3], grad_compact4[4], grad_compact4[5]       <L 73>
                        var_83 = wp::extract(var_81, var_82);
                        var_85 = wp::extract(var_81, var_84);
                        var_87 = wp::extract(var_81, var_86);
                        var_89 = wp::extract(var_81, var_88);
                        var_91 = wp::extract(var_81, var_90);
                        var_93 = wp::extract(var_81, var_92);
                        var_94 = wp::vec_t<12,wp::float64>({var_4, var_4, var_4, var_83, var_85, var_87, var_4, var_4, var_4, var_89, var_91, var_93});
                    }
                    var_95 = wp::where(var_80, var_94, var_78);
                    if (!var_80) {
                        // elif dt == 5:                                                      <L 75>
                        var_97 = (var_2 == var_96);
                        if (var_97) {
                            // grad_compact5 = point_line_distance_gradient(ea1, eb0, eb1)       <L 76>
                            var_98 = point_line_distance_gradient_0(var_ea1, var_eb0, var_eb1);
                            // grad = wp.vector(                                              <L 77>
                            // _0,                                                            <L 78>
                            // _0,                                                            <L 79>
                            // _0,                                                            <L 80>
                            // grad_compact5[0],                                              <L 81>
                            var_100 = wp::extract(var_98, var_99);
                            // grad_compact5[1],                                              <L 82>
                            var_102 = wp::extract(var_98, var_101);
                            // grad_compact5[2],                                              <L 83>
                            var_104 = wp::extract(var_98, var_103);
                            // grad_compact5[3],                                              <L 84>
                            var_106 = wp::extract(var_98, var_105);
                            // grad_compact5[4],                                              <L 85>
                            var_108 = wp::extract(var_98, var_107);
                            // grad_compact5[5],                                              <L 86>
                            var_110 = wp::extract(var_98, var_109);
                            // grad_compact5[6],                                              <L 87>
                            var_112 = wp::extract(var_98, var_111);
                            // grad_compact5[7],                                              <L 88>
                            var_114 = wp::extract(var_98, var_113);
                            // grad_compact5[8],                                              <L 89>
                            var_116 = wp::extract(var_98, var_115);
                            var_117 = wp::vec_t<12,wp::float64>({var_4, var_4, var_4, var_100, var_102, var_104, var_106, var_108, var_110, var_112, var_114, var_116});
                        }
                        var_118 = wp::where(var_97, var_117, var_95);
                        if (!var_97) {
                            // elif dt == 6:                                                  <L 91>
                            var_120 = (var_2 == var_119);
                            if (var_120) {
                                // grad_compact6 = point_line_distance_gradient(eb0, ea0, ea1)       <L 92>
                                var_121 = point_line_distance_gradient_0(var_eb0, var_ea0, var_ea1);
                                // grad = wp.vector(                                          <L 93>
                                // grad_compact6[3],                                          <L 94>
                                var_123 = wp::extract(var_121, var_122);
                                // grad_compact6[4],                                          <L 95>
                                var_125 = wp::extract(var_121, var_124);
                                // grad_compact6[5],                                          <L 96>
                                var_127 = wp::extract(var_121, var_126);
                                // grad_compact6[6],                                          <L 97>
                                var_129 = wp::extract(var_121, var_128);
                                // grad_compact6[7],                                          <L 98>
                                var_131 = wp::extract(var_121, var_130);
                                // grad_compact6[8],                                          <L 99>
                                var_133 = wp::extract(var_121, var_132);
                                // grad_compact6[0],                                          <L 100>
                                var_135 = wp::extract(var_121, var_134);
                                // grad_compact6[1],                                          <L 101>
                                var_137 = wp::extract(var_121, var_136);
                                // grad_compact6[2],                                          <L 102>
                                var_139 = wp::extract(var_121, var_138);
                                // _0,                                                        <L 103>
                                // _0,                                                        <L 104>
                                // _0,                                                        <L 105>
                                var_140 = wp::vec_t<12,wp::float64>({var_123, var_125, var_127, var_129, var_131, var_133, var_135, var_137, var_139, var_4, var_4, var_4});
                            }
                            var_141 = wp::where(var_120, var_140, var_118);
                            if (!var_120) {
                                // elif dt == 7:                                              <L 107>
                                var_143 = (var_2 == var_142);
                                if (var_143) {
                                    // grad_compact7 = point_line_distance_gradient(eb1, ea0, ea1)       <L 108>
                                    var_144 = point_line_distance_gradient_0(var_eb1, var_ea0, var_ea1);
                                    // grad = wp.vector(                                      <L 109>
                                    // grad_compact7[3],                                      <L 110>
                                    var_146 = wp::extract(var_144, var_145);
                                    // grad_compact7[4],                                      <L 111>
                                    var_148 = wp::extract(var_144, var_147);
                                    // grad_compact7[5],                                      <L 112>
                                    var_150 = wp::extract(var_144, var_149);
                                    // grad_compact7[6],                                      <L 113>
                                    var_152 = wp::extract(var_144, var_151);
                                    // grad_compact7[7],                                      <L 114>
                                    var_154 = wp::extract(var_144, var_153);
                                    // grad_compact7[8],                                      <L 115>
                                    var_156 = wp::extract(var_144, var_155);
                                    // _0,                                                    <L 116>
                                    // _0,                                                    <L 117>
                                    // _0,                                                    <L 118>
                                    // grad_compact7[0],                                      <L 119>
                                    var_158 = wp::extract(var_144, var_157);
                                    // grad_compact7[1],                                      <L 120>
                                    var_160 = wp::extract(var_144, var_159);
                                    // grad_compact7[2],                                      <L 121>
                                    var_162 = wp::extract(var_144, var_161);
                                    var_163 = wp::vec_t<12,wp::float64>({var_146, var_148, var_150, var_152, var_154, var_156, var_4, var_4, var_4, var_158, var_160, var_162});
                                }
                                var_164 = wp::where(var_143, var_163, var_141);
                                if (!var_143) {
                                    // elif dt == 8:                                          <L 123>
                                    var_166 = (var_2 == var_165);
                                    if (var_166) {
                                        // grad = line_line_distance_gradient(ea0, ea1, eb0, eb1)       <L 124>
                                        var_167 = line_line_distance_gradient_0(var_ea0, var_ea1, var_eb0, var_eb1);
                                    }
                                    var_168 = wp::where(var_166, var_167, var_164);
                                }
                                var_169 = wp::where(var_143, var_164, var_168);
                            }
                            var_170 = wp::where(var_120, var_141, var_169);
                        }
                        var_171 = wp::where(var_97, var_118, var_170);
                    }
                    var_172 = wp::where(var_80, var_95, var_171);
                }
                var_173 = wp::where(var_63, var_78, var_172);
            }
            var_174 = wp::where(var_40, var_61, var_173);
        }
        var_175 = wp::where(var_23, var_38, var_174);
    }
    var_176 = wp::where(var_6, var_21, var_175);
    // return grad                                                                            <L 126>
    goto label0;
    //---------
    // reverse
    label0:;
    adj_176 += adj_ret;
    // adj: return grad                                                                       <L 126>
    wp::adj_where(var_6, var_21, var_175, adj_6, adj_21, adj_175, adj_176);
    if (!var_6) {
        wp::adj_where(var_23, var_38, var_174, adj_23, adj_38, adj_174, adj_175);
        if (!var_23) {
            wp::adj_where(var_40, var_61, var_173, adj_40, adj_61, adj_173, adj_174);
            if (!var_40) {
                wp::adj_where(var_63, var_78, var_172, adj_63, adj_78, adj_172, adj_173);
                if (!var_63) {
                    wp::adj_where(var_80, var_95, var_171, adj_80, adj_95, adj_171, adj_172);
                    if (!var_80) {
                        wp::adj_where(var_97, var_118, var_170, adj_97, adj_118, adj_170, adj_171);
                        if (!var_97) {
                            wp::adj_where(var_120, var_141, var_169, adj_120, adj_141, adj_169, adj_170);
                            if (!var_120) {
                                wp::adj_where(var_143, var_164, var_168, adj_143, adj_164, adj_168, adj_169);
                                if (!var_143) {
                                    wp::adj_where(var_166, var_167, var_164, adj_166, adj_167, adj_164, adj_168);
                                    if (var_166) {
                                        adj_line_line_distance_gradient_0(var_ea0, var_ea1, var_eb0, var_eb1, adj_ea0, adj_ea1, adj_eb0, adj_eb1, adj_167);
                                        // adj: grad = line_line_distance_gradient(ea0, ea1, eb0, eb1)  <L 124>
                                    }
                                    // adj: elif dt == 8:                                     <L 123>
                                }
                                wp::adj_where(var_143, var_163, var_141, adj_143, adj_163, adj_141, adj_164);
                                if (var_143) {
                                    wp::adj_vec_t({var_146, var_148, var_150, var_152, var_154, var_156, var_4, var_4, var_4, var_158, var_160, var_162}, {&adj_146, &adj_148, &adj_150, &adj_152, &adj_154, &adj_156, &adj_4, &adj_4, &adj_4, &adj_158, &adj_160, &adj_162}, adj_163);
                                    wp::adj_extract(var_144, var_161, adj_144, adj_161, adj_162);
                                    // adj: grad_compact7[2],                                 <L 121>
                                    wp::adj_extract(var_144, var_159, adj_144, adj_159, adj_160);
                                    // adj: grad_compact7[1],                                 <L 120>
                                    wp::adj_extract(var_144, var_157, adj_144, adj_157, adj_158);
                                    // adj: grad_compact7[0],                                 <L 119>
                                    // adj: _0,                                               <L 118>
                                    // adj: _0,                                               <L 117>
                                    // adj: _0,                                               <L 116>
                                    wp::adj_extract(var_144, var_155, adj_144, adj_155, adj_156);
                                    // adj: grad_compact7[8],                                 <L 115>
                                    wp::adj_extract(var_144, var_153, adj_144, adj_153, adj_154);
                                    // adj: grad_compact7[7],                                 <L 114>
                                    wp::adj_extract(var_144, var_151, adj_144, adj_151, adj_152);
                                    // adj: grad_compact7[6],                                 <L 113>
                                    wp::adj_extract(var_144, var_149, adj_144, adj_149, adj_150);
                                    // adj: grad_compact7[5],                                 <L 112>
                                    wp::adj_extract(var_144, var_147, adj_144, adj_147, adj_148);
                                    // adj: grad_compact7[4],                                 <L 111>
                                    wp::adj_extract(var_144, var_145, adj_144, adj_145, adj_146);
                                    // adj: grad_compact7[3],                                 <L 110>
                                    // adj: grad = wp.vector(                                 <L 109>
                                    adj_point_line_distance_gradient_0(var_eb1, var_ea0, var_ea1, adj_eb1, adj_ea0, adj_ea1, adj_144);
                                    // adj: grad_compact7 = point_line_distance_gradient(eb1, ea0, ea1)  <L 108>
                                }
                                // adj: elif dt == 7:                                         <L 107>
                            }
                            wp::adj_where(var_120, var_140, var_118, adj_120, adj_140, adj_118, adj_141);
                            if (var_120) {
                                wp::adj_vec_t({var_123, var_125, var_127, var_129, var_131, var_133, var_135, var_137, var_139, var_4, var_4, var_4}, {&adj_123, &adj_125, &adj_127, &adj_129, &adj_131, &adj_133, &adj_135, &adj_137, &adj_139, &adj_4, &adj_4, &adj_4}, adj_140);
                                // adj: _0,                                                   <L 105>
                                // adj: _0,                                                   <L 104>
                                // adj: _0,                                                   <L 103>
                                wp::adj_extract(var_121, var_138, adj_121, adj_138, adj_139);
                                // adj: grad_compact6[2],                                     <L 102>
                                wp::adj_extract(var_121, var_136, adj_121, adj_136, adj_137);
                                // adj: grad_compact6[1],                                     <L 101>
                                wp::adj_extract(var_121, var_134, adj_121, adj_134, adj_135);
                                // adj: grad_compact6[0],                                     <L 100>
                                wp::adj_extract(var_121, var_132, adj_121, adj_132, adj_133);
                                // adj: grad_compact6[8],                                     <L 99>
                                wp::adj_extract(var_121, var_130, adj_121, adj_130, adj_131);
                                // adj: grad_compact6[7],                                     <L 98>
                                wp::adj_extract(var_121, var_128, adj_121, adj_128, adj_129);
                                // adj: grad_compact6[6],                                     <L 97>
                                wp::adj_extract(var_121, var_126, adj_121, adj_126, adj_127);
                                // adj: grad_compact6[5],                                     <L 96>
                                wp::adj_extract(var_121, var_124, adj_121, adj_124, adj_125);
                                // adj: grad_compact6[4],                                     <L 95>
                                wp::adj_extract(var_121, var_122, adj_121, adj_122, adj_123);
                                // adj: grad_compact6[3],                                     <L 94>
                                // adj: grad = wp.vector(                                     <L 93>
                                adj_point_line_distance_gradient_0(var_eb0, var_ea0, var_ea1, adj_eb0, adj_ea0, adj_ea1, adj_121);
                                // adj: grad_compact6 = point_line_distance_gradient(eb0, ea0, ea1)  <L 92>
                            }
                            // adj: elif dt == 6:                                             <L 91>
                        }
                        wp::adj_where(var_97, var_117, var_95, adj_97, adj_117, adj_95, adj_118);
                        if (var_97) {
                            wp::adj_vec_t({var_4, var_4, var_4, var_100, var_102, var_104, var_106, var_108, var_110, var_112, var_114, var_116}, {&adj_4, &adj_4, &adj_4, &adj_100, &adj_102, &adj_104, &adj_106, &adj_108, &adj_110, &adj_112, &adj_114, &adj_116}, adj_117);
                            wp::adj_extract(var_98, var_115, adj_98, adj_115, adj_116);
                            // adj: grad_compact5[8],                                         <L 89>
                            wp::adj_extract(var_98, var_113, adj_98, adj_113, adj_114);
                            // adj: grad_compact5[7],                                         <L 88>
                            wp::adj_extract(var_98, var_111, adj_98, adj_111, adj_112);
                            // adj: grad_compact5[6],                                         <L 87>
                            wp::adj_extract(var_98, var_109, adj_98, adj_109, adj_110);
                            // adj: grad_compact5[5],                                         <L 86>
                            wp::adj_extract(var_98, var_107, adj_98, adj_107, adj_108);
                            // adj: grad_compact5[4],                                         <L 85>
                            wp::adj_extract(var_98, var_105, adj_98, adj_105, adj_106);
                            // adj: grad_compact5[3],                                         <L 84>
                            wp::adj_extract(var_98, var_103, adj_98, adj_103, adj_104);
                            // adj: grad_compact5[2],                                         <L 83>
                            wp::adj_extract(var_98, var_101, adj_98, adj_101, adj_102);
                            // adj: grad_compact5[1],                                         <L 82>
                            wp::adj_extract(var_98, var_99, adj_98, adj_99, adj_100);
                            // adj: grad_compact5[0],                                         <L 81>
                            // adj: _0,                                                       <L 80>
                            // adj: _0,                                                       <L 79>
                            // adj: _0,                                                       <L 78>
                            // adj: grad = wp.vector(                                         <L 77>
                            adj_point_line_distance_gradient_0(var_ea1, var_eb0, var_eb1, adj_ea1, adj_eb0, adj_eb1, adj_98);
                            // adj: grad_compact5 = point_line_distance_gradient(ea1, eb0, eb1)  <L 76>
                        }
                        // adj: elif dt == 5:                                                 <L 75>
                    }
                    wp::adj_where(var_80, var_94, var_78, adj_80, adj_94, adj_78, adj_95);
                    if (var_80) {
                        wp::adj_vec_t({var_4, var_4, var_4, var_83, var_85, var_87, var_4, var_4, var_4, var_89, var_91, var_93}, {&adj_4, &adj_4, &adj_4, &adj_83, &adj_85, &adj_87, &adj_4, &adj_4, &adj_4, &adj_89, &adj_91, &adj_93}, adj_94);
                        wp::adj_extract(var_81, var_92, adj_81, adj_92, adj_93);
                        wp::adj_extract(var_81, var_90, adj_81, adj_90, adj_91);
                        wp::adj_extract(var_81, var_88, adj_81, adj_88, adj_89);
                        wp::adj_extract(var_81, var_86, adj_81, adj_86, adj_87);
                        wp::adj_extract(var_81, var_84, adj_81, adj_84, adj_85);
                        wp::adj_extract(var_81, var_82, adj_81, adj_82, adj_83);
                        // adj: _0, _0, _0, grad_compact4[0], grad_compact4[1], grad_compact4[2], _0, _0, _0, grad_compact4[3], grad_compact4[4], grad_compact4[5]  <L 73>
                        // adj: grad = wp.vector(                                             <L 72>
                        adj_point_point_distance_gradient_0(var_ea1, var_eb1, adj_ea1, adj_eb1, adj_81);
                        // adj: grad_compact4 = point_point_distance_gradient(ea1, eb1)       <L 71>
                    }
                    // adj: elif dt == 4:                                                     <L 70>
                }
                wp::adj_where(var_63, var_77, var_61, adj_63, adj_77, adj_61, adj_78);
                if (var_63) {
                    wp::adj_vec_t({var_4, var_4, var_4, var_66, var_68, var_70, var_72, var_74, var_76, var_4, var_4, var_4}, {&adj_4, &adj_4, &adj_4, &adj_66, &adj_68, &adj_70, &adj_72, &adj_74, &adj_76, &adj_4, &adj_4, &adj_4}, adj_77);
                    wp::adj_extract(var_64, var_75, adj_64, adj_75, adj_76);
                    wp::adj_extract(var_64, var_73, adj_64, adj_73, adj_74);
                    wp::adj_extract(var_64, var_71, adj_64, adj_71, adj_72);
                    wp::adj_extract(var_64, var_69, adj_64, adj_69, adj_70);
                    wp::adj_extract(var_64, var_67, adj_64, adj_67, adj_68);
                    wp::adj_extract(var_64, var_65, adj_64, adj_65, adj_66);
                    // adj: _0, _0, _0, grad_compact3[0], grad_compact3[1], grad_compact3[2], grad_compact3[3], grad_compact3[4], grad_compact3[5], _0, _0, _0  <L 68>
                    // adj: grad = wp.vector(                                                 <L 67>
                    adj_point_point_distance_gradient_0(var_ea1, var_eb0, adj_ea1, adj_eb0, adj_64);
                    // adj: grad_compact3 = point_point_distance_gradient(ea1, eb0)           <L 66>
                }
                // adj: elif dt == 3:                                                         <L 65>
            }
            wp::adj_where(var_40, var_60, var_38, adj_40, adj_60, adj_38, adj_61);
            if (var_40) {
                wp::adj_vec_t({var_43, var_45, var_47, var_4, var_4, var_4, var_49, var_51, var_53, var_55, var_57, var_59}, {&adj_43, &adj_45, &adj_47, &adj_4, &adj_4, &adj_4, &adj_49, &adj_51, &adj_53, &adj_55, &adj_57, &adj_59}, adj_60);
                wp::adj_extract(var_41, var_58, adj_41, adj_58, adj_59);
                // adj: grad_compact2[8],                                                     <L 63>
                wp::adj_extract(var_41, var_56, adj_41, adj_56, adj_57);
                // adj: grad_compact2[7],                                                     <L 62>
                wp::adj_extract(var_41, var_54, adj_41, adj_54, adj_55);
                // adj: grad_compact2[6],                                                     <L 61>
                wp::adj_extract(var_41, var_52, adj_41, adj_52, adj_53);
                // adj: grad_compact2[5],                                                     <L 60>
                wp::adj_extract(var_41, var_50, adj_41, adj_50, adj_51);
                // adj: grad_compact2[4],                                                     <L 59>
                wp::adj_extract(var_41, var_48, adj_41, adj_48, adj_49);
                // adj: grad_compact2[3],                                                     <L 58>
                // adj: _0,                                                                   <L 57>
                // adj: _0,                                                                   <L 56>
                // adj: _0,                                                                   <L 55>
                wp::adj_extract(var_41, var_46, adj_41, adj_46, adj_47);
                // adj: grad_compact2[2],                                                     <L 54>
                wp::adj_extract(var_41, var_44, adj_41, adj_44, adj_45);
                // adj: grad_compact2[1],                                                     <L 53>
                wp::adj_extract(var_41, var_42, adj_41, adj_42, adj_43);
                // adj: grad_compact2[0],                                                     <L 52>
                // adj: grad = wp.vector(                                                     <L 51>
                adj_point_line_distance_gradient_0(var_ea0, var_eb0, var_eb1, adj_ea0, adj_eb0, adj_eb1, adj_41);
                // adj: grad_compact2 = point_line_distance_gradient(ea0, eb0, eb1)           <L 50>
            }
            // adj: elif dt == 2:                                                             <L 49>
        }
        wp::adj_where(var_23, var_37, var_21, adj_23, adj_37, adj_21, adj_38);
        if (var_23) {
            wp::adj_vec_t({var_26, var_28, var_30, var_4, var_4, var_4, var_4, var_4, var_4, var_32, var_34, var_36}, {&adj_26, &adj_28, &adj_30, &adj_4, &adj_4, &adj_4, &adj_4, &adj_4, &adj_4, &adj_32, &adj_34, &adj_36}, adj_37);
            wp::adj_extract(var_24, var_35, adj_24, adj_35, adj_36);
            wp::adj_extract(var_24, var_33, adj_24, adj_33, adj_34);
            wp::adj_extract(var_24, var_31, adj_24, adj_31, adj_32);
            wp::adj_extract(var_24, var_29, adj_24, adj_29, adj_30);
            wp::adj_extract(var_24, var_27, adj_24, adj_27, adj_28);
            wp::adj_extract(var_24, var_25, adj_24, adj_25, adj_26);
            // adj: grad_compact1[0], grad_compact1[1], grad_compact1[2], _0, _0, _0, _0, _0, _0, grad_compact1[3], grad_compact1[4], grad_compact1[5]  <L 47>
            // adj: grad = wp.vector(                                                         <L 46>
            adj_point_point_distance_gradient_0(var_ea0, var_eb1, adj_ea0, adj_eb1, adj_24);
            // adj: grad_compact1 = point_point_distance_gradient(ea0, eb1)                   <L 45>
        }
        // adj: elif dt == 1:                                                                 <L 44>
    }
    wp::adj_where(var_6, var_20, var_1, adj_6, adj_20, adj_1, adj_21);
    if (var_6) {
        wp::adj_vec_t({var_9, var_11, var_13, var_4, var_4, var_4, var_15, var_17, var_19, var_4, var_4, var_4}, {&adj_9, &adj_11, &adj_13, &adj_4, &adj_4, &adj_4, &adj_15, &adj_17, &adj_19, &adj_4, &adj_4, &adj_4}, adj_20);
        wp::adj_extract(var_7, var_18, adj_7, adj_18, adj_19);
        wp::adj_extract(var_7, var_16, adj_7, adj_16, adj_17);
        wp::adj_extract(var_7, var_14, adj_7, adj_14, adj_15);
        wp::adj_extract(var_7, var_12, adj_7, adj_12, adj_13);
        wp::adj_extract(var_7, var_10, adj_7, adj_10, adj_11);
        wp::adj_extract(var_7, var_8, adj_7, adj_8, adj_9);
        // adj: grad_compact0[0], grad_compact0[1], grad_compact0[2], _0, _0, _0, grad_compact0[3], grad_compact0[4], grad_compact0[5], _0, _0, _0  <L 42>
        // adj: grad = wp.vector(                                                             <L 41>
        adj_point_point_distance_gradient_0(var_ea0, var_eb0, adj_ea0, adj_eb0, adj_7);
        // adj: grad_compact0 = point_point_distance_gradient(ea0, eb0)                       <L 40>
    }
    // adj: if dt == 0:                                                                       <L 39>
    wp::adj_float64(var_3, adj_3, adj_4);
    // adj: _0 = wp.float64(0.0)                                                              <L 38>
    adj_edge_edge_distance_type_0(var_ea0, var_ea1, var_eb0, var_eb1, adj_ea0, adj_ea1, adj_eb0, adj_eb1, adj_2);
    // adj: dt = edge_edge_distance_type(ea0, ea1, eb0, eb1)                                  <L 37>
    // adj: grad = wp.vec(length=12, dtype=wp.float64)                                        <L 36>
    // adj: def edge_edge_distance_gradient(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):  <L 35>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/barrier.py:4
static CUDA_CALLABLE wp::float64 barrier_0(
    wp::float64 var_d2,
    wp::float64 var_dHat2,
    wp::float64 var_kappa)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    wp::float64 var_1;
    bool var_2;
    wp::float64 var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    wp::float64 var_6;
    wp::float64 var_7;
    wp::float64 var_8;
    wp::float64 var_9;
    wp::float64 var_10;
    wp::float64 var_11;
    wp::float64 var_12;
    //---------
    // forward
    // def barrier(d2: wp.float64, dHat2: wp.float64, kappa: wp.float64):                     <L 5>
    // e = wp.float64(0.0)                                                                    <L 6>
    var_1 = wp::float64(var_0);
    // if d2 < dHat2:                                                                         <L 7>
    var_2 = (var_d2 < var_dHat2);
    if (var_2) {
        // t2 = d2 - dHat2                                                                    <L 8>
        var_3 = wp::sub(var_d2, var_dHat2);
        // e = -kappa * (t2 / dHat2) * (t2 / dHat2) * wp.log(d2 / dHat2)                      <L 9>
        var_4 = wp::neg(var_kappa);
        var_5 = wp::div(var_3, var_dHat2);
        var_6 = wp::mul(var_4, var_5);
        var_7 = wp::div(var_3, var_dHat2);
        var_8 = wp::mul(var_6, var_7);
        var_9 = wp::div(var_d2, var_dHat2);
        var_10 = wp::log(var_9);
        var_11 = wp::mul(var_8, var_10);
    }
    var_12 = wp::where(var_2, var_11, var_1);
    // return e                                                                               <L 10>
    return var_12;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/barrier.py:4
static CUDA_CALLABLE void adj_barrier_0(
    wp::float64 var_d2,
    wp::float64 var_dHat2,
    wp::float64 var_kappa,
    wp::float64 & adj_d2,
    wp::float64 & adj_dHat2,
    wp::float64 & adj_kappa,
    wp::float64 & adj_ret)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    wp::float64 var_1;
    bool var_2;
    wp::float64 var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    wp::float64 var_6;
    wp::float64 var_7;
    wp::float64 var_8;
    wp::float64 var_9;
    wp::float64 var_10;
    wp::float64 var_11;
    wp::float64 var_12;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    wp::float64 adj_1 = {};
    bool adj_2 = {};
    wp::float64 adj_3 = {};
    wp::float64 adj_4 = {};
    wp::float64 adj_5 = {};
    wp::float64 adj_6 = {};
    wp::float64 adj_7 = {};
    wp::float64 adj_8 = {};
    wp::float64 adj_9 = {};
    wp::float64 adj_10 = {};
    wp::float64 adj_11 = {};
    wp::float64 adj_12 = {};
    //---------
    // forward
    // def barrier(d2: wp.float64, dHat2: wp.float64, kappa: wp.float64):                     <L 5>
    // e = wp.float64(0.0)                                                                    <L 6>
    var_1 = wp::float64(var_0);
    // if d2 < dHat2:                                                                         <L 7>
    var_2 = (var_d2 < var_dHat2);
    if (var_2) {
        // t2 = d2 - dHat2                                                                    <L 8>
        var_3 = wp::sub(var_d2, var_dHat2);
        // e = -kappa * (t2 / dHat2) * (t2 / dHat2) * wp.log(d2 / dHat2)                      <L 9>
        var_4 = wp::neg(var_kappa);
        var_5 = wp::div(var_3, var_dHat2);
        var_6 = wp::mul(var_4, var_5);
        var_7 = wp::div(var_3, var_dHat2);
        var_8 = wp::mul(var_6, var_7);
        var_9 = wp::div(var_d2, var_dHat2);
        var_10 = wp::log(var_9);
        var_11 = wp::mul(var_8, var_10);
    }
    var_12 = wp::where(var_2, var_11, var_1);
    // return e                                                                               <L 10>
    goto label0;
    //---------
    // reverse
    label0:;
    adj_12 += adj_ret;
    // adj: return e                                                                          <L 10>
    wp::adj_where(var_2, var_11, var_1, adj_2, adj_11, adj_1, adj_12);
    if (var_2) {
        wp::adj_mul(var_8, var_10, adj_8, adj_10, adj_11);
        wp::adj_log(var_9, adj_9, adj_10);
        wp::adj_div(var_d2, var_dHat2, var_9, adj_d2, adj_dHat2, adj_9);
        wp::adj_mul(var_6, var_7, adj_6, adj_7, adj_8);
        wp::adj_div(var_3, var_dHat2, var_7, adj_3, adj_dHat2, adj_7);
        wp::adj_mul(var_4, var_5, adj_4, adj_5, adj_6);
        wp::adj_div(var_3, var_dHat2, var_5, adj_3, adj_dHat2, adj_5);
        wp::adj_neg(var_kappa, adj_kappa, adj_4);
        // adj: e = -kappa * (t2 / dHat2) * (t2 / dHat2) * wp.log(d2 / dHat2)                 <L 9>
        wp::adj_sub(var_d2, var_dHat2, adj_d2, adj_dHat2, adj_3);
        // adj: t2 = d2 - dHat2                                                               <L 8>
    }
    // adj: if d2 < dHat2:                                                                    <L 7>
    wp::adj_float64(var_0, adj_0, adj_1);
    // adj: e = wp.float64(0.0)                                                               <L 6>
    // adj: def barrier(d2: wp.float64, dHat2: wp.float64, kappa: wp.float64):                <L 5>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/edge_edge_mullifier.py:303
static CUDA_CALLABLE wp::float64 g_EEM_0(
    wp::float64 var_input,
    wp::float64 var_eps_x)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 1.0;
    wp::float64 var_1;
    wp::float64 var_2;
    const wp::float32 var_3 = 2.0;
    wp::float64 var_4;
    wp::float64 var_5;
    wp::float64 var_6;
    wp::float64 var_7;
    const wp::float32 var_8 = 1.0;
    wp::float64 var_9;
    wp::float64 var_10;
    wp::float64 var_11;
    //---------
    // forward
    // def g_EEM(input: wp.float64, eps_x: wp.float64):                                       <L 304>
    // one_div_eps_x = wp.float64(1.0) / eps_x                                                <L 305>
    var_1 = wp::float64(var_0);
    var_2 = wp::div(var_1, var_eps_x);
    // return wp.float64(2.0) * one_div_eps_x * (-one_div_eps_x * input + wp.float64(1.0))       <L 306>
    var_4 = wp::float64(var_3);
    var_5 = wp::mul(var_4, var_2);
    var_6 = wp::neg(var_2);
    var_7 = wp::mul(var_6, var_input);
    var_9 = wp::float64(var_8);
    var_10 = wp::add(var_7, var_9);
    var_11 = wp::mul(var_5, var_10);
    return var_11;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/edge_edge_mullifier.py:303
static CUDA_CALLABLE void adj_g_EEM_0(
    wp::float64 var_input,
    wp::float64 var_eps_x,
    wp::float64 & adj_input,
    wp::float64 & adj_eps_x,
    wp::float64 & adj_ret)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 1.0;
    wp::float64 var_1;
    wp::float64 var_2;
    const wp::float32 var_3 = 2.0;
    wp::float64 var_4;
    wp::float64 var_5;
    wp::float64 var_6;
    wp::float64 var_7;
    const wp::float32 var_8 = 1.0;
    wp::float64 var_9;
    wp::float64 var_10;
    wp::float64 var_11;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    wp::float64 adj_1 = {};
    wp::float64 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float64 adj_4 = {};
    wp::float64 adj_5 = {};
    wp::float64 adj_6 = {};
    wp::float64 adj_7 = {};
    wp::float32 adj_8 = {};
    wp::float64 adj_9 = {};
    wp::float64 adj_10 = {};
    wp::float64 adj_11 = {};
    //---------
    // forward
    // def g_EEM(input: wp.float64, eps_x: wp.float64):                                       <L 304>
    // one_div_eps_x = wp.float64(1.0) / eps_x                                                <L 305>
    var_1 = wp::float64(var_0);
    var_2 = wp::div(var_1, var_eps_x);
    // return wp.float64(2.0) * one_div_eps_x * (-one_div_eps_x * input + wp.float64(1.0))       <L 306>
    var_4 = wp::float64(var_3);
    var_5 = wp::mul(var_4, var_2);
    var_6 = wp::neg(var_2);
    var_7 = wp::mul(var_6, var_input);
    var_9 = wp::float64(var_8);
    var_10 = wp::add(var_7, var_9);
    var_11 = wp::mul(var_5, var_10);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_11 += adj_ret;
    wp::adj_mul(var_5, var_10, adj_5, adj_10, adj_11);
    wp::adj_add(var_7, var_9, adj_7, adj_9, adj_10);
    wp::adj_float64(var_8, adj_8, adj_9);
    wp::adj_mul(var_6, var_input, adj_6, adj_input, adj_7);
    wp::adj_neg(var_2, adj_2, adj_6);
    wp::adj_mul(var_4, var_2, adj_4, adj_2, adj_5);
    wp::adj_float64(var_3, adj_3, adj_4);
    // adj: return wp.float64(2.0) * one_div_eps_x * (-one_div_eps_x * input + wp.float64(1.0))  <L 306>
    wp::adj_div(var_1, var_eps_x, var_2, adj_1, adj_eps_x, adj_2);
    wp::adj_float64(var_0, adj_0, adj_1);
    // adj: one_div_eps_x = wp.float64(1.0) / eps_x                                           <L 305>
    // adj: def g_EEM(input: wp.float64, eps_x: wp.float64):                                  <L 304>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/edge_edge_mullifier.py:4
static CUDA_CALLABLE wp::vec_t<12,wp::float64> g_EECN2_0(
    wp::float64 var_v01,
    wp::float64 var_v02,
    wp::float64 var_v03,
    wp::float64 var_v11,
    wp::float64 var_v12,
    wp::float64 var_v13,
    wp::float64 var_v21,
    wp::float64 var_v22,
    wp::float64 var_v23,
    wp::float64 var_v31,
    wp::float64 var_v32,
    wp::float64 var_v33)
{
    //---------
    // primal vars
    wp::float64 var_0;
    wp::float64 var_1;
    wp::float64 var_2;
    wp::float64 var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    wp::float64 var_6;
    wp::float64 var_7;
    wp::float64 var_8;
    wp::float64 var_9;
    wp::float64 var_10;
    wp::float64 var_11;
    wp::float64 var_12;
    wp::float64 var_13;
    wp::float64 var_14;
    wp::float64 var_15;
    wp::float64 var_16;
    wp::float64 var_17;
    wp::float64 var_18;
    wp::float64 var_19;
    wp::float64 var_20;
    wp::float64 var_21;
    wp::float64 var_22;
    wp::float64 var_23;
    wp::float64 var_24;
    const wp::float32 var_25 = 2.0;
    wp::float64 var_26;
    wp::float64 var_27;
    wp::float64 var_28;
    const wp::float32 var_29 = 2.0;
    wp::float64 var_30;
    wp::float64 var_31;
    wp::float64 var_32;
    const wp::float32 var_33 = 2.0;
    wp::float64 var_34;
    wp::float64 var_35;
    wp::float64 var_36;
    const wp::float32 var_37 = 2.0;
    wp::float64 var_38;
    wp::float64 var_39;
    wp::float64 var_40;
    const wp::float32 var_41 = 2.0;
    wp::float64 var_42;
    wp::float64 var_43;
    wp::float64 var_44;
    const wp::float32 var_45 = 2.0;
    wp::float64 var_46;
    wp::float64 var_47;
    wp::float64 var_48;
    const wp::float32 var_49 = 2.0;
    wp::float64 var_50;
    wp::float64 var_51;
    wp::float64 var_52;
    const wp::float32 var_53 = 2.0;
    wp::float64 var_54;
    wp::float64 var_55;
    wp::float64 var_56;
    const wp::float32 var_57 = 2.0;
    wp::float64 var_58;
    wp::float64 var_59;
    wp::float64 var_60;
    const wp::float32 var_61 = 2.0;
    wp::float64 var_62;
    wp::float64 var_63;
    wp::float64 var_64;
    const wp::float32 var_65 = 2.0;
    wp::float64 var_66;
    wp::float64 var_67;
    wp::float64 var_68;
    const wp::float32 var_69 = 2.0;
    wp::float64 var_70;
    wp::float64 var_71;
    wp::float64 var_72;
    wp::float64 var_73;
    wp::float64 var_74;
    wp::float64 var_75;
    wp::float64 var_76;
    wp::float64 var_77;
    wp::float64 var_78;
    wp::float64 var_79;
    wp::float64 var_80;
    wp::float64 var_81;
    wp::float64 var_82;
    wp::float64 var_83;
    wp::float64 var_84;
    wp::float64 var_85;
    wp::float64 var_86;
    wp::float64 var_87;
    wp::float64 var_88;
    wp::float64 var_89;
    wp::vec_t<12,wp::float64> var_90;
    //---------
    // forward
    // def g_EECN2(                                                                           <L 5>
    // t8 = -v11 + v01                                                                        <L 23>
    var_0 = wp::neg(var_v11);
    var_1 = wp::add(var_0, var_v01);
    // t9 = -v12 + v02                                                                        <L 24>
    var_2 = wp::neg(var_v12);
    var_3 = wp::add(var_2, var_v02);
    // t10 = -v13 + v03                                                                       <L 25>
    var_4 = wp::neg(var_v13);
    var_5 = wp::add(var_4, var_v03);
    // t11 = -v31 + v21                                                                       <L 26>
    var_6 = wp::neg(var_v31);
    var_7 = wp::add(var_6, var_v21);
    // t12 = -v32 + v22                                                                       <L 27>
    var_8 = wp::neg(var_v32);
    var_9 = wp::add(var_8, var_v22);
    // t13 = -v33 + v23                                                                       <L 28>
    var_10 = wp::neg(var_v33);
    var_11 = wp::add(var_10, var_v23);
    // t23 = t8 * t12 + -(t9 * t11)                                                           <L 29>
    var_12 = wp::mul(var_1, var_9);
    var_13 = wp::mul(var_3, var_7);
    var_14 = wp::neg(var_13);
    var_15 = wp::add(var_12, var_14);
    // t24 = t8 * t13 + -(t10 * t11)                                                          <L 30>
    var_16 = wp::mul(var_1, var_11);
    var_17 = wp::mul(var_5, var_7);
    var_18 = wp::neg(var_17);
    var_19 = wp::add(var_16, var_18);
    // t25 = t9 * t13 + -(t10 * t12)                                                          <L 31>
    var_20 = wp::mul(var_3, var_11);
    var_21 = wp::mul(var_5, var_9);
    var_22 = wp::neg(var_21);
    var_23 = wp::add(var_20, var_22);
    // t26 = t8 * t23 * wp.float64(2.0)                                                       <L 32>
    var_24 = wp::mul(var_1, var_15);
    var_26 = wp::float64(var_25);
    var_27 = wp::mul(var_24, var_26);
    // t27 = t9 * t23 * wp.float64(2.0)                                                       <L 33>
    var_28 = wp::mul(var_3, var_15);
    var_30 = wp::float64(var_29);
    var_31 = wp::mul(var_28, var_30);
    // t28 = t8 * t24 * wp.float64(2.0)                                                       <L 34>
    var_32 = wp::mul(var_1, var_19);
    var_34 = wp::float64(var_33);
    var_35 = wp::mul(var_32, var_34);
    // t29 = t10 * t24 * wp.float64(2.0)                                                      <L 35>
    var_36 = wp::mul(var_5, var_19);
    var_38 = wp::float64(var_37);
    var_39 = wp::mul(var_36, var_38);
    // t30 = t9 * t25 * wp.float64(2.0)                                                       <L 36>
    var_40 = wp::mul(var_3, var_23);
    var_42 = wp::float64(var_41);
    var_43 = wp::mul(var_40, var_42);
    // t31 = t10 * t25 * wp.float64(2.0)                                                      <L 37>
    var_44 = wp::mul(var_5, var_23);
    var_46 = wp::float64(var_45);
    var_47 = wp::mul(var_44, var_46);
    // t32 = t11 * t23 * wp.float64(2.0)                                                      <L 38>
    var_48 = wp::mul(var_7, var_15);
    var_50 = wp::float64(var_49);
    var_51 = wp::mul(var_48, var_50);
    // t33 = t12 * t23 * wp.float64(2.0)                                                      <L 39>
    var_52 = wp::mul(var_9, var_15);
    var_54 = wp::float64(var_53);
    var_55 = wp::mul(var_52, var_54);
    // t23 = t11 * t24 * wp.float64(2.0)                                                      <L 40>
    var_56 = wp::mul(var_7, var_19);
    var_58 = wp::float64(var_57);
    var_59 = wp::mul(var_56, var_58);
    // t10 = t13 * t24 * wp.float64(2.0)                                                      <L 41>
    var_60 = wp::mul(var_11, var_19);
    var_62 = wp::float64(var_61);
    var_63 = wp::mul(var_60, var_62);
    // t9 = t12 * t25 * wp.float64(2.0)                                                       <L 42>
    var_64 = wp::mul(var_9, var_23);
    var_66 = wp::float64(var_65);
    var_67 = wp::mul(var_64, var_66);
    // t8 = t13 * t25 * wp.float64(2.0)                                                       <L 43>
    var_68 = wp::mul(var_11, var_23);
    var_70 = wp::float64(var_69);
    var_71 = wp::mul(var_68, var_70);
    // g = wp.vector(                                                                         <L 44>
    // t33 + t10,                                                                             <L 45>
    var_72 = wp::add(var_55, var_63);
    // -t32 + t8,                                                                             <L 46>
    var_73 = wp::neg(var_51);
    var_74 = wp::add(var_73, var_71);
    // -t23 - t9,                                                                             <L 47>
    var_75 = wp::neg(var_59);
    var_76 = wp::sub(var_75, var_67);
    // -t33 - t10,                                                                            <L 48>
    var_77 = wp::neg(var_55);
    var_78 = wp::sub(var_77, var_63);
    // t32 - t8,                                                                              <L 49>
    var_79 = wp::sub(var_51, var_71);
    // t23 + t9,                                                                              <L 50>
    var_80 = wp::add(var_59, var_67);
    // -t27 - t29,                                                                            <L 51>
    var_81 = wp::neg(var_31);
    var_82 = wp::sub(var_81, var_39);
    // t26 - t31,                                                                             <L 52>
    var_83 = wp::sub(var_27, var_47);
    // t28 + t30,                                                                             <L 53>
    var_84 = wp::add(var_35, var_43);
    // t27 + t29,                                                                             <L 54>
    var_85 = wp::add(var_31, var_39);
    // -t26 + t31,                                                                            <L 55>
    var_86 = wp::neg(var_27);
    var_87 = wp::add(var_86, var_47);
    // -t28 - t30,                                                                            <L 56>
    var_88 = wp::neg(var_35);
    var_89 = wp::sub(var_88, var_43);
    var_90 = wp::vec_t<12,wp::float64>({var_72, var_74, var_76, var_78, var_79, var_80, var_82, var_83, var_84, var_85, var_87, var_89});
    // return g                                                                               <L 58>
    return var_90;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/edge_edge_mullifier.py:4
static CUDA_CALLABLE void adj_g_EECN2_0(
    wp::float64 var_v01,
    wp::float64 var_v02,
    wp::float64 var_v03,
    wp::float64 var_v11,
    wp::float64 var_v12,
    wp::float64 var_v13,
    wp::float64 var_v21,
    wp::float64 var_v22,
    wp::float64 var_v23,
    wp::float64 var_v31,
    wp::float64 var_v32,
    wp::float64 var_v33,
    wp::float64 & adj_v01,
    wp::float64 & adj_v02,
    wp::float64 & adj_v03,
    wp::float64 & adj_v11,
    wp::float64 & adj_v12,
    wp::float64 & adj_v13,
    wp::float64 & adj_v21,
    wp::float64 & adj_v22,
    wp::float64 & adj_v23,
    wp::float64 & adj_v31,
    wp::float64 & adj_v32,
    wp::float64 & adj_v33,
    wp::vec_t<12,wp::float64> & adj_ret)
{
    //---------
    // primal vars
    wp::float64 var_0;
    wp::float64 var_1;
    wp::float64 var_2;
    wp::float64 var_3;
    wp::float64 var_4;
    wp::float64 var_5;
    wp::float64 var_6;
    wp::float64 var_7;
    wp::float64 var_8;
    wp::float64 var_9;
    wp::float64 var_10;
    wp::float64 var_11;
    wp::float64 var_12;
    wp::float64 var_13;
    wp::float64 var_14;
    wp::float64 var_15;
    wp::float64 var_16;
    wp::float64 var_17;
    wp::float64 var_18;
    wp::float64 var_19;
    wp::float64 var_20;
    wp::float64 var_21;
    wp::float64 var_22;
    wp::float64 var_23;
    wp::float64 var_24;
    const wp::float32 var_25 = 2.0;
    wp::float64 var_26;
    wp::float64 var_27;
    wp::float64 var_28;
    const wp::float32 var_29 = 2.0;
    wp::float64 var_30;
    wp::float64 var_31;
    wp::float64 var_32;
    const wp::float32 var_33 = 2.0;
    wp::float64 var_34;
    wp::float64 var_35;
    wp::float64 var_36;
    const wp::float32 var_37 = 2.0;
    wp::float64 var_38;
    wp::float64 var_39;
    wp::float64 var_40;
    const wp::float32 var_41 = 2.0;
    wp::float64 var_42;
    wp::float64 var_43;
    wp::float64 var_44;
    const wp::float32 var_45 = 2.0;
    wp::float64 var_46;
    wp::float64 var_47;
    wp::float64 var_48;
    const wp::float32 var_49 = 2.0;
    wp::float64 var_50;
    wp::float64 var_51;
    wp::float64 var_52;
    const wp::float32 var_53 = 2.0;
    wp::float64 var_54;
    wp::float64 var_55;
    wp::float64 var_56;
    const wp::float32 var_57 = 2.0;
    wp::float64 var_58;
    wp::float64 var_59;
    wp::float64 var_60;
    const wp::float32 var_61 = 2.0;
    wp::float64 var_62;
    wp::float64 var_63;
    wp::float64 var_64;
    const wp::float32 var_65 = 2.0;
    wp::float64 var_66;
    wp::float64 var_67;
    wp::float64 var_68;
    const wp::float32 var_69 = 2.0;
    wp::float64 var_70;
    wp::float64 var_71;
    wp::float64 var_72;
    wp::float64 var_73;
    wp::float64 var_74;
    wp::float64 var_75;
    wp::float64 var_76;
    wp::float64 var_77;
    wp::float64 var_78;
    wp::float64 var_79;
    wp::float64 var_80;
    wp::float64 var_81;
    wp::float64 var_82;
    wp::float64 var_83;
    wp::float64 var_84;
    wp::float64 var_85;
    wp::float64 var_86;
    wp::float64 var_87;
    wp::float64 var_88;
    wp::float64 var_89;
    wp::vec_t<12,wp::float64> var_90;
    //---------
    // dual vars
    wp::float64 adj_0 = {};
    wp::float64 adj_1 = {};
    wp::float64 adj_2 = {};
    wp::float64 adj_3 = {};
    wp::float64 adj_4 = {};
    wp::float64 adj_5 = {};
    wp::float64 adj_6 = {};
    wp::float64 adj_7 = {};
    wp::float64 adj_8 = {};
    wp::float64 adj_9 = {};
    wp::float64 adj_10 = {};
    wp::float64 adj_11 = {};
    wp::float64 adj_12 = {};
    wp::float64 adj_13 = {};
    wp::float64 adj_14 = {};
    wp::float64 adj_15 = {};
    wp::float64 adj_16 = {};
    wp::float64 adj_17 = {};
    wp::float64 adj_18 = {};
    wp::float64 adj_19 = {};
    wp::float64 adj_20 = {};
    wp::float64 adj_21 = {};
    wp::float64 adj_22 = {};
    wp::float64 adj_23 = {};
    wp::float64 adj_24 = {};
    wp::float32 adj_25 = {};
    wp::float64 adj_26 = {};
    wp::float64 adj_27 = {};
    wp::float64 adj_28 = {};
    wp::float32 adj_29 = {};
    wp::float64 adj_30 = {};
    wp::float64 adj_31 = {};
    wp::float64 adj_32 = {};
    wp::float32 adj_33 = {};
    wp::float64 adj_34 = {};
    wp::float64 adj_35 = {};
    wp::float64 adj_36 = {};
    wp::float32 adj_37 = {};
    wp::float64 adj_38 = {};
    wp::float64 adj_39 = {};
    wp::float64 adj_40 = {};
    wp::float32 adj_41 = {};
    wp::float64 adj_42 = {};
    wp::float64 adj_43 = {};
    wp::float64 adj_44 = {};
    wp::float32 adj_45 = {};
    wp::float64 adj_46 = {};
    wp::float64 adj_47 = {};
    wp::float64 adj_48 = {};
    wp::float32 adj_49 = {};
    wp::float64 adj_50 = {};
    wp::float64 adj_51 = {};
    wp::float64 adj_52 = {};
    wp::float32 adj_53 = {};
    wp::float64 adj_54 = {};
    wp::float64 adj_55 = {};
    wp::float64 adj_56 = {};
    wp::float32 adj_57 = {};
    wp::float64 adj_58 = {};
    wp::float64 adj_59 = {};
    wp::float64 adj_60 = {};
    wp::float32 adj_61 = {};
    wp::float64 adj_62 = {};
    wp::float64 adj_63 = {};
    wp::float64 adj_64 = {};
    wp::float32 adj_65 = {};
    wp::float64 adj_66 = {};
    wp::float64 adj_67 = {};
    wp::float64 adj_68 = {};
    wp::float32 adj_69 = {};
    wp::float64 adj_70 = {};
    wp::float64 adj_71 = {};
    wp::float64 adj_72 = {};
    wp::float64 adj_73 = {};
    wp::float64 adj_74 = {};
    wp::float64 adj_75 = {};
    wp::float64 adj_76 = {};
    wp::float64 adj_77 = {};
    wp::float64 adj_78 = {};
    wp::float64 adj_79 = {};
    wp::float64 adj_80 = {};
    wp::float64 adj_81 = {};
    wp::float64 adj_82 = {};
    wp::float64 adj_83 = {};
    wp::float64 adj_84 = {};
    wp::float64 adj_85 = {};
    wp::float64 adj_86 = {};
    wp::float64 adj_87 = {};
    wp::float64 adj_88 = {};
    wp::float64 adj_89 = {};
    wp::vec_t<12,wp::float64> adj_90 = {};
    //---------
    // forward
    // def g_EECN2(                                                                           <L 5>
    // t8 = -v11 + v01                                                                        <L 23>
    var_0 = wp::neg(var_v11);
    var_1 = wp::add(var_0, var_v01);
    // t9 = -v12 + v02                                                                        <L 24>
    var_2 = wp::neg(var_v12);
    var_3 = wp::add(var_2, var_v02);
    // t10 = -v13 + v03                                                                       <L 25>
    var_4 = wp::neg(var_v13);
    var_5 = wp::add(var_4, var_v03);
    // t11 = -v31 + v21                                                                       <L 26>
    var_6 = wp::neg(var_v31);
    var_7 = wp::add(var_6, var_v21);
    // t12 = -v32 + v22                                                                       <L 27>
    var_8 = wp::neg(var_v32);
    var_9 = wp::add(var_8, var_v22);
    // t13 = -v33 + v23                                                                       <L 28>
    var_10 = wp::neg(var_v33);
    var_11 = wp::add(var_10, var_v23);
    // t23 = t8 * t12 + -(t9 * t11)                                                           <L 29>
    var_12 = wp::mul(var_1, var_9);
    var_13 = wp::mul(var_3, var_7);
    var_14 = wp::neg(var_13);
    var_15 = wp::add(var_12, var_14);
    // t24 = t8 * t13 + -(t10 * t11)                                                          <L 30>
    var_16 = wp::mul(var_1, var_11);
    var_17 = wp::mul(var_5, var_7);
    var_18 = wp::neg(var_17);
    var_19 = wp::add(var_16, var_18);
    // t25 = t9 * t13 + -(t10 * t12)                                                          <L 31>
    var_20 = wp::mul(var_3, var_11);
    var_21 = wp::mul(var_5, var_9);
    var_22 = wp::neg(var_21);
    var_23 = wp::add(var_20, var_22);
    // t26 = t8 * t23 * wp.float64(2.0)                                                       <L 32>
    var_24 = wp::mul(var_1, var_15);
    var_26 = wp::float64(var_25);
    var_27 = wp::mul(var_24, var_26);
    // t27 = t9 * t23 * wp.float64(2.0)                                                       <L 33>
    var_28 = wp::mul(var_3, var_15);
    var_30 = wp::float64(var_29);
    var_31 = wp::mul(var_28, var_30);
    // t28 = t8 * t24 * wp.float64(2.0)                                                       <L 34>
    var_32 = wp::mul(var_1, var_19);
    var_34 = wp::float64(var_33);
    var_35 = wp::mul(var_32, var_34);
    // t29 = t10 * t24 * wp.float64(2.0)                                                      <L 35>
    var_36 = wp::mul(var_5, var_19);
    var_38 = wp::float64(var_37);
    var_39 = wp::mul(var_36, var_38);
    // t30 = t9 * t25 * wp.float64(2.0)                                                       <L 36>
    var_40 = wp::mul(var_3, var_23);
    var_42 = wp::float64(var_41);
    var_43 = wp::mul(var_40, var_42);
    // t31 = t10 * t25 * wp.float64(2.0)                                                      <L 37>
    var_44 = wp::mul(var_5, var_23);
    var_46 = wp::float64(var_45);
    var_47 = wp::mul(var_44, var_46);
    // t32 = t11 * t23 * wp.float64(2.0)                                                      <L 38>
    var_48 = wp::mul(var_7, var_15);
    var_50 = wp::float64(var_49);
    var_51 = wp::mul(var_48, var_50);
    // t33 = t12 * t23 * wp.float64(2.0)                                                      <L 39>
    var_52 = wp::mul(var_9, var_15);
    var_54 = wp::float64(var_53);
    var_55 = wp::mul(var_52, var_54);
    // t23 = t11 * t24 * wp.float64(2.0)                                                      <L 40>
    var_56 = wp::mul(var_7, var_19);
    var_58 = wp::float64(var_57);
    var_59 = wp::mul(var_56, var_58);
    // t10 = t13 * t24 * wp.float64(2.0)                                                      <L 41>
    var_60 = wp::mul(var_11, var_19);
    var_62 = wp::float64(var_61);
    var_63 = wp::mul(var_60, var_62);
    // t9 = t12 * t25 * wp.float64(2.0)                                                       <L 42>
    var_64 = wp::mul(var_9, var_23);
    var_66 = wp::float64(var_65);
    var_67 = wp::mul(var_64, var_66);
    // t8 = t13 * t25 * wp.float64(2.0)                                                       <L 43>
    var_68 = wp::mul(var_11, var_23);
    var_70 = wp::float64(var_69);
    var_71 = wp::mul(var_68, var_70);
    // g = wp.vector(                                                                         <L 44>
    // t33 + t10,                                                                             <L 45>
    var_72 = wp::add(var_55, var_63);
    // -t32 + t8,                                                                             <L 46>
    var_73 = wp::neg(var_51);
    var_74 = wp::add(var_73, var_71);
    // -t23 - t9,                                                                             <L 47>
    var_75 = wp::neg(var_59);
    var_76 = wp::sub(var_75, var_67);
    // -t33 - t10,                                                                            <L 48>
    var_77 = wp::neg(var_55);
    var_78 = wp::sub(var_77, var_63);
    // t32 - t8,                                                                              <L 49>
    var_79 = wp::sub(var_51, var_71);
    // t23 + t9,                                                                              <L 50>
    var_80 = wp::add(var_59, var_67);
    // -t27 - t29,                                                                            <L 51>
    var_81 = wp::neg(var_31);
    var_82 = wp::sub(var_81, var_39);
    // t26 - t31,                                                                             <L 52>
    var_83 = wp::sub(var_27, var_47);
    // t28 + t30,                                                                             <L 53>
    var_84 = wp::add(var_35, var_43);
    // t27 + t29,                                                                             <L 54>
    var_85 = wp::add(var_31, var_39);
    // -t26 + t31,                                                                            <L 55>
    var_86 = wp::neg(var_27);
    var_87 = wp::add(var_86, var_47);
    // -t28 - t30,                                                                            <L 56>
    var_88 = wp::neg(var_35);
    var_89 = wp::sub(var_88, var_43);
    var_90 = wp::vec_t<12,wp::float64>({var_72, var_74, var_76, var_78, var_79, var_80, var_82, var_83, var_84, var_85, var_87, var_89});
    // return g                                                                               <L 58>
    goto label0;
    //---------
    // reverse
    label0:;
    adj_90 += adj_ret;
    // adj: return g                                                                          <L 58>
    wp::adj_vec_t({var_72, var_74, var_76, var_78, var_79, var_80, var_82, var_83, var_84, var_85, var_87, var_89}, {&adj_72, &adj_74, &adj_76, &adj_78, &adj_79, &adj_80, &adj_82, &adj_83, &adj_84, &adj_85, &adj_87, &adj_89}, adj_90);
    wp::adj_sub(var_88, var_43, adj_88, adj_43, adj_89);
    wp::adj_neg(var_35, adj_35, adj_88);
    // adj: -t28 - t30,                                                                       <L 56>
    wp::adj_add(var_86, var_47, adj_86, adj_47, adj_87);
    wp::adj_neg(var_27, adj_27, adj_86);
    // adj: -t26 + t31,                                                                       <L 55>
    wp::adj_add(var_31, var_39, adj_31, adj_39, adj_85);
    // adj: t27 + t29,                                                                        <L 54>
    wp::adj_add(var_35, var_43, adj_35, adj_43, adj_84);
    // adj: t28 + t30,                                                                        <L 53>
    wp::adj_sub(var_27, var_47, adj_27, adj_47, adj_83);
    // adj: t26 - t31,                                                                        <L 52>
    wp::adj_sub(var_81, var_39, adj_81, adj_39, adj_82);
    wp::adj_neg(var_31, adj_31, adj_81);
    // adj: -t27 - t29,                                                                       <L 51>
    wp::adj_add(var_59, var_67, adj_59, adj_67, adj_80);
    // adj: t23 + t9,                                                                         <L 50>
    wp::adj_sub(var_51, var_71, adj_51, adj_71, adj_79);
    // adj: t32 - t8,                                                                         <L 49>
    wp::adj_sub(var_77, var_63, adj_77, adj_63, adj_78);
    wp::adj_neg(var_55, adj_55, adj_77);
    // adj: -t33 - t10,                                                                       <L 48>
    wp::adj_sub(var_75, var_67, adj_75, adj_67, adj_76);
    wp::adj_neg(var_59, adj_59, adj_75);
    // adj: -t23 - t9,                                                                        <L 47>
    wp::adj_add(var_73, var_71, adj_73, adj_71, adj_74);
    wp::adj_neg(var_51, adj_51, adj_73);
    // adj: -t32 + t8,                                                                        <L 46>
    wp::adj_add(var_55, var_63, adj_55, adj_63, adj_72);
    // adj: t33 + t10,                                                                        <L 45>
    // adj: g = wp.vector(                                                                    <L 44>
    wp::adj_mul(var_68, var_70, adj_68, adj_70, adj_71);
    wp::adj_float64(var_69, adj_69, adj_70);
    wp::adj_mul(var_11, var_23, adj_11, adj_23, adj_68);
    // adj: t8 = t13 * t25 * wp.float64(2.0)                                                  <L 43>
    wp::adj_mul(var_64, var_66, adj_64, adj_66, adj_67);
    wp::adj_float64(var_65, adj_65, adj_66);
    wp::adj_mul(var_9, var_23, adj_9, adj_23, adj_64);
    // adj: t9 = t12 * t25 * wp.float64(2.0)                                                  <L 42>
    wp::adj_mul(var_60, var_62, adj_60, adj_62, adj_63);
    wp::adj_float64(var_61, adj_61, adj_62);
    wp::adj_mul(var_11, var_19, adj_11, adj_19, adj_60);
    // adj: t10 = t13 * t24 * wp.float64(2.0)                                                 <L 41>
    wp::adj_mul(var_56, var_58, adj_56, adj_58, adj_59);
    wp::adj_float64(var_57, adj_57, adj_58);
    wp::adj_mul(var_7, var_19, adj_7, adj_19, adj_56);
    // adj: t23 = t11 * t24 * wp.float64(2.0)                                                 <L 40>
    wp::adj_mul(var_52, var_54, adj_52, adj_54, adj_55);
    wp::adj_float64(var_53, adj_53, adj_54);
    wp::adj_mul(var_9, var_15, adj_9, adj_15, adj_52);
    // adj: t33 = t12 * t23 * wp.float64(2.0)                                                 <L 39>
    wp::adj_mul(var_48, var_50, adj_48, adj_50, adj_51);
    wp::adj_float64(var_49, adj_49, adj_50);
    wp::adj_mul(var_7, var_15, adj_7, adj_15, adj_48);
    // adj: t32 = t11 * t23 * wp.float64(2.0)                                                 <L 38>
    wp::adj_mul(var_44, var_46, adj_44, adj_46, adj_47);
    wp::adj_float64(var_45, adj_45, adj_46);
    wp::adj_mul(var_5, var_23, adj_5, adj_23, adj_44);
    // adj: t31 = t10 * t25 * wp.float64(2.0)                                                 <L 37>
    wp::adj_mul(var_40, var_42, adj_40, adj_42, adj_43);
    wp::adj_float64(var_41, adj_41, adj_42);
    wp::adj_mul(var_3, var_23, adj_3, adj_23, adj_40);
    // adj: t30 = t9 * t25 * wp.float64(2.0)                                                  <L 36>
    wp::adj_mul(var_36, var_38, adj_36, adj_38, adj_39);
    wp::adj_float64(var_37, adj_37, adj_38);
    wp::adj_mul(var_5, var_19, adj_5, adj_19, adj_36);
    // adj: t29 = t10 * t24 * wp.float64(2.0)                                                 <L 35>
    wp::adj_mul(var_32, var_34, adj_32, adj_34, adj_35);
    wp::adj_float64(var_33, adj_33, adj_34);
    wp::adj_mul(var_1, var_19, adj_1, adj_19, adj_32);
    // adj: t28 = t8 * t24 * wp.float64(2.0)                                                  <L 34>
    wp::adj_mul(var_28, var_30, adj_28, adj_30, adj_31);
    wp::adj_float64(var_29, adj_29, adj_30);
    wp::adj_mul(var_3, var_15, adj_3, adj_15, adj_28);
    // adj: t27 = t9 * t23 * wp.float64(2.0)                                                  <L 33>
    wp::adj_mul(var_24, var_26, adj_24, adj_26, adj_27);
    wp::adj_float64(var_25, adj_25, adj_26);
    wp::adj_mul(var_1, var_15, adj_1, adj_15, adj_24);
    // adj: t26 = t8 * t23 * wp.float64(2.0)                                                  <L 32>
    wp::adj_add(var_20, var_22, adj_20, adj_22, adj_23);
    wp::adj_neg(var_21, adj_21, adj_22);
    wp::adj_mul(var_5, var_9, adj_5, adj_9, adj_21);
    wp::adj_mul(var_3, var_11, adj_3, adj_11, adj_20);
    // adj: t25 = t9 * t13 + -(t10 * t12)                                                     <L 31>
    wp::adj_add(var_16, var_18, adj_16, adj_18, adj_19);
    wp::adj_neg(var_17, adj_17, adj_18);
    wp::adj_mul(var_5, var_7, adj_5, adj_7, adj_17);
    wp::adj_mul(var_1, var_11, adj_1, adj_11, adj_16);
    // adj: t24 = t8 * t13 + -(t10 * t11)                                                     <L 30>
    wp::adj_add(var_12, var_14, adj_12, adj_14, adj_15);
    wp::adj_neg(var_13, adj_13, adj_14);
    wp::adj_mul(var_3, var_7, adj_3, adj_7, adj_13);
    wp::adj_mul(var_1, var_9, adj_1, adj_9, adj_12);
    // adj: t23 = t8 * t12 + -(t9 * t11)                                                      <L 29>
    wp::adj_add(var_10, var_v23, adj_10, adj_v23, adj_11);
    wp::adj_neg(var_v33, adj_v33, adj_10);
    // adj: t13 = -v33 + v23                                                                  <L 28>
    wp::adj_add(var_8, var_v22, adj_8, adj_v22, adj_9);
    wp::adj_neg(var_v32, adj_v32, adj_8);
    // adj: t12 = -v32 + v22                                                                  <L 27>
    wp::adj_add(var_6, var_v21, adj_6, adj_v21, adj_7);
    wp::adj_neg(var_v31, adj_v31, adj_6);
    // adj: t11 = -v31 + v21                                                                  <L 26>
    wp::adj_add(var_4, var_v03, adj_4, adj_v03, adj_5);
    wp::adj_neg(var_v13, adj_v13, adj_4);
    // adj: t10 = -v13 + v03                                                                  <L 25>
    wp::adj_add(var_2, var_v02, adj_2, adj_v02, adj_3);
    wp::adj_neg(var_v12, adj_v12, adj_2);
    // adj: t9 = -v12 + v02                                                                   <L 24>
    wp::adj_add(var_0, var_v01, adj_0, adj_v01, adj_1);
    wp::adj_neg(var_v11, adj_v11, adj_0);
    // adj: t8 = -v11 + v01                                                                   <L 23>
    // adj: def g_EECN2(                                                                      <L 5>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/edge_edge_mullifier.py:322
static CUDA_CALLABLE wp::vec_t<12,wp::float64> edge_edge_cross_norm2_gradient_0(
    wp::vec_t<3,wp::float64> var_ea0,
    wp::vec_t<3,wp::float64> var_ea1,
    wp::vec_t<3,wp::float64> var_eb0,
    wp::vec_t<3,wp::float64> var_eb1)
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
    // def edge_edge_cross_norm2_gradient(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):       <L 323>
    // return g_EECN2(                                                                        <L 324>
    // ea0[0],                                                                                <L 325>
    var_1 = wp::extract(var_ea0, var_0);
    // ea0[1],                                                                                <L 326>
    var_3 = wp::extract(var_ea0, var_2);
    // ea0[2],                                                                                <L 327>
    var_5 = wp::extract(var_ea0, var_4);
    // ea1[0],                                                                                <L 328>
    var_7 = wp::extract(var_ea1, var_6);
    // ea1[1],                                                                                <L 329>
    var_9 = wp::extract(var_ea1, var_8);
    // ea1[2],                                                                                <L 330>
    var_11 = wp::extract(var_ea1, var_10);
    // eb0[0],                                                                                <L 331>
    var_13 = wp::extract(var_eb0, var_12);
    // eb0[1],                                                                                <L 332>
    var_15 = wp::extract(var_eb0, var_14);
    // eb0[2],                                                                                <L 333>
    var_17 = wp::extract(var_eb0, var_16);
    // eb1[0],                                                                                <L 334>
    var_19 = wp::extract(var_eb1, var_18);
    // eb1[1],                                                                                <L 335>
    var_21 = wp::extract(var_eb1, var_20);
    // eb1[2],                                                                                <L 336>
    var_23 = wp::extract(var_eb1, var_22);
    var_24 = g_EECN2_0(var_1, var_3, var_5, var_7, var_9, var_11, var_13, var_15, var_17, var_19, var_21, var_23);
    return var_24;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/edge_edge_mullifier.py:322
static CUDA_CALLABLE void adj_edge_edge_cross_norm2_gradient_0(
    wp::vec_t<3,wp::float64> var_ea0,
    wp::vec_t<3,wp::float64> var_ea1,
    wp::vec_t<3,wp::float64> var_eb0,
    wp::vec_t<3,wp::float64> var_eb1,
    wp::vec_t<3,wp::float64> & adj_ea0,
    wp::vec_t<3,wp::float64> & adj_ea1,
    wp::vec_t<3,wp::float64> & adj_eb0,
    wp::vec_t<3,wp::float64> & adj_eb1,
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
    // def edge_edge_cross_norm2_gradient(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):       <L 323>
    // return g_EECN2(                                                                        <L 324>
    // ea0[0],                                                                                <L 325>
    var_1 = wp::extract(var_ea0, var_0);
    // ea0[1],                                                                                <L 326>
    var_3 = wp::extract(var_ea0, var_2);
    // ea0[2],                                                                                <L 327>
    var_5 = wp::extract(var_ea0, var_4);
    // ea1[0],                                                                                <L 328>
    var_7 = wp::extract(var_ea1, var_6);
    // ea1[1],                                                                                <L 329>
    var_9 = wp::extract(var_ea1, var_8);
    // ea1[2],                                                                                <L 330>
    var_11 = wp::extract(var_ea1, var_10);
    // eb0[0],                                                                                <L 331>
    var_13 = wp::extract(var_eb0, var_12);
    // eb0[1],                                                                                <L 332>
    var_15 = wp::extract(var_eb0, var_14);
    // eb0[2],                                                                                <L 333>
    var_17 = wp::extract(var_eb0, var_16);
    // eb1[0],                                                                                <L 334>
    var_19 = wp::extract(var_eb1, var_18);
    // eb1[1],                                                                                <L 335>
    var_21 = wp::extract(var_eb1, var_20);
    // eb1[2],                                                                                <L 336>
    var_23 = wp::extract(var_eb1, var_22);
    var_24 = g_EECN2_0(var_1, var_3, var_5, var_7, var_9, var_11, var_13, var_15, var_17, var_19, var_21, var_23);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_24 += adj_ret;
    adj_g_EECN2_0(var_1, var_3, var_5, var_7, var_9, var_11, var_13, var_15, var_17, var_19, var_21, var_23, adj_1, adj_3, adj_5, adj_7, adj_9, adj_11, adj_13, adj_15, adj_17, adj_19, adj_21, adj_23, adj_24);
    wp::adj_extract(var_eb1, var_22, adj_eb1, adj_22, adj_23);
    // adj: eb1[2],                                                                           <L 336>
    wp::adj_extract(var_eb1, var_20, adj_eb1, adj_20, adj_21);
    // adj: eb1[1],                                                                           <L 335>
    wp::adj_extract(var_eb1, var_18, adj_eb1, adj_18, adj_19);
    // adj: eb1[0],                                                                           <L 334>
    wp::adj_extract(var_eb0, var_16, adj_eb0, adj_16, adj_17);
    // adj: eb0[2],                                                                           <L 333>
    wp::adj_extract(var_eb0, var_14, adj_eb0, adj_14, adj_15);
    // adj: eb0[1],                                                                           <L 332>
    wp::adj_extract(var_eb0, var_12, adj_eb0, adj_12, adj_13);
    // adj: eb0[0],                                                                           <L 331>
    wp::adj_extract(var_ea1, var_10, adj_ea1, adj_10, adj_11);
    // adj: ea1[2],                                                                           <L 330>
    wp::adj_extract(var_ea1, var_8, adj_ea1, adj_8, adj_9);
    // adj: ea1[1],                                                                           <L 329>
    wp::adj_extract(var_ea1, var_6, adj_ea1, adj_6, adj_7);
    // adj: ea1[0],                                                                           <L 328>
    wp::adj_extract(var_ea0, var_4, adj_ea0, adj_4, adj_5);
    // adj: ea0[2],                                                                           <L 327>
    wp::adj_extract(var_ea0, var_2, adj_ea0, adj_2, adj_3);
    // adj: ea0[1],                                                                           <L 326>
    wp::adj_extract(var_ea0, var_0, adj_ea0, adj_0, adj_1);
    // adj: ea0[0],                                                                           <L 325>
    // adj: return g_EECN2(                                                                   <L 324>
    // adj: def edge_edge_cross_norm2_gradient(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d):  <L 323>
    return;
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/edge_edge_mullifier.py:374
static CUDA_CALLABLE wp::vec_t<12,wp::float64> edge_edge_mollifier_gradient_0(
    wp::vec_t<3,wp::float64> var_ea0,
    wp::vec_t<3,wp::float64> var_ea1,
    wp::vec_t<3,wp::float64> var_eb0,
    wp::vec_t<3,wp::float64> var_eb1,
    wp::float64 var_eps_x)
{
    //---------
    // primal vars
    wp::float64 var_0;
    bool var_1;
    wp::float64 var_2;
    wp::vec_t<12,wp::float64> var_3;
    wp::vec_t<12,wp::float64> var_4;
    const wp::int32 var_5 = 12;
    wp::vec_t<12,wp::float64> var_6;
    const wp::int32 var_7 = 0;
    const wp::float32 var_8 = 0.0;
    wp::float64 var_9;
    const wp::int32 var_10 = 1;
    const wp::float32 var_11 = 0.0;
    wp::float64 var_12;
    const wp::int32 var_13 = 2;
    const wp::float32 var_14 = 0.0;
    wp::float64 var_15;
    const wp::int32 var_16 = 3;
    const wp::float32 var_17 = 0.0;
    wp::float64 var_18;
    const wp::int32 var_19 = 4;
    const wp::float32 var_20 = 0.0;
    wp::float64 var_21;
    const wp::int32 var_22 = 5;
    const wp::float32 var_23 = 0.0;
    wp::float64 var_24;
    const wp::int32 var_25 = 6;
    const wp::float32 var_26 = 0.0;
    wp::float64 var_27;
    const wp::int32 var_28 = 7;
    const wp::float32 var_29 = 0.0;
    wp::float64 var_30;
    const wp::int32 var_31 = 8;
    const wp::float32 var_32 = 0.0;
    wp::float64 var_33;
    const wp::int32 var_34 = 9;
    const wp::float32 var_35 = 0.0;
    wp::float64 var_36;
    const wp::int32 var_37 = 10;
    const wp::float32 var_38 = 0.0;
    wp::float64 var_39;
    const wp::int32 var_40 = 11;
    const wp::float32 var_41 = 0.0;
    wp::float64 var_42;
    wp::vec_t<12,wp::float64> var_43;
    //---------
    // forward
    // def edge_edge_mollifier_gradient(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d, eps_x: wp.float64):       <L 375>
    // EECrossSqNorm = edge_edge_cross_norm2(ea0, ea1, eb0, eb1)                              <L 376>
    var_0 = edge_edge_cross_norm2_0(var_ea0, var_ea1, var_eb0, var_eb1);
    // if EECrossSqNorm < eps_x:                                                              <L 377>
    var_1 = (var_0 < var_eps_x);
    if (var_1) {
        // q_g = g_EEM(EECrossSqNorm, eps_x)                                                  <L 378>
        var_2 = g_EEM_0(var_0, var_eps_x);
        // grad = q_g * edge_edge_cross_norm2_gradient(ea0, ea1, eb0, eb1)                    <L 379>
        var_3 = edge_edge_cross_norm2_gradient_0(var_ea0, var_ea1, var_eb0, var_eb1);
        var_4 = wp::mul(var_2, var_3);
        // return grad                                                                        <L 380>
        return var_4;
    }
    if (!var_1) {
        // grad = wp.vec(length=12, dtype=wp.float64)                                         <L 382>
        var_6 = wp::vec_t<12,wp::float64>();
        // for i in range(12):                                                                <L 383>
        // grad[i] = wp.float64(0.0)                                                          <L 384>
        var_9 = wp::float64(var_8);
        wp::assign_inplace(var_6, var_7, var_9);
        var_12 = wp::float64(var_11);
        wp::assign_inplace(var_6, var_10, var_12);
        var_15 = wp::float64(var_14);
        wp::assign_inplace(var_6, var_13, var_15);
        var_18 = wp::float64(var_17);
        wp::assign_inplace(var_6, var_16, var_18);
        var_21 = wp::float64(var_20);
        wp::assign_inplace(var_6, var_19, var_21);
        var_24 = wp::float64(var_23);
        wp::assign_inplace(var_6, var_22, var_24);
        var_27 = wp::float64(var_26);
        wp::assign_inplace(var_6, var_25, var_27);
        var_30 = wp::float64(var_29);
        wp::assign_inplace(var_6, var_28, var_30);
        var_33 = wp::float64(var_32);
        wp::assign_inplace(var_6, var_31, var_33);
        var_36 = wp::float64(var_35);
        wp::assign_inplace(var_6, var_34, var_36);
        var_39 = wp::float64(var_38);
        wp::assign_inplace(var_6, var_37, var_39);
        var_42 = wp::float64(var_41);
        wp::assign_inplace(var_6, var_40, var_42);
        // return grad                                                                        <L 385>
        return var_6;
    }
    var_43 = wp::where(var_1, var_4, var_6);
    return {};
}


// /home/changyu/actions-runner/_work/Taccel-Action/Taccel-Action/turbo/warp_ipc/contact/edge_edge_mullifier.py:374
static CUDA_CALLABLE void adj_edge_edge_mollifier_gradient_0(
    wp::vec_t<3,wp::float64> var_ea0,
    wp::vec_t<3,wp::float64> var_ea1,
    wp::vec_t<3,wp::float64> var_eb0,
    wp::vec_t<3,wp::float64> var_eb1,
    wp::float64 var_eps_x,
    wp::vec_t<3,wp::float64> & adj_ea0,
    wp::vec_t<3,wp::float64> & adj_ea1,
    wp::vec_t<3,wp::float64> & adj_eb0,
    wp::vec_t<3,wp::float64> & adj_eb1,
    wp::float64 & adj_eps_x,
    wp::vec_t<12,wp::float64> & adj_ret)
{
    //---------
    // primal vars
    wp::float64 var_0;
    bool var_1;
    wp::float64 var_2;
    wp::vec_t<12,wp::float64> var_3;
    wp::vec_t<12,wp::float64> var_4;
    const wp::int32 var_5 = 12;
    wp::vec_t<12,wp::float64> var_6;
    const wp::int32 var_7 = 0;
    const wp::float32 var_8 = 0.0;
    wp::float64 var_9;
    const wp::int32 var_10 = 1;
    const wp::float32 var_11 = 0.0;
    wp::float64 var_12;
    const wp::int32 var_13 = 2;
    const wp::float32 var_14 = 0.0;
    wp::float64 var_15;
    const wp::int32 var_16 = 3;
    const wp::float32 var_17 = 0.0;
    wp::float64 var_18;
    const wp::int32 var_19 = 4;
    const wp::float32 var_20 = 0.0;
    wp::float64 var_21;
    const wp::int32 var_22 = 5;
    const wp::float32 var_23 = 0.0;
    wp::float64 var_24;
    const wp::int32 var_25 = 6;
    const wp::float32 var_26 = 0.0;
    wp::float64 var_27;
    const wp::int32 var_28 = 7;
    const wp::float32 var_29 = 0.0;
    wp::float64 var_30;
    const wp::int32 var_31 = 8;
    const wp::float32 var_32 = 0.0;
    wp::float64 var_33;
    const wp::int32 var_34 = 9;
    const wp::float32 var_35 = 0.0;
    wp::float64 var_36;
    const wp::int32 var_37 = 10;
    const wp::float32 var_38 = 0.0;
    wp::float64 var_39;
    const wp::int32 var_40 = 11;
    const wp::float32 var_41 = 0.0;
    wp::float64 var_42;
    wp::vec_t<12,wp::float64> var_43;
    //---------
    // dual vars
    wp::float64 adj_0 = {};
    bool adj_1 = {};
    wp::float64 adj_2 = {};
    wp::vec_t<12,wp::float64> adj_3 = {};
    wp::vec_t<12,wp::float64> adj_4 = {};
    wp::int32 adj_5 = {};
    wp::vec_t<12,wp::float64> adj_6 = {};
    wp::int32 adj_7 = {};
    wp::float32 adj_8 = {};
    wp::float64 adj_9 = {};
    wp::int32 adj_10 = {};
    wp::float32 adj_11 = {};
    wp::float64 adj_12 = {};
    wp::int32 adj_13 = {};
    wp::float32 adj_14 = {};
    wp::float64 adj_15 = {};
    wp::int32 adj_16 = {};
    wp::float32 adj_17 = {};
    wp::float64 adj_18 = {};
    wp::int32 adj_19 = {};
    wp::float32 adj_20 = {};
    wp::float64 adj_21 = {};
    wp::int32 adj_22 = {};
    wp::float32 adj_23 = {};
    wp::float64 adj_24 = {};
    wp::int32 adj_25 = {};
    wp::float32 adj_26 = {};
    wp::float64 adj_27 = {};
    wp::int32 adj_28 = {};
    wp::float32 adj_29 = {};
    wp::float64 adj_30 = {};
    wp::int32 adj_31 = {};
    wp::float32 adj_32 = {};
    wp::float64 adj_33 = {};
    wp::int32 adj_34 = {};
    wp::float32 adj_35 = {};
    wp::float64 adj_36 = {};
    wp::int32 adj_37 = {};
    wp::float32 adj_38 = {};
    wp::float64 adj_39 = {};
    wp::int32 adj_40 = {};
    wp::float32 adj_41 = {};
    wp::float64 adj_42 = {};
    wp::vec_t<12,wp::float64> adj_43 = {};
    //---------
    // forward
    // def edge_edge_mollifier_gradient(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d, eps_x: wp.float64):       <L 375>
    // EECrossSqNorm = edge_edge_cross_norm2(ea0, ea1, eb0, eb1)                              <L 376>
    var_0 = edge_edge_cross_norm2_0(var_ea0, var_ea1, var_eb0, var_eb1);
    // if EECrossSqNorm < eps_x:                                                              <L 377>
    var_1 = (var_0 < var_eps_x);
    if (var_1) {
        // q_g = g_EEM(EECrossSqNorm, eps_x)                                                  <L 378>
        var_2 = g_EEM_0(var_0, var_eps_x);
        // grad = q_g * edge_edge_cross_norm2_gradient(ea0, ea1, eb0, eb1)                    <L 379>
        var_3 = edge_edge_cross_norm2_gradient_0(var_ea0, var_ea1, var_eb0, var_eb1);
        var_4 = wp::mul(var_2, var_3);
        // return grad                                                                        <L 380>
        goto label0;
    }
    if (!var_1) {
        // grad = wp.vec(length=12, dtype=wp.float64)                                         <L 382>
        var_6 = wp::vec_t<12,wp::float64>();
        // for i in range(12):                                                                <L 383>
        // grad[i] = wp.float64(0.0)                                                          <L 384>
        var_9 = wp::float64(var_8);
        wp::assign_inplace(var_6, var_7, var_9);
        var_12 = wp::float64(var_11);
        wp::assign_inplace(var_6, var_10, var_12);
        var_15 = wp::float64(var_14);
        wp::assign_inplace(var_6, var_13, var_15);
        var_18 = wp::float64(var_17);
        wp::assign_inplace(var_6, var_16, var_18);
        var_21 = wp::float64(var_20);
        wp::assign_inplace(var_6, var_19, var_21);
        var_24 = wp::float64(var_23);
        wp::assign_inplace(var_6, var_22, var_24);
        var_27 = wp::float64(var_26);
        wp::assign_inplace(var_6, var_25, var_27);
        var_30 = wp::float64(var_29);
        wp::assign_inplace(var_6, var_28, var_30);
        var_33 = wp::float64(var_32);
        wp::assign_inplace(var_6, var_31, var_33);
        var_36 = wp::float64(var_35);
        wp::assign_inplace(var_6, var_34, var_36);
        var_39 = wp::float64(var_38);
        wp::assign_inplace(var_6, var_37, var_39);
        var_42 = wp::float64(var_41);
        wp::assign_inplace(var_6, var_40, var_42);
        // return grad                                                                        <L 385>
        goto label1;
    }
    var_43 = wp::where(var_1, var_4, var_6);
    //---------
    // reverse
    wp::adj_where(var_1, var_4, var_6, adj_1, adj_4, adj_6, adj_43);
    if (!var_1) {
        label1:;
        adj_6 += adj_ret;
        // adj: return grad                                                                   <L 385>
        wp::adj_assign_inplace(var_6, var_40, var_42, adj_6, adj_40, adj_42);
        wp::adj_float64(var_41, adj_41, adj_42);
        wp::adj_assign_inplace(var_6, var_37, var_39, adj_6, adj_37, adj_39);
        wp::adj_float64(var_38, adj_38, adj_39);
        wp::adj_assign_inplace(var_6, var_34, var_36, adj_6, adj_34, adj_36);
        wp::adj_float64(var_35, adj_35, adj_36);
        wp::adj_assign_inplace(var_6, var_31, var_33, adj_6, adj_31, adj_33);
        wp::adj_float64(var_32, adj_32, adj_33);
        wp::adj_assign_inplace(var_6, var_28, var_30, adj_6, adj_28, adj_30);
        wp::adj_float64(var_29, adj_29, adj_30);
        wp::adj_assign_inplace(var_6, var_25, var_27, adj_6, adj_25, adj_27);
        wp::adj_float64(var_26, adj_26, adj_27);
        wp::adj_assign_inplace(var_6, var_22, var_24, adj_6, adj_22, adj_24);
        wp::adj_float64(var_23, adj_23, adj_24);
        wp::adj_assign_inplace(var_6, var_19, var_21, adj_6, adj_19, adj_21);
        wp::adj_float64(var_20, adj_20, adj_21);
        wp::adj_assign_inplace(var_6, var_16, var_18, adj_6, adj_16, adj_18);
        wp::adj_float64(var_17, adj_17, adj_18);
        wp::adj_assign_inplace(var_6, var_13, var_15, adj_6, adj_13, adj_15);
        wp::adj_float64(var_14, adj_14, adj_15);
        wp::adj_assign_inplace(var_6, var_10, var_12, adj_6, adj_10, adj_12);
        wp::adj_float64(var_11, adj_11, adj_12);
        wp::adj_assign_inplace(var_6, var_7, var_9, adj_6, adj_7, adj_9);
        wp::adj_float64(var_8, adj_8, adj_9);
        // adj: grad[i] = wp.float64(0.0)                                                     <L 384>
        // adj: for i in range(12):                                                           <L 383>
        // adj: grad = wp.vec(length=12, dtype=wp.float64)                                    <L 382>
    }
    if (var_1) {
        label0:;
        adj_4 += adj_ret;
        // adj: return grad                                                                   <L 380>
        wp::adj_mul(var_2, var_3, adj_2, adj_3, adj_4);
        adj_edge_edge_cross_norm2_gradient_0(var_ea0, var_ea1, var_eb0, var_eb1, adj_ea0, adj_ea1, adj_eb0, adj_eb1, adj_3);
        // adj: grad = q_g * edge_edge_cross_norm2_gradient(ea0, ea1, eb0, eb1)               <L 379>
        adj_g_EEM_0(var_0, var_eps_x, adj_0, adj_eps_x, adj_2);
        // adj: q_g = g_EEM(EECrossSqNorm, eps_x)                                             <L 378>
    }
    // adj: if EECrossSqNorm < eps_x:                                                         <L 377>
    adj_edge_edge_cross_norm2_0(var_ea0, var_ea1, var_eb0, var_eb1, adj_ea0, adj_ea1, adj_eb0, adj_eb1, adj_0);
    // adj: EECrossSqNorm = edge_edge_cross_norm2(ea0, ea1, eb0, eb1)                         <L 376>
    // adj: def edge_edge_mollifier_gradient(ea0: wp.vec3d, ea1: wp.vec3d, eb0: wp.vec3d, eb1: wp.vec3d, eps_x: wp.float64):  <L 375>
    return;
}



extern "C" __global__ void grad_IPC_hs_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float64>> var_gradient_x,
    wp::array_t<wp::float64> var_hs_lambda,
    wp::array_t<wp::int32> var_hs_node,
    wp::array_t<wp::int32> var_hs_ground,
    wp::array_t<wp::vec_t<3,wp::float64>> var_x,
    wp::array_t<wp::int32> var_surf_vi,
    wp::array_t<wp::vec_t<3,wp::float64>> var_hat_x,
    wp::array_t<wp::float64> var_node_xi,
    wp::array_t<wp::float64> var_node_area,
    wp::array_t<wp::vec_t<3,wp::float64>> var_half_space_n,
    wp::array_t<wp::vec_t<3,wp::float64>> var_half_space_o,
    wp::array_t<wp::float64> var_half_space_mu,
    wp::float64 var_dhat,
    wp::float64 var_hat_h,
    wp::float64 var_kappa,
    wp::float64 var_scale,
    wp::float64 var_epsv,
    wp::int32 var_E,
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
        wp::int32* var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        wp::float64* var_4;
        wp::float64 var_5;
        wp::float64 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 1;
        bool var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 2;
        bool var_20;
        wp::int32 var_21;
        bool var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::vec_t<3,wp::float64>* var_26;
        wp::vec_t<3,wp::float64> var_27;
        wp::vec_t<3,wp::float64> var_28;
        wp::vec_t<3,wp::float64>* var_29;
        wp::vec_t<3,wp::float64> var_30;
        wp::vec_t<3,wp::float64> var_31;
        wp::vec_t<3,wp::float64>* var_32;
        wp::vec_t<3,wp::float64> var_33;
        wp::vec_t<3,wp::float64> var_34;
        wp::vec_t<3,wp::float64>* var_35;
        wp::vec_t<3,wp::float64> var_36;
        wp::vec_t<3,wp::float64> var_37;
        wp::vec_t<3,wp::float64> var_38;
        wp::float64 var_39;
        wp::float64 var_40;
        wp::vec_t<3,wp::float64> var_41;
        const wp::int32 var_42 = 0;
        bool var_43;
        wp::float64* var_44;
        wp::float64 var_45;
        wp::float64 var_46;
        wp::float64 var_47;
        wp::float64 var_48;
        wp::vec_t<3,wp::float64> var_49;
        wp::vec_t<3,wp::float64> var_50;
        wp::vec_t<3,wp::float64> var_51;
        wp::float64* var_52;
        wp::float64* var_53;
        wp::float64 var_54;
        wp::float64 var_55;
        wp::float64 var_56;
        wp::float64 var_57;
        wp::vec_t<3,wp::float64> var_58;
        wp::vec_t<3,wp::float64> var_59;
        wp::vec_t<3,wp::float64> var_60;
        //---------
        // forward
        // def grad_IPC_hs(                                                                       <L 18>
        // idx = wp.tid()                                                                         <L 40>
        var_0 = builtin_tid1d();
        // svi = hs_node[idx]                                                                     <L 42>
        var_1 = wp::address(var_hs_node, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // xi = node_xi[svi]                                                                      <L 43>
        var_4 = wp::address(var_node_xi, var_2);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // xI = surf_vi[svi]                                                                      <L 44>
        var_7 = wp::address(var_surf_vi, var_2);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // if (env_states[node2env[xI]] == ENV_STATE_INVALID) or (env_states[node2env[xI]] == ENV_STATE_NEWTON_SOLVED):       <L 45>
        var_10 = wp::address(var_node2env, var_8);
        var_12 = wp::load(var_10);
        var_11 = wp::address(var_env_states, var_12);
        var_15 = wp::load(var_11);
        var_14 = (var_15 == var_13);
        var_16 = wp::address(var_node2env, var_8);
        var_18 = wp::load(var_16);
        var_17 = wp::address(var_env_states, var_18);
        var_21 = wp::load(var_17);
        var_20 = (var_21 == var_19);
        var_22 = var_14 || var_20;
        if (var_22) {
            // return                                                                             <L 46>
            continue;
        }
        // hs = hs_ground[idx]                                                                    <L 47>
        var_23 = wp::address(var_hs_ground, var_0);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // n = half_space_n[hs]                                                                   <L 48>
        var_26 = wp::address(var_half_space_n, var_24);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // o = half_space_o[hs]                                                                   <L 49>
        var_29 = wp::address(var_half_space_o, var_24);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // xIcoord = x[xI]                                                                        <L 50>
        var_32 = wp::address(var_x, var_8);
        var_34 = wp::load(var_32);
        var_33 = wp::copy(var_34);
        // hat_xIcoord = hat_x[xI]                                                                <L 51>
        var_35 = wp::address(var_hat_x, var_8);
        var_37 = wp::load(var_35);
        var_36 = wp::copy(var_37);
        // d = wp.dot(xIcoord - o, n) - xi                                                        <L 52>
        var_38 = wp::sub(var_33, var_30);
        var_39 = wp::dot(var_38, var_27);
        var_40 = wp::sub(var_39, var_5);
        // grad_b = wp.vec3d()                                                                    <L 53>
        var_41 = wp::vec_t<3,wp::float64>();
        // if E == BARRIER:                                                                       <L 54>
        var_43 = (var_E == var_42);
        if (var_43) {
            // db_div_dd = node_area[svi] * dhat * barrier.barrier_gradient(d, dhat, kappa)       <L 55>
            var_44 = wp::address(var_node_area, var_2);
            var_46 = wp::load(var_44);
            var_45 = wp::mul(var_46, var_dhat);
            var_47 = barrier_gradient_0(var_40, var_dhat, var_kappa);
            var_48 = wp::mul(var_45, var_47);
            // grad_b = db_div_dd * n * scale                                                     <L 56>
            var_49 = wp::mul(var_48, var_27);
            var_50 = wp::mul(var_49, var_scale);
        }
        var_51 = wp::where(var_43, var_50, var_41);
        if (!var_43) {
            // coeff = half_space_mu[hs] * hs_lambda[idx] * scale                                 <L 58>
            var_52 = wp::address(var_half_space_mu, var_24);
            var_53 = wp::address(var_hs_lambda, var_0);
            var_55 = wp::load(var_52);
            var_56 = wp::load(var_53);
            var_54 = wp::mul(var_55, var_56);
            var_57 = wp::mul(var_54, var_scale);
            // grad_b = friction.friction_gradient(n, xIcoord, hat_xIcoord, hat_h, epsv, coeff)       <L 59>
            var_58 = friction_gradient_0(var_27, var_33, var_36, var_hat_h, var_epsv, var_57);
        }
        var_59 = wp::where(var_43, var_51, var_58);
        // wp.atomic_add(gradient_x, xI, grad_b)                                                  <L 61>
        var_60 = wp::atomic_add(var_gradient_x, var_8, var_59);
    }
}



extern "C" __global__ void grad_IPC_hs_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float64>> var_gradient_x,
    wp::array_t<wp::float64> var_hs_lambda,
    wp::array_t<wp::int32> var_hs_node,
    wp::array_t<wp::int32> var_hs_ground,
    wp::array_t<wp::vec_t<3,wp::float64>> var_x,
    wp::array_t<wp::int32> var_surf_vi,
    wp::array_t<wp::vec_t<3,wp::float64>> var_hat_x,
    wp::array_t<wp::float64> var_node_xi,
    wp::array_t<wp::float64> var_node_area,
    wp::array_t<wp::vec_t<3,wp::float64>> var_half_space_n,
    wp::array_t<wp::vec_t<3,wp::float64>> var_half_space_o,
    wp::array_t<wp::float64> var_half_space_mu,
    wp::float64 var_dhat,
    wp::float64 var_hat_h,
    wp::float64 var_kappa,
    wp::float64 var_scale,
    wp::float64 var_epsv,
    wp::int32 var_E,
    wp::array_t<wp::int32> var_node2env,
    wp::array_t<wp::int32> var_env_states,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_gradient_x,
    wp::array_t<wp::float64> adj_hs_lambda,
    wp::array_t<wp::int32> adj_hs_node,
    wp::array_t<wp::int32> adj_hs_ground,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_x,
    wp::array_t<wp::int32> adj_surf_vi,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_hat_x,
    wp::array_t<wp::float64> adj_node_xi,
    wp::array_t<wp::float64> adj_node_area,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_half_space_n,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_half_space_o,
    wp::array_t<wp::float64> adj_half_space_mu,
    wp::float64 adj_dhat,
    wp::float64 adj_hat_h,
    wp::float64 adj_kappa,
    wp::float64 adj_scale,
    wp::float64 adj_epsv,
    wp::int32 adj_E,
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
        wp::int32* var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        wp::float64* var_4;
        wp::float64 var_5;
        wp::float64 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 1;
        bool var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 2;
        bool var_20;
        wp::int32 var_21;
        bool var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::vec_t<3,wp::float64>* var_26;
        wp::vec_t<3,wp::float64> var_27;
        wp::vec_t<3,wp::float64> var_28;
        wp::vec_t<3,wp::float64>* var_29;
        wp::vec_t<3,wp::float64> var_30;
        wp::vec_t<3,wp::float64> var_31;
        wp::vec_t<3,wp::float64>* var_32;
        wp::vec_t<3,wp::float64> var_33;
        wp::vec_t<3,wp::float64> var_34;
        wp::vec_t<3,wp::float64>* var_35;
        wp::vec_t<3,wp::float64> var_36;
        wp::vec_t<3,wp::float64> var_37;
        wp::vec_t<3,wp::float64> var_38;
        wp::float64 var_39;
        wp::float64 var_40;
        wp::vec_t<3,wp::float64> var_41;
        const wp::int32 var_42 = 0;
        bool var_43;
        wp::float64* var_44;
        wp::float64 var_45;
        wp::float64 var_46;
        wp::float64 var_47;
        wp::float64 var_48;
        wp::vec_t<3,wp::float64> var_49;
        wp::vec_t<3,wp::float64> var_50;
        wp::vec_t<3,wp::float64> var_51;
        wp::float64* var_52;
        wp::float64* var_53;
        wp::float64 var_54;
        wp::float64 var_55;
        wp::float64 var_56;
        wp::float64 var_57;
        wp::vec_t<3,wp::float64> var_58;
        wp::vec_t<3,wp::float64> var_59;
        wp::vec_t<3,wp::float64> var_60;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::float64 adj_4 = {};
        wp::float64 adj_5 = {};
        wp::float64 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        bool adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        bool adj_20 = {};
        wp::int32 adj_21 = {};
        bool adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
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
        wp::float64 adj_39 = {};
        wp::float64 adj_40 = {};
        wp::vec_t<3,wp::float64> adj_41 = {};
        wp::int32 adj_42 = {};
        bool adj_43 = {};
        wp::float64 adj_44 = {};
        wp::float64 adj_45 = {};
        wp::float64 adj_46 = {};
        wp::float64 adj_47 = {};
        wp::float64 adj_48 = {};
        wp::vec_t<3,wp::float64> adj_49 = {};
        wp::vec_t<3,wp::float64> adj_50 = {};
        wp::vec_t<3,wp::float64> adj_51 = {};
        wp::float64 adj_52 = {};
        wp::float64 adj_53 = {};
        wp::float64 adj_54 = {};
        wp::float64 adj_55 = {};
        wp::float64 adj_56 = {};
        wp::float64 adj_57 = {};
        wp::vec_t<3,wp::float64> adj_58 = {};
        wp::vec_t<3,wp::float64> adj_59 = {};
        wp::vec_t<3,wp::float64> adj_60 = {};
        //---------
        // forward
        // def grad_IPC_hs(                                                                       <L 18>
        // idx = wp.tid()                                                                         <L 40>
        var_0 = builtin_tid1d();
        // svi = hs_node[idx]                                                                     <L 42>
        var_1 = wp::address(var_hs_node, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // xi = node_xi[svi]                                                                      <L 43>
        var_4 = wp::address(var_node_xi, var_2);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // xI = surf_vi[svi]                                                                      <L 44>
        var_7 = wp::address(var_surf_vi, var_2);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // if (env_states[node2env[xI]] == ENV_STATE_INVALID) or (env_states[node2env[xI]] == ENV_STATE_NEWTON_SOLVED):       <L 45>
        var_10 = wp::address(var_node2env, var_8);
        var_12 = wp::load(var_10);
        var_11 = wp::address(var_env_states, var_12);
        var_15 = wp::load(var_11);
        var_14 = (var_15 == var_13);
        var_16 = wp::address(var_node2env, var_8);
        var_18 = wp::load(var_16);
        var_17 = wp::address(var_env_states, var_18);
        var_21 = wp::load(var_17);
        var_20 = (var_21 == var_19);
        var_22 = var_14 || var_20;
        if (var_22) {
            // return                                                                             <L 46>
            goto label0;
        }
        // hs = hs_ground[idx]                                                                    <L 47>
        var_23 = wp::address(var_hs_ground, var_0);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // n = half_space_n[hs]                                                                   <L 48>
        var_26 = wp::address(var_half_space_n, var_24);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // o = half_space_o[hs]                                                                   <L 49>
        var_29 = wp::address(var_half_space_o, var_24);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // xIcoord = x[xI]                                                                        <L 50>
        var_32 = wp::address(var_x, var_8);
        var_34 = wp::load(var_32);
        var_33 = wp::copy(var_34);
        // hat_xIcoord = hat_x[xI]                                                                <L 51>
        var_35 = wp::address(var_hat_x, var_8);
        var_37 = wp::load(var_35);
        var_36 = wp::copy(var_37);
        // d = wp.dot(xIcoord - o, n) - xi                                                        <L 52>
        var_38 = wp::sub(var_33, var_30);
        var_39 = wp::dot(var_38, var_27);
        var_40 = wp::sub(var_39, var_5);
        // grad_b = wp.vec3d()                                                                    <L 53>
        var_41 = wp::vec_t<3,wp::float64>();
        // if E == BARRIER:                                                                       <L 54>
        var_43 = (var_E == var_42);
        if (var_43) {
            // db_div_dd = node_area[svi] * dhat * barrier.barrier_gradient(d, dhat, kappa)       <L 55>
            var_44 = wp::address(var_node_area, var_2);
            var_46 = wp::load(var_44);
            var_45 = wp::mul(var_46, var_dhat);
            var_47 = barrier_gradient_0(var_40, var_dhat, var_kappa);
            var_48 = wp::mul(var_45, var_47);
            // grad_b = db_div_dd * n * scale                                                     <L 56>
            var_49 = wp::mul(var_48, var_27);
            var_50 = wp::mul(var_49, var_scale);
        }
        var_51 = wp::where(var_43, var_50, var_41);
        if (!var_43) {
            // coeff = half_space_mu[hs] * hs_lambda[idx] * scale                                 <L 58>
            var_52 = wp::address(var_half_space_mu, var_24);
            var_53 = wp::address(var_hs_lambda, var_0);
            var_55 = wp::load(var_52);
            var_56 = wp::load(var_53);
            var_54 = wp::mul(var_55, var_56);
            var_57 = wp::mul(var_54, var_scale);
            // grad_b = friction.friction_gradient(n, xIcoord, hat_xIcoord, hat_h, epsv, coeff)       <L 59>
            var_58 = friction_gradient_0(var_27, var_33, var_36, var_hat_h, var_epsv, var_57);
        }
        var_59 = wp::where(var_43, var_51, var_58);
        // wp.atomic_add(gradient_x, xI, grad_b)                                                  <L 61>
        // var_60 = wp::atomic_add(var_gradient_x, var_8, var_59);
        //---------
        // reverse
        wp::adj_atomic_add(var_gradient_x, var_8, var_59, adj_gradient_x, adj_8, adj_59, adj_60);
        // adj: wp.atomic_add(gradient_x, xI, grad_b)                                             <L 61>
        wp::adj_where(var_43, var_51, var_58, adj_43, adj_51, adj_58, adj_59);
        if (!var_43) {
            adj_friction_gradient_0(var_27, var_33, var_36, var_hat_h, var_epsv, var_57, adj_27, adj_33, adj_36, adj_hat_h, adj_epsv, adj_57, adj_58);
            // adj: grad_b = friction.friction_gradient(n, xIcoord, hat_xIcoord, hat_h, epsv, coeff)  <L 59>
            wp::adj_mul(var_54, var_scale, adj_54, adj_scale, adj_57);
            wp::adj_mul(var_55, var_56, adj_52, adj_53, adj_54);
            wp::adj_load(var_53, adj_53, adj_56);
            wp::adj_load(var_52, adj_52, adj_55);
            wp::adj_address(var_hs_lambda, var_0, adj_hs_lambda, adj_0, adj_53);
            wp::adj_address(var_half_space_mu, var_24, adj_half_space_mu, adj_24, adj_52);
            // adj: coeff = half_space_mu[hs] * hs_lambda[idx] * scale                            <L 58>
        }
        wp::adj_where(var_43, var_50, var_41, adj_43, adj_50, adj_41, adj_51);
        if (var_43) {
            wp::adj_mul(var_49, var_scale, adj_49, adj_scale, adj_50);
            wp::adj_mul(var_48, var_27, adj_48, adj_27, adj_49);
            // adj: grad_b = db_div_dd * n * scale                                                <L 56>
            wp::adj_mul(var_45, var_47, adj_45, adj_47, adj_48);
            adj_barrier_gradient_0(var_40, var_dhat, var_kappa, adj_40, adj_dhat, adj_kappa, adj_47);
            wp::adj_mul(var_46, var_dhat, adj_44, adj_dhat, adj_45);
            wp::adj_load(var_44, adj_44, adj_46);
            wp::adj_address(var_node_area, var_2, adj_node_area, adj_2, adj_44);
            // adj: db_div_dd = node_area[svi] * dhat * barrier.barrier_gradient(d, dhat, kappa)  <L 55>
        }
        // adj: if E == BARRIER:                                                                  <L 54>
        // adj: grad_b = wp.vec3d()                                                               <L 53>
        wp::adj_sub(var_39, var_5, adj_39, adj_5, adj_40);
        wp::adj_dot(var_38, var_27, adj_38, adj_27, adj_39);
        wp::adj_sub(var_33, var_30, adj_33, adj_30, adj_38);
        // adj: d = wp.dot(xIcoord - o, n) - xi                                                   <L 52>
        wp::adj_copy(var_37, adj_35, adj_36);
        wp::adj_load(var_35, adj_35, adj_37);
        wp::adj_address(var_hat_x, var_8, adj_hat_x, adj_8, adj_35);
        // adj: hat_xIcoord = hat_x[xI]                                                           <L 51>
        wp::adj_copy(var_34, adj_32, adj_33);
        wp::adj_load(var_32, adj_32, adj_34);
        wp::adj_address(var_x, var_8, adj_x, adj_8, adj_32);
        // adj: xIcoord = x[xI]                                                                   <L 50>
        wp::adj_copy(var_31, adj_29, adj_30);
        wp::adj_load(var_29, adj_29, adj_31);
        wp::adj_address(var_half_space_o, var_24, adj_half_space_o, adj_24, adj_29);
        // adj: o = half_space_o[hs]                                                              <L 49>
        wp::adj_copy(var_28, adj_26, adj_27);
        wp::adj_load(var_26, adj_26, adj_28);
        wp::adj_address(var_half_space_n, var_24, adj_half_space_n, adj_24, adj_26);
        // adj: n = half_space_n[hs]                                                              <L 48>
        wp::adj_copy(var_25, adj_23, adj_24);
        wp::adj_load(var_23, adj_23, adj_25);
        wp::adj_address(var_hs_ground, var_0, adj_hs_ground, adj_0, adj_23);
        // adj: hs = hs_ground[idx]                                                               <L 47>
        if (var_22) {
            label0:;
            // adj: return                                                                        <L 46>
        }
        wp::adj_load(var_17, adj_17, adj_21);
        wp::adj_address(var_env_states, var_18, adj_env_states, adj_16, adj_17);
        wp::adj_load(var_16, adj_16, adj_18);
        wp::adj_address(var_node2env, var_8, adj_node2env, adj_8, adj_16);
        wp::adj_load(var_11, adj_11, adj_15);
        wp::adj_address(var_env_states, var_12, adj_env_states, adj_10, adj_11);
        wp::adj_load(var_10, adj_10, adj_12);
        wp::adj_address(var_node2env, var_8, adj_node2env, adj_8, adj_10);
        // adj: if (env_states[node2env[xI]] == ENV_STATE_INVALID) or (env_states[node2env[xI]] == ENV_STATE_NEWTON_SOLVED):  <L 45>
        wp::adj_copy(var_9, adj_7, adj_8);
        wp::adj_load(var_7, adj_7, adj_9);
        wp::adj_address(var_surf_vi, var_2, adj_surf_vi, adj_2, adj_7);
        // adj: xI = surf_vi[svi]                                                                 <L 44>
        wp::adj_copy(var_6, adj_4, adj_5);
        wp::adj_load(var_4, adj_4, adj_6);
        wp::adj_address(var_node_xi, var_2, adj_node_xi, adj_2, adj_4);
        // adj: xi = node_xi[svi]                                                                 <L 43>
        wp::adj_copy(var_3, adj_1, adj_2);
        wp::adj_load(var_1, adj_1, adj_3);
        wp::adj_address(var_hs_node, var_0, adj_hs_node, adj_0, adj_1);
        // adj: svi = hs_node[idx]                                                                <L 42>
        // adj: idx = wp.tid()                                                                    <L 40>
        // adj: def grad_IPC_hs(                                                                  <L 18>
        continue;
    }
}



extern "C" __global__ void grad_IPC_collisions_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float64>> var_gradient_x,
    wp::array_t<wp::float64> var_c_lambda,
    wp::array_t<wp::vec_t<2,wp::float64>> var_closest_points,
    wp::array_t<wp::vec_t<3,wp::float64>> var_normal,
    wp::array_t<wp::int32> var_nodeI,
    wp::array_t<wp::int32> var_nodeJ,
    wp::array_t<wp::int32> var_bodyI,
    wp::array_t<wp::int32> var_bodyJ,
    wp::array_t<wp::int32> var_collision_type,
    wp::array_t<wp::vec_t<3,wp::float64>> var_x,
    wp::array_t<wp::int32> var_surf_vi,
    wp::array_t<wp::vec_t<3,wp::float64>> var_hat_x,
    wp::array_t<wp::vec_t<3,wp::float64>> var_X,
    wp::array_t<wp::float64> var_mu_body,
    wp::array_t<wp::float64> var_node_xi,
    wp::array_t<wp::float64> var_edge_xi,
    wp::array_t<wp::float64> var_face_xi,
    wp::array_t<wp::float64> var_node_area,
    wp::array_t<wp::float64> var_edge_area,
    wp::array_t<wp::vec_t<2,wp::int32>> var_edge,
    wp::array_t<wp::vec_t<3,wp::int32>> var_face,
    wp::float64 var_dhat,
    wp::float64 var_hat_h,
    wp::float64 var_kappa,
    wp::float64 var_int_w_PTEE,
    wp::float64 var_scale,
    wp::float64 var_epsv,
    wp::int32 var_E,
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
        const wp::int32 var_1 = 12;
        wp::vec_t<12,wp::float64> var_2;
        wp::vec_t<4,wp::int32> var_3;
        wp::int32* var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        const wp::int32 var_9 = 1;
        bool var_10;
        wp::int32 var_11;
        bool var_12;
        wp::int32* var_13;
        bool var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32* var_21;
        bool var_22;
        wp::int32 var_23;
        wp::int32* var_24;
        wp::int32* var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::vec_t<3,wp::int32>* var_29;
        wp::vec_t<3,wp::int32> var_30;
        wp::vec_t<3,wp::int32> var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        wp::int32* var_35;
        wp::int32* var_36;
        wp::int32 var_37;
        const wp::int32 var_38 = 1;
        bool var_39;
        wp::int32 var_40;
        wp::int32* var_41;
        wp::int32* var_42;
        wp::int32 var_43;
        const wp::int32 var_44 = 2;
        bool var_45;
        wp::int32 var_46;
        bool var_47;
        const wp::int32 var_48 = 0;
        wp::int32 var_49;
        const wp::int32 var_50 = 1;
        wp::int32 var_51;
        const wp::int32 var_52 = 2;
        wp::int32 var_53;
        wp::vec_t<4,wp::int32> var_54;
        wp::vec_t<3,wp::float64>* var_55;
        wp::vec_t<3,wp::float64> var_56;
        wp::vec_t<3,wp::float64> var_57;
        const wp::int32 var_58 = 0;
        wp::int32 var_59;
        wp::vec_t<3,wp::float64>* var_60;
        wp::vec_t<3,wp::float64> var_61;
        wp::vec_t<3,wp::float64> var_62;
        const wp::int32 var_63 = 1;
        wp::int32 var_64;
        wp::vec_t<3,wp::float64>* var_65;
        wp::vec_t<3,wp::float64> var_66;
        wp::vec_t<3,wp::float64> var_67;
        const wp::int32 var_68 = 2;
        wp::int32 var_69;
        wp::vec_t<3,wp::float64>* var_70;
        wp::vec_t<3,wp::float64> var_71;
        wp::vec_t<3,wp::float64> var_72;
        wp::vec_t<3,wp::float64>* var_73;
        wp::vec_t<3,wp::float64> var_74;
        wp::vec_t<3,wp::float64> var_75;
        const wp::int32 var_76 = 0;
        wp::int32 var_77;
        wp::vec_t<3,wp::float64>* var_78;
        wp::vec_t<3,wp::float64> var_79;
        wp::vec_t<3,wp::float64> var_80;
        const wp::int32 var_81 = 1;
        wp::int32 var_82;
        wp::vec_t<3,wp::float64>* var_83;
        wp::vec_t<3,wp::float64> var_84;
        wp::vec_t<3,wp::float64> var_85;
        const wp::int32 var_86 = 2;
        wp::int32 var_87;
        wp::vec_t<3,wp::float64>* var_88;
        wp::vec_t<3,wp::float64> var_89;
        wp::vec_t<3,wp::float64> var_90;
        wp::float64* var_91;
        wp::float64* var_92;
        wp::float64 var_93;
        wp::float64 var_94;
        wp::float64 var_95;
        wp::float64 var_96;
        wp::float64 var_97;
        wp::float64 var_98;
        wp::vec_t<12,wp::float64> var_99;
        wp::float64 var_100;
        const wp::float32 var_101 = 2.0;
        wp::float64 var_102;
        wp::float64 var_103;
        wp::float64 var_104;
        wp::float64 var_105;
        const wp::int32 var_106 = 0;
        bool var_107;
        wp::float64 var_108;
        wp::float64* var_109;
        wp::float64 var_110;
        wp::float64 var_111;
        wp::float64 var_112;
        wp::float64 var_113;
        wp::float64 var_114;
        wp::vec_t<12,wp::float64> var_115;
        wp::vec_t<12,wp::float64> var_116;
        const wp::int32 var_117 = 0;
        wp::vec_t<2,wp::float64>* var_118;
        wp::float64 var_119;
        wp::vec_t<2,wp::float64> var_120;
        const wp::int32 var_121 = 1;
        wp::vec_t<2,wp::float64>* var_122;
        wp::float64 var_123;
        wp::vec_t<2,wp::float64> var_124;
        const wp::float32 var_125 = 1.0;
        wp::float64 var_126;
        wp::float64 var_127;
        wp::float64 var_128;
        wp::vec_t<3,wp::float64> var_129;
        wp::vec_t<3,wp::float64> var_130;
        wp::vec_t<3,wp::float64> var_131;
        wp::vec_t<3,wp::float64> var_132;
        wp::vec_t<3,wp::float64> var_133;
        wp::vec_t<3,wp::float64> var_134;
        const wp::float32 var_135 = 1.0;
        wp::float64 var_136;
        wp::float64 var_137;
        wp::float64 var_138;
        wp::vec_t<3,wp::float64> var_139;
        wp::vec_t<3,wp::float64> var_140;
        wp::vec_t<3,wp::float64> var_141;
        wp::vec_t<3,wp::float64> var_142;
        wp::vec_t<3,wp::float64> var_143;
        wp::vec_t<3,wp::float64> var_144;
        const wp::float32 var_145 = 1.0;
        wp::float64 var_146;
        const wp::float32 var_147 = 1.0;
        wp::float64 var_148;
        wp::float64 var_149;
        wp::float64 var_150;
        wp::float64 var_151;
        wp::float64 var_152;
        wp::float64 var_153;
        wp::vec_t<4,wp::float64> var_154;
        wp::int32* var_155;
        wp::float64* var_156;
        wp::int32 var_157;
        wp::int32* var_158;
        wp::float64* var_159;
        wp::int32 var_160;
        wp::float64 var_161;
        wp::float64 var_162;
        wp::float64 var_163;
        wp::float64* var_164;
        wp::float64 var_165;
        wp::float64 var_166;
        wp::float64 var_167;
        wp::vec_t<3,wp::float64>* var_168;
        wp::vec_t<3,wp::float64> var_169;
        wp::vec_t<3,wp::float64> var_170;
        const wp::int32 var_171 = 0;
        wp::float64 var_172;
        const wp::int32 var_173 = 0;
        wp::float64 var_174;
        wp::float64 var_175;
        const wp::int32 var_176 = 1;
        wp::float64 var_177;
        const wp::int32 var_178 = 0;
        wp::float64 var_179;
        wp::float64 var_180;
        const wp::int32 var_181 = 2;
        wp::float64 var_182;
        const wp::int32 var_183 = 0;
        wp::float64 var_184;
        wp::float64 var_185;
        const wp::int32 var_186 = 0;
        wp::float64 var_187;
        const wp::int32 var_188 = 1;
        wp::float64 var_189;
        wp::float64 var_190;
        const wp::int32 var_191 = 1;
        wp::float64 var_192;
        const wp::int32 var_193 = 1;
        wp::float64 var_194;
        wp::float64 var_195;
        const wp::int32 var_196 = 2;
        wp::float64 var_197;
        const wp::int32 var_198 = 1;
        wp::float64 var_199;
        wp::float64 var_200;
        const wp::int32 var_201 = 0;
        wp::float64 var_202;
        const wp::int32 var_203 = 2;
        wp::float64 var_204;
        wp::float64 var_205;
        const wp::int32 var_206 = 1;
        wp::float64 var_207;
        const wp::int32 var_208 = 2;
        wp::float64 var_209;
        wp::float64 var_210;
        const wp::int32 var_211 = 2;
        wp::float64 var_212;
        const wp::int32 var_213 = 2;
        wp::float64 var_214;
        wp::float64 var_215;
        const wp::int32 var_216 = 0;
        wp::float64 var_217;
        const wp::int32 var_218 = 3;
        wp::float64 var_219;
        wp::float64 var_220;
        const wp::int32 var_221 = 1;
        wp::float64 var_222;
        const wp::int32 var_223 = 3;
        wp::float64 var_224;
        wp::float64 var_225;
        const wp::int32 var_226 = 2;
        wp::float64 var_227;
        const wp::int32 var_228 = 3;
        wp::float64 var_229;
        wp::float64 var_230;
        wp::vec_t<12,wp::float64> var_231;
        wp::vec_t<12,wp::float64> var_232;
        wp::vec_t<12,wp::float64> var_233;
        wp::vec_t<4,wp::int32> var_234;
        wp::int32* var_235;
        wp::int32 var_236;
        wp::int32 var_237;
        wp::int32* var_238;
        wp::int32 var_239;
        wp::int32 var_240;
        wp::vec_t<2,wp::int32>* var_241;
        wp::vec_t<2,wp::int32> var_242;
        wp::vec_t<2,wp::int32> var_243;
        wp::vec_t<2,wp::int32>* var_244;
        wp::vec_t<2,wp::int32> var_245;
        wp::vec_t<2,wp::int32> var_246;
        const wp::int32 var_247 = 0;
        wp::int32 var_248;
        const wp::int32 var_249 = 1;
        wp::int32 var_250;
        const wp::int32 var_251 = 0;
        wp::int32 var_252;
        const wp::int32 var_253 = 1;
        wp::int32 var_254;
        wp::vec_t<4,wp::int32> var_255;
        wp::float64* var_256;
        wp::float64* var_257;
        wp::float64 var_258;
        wp::float64 var_259;
        wp::float64 var_260;
        const wp::int32 var_261 = 0;
        wp::int32 var_262;
        wp::int32* var_263;
        wp::int32* var_264;
        wp::int32 var_265;
        bool var_266;
        wp::int32 var_267;
        const wp::int32 var_268 = 0;
        wp::int32 var_269;
        wp::int32* var_270;
        wp::int32* var_271;
        wp::int32 var_272;
        bool var_273;
        wp::int32 var_274;
        bool var_275;
        const wp::int32 var_276 = 0;
        wp::int32 var_277;
        wp::vec_t<3,wp::float64>* var_278;
        wp::vec_t<3,wp::float64> var_279;
        wp::vec_t<3,wp::float64> var_280;
        const wp::int32 var_281 = 1;
        wp::int32 var_282;
        wp::vec_t<3,wp::float64>* var_283;
        wp::vec_t<3,wp::float64> var_284;
        wp::vec_t<3,wp::float64> var_285;
        const wp::int32 var_286 = 0;
        wp::int32 var_287;
        wp::vec_t<3,wp::float64>* var_288;
        wp::vec_t<3,wp::float64> var_289;
        wp::vec_t<3,wp::float64> var_290;
        const wp::int32 var_291 = 1;
        wp::int32 var_292;
        wp::vec_t<3,wp::float64>* var_293;
        wp::vec_t<3,wp::float64> var_294;
        wp::vec_t<3,wp::float64> var_295;
        wp::float64 var_296;
        wp::float64 var_297;
        wp::float64 var_298;
        const wp::int32 var_299 = 0;
        wp::int32 var_300;
        wp::vec_t<3,wp::float64>* var_301;
        wp::vec_t<3,wp::float64> var_302;
        wp::vec_t<3,wp::float64> var_303;
        const wp::int32 var_304 = 1;
        wp::int32 var_305;
        wp::vec_t<3,wp::float64>* var_306;
        wp::vec_t<3,wp::float64> var_307;
        wp::vec_t<3,wp::float64> var_308;
        const wp::int32 var_309 = 0;
        wp::int32 var_310;
        wp::vec_t<3,wp::float64>* var_311;
        wp::vec_t<3,wp::float64> var_312;
        wp::vec_t<3,wp::float64> var_313;
        const wp::int32 var_314 = 1;
        wp::int32 var_315;
        wp::vec_t<3,wp::float64>* var_316;
        wp::vec_t<3,wp::float64> var_317;
        wp::vec_t<3,wp::float64> var_318;
        wp::float64 var_319;
        wp::float64 var_320;
        wp::float64 var_321;
        const wp::float32 var_322 = 2.0;
        wp::float64 var_323;
        wp::float64 var_324;
        wp::float64 var_325;
        wp::float64 var_326;
        bool var_327;
        wp::float64 var_328;
        wp::float64* var_329;
        wp::float64* var_330;
        wp::float64 var_331;
        wp::float64 var_332;
        wp::float64 var_333;
        wp::float64 var_334;
        wp::float64 var_335;
        wp::float64 var_336;
        wp::float64 var_337;
        wp::vec_t<12,wp::float64> var_338;
        wp::vec_t<12,wp::float64> var_339;
        const wp::float32 var_340 = 1.0;
        wp::float64 var_341;
        bool var_342;
        wp::float64 var_343;
        wp::float64* var_344;
        wp::float64* var_345;
        wp::float64 var_346;
        wp::float64 var_347;
        wp::float64 var_348;
        wp::float64 var_349;
        wp::float64 var_350;
        wp::float64 var_351;
        wp::float64 var_352;
        wp::vec_t<12,wp::float64> var_353;
        wp::vec_t<12,wp::float64> var_354;
        wp::vec_t<12,wp::float64> var_355;
        wp::vec_t<12,wp::float64> var_356;
        wp::vec_t<12,wp::float64> var_357;
        wp::vec_t<12,wp::float64> var_358;
        wp::float64 var_359;
        const wp::int32 var_360 = 0;
        wp::int32 var_361;
        wp::vec_t<3,wp::float64>* var_362;
        wp::vec_t<3,wp::float64> var_363;
        wp::vec_t<3,wp::float64> var_364;
        const wp::int32 var_365 = 1;
        wp::int32 var_366;
        wp::vec_t<3,wp::float64>* var_367;
        wp::vec_t<3,wp::float64> var_368;
        wp::vec_t<3,wp::float64> var_369;
        const wp::int32 var_370 = 0;
        wp::int32 var_371;
        wp::vec_t<3,wp::float64>* var_372;
        wp::vec_t<3,wp::float64> var_373;
        wp::vec_t<3,wp::float64> var_374;
        const wp::int32 var_375 = 1;
        wp::int32 var_376;
        wp::vec_t<3,wp::float64>* var_377;
        wp::vec_t<3,wp::float64> var_378;
        wp::vec_t<3,wp::float64> var_379;
        const wp::int32 var_380 = 0;
        wp::vec_t<2,wp::float64>* var_381;
        wp::float64 var_382;
        wp::vec_t<2,wp::float64> var_383;
        const wp::int32 var_384 = 1;
        wp::vec_t<2,wp::float64>* var_385;
        wp::float64 var_386;
        wp::vec_t<2,wp::float64> var_387;
        const wp::float32 var_388 = 1.0;
        wp::float64 var_389;
        wp::float64 var_390;
        const wp::int32 var_391 = 0;
        wp::float64 var_392;
        wp::float64 var_393;
        const wp::int32 var_394 = 0;
        wp::float64 var_395;
        wp::float64 var_396;
        wp::float64 var_397;
        const wp::float32 var_398 = 1.0;
        wp::float64 var_399;
        wp::float64 var_400;
        const wp::int32 var_401 = 0;
        wp::float64 var_402;
        wp::float64 var_403;
        wp::float64 var_404;
        const wp::int32 var_405 = 0;
        wp::float64 var_406;
        wp::float64 var_407;
        wp::float64 var_408;
        const wp::float32 var_409 = 1.0;
        wp::float64 var_410;
        wp::float64 var_411;
        const wp::int32 var_412 = 1;
        wp::float64 var_413;
        wp::float64 var_414;
        const wp::int32 var_415 = 1;
        wp::float64 var_416;
        wp::float64 var_417;
        wp::float64 var_418;
        const wp::float32 var_419 = 1.0;
        wp::float64 var_420;
        wp::float64 var_421;
        const wp::int32 var_422 = 1;
        wp::float64 var_423;
        wp::float64 var_424;
        wp::float64 var_425;
        const wp::int32 var_426 = 1;
        wp::float64 var_427;
        wp::float64 var_428;
        wp::float64 var_429;
        const wp::float32 var_430 = 1.0;
        wp::float64 var_431;
        wp::float64 var_432;
        const wp::int32 var_433 = 2;
        wp::float64 var_434;
        wp::float64 var_435;
        const wp::int32 var_436 = 2;
        wp::float64 var_437;
        wp::float64 var_438;
        wp::float64 var_439;
        const wp::float32 var_440 = 1.0;
        wp::float64 var_441;
        wp::float64 var_442;
        const wp::int32 var_443 = 2;
        wp::float64 var_444;
        wp::float64 var_445;
        wp::float64 var_446;
        const wp::int32 var_447 = 2;
        wp::float64 var_448;
        wp::float64 var_449;
        wp::float64 var_450;
        wp::vec_t<3,wp::float64> var_451;
        const wp::float32 var_452 = 1.0;
        wp::float64 var_453;
        wp::float64 var_454;
        const wp::int32 var_455 = 0;
        wp::float64 var_456;
        wp::float64 var_457;
        const wp::int32 var_458 = 0;
        wp::float64 var_459;
        wp::float64 var_460;
        wp::float64 var_461;
        const wp::float32 var_462 = 1.0;
        wp::float64 var_463;
        wp::float64 var_464;
        const wp::int32 var_465 = 0;
        wp::float64 var_466;
        wp::float64 var_467;
        wp::float64 var_468;
        const wp::int32 var_469 = 0;
        wp::float64 var_470;
        wp::float64 var_471;
        wp::float64 var_472;
        const wp::float32 var_473 = 1.0;
        wp::float64 var_474;
        wp::float64 var_475;
        const wp::int32 var_476 = 1;
        wp::float64 var_477;
        wp::float64 var_478;
        const wp::int32 var_479 = 1;
        wp::float64 var_480;
        wp::float64 var_481;
        wp::float64 var_482;
        const wp::float32 var_483 = 1.0;
        wp::float64 var_484;
        wp::float64 var_485;
        const wp::int32 var_486 = 1;
        wp::float64 var_487;
        wp::float64 var_488;
        wp::float64 var_489;
        const wp::int32 var_490 = 1;
        wp::float64 var_491;
        wp::float64 var_492;
        wp::float64 var_493;
        const wp::float32 var_494 = 1.0;
        wp::float64 var_495;
        wp::float64 var_496;
        const wp::int32 var_497 = 2;
        wp::float64 var_498;
        wp::float64 var_499;
        const wp::int32 var_500 = 2;
        wp::float64 var_501;
        wp::float64 var_502;
        wp::float64 var_503;
        const wp::float32 var_504 = 1.0;
        wp::float64 var_505;
        wp::float64 var_506;
        const wp::int32 var_507 = 2;
        wp::float64 var_508;
        wp::float64 var_509;
        wp::float64 var_510;
        const wp::int32 var_511 = 2;
        wp::float64 var_512;
        wp::float64 var_513;
        wp::float64 var_514;
        wp::vec_t<3,wp::float64> var_515;
        wp::int32* var_516;
        wp::float64* var_517;
        wp::int32 var_518;
        wp::int32* var_519;
        wp::float64* var_520;
        wp::int32 var_521;
        wp::float64 var_522;
        wp::float64 var_523;
        wp::float64 var_524;
        wp::float64* var_525;
        wp::float64 var_526;
        wp::float64 var_527;
        wp::float64 var_528;
        wp::vec_t<3,wp::float64>* var_529;
        wp::vec_t<3,wp::float64> var_530;
        wp::vec_t<3,wp::float64> var_531;
        const wp::int32 var_532 = 0;
        wp::float64 var_533;
        const wp::float32 var_534 = 1.0;
        wp::float64 var_535;
        wp::float64 var_536;
        wp::float64 var_537;
        const wp::int32 var_538 = 1;
        wp::float64 var_539;
        const wp::float32 var_540 = 1.0;
        wp::float64 var_541;
        wp::float64 var_542;
        wp::float64 var_543;
        const wp::int32 var_544 = 2;
        wp::float64 var_545;
        const wp::float32 var_546 = 1.0;
        wp::float64 var_547;
        wp::float64 var_548;
        wp::float64 var_549;
        const wp::int32 var_550 = 0;
        wp::float64 var_551;
        wp::float64 var_552;
        const wp::int32 var_553 = 1;
        wp::float64 var_554;
        wp::float64 var_555;
        const wp::int32 var_556 = 2;
        wp::float64 var_557;
        wp::float64 var_558;
        const wp::int32 var_559 = 0;
        wp::float64 var_560;
        const wp::float32 var_561 = 1.0;
        wp::float64 var_562;
        wp::float64 var_563;
        wp::float64 var_564;
        wp::float64 var_565;
        const wp::int32 var_566 = 1;
        wp::float64 var_567;
        const wp::float32 var_568 = 1.0;
        wp::float64 var_569;
        wp::float64 var_570;
        wp::float64 var_571;
        wp::float64 var_572;
        const wp::int32 var_573 = 2;
        wp::float64 var_574;
        const wp::float32 var_575 = 1.0;
        wp::float64 var_576;
        wp::float64 var_577;
        wp::float64 var_578;
        wp::float64 var_579;
        const wp::int32 var_580 = 0;
        wp::float64 var_581;
        wp::float64 var_582;
        wp::float64 var_583;
        const wp::int32 var_584 = 1;
        wp::float64 var_585;
        wp::float64 var_586;
        wp::float64 var_587;
        const wp::int32 var_588 = 2;
        wp::float64 var_589;
        wp::float64 var_590;
        wp::float64 var_591;
        wp::vec_t<12,wp::float64> var_592;
        wp::vec_t<12,wp::float64> var_593;
        wp::float64 var_594;
        wp::float64 var_595;
        wp::vec_t<3,wp::float64> var_596;
        wp::vec_t<3,wp::float64> var_597;
        wp::float64 var_598;
        wp::vec_t<3,wp::float64> var_599;
        wp::vec_t<12,wp::float64> var_600;
        wp::vec_t<4,wp::int32> var_601;
        wp::float64 var_602;
        wp::float64 var_603;
        wp::float64 var_604;
        wp::float64 var_605;
        wp::float64 var_606;
        wp::float64 var_607;
        wp::vec_t<3,wp::float64> var_608;
        wp::vec_t<3,wp::float64> var_609;
        wp::float64 var_610;
        wp::vec_t<3,wp::float64> var_611;
        const wp::int32 var_612 = 0;
        const wp::int32 var_613 = 3;
        wp::int32 var_614;
        const wp::int32 var_615 = 0;
        wp::int32 var_616;
        wp::float64 var_617;
        const wp::int32 var_618 = 3;
        wp::int32 var_619;
        const wp::int32 var_620 = 1;
        wp::int32 var_621;
        wp::float64 var_622;
        const wp::int32 var_623 = 3;
        wp::int32 var_624;
        const wp::int32 var_625 = 2;
        wp::int32 var_626;
        wp::float64 var_627;
        wp::vec_t<3,wp::float64> var_628;
        wp::int32 var_629;
        wp::vec_t<3,wp::float64> var_630;
        const wp::int32 var_631 = 1;
        const wp::int32 var_632 = 3;
        wp::int32 var_633;
        const wp::int32 var_634 = 0;
        wp::int32 var_635;
        wp::float64 var_636;
        const wp::int32 var_637 = 3;
        wp::int32 var_638;
        const wp::int32 var_639 = 1;
        wp::int32 var_640;
        wp::float64 var_641;
        const wp::int32 var_642 = 3;
        wp::int32 var_643;
        const wp::int32 var_644 = 2;
        wp::int32 var_645;
        wp::float64 var_646;
        wp::vec_t<3,wp::float64> var_647;
        wp::int32 var_648;
        wp::vec_t<3,wp::float64> var_649;
        const wp::int32 var_650 = 2;
        const wp::int32 var_651 = 3;
        wp::int32 var_652;
        const wp::int32 var_653 = 0;
        wp::int32 var_654;
        wp::float64 var_655;
        const wp::int32 var_656 = 3;
        wp::int32 var_657;
        const wp::int32 var_658 = 1;
        wp::int32 var_659;
        wp::float64 var_660;
        const wp::int32 var_661 = 3;
        wp::int32 var_662;
        const wp::int32 var_663 = 2;
        wp::int32 var_664;
        wp::float64 var_665;
        wp::vec_t<3,wp::float64> var_666;
        wp::int32 var_667;
        wp::vec_t<3,wp::float64> var_668;
        const wp::int32 var_669 = 3;
        const wp::int32 var_670 = 3;
        wp::int32 var_671;
        const wp::int32 var_672 = 0;
        wp::int32 var_673;
        wp::float64 var_674;
        const wp::int32 var_675 = 3;
        wp::int32 var_676;
        const wp::int32 var_677 = 1;
        wp::int32 var_678;
        wp::float64 var_679;
        const wp::int32 var_680 = 3;
        wp::int32 var_681;
        const wp::int32 var_682 = 2;
        wp::int32 var_683;
        wp::float64 var_684;
        wp::vec_t<3,wp::float64> var_685;
        wp::int32 var_686;
        wp::vec_t<3,wp::float64> var_687;
        //---------
        // forward
        // def grad_IPC_collisions(                                                               <L 65>
        // idx = wp.tid()                                                                         <L 97>
        var_0 = builtin_tid1d();
        // grad_b = wp.vec(length=12, dtype=wp.float64)                                           <L 99>
        var_2 = wp::vec_t<12,wp::float64>();
        // vInd = wp.vec4i()                                                                      <L 100>
        var_3 = wp::vec_t<4,wp::int32>();
        // if collision_type[idx] == TRIANGLE_POINT_PAIR or collision_type[idx] == POINT_TRIANGLE_PAIR:       <L 101>
        var_4 = wp::address(var_collision_type, var_0);
        var_7 = wp::load(var_4);
        var_6 = (var_7 == var_5);
        var_8 = wp::address(var_collision_type, var_0);
        var_11 = wp::load(var_8);
        var_10 = (var_11 == var_9);
        var_12 = var_6 || var_10;
        if (var_12) {
            // svi = wp.select(collision_type[idx] == POINT_TRIANGLE_PAIR, nodeJ[idx], nodeI[idx])       <L 102>
            var_13 = wp::address(var_collision_type, var_0);
            var_15 = wp::load(var_13);
            var_14 = (var_15 == var_9);
            var_16 = wp::address(var_nodeJ, var_0);
            var_17 = wp::address(var_nodeI, var_0);
            var_19 = wp::load(var_16);
            var_20 = wp::load(var_17);
            var_18 = wp::select(var_14, var_19, var_20);
            // faceJ = wp.select(collision_type[idx] == POINT_TRIANGLE_PAIR, nodeI[idx], nodeJ[idx])       <L 103>
            var_21 = wp::address(var_collision_type, var_0);
            var_23 = wp::load(var_21);
            var_22 = (var_23 == var_9);
            var_24 = wp::address(var_nodeI, var_0);
            var_25 = wp::address(var_nodeJ, var_0);
            var_27 = wp::load(var_24);
            var_28 = wp::load(var_25);
            var_26 = wp::select(var_22, var_27, var_28);
            // faceVInd = face[faceJ]                                                             <L 104>
            var_29 = wp::address(var_face, var_26);
            var_31 = wp::load(var_29);
            var_30 = wp::copy(var_31);
            // xI = surf_vi[svi]                                                                  <L 105>
            var_32 = wp::address(var_surf_vi, var_18);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // if (env_states[node2env[xI]] == ENV_STATE_INVALID) or (env_states[node2env[xI]] == ENV_STATE_NEWTON_SOLVED):       <L 106>
            var_35 = wp::address(var_node2env, var_33);
            var_37 = wp::load(var_35);
            var_36 = wp::address(var_env_states, var_37);
            var_40 = wp::load(var_36);
            var_39 = (var_40 == var_38);
            var_41 = wp::address(var_node2env, var_33);
            var_43 = wp::load(var_41);
            var_42 = wp::address(var_env_states, var_43);
            var_46 = wp::load(var_42);
            var_45 = (var_46 == var_44);
            var_47 = var_39 || var_45;
            if (var_47) {
                // return                                                                         <L 107>
                continue;
            }
            // vInd = wp.vec4i(xI, faceVInd[0], faceVInd[1], faceVInd[2])                         <L 108>
            var_49 = wp::extract(var_30, var_48);
            var_51 = wp::extract(var_30, var_50);
            var_53 = wp::extract(var_30, var_52);
            var_54 = wp::vec_t<4,wp::int32>(var_33, var_49, var_51, var_53);
            // p = x[xI]                                                                          <L 109>
            var_55 = wp::address(var_x, var_33);
            var_57 = wp::load(var_55);
            var_56 = wp::copy(var_57);
            // t0 = x[faceVInd[0]]                                                                <L 110>
            var_59 = wp::extract(var_30, var_58);
            var_60 = wp::address(var_x, var_59);
            var_62 = wp::load(var_60);
            var_61 = wp::copy(var_62);
            // t1 = x[faceVInd[1]]                                                                <L 111>
            var_64 = wp::extract(var_30, var_63);
            var_65 = wp::address(var_x, var_64);
            var_67 = wp::load(var_65);
            var_66 = wp::copy(var_67);
            // t2 = x[faceVInd[2]]                                                                <L 112>
            var_69 = wp::extract(var_30, var_68);
            var_70 = wp::address(var_x, var_69);
            var_72 = wp::load(var_70);
            var_71 = wp::copy(var_72);
            // hat_p = hat_x[xI]                                                                  <L 113>
            var_73 = wp::address(var_hat_x, var_33);
            var_75 = wp::load(var_73);
            var_74 = wp::copy(var_75);
            // hat_t0 = hat_x[faceVInd[0]]                                                        <L 114>
            var_77 = wp::extract(var_30, var_76);
            var_78 = wp::address(var_hat_x, var_77);
            var_80 = wp::load(var_78);
            var_79 = wp::copy(var_80);
            // hat_t1 = hat_x[faceVInd[1]]                                                        <L 115>
            var_82 = wp::extract(var_30, var_81);
            var_83 = wp::address(var_hat_x, var_82);
            var_85 = wp::load(var_83);
            var_84 = wp::copy(var_85);
            // hat_t2 = hat_x[faceVInd[2]]                                                        <L 116>
            var_87 = wp::extract(var_30, var_86);
            var_88 = wp::address(var_hat_x, var_87);
            var_90 = wp::load(var_88);
            var_89 = wp::copy(var_90);
            // xi = node_xi[svi] + face_xi[faceJ]                                                 <L 117>
            var_91 = wp::address(var_node_xi, var_18);
            var_92 = wp::address(var_face_xi, var_26);
            var_94 = wp::load(var_91);
            var_95 = wp::load(var_92);
            var_93 = wp::add(var_94, var_95);
            // dsq = point_triangle_distance(p, t0, t1, t2) - xi * xi                             <L 118>
            var_96 = point_triangle_distance_0(var_56, var_61, var_66, var_71);
            var_97 = wp::mul(var_93, var_93);
            var_98 = wp::sub(var_96, var_97);
            // grad_b = point_triangle_distance_gradient(p, t0, t1, t2)                           <L 119>
            var_99 = point_triangle_distance_gradient_0(var_56, var_61, var_66, var_71);
            // activeGap2 = dhat * dhat + wp.float64(2.0) * dhat * xi                             <L 122>
            var_100 = wp::mul(var_dhat, var_dhat);
            var_102 = wp::float64(var_101);
            var_103 = wp::mul(var_102, var_dhat);
            var_104 = wp::mul(var_103, var_93);
            var_105 = wp::add(var_100, var_104);
            // if E == BARRIER:                                                                   <L 123>
            var_107 = (var_E == var_106);
            if (var_107) {
                // db_div_dd = scale * int_w_PTEE * node_area[svi] * dhat * barrier.barrier_gradient(dsq, activeGap2, kappa)       <L 124>
                var_108 = wp::mul(var_scale, var_int_w_PTEE);
                var_109 = wp::address(var_node_area, var_18);
                var_111 = wp::load(var_109);
                var_110 = wp::mul(var_108, var_111);
                var_112 = wp::mul(var_110, var_dhat);
                var_113 = barrier_gradient_0(var_98, var_105, var_kappa);
                var_114 = wp::mul(var_112, var_113);
                // grad_b *= db_div_dd                                                            <L 125>
                var_115 = wp::mul(var_99, var_114);
            }
            var_116 = wp::where(var_107, var_115, var_99);
            if (!var_107) {
                // u = closest_points[idx][0]                                                     <L 127>
                var_118 = wp::address(var_closest_points, var_0);
                var_120 = wp::load(var_118);
                var_119 = wp::extract(var_120, var_117);
                // v = closest_points[idx][1]                                                     <L 128>
                var_122 = wp::address(var_closest_points, var_0);
                var_124 = wp::load(var_122);
                var_123 = wp::extract(var_124, var_121);
                // xIcoord = p - (wp.float64(1.0) - u - v) * t0 - u * t1 - v * t2                 <L 129>
                var_126 = wp::float64(var_125);
                var_127 = wp::sub(var_126, var_119);
                var_128 = wp::sub(var_127, var_123);
                var_129 = wp::mul(var_128, var_61);
                var_130 = wp::sub(var_56, var_129);
                var_131 = wp::mul(var_119, var_66);
                var_132 = wp::sub(var_130, var_131);
                var_133 = wp::mul(var_123, var_71);
                var_134 = wp::sub(var_132, var_133);
                // hat_xIcoord = hat_p - (wp.float64(1.0) - u - v) * hat_t0 - u * hat_t1 - v * hat_t2       <L 130>
                var_136 = wp::float64(var_135);
                var_137 = wp::sub(var_136, var_119);
                var_138 = wp::sub(var_137, var_123);
                var_139 = wp::mul(var_138, var_79);
                var_140 = wp::sub(var_74, var_139);
                var_141 = wp::mul(var_119, var_84);
                var_142 = wp::sub(var_140, var_141);
                var_143 = wp::mul(var_123, var_89);
                var_144 = wp::sub(var_142, var_143);
                // weights = wp.vec4d(wp.float64(1.0), -(wp.float64(1.0) - u - v), -u, -v)        <L 131>
                var_146 = wp::float64(var_145);
                var_148 = wp::float64(var_147);
                var_149 = wp::sub(var_148, var_119);
                var_150 = wp::sub(var_149, var_123);
                var_151 = wp::neg(var_150);
                var_152 = wp::neg(var_119);
                var_153 = wp::neg(var_123);
                var_154 = wp::vec_t<4,wp::float64>(var_146, var_151, var_152, var_153);
                // coeff = get_combined_coulomb_friction(mu_body[bodyI[idx]], mu_body[bodyJ[idx]]) * c_lambda[idx] * scale       <L 132>
                var_155 = wp::address(var_bodyI, var_0);
                var_157 = wp::load(var_155);
                var_156 = wp::address(var_mu_body, var_157);
                var_158 = wp::address(var_bodyJ, var_0);
                var_160 = wp::load(var_158);
                var_159 = wp::address(var_mu_body, var_160);
                var_162 = wp::load(var_156);
                var_163 = wp::load(var_159);
                var_161 = get_combined_coulomb_friction_0(var_162, var_163);
                var_164 = wp::address(var_c_lambda, var_0);
                var_166 = wp::load(var_164);
                var_165 = wp::mul(var_161, var_166);
                var_167 = wp::mul(var_165, var_scale);
                // grad_f = friction.friction_gradient(normal[idx], xIcoord, hat_xIcoord, hat_h, epsv, coeff)       <L 133>
                var_168 = wp::address(var_normal, var_0);
                var_170 = wp::load(var_168);
                var_169 = friction_gradient_0(var_170, var_134, var_144, var_hat_h, var_epsv, var_167);
                // grad_b = vec12d(                                                               <L 134>
                // grad_f[0] * weights[0],                                                        <L 135>
                var_172 = wp::extract(var_169, var_171);
                var_174 = wp::extract(var_154, var_173);
                var_175 = wp::mul(var_172, var_174);
                // grad_f[1] * weights[0],                                                        <L 136>
                var_177 = wp::extract(var_169, var_176);
                var_179 = wp::extract(var_154, var_178);
                var_180 = wp::mul(var_177, var_179);
                // grad_f[2] * weights[0],                                                        <L 137>
                var_182 = wp::extract(var_169, var_181);
                var_184 = wp::extract(var_154, var_183);
                var_185 = wp::mul(var_182, var_184);
                // grad_f[0] * weights[1],                                                        <L 138>
                var_187 = wp::extract(var_169, var_186);
                var_189 = wp::extract(var_154, var_188);
                var_190 = wp::mul(var_187, var_189);
                // grad_f[1] * weights[1],                                                        <L 139>
                var_192 = wp::extract(var_169, var_191);
                var_194 = wp::extract(var_154, var_193);
                var_195 = wp::mul(var_192, var_194);
                // grad_f[2] * weights[1],                                                        <L 140>
                var_197 = wp::extract(var_169, var_196);
                var_199 = wp::extract(var_154, var_198);
                var_200 = wp::mul(var_197, var_199);
                // grad_f[0] * weights[2],                                                        <L 141>
                var_202 = wp::extract(var_169, var_201);
                var_204 = wp::extract(var_154, var_203);
                var_205 = wp::mul(var_202, var_204);
                // grad_f[1] * weights[2],                                                        <L 142>
                var_207 = wp::extract(var_169, var_206);
                var_209 = wp::extract(var_154, var_208);
                var_210 = wp::mul(var_207, var_209);
                // grad_f[2] * weights[2],                                                        <L 143>
                var_212 = wp::extract(var_169, var_211);
                var_214 = wp::extract(var_154, var_213);
                var_215 = wp::mul(var_212, var_214);
                // grad_f[0] * weights[3],                                                        <L 144>
                var_217 = wp::extract(var_169, var_216);
                var_219 = wp::extract(var_154, var_218);
                var_220 = wp::mul(var_217, var_219);
                // grad_f[1] * weights[3],                                                        <L 145>
                var_222 = wp::extract(var_169, var_221);
                var_224 = wp::extract(var_154, var_223);
                var_225 = wp::mul(var_222, var_224);
                // grad_f[2] * weights[3],                                                        <L 146>
                var_227 = wp::extract(var_169, var_226);
                var_229 = wp::extract(var_154, var_228);
                var_230 = wp::mul(var_227, var_229);
                var_231 = wp::vec_t<12,wp::float64>({var_175, var_180, var_185, var_190, var_195, var_200, var_205, var_210, var_215, var_220, var_225, var_230});
            }
            var_232 = wp::where(var_107, var_116, var_231);
        }
        var_233 = wp::where(var_12, var_232, var_2);
        var_234 = wp::where(var_12, var_54, var_3);
        if (!var_12) {
            // edgeI = nodeI[idx]                                                                 <L 152>
            var_235 = wp::address(var_nodeI, var_0);
            var_237 = wp::load(var_235);
            var_236 = wp::copy(var_237);
            // edgeJ = nodeJ[idx]                                                                 <L 153>
            var_238 = wp::address(var_nodeJ, var_0);
            var_240 = wp::load(var_238);
            var_239 = wp::copy(var_240);
            // edgeVIndI = edge[edgeI]                                                            <L 154>
            var_241 = wp::address(var_edge, var_236);
            var_243 = wp::load(var_241);
            var_242 = wp::copy(var_243);
            // edgeVIndJ = edge[edgeJ]                                                            <L 155>
            var_244 = wp::address(var_edge, var_239);
            var_246 = wp::load(var_244);
            var_245 = wp::copy(var_246);
            // vInd = wp.vec4i(edgeVIndI[0], edgeVIndI[1], edgeVIndJ[0], edgeVIndJ[1])            <L 156>
            var_248 = wp::extract(var_242, var_247);
            var_250 = wp::extract(var_242, var_249);
            var_252 = wp::extract(var_245, var_251);
            var_254 = wp::extract(var_245, var_253);
            var_255 = wp::vec_t<4,wp::int32>(var_248, var_250, var_252, var_254);
            // xi = edge_xi[edgeI] + edge_xi[edgeJ]                                               <L 157>
            var_256 = wp::address(var_edge_xi, var_236);
            var_257 = wp::address(var_edge_xi, var_239);
            var_259 = wp::load(var_256);
            var_260 = wp::load(var_257);
            var_258 = wp::add(var_259, var_260);
            // if (env_states[node2env[edgeVIndI[0]]] == ENV_STATE_INVALID) or (env_states[node2env[edgeVIndI[0]]] == ENV_STATE_NEWTON_SOLVED):       <L 158>
            var_262 = wp::extract(var_242, var_261);
            var_263 = wp::address(var_node2env, var_262);
            var_265 = wp::load(var_263);
            var_264 = wp::address(var_env_states, var_265);
            var_267 = wp::load(var_264);
            var_266 = (var_267 == var_38);
            var_269 = wp::extract(var_242, var_268);
            var_270 = wp::address(var_node2env, var_269);
            var_272 = wp::load(var_270);
            var_271 = wp::address(var_env_states, var_272);
            var_274 = wp::load(var_271);
            var_273 = (var_274 == var_44);
            var_275 = var_266 || var_273;
            if (var_275) {
                // return                                                                         <L 159>
                continue;
            }
            // ea0 = x[edgeVIndI[0]]                                                              <L 160>
            var_277 = wp::extract(var_242, var_276);
            var_278 = wp::address(var_x, var_277);
            var_280 = wp::load(var_278);
            var_279 = wp::copy(var_280);
            // ea1 = x[edgeVIndI[1]]                                                              <L 161>
            var_282 = wp::extract(var_242, var_281);
            var_283 = wp::address(var_x, var_282);
            var_285 = wp::load(var_283);
            var_284 = wp::copy(var_285);
            // eb0 = x[edgeVIndJ[0]]                                                              <L 162>
            var_287 = wp::extract(var_245, var_286);
            var_288 = wp::address(var_x, var_287);
            var_290 = wp::load(var_288);
            var_289 = wp::copy(var_290);
            // eb1 = x[edgeVIndJ[1]]                                                              <L 163>
            var_292 = wp::extract(var_245, var_291);
            var_293 = wp::address(var_x, var_292);
            var_295 = wp::load(var_293);
            var_294 = wp::copy(var_295);
            // dsq = edge_edge_distance(ea0, ea1, eb0, eb1) - xi * xi                             <L 164>
            var_296 = edge_edge_distance_0(var_279, var_284, var_289, var_294);
            var_297 = wp::mul(var_258, var_258);
            var_298 = wp::sub(var_296, var_297);
            // EA0 = X[edgeVIndI[0]]                                                              <L 166>
            var_300 = wp::extract(var_242, var_299);
            var_301 = wp::address(var_X, var_300);
            var_303 = wp::load(var_301);
            var_302 = wp::copy(var_303);
            // EA1 = X[edgeVIndI[1]]                                                              <L 167>
            var_305 = wp::extract(var_242, var_304);
            var_306 = wp::address(var_X, var_305);
            var_308 = wp::load(var_306);
            var_307 = wp::copy(var_308);
            // EB0 = X[edgeVIndJ[0]]                                                              <L 168>
            var_310 = wp::extract(var_245, var_309);
            var_311 = wp::address(var_X, var_310);
            var_313 = wp::load(var_311);
            var_312 = wp::copy(var_313);
            // EB1 = X[edgeVIndJ[1]]                                                              <L 169>
            var_315 = wp::extract(var_245, var_314);
            var_316 = wp::address(var_X, var_315);
            var_318 = wp::load(var_316);
            var_317 = wp::copy(var_318);
            // eps_x = edge_edge_mollifier_threshold(EA0, EA1, EB0, EB1)                          <L 170>
            var_319 = edge_edge_mollifier_threshold_0(var_302, var_307, var_312, var_317);
            // m = edge_edge_mollifier(ea0, ea1, eb0, eb1, eps_x)                                 <L 171>
            var_320 = edge_edge_mollifier_0(var_279, var_284, var_289, var_294, var_319);
            // activeGap2 = dhat * dhat + wp.float64(2.0) * dhat * xi                             <L 173>
            var_321 = wp::mul(var_dhat, var_dhat);
            var_323 = wp::float64(var_322);
            var_324 = wp::mul(var_323, var_dhat);
            var_325 = wp::mul(var_324, var_258);
            var_326 = wp::add(var_321, var_325);
            // if E == BARRIER:                                                                   <L 175>
            var_327 = (var_E == var_106);
            if (var_327) {
                // db_div_dd = scale * int_w_PTEE * (edge_area[edgeI] + edge_area[edgeJ]) * dhat * barrier.barrier_gradient(dsq, activeGap2, kappa)       <L 176>
                var_328 = wp::mul(var_scale, var_int_w_PTEE);
                var_329 = wp::address(var_edge_area, var_236);
                var_330 = wp::address(var_edge_area, var_239);
                var_332 = wp::load(var_329);
                var_333 = wp::load(var_330);
                var_331 = wp::add(var_332, var_333);
                var_334 = wp::mul(var_328, var_331);
                var_335 = wp::mul(var_334, var_dhat);
                var_336 = barrier_gradient_0(var_298, var_326, var_kappa);
                var_337 = wp::mul(var_335, var_336);
                // grad_b = edge_edge_distance_gradient(ea0, ea1, eb0, eb1)                       <L 177>
                var_338 = edge_edge_distance_gradient_0(var_279, var_284, var_289, var_294);
                // grad_b *= db_div_dd                                                            <L 178>
                var_339 = wp::mul(var_338, var_337);
                // if m < wp.float64(1.0):                                                        <L 179>
                var_341 = wp::float64(var_340);
                var_342 = (var_320 < var_341);
                if (var_342) {
                    // b = scale * int_w_PTEE * (edge_area[edgeI] + edge_area[edgeJ]) * dhat * barrier.barrier(dsq, activeGap2, kappa)       <L 180>
                    var_343 = wp::mul(var_scale, var_int_w_PTEE);
                    var_344 = wp::address(var_edge_area, var_236);
                    var_345 = wp::address(var_edge_area, var_239);
                    var_347 = wp::load(var_344);
                    var_348 = wp::load(var_345);
                    var_346 = wp::add(var_347, var_348);
                    var_349 = wp::mul(var_343, var_346);
                    var_350 = wp::mul(var_349, var_dhat);
                    var_351 = barrier_0(var_298, var_326, var_kappa);
                    var_352 = wp::mul(var_350, var_351);
                    // grad_m = edge_edge_mollifier_gradient(ea0, ea1, eb0, eb1, eps_x)           <L 181>
                    var_353 = edge_edge_mollifier_gradient_0(var_279, var_284, var_289, var_294, var_319);
                    // grad_b = m * grad_b + grad_m * b                                           <L 182>
                    var_354 = wp::mul(var_320, var_339);
                    var_355 = wp::mul(var_353, var_352);
                    var_356 = wp::add(var_354, var_355);
                }
                var_357 = wp::where(var_342, var_356, var_339);
            }
            var_358 = wp::where(var_327, var_357, var_233);
            var_359 = wp::where(var_327, var_337, var_114);
            if (!var_327) {
                // hat_ea0 = hat_x[edgeVIndI[0]]                                                  <L 184>
                var_361 = wp::extract(var_242, var_360);
                var_362 = wp::address(var_hat_x, var_361);
                var_364 = wp::load(var_362);
                var_363 = wp::copy(var_364);
                // hat_ea1 = hat_x[edgeVIndI[1]]                                                  <L 185>
                var_366 = wp::extract(var_242, var_365);
                var_367 = wp::address(var_hat_x, var_366);
                var_369 = wp::load(var_367);
                var_368 = wp::copy(var_369);
                // hat_eb0 = hat_x[edgeVIndJ[0]]                                                  <L 186>
                var_371 = wp::extract(var_245, var_370);
                var_372 = wp::address(var_hat_x, var_371);
                var_374 = wp::load(var_372);
                var_373 = wp::copy(var_374);
                // hat_eb1 = hat_x[edgeVIndJ[1]]                                                  <L 187>
                var_376 = wp::extract(var_245, var_375);
                var_377 = wp::address(var_hat_x, var_376);
                var_379 = wp::load(var_377);
                var_378 = wp::copy(var_379);
                // u = closest_points[idx][0]                                                     <L 188>
                var_381 = wp::address(var_closest_points, var_0);
                var_383 = wp::load(var_381);
                var_382 = wp::extract(var_383, var_380);
                // v = closest_points[idx][1]                                                     <L 189>
                var_385 = wp::address(var_closest_points, var_0);
                var_387 = wp::load(var_385);
                var_386 = wp::extract(var_387, var_384);
                // xIcoord = wp.vec3d(                                                            <L 190>
                // (wp.float64(1.0) - u) * ea0[0] + u * ea1[0] - (wp.float64(1.0) - v) * eb0[0] - v * eb1[0],       <L 191>
                var_389 = wp::float64(var_388);
                var_390 = wp::sub(var_389, var_382);
                var_392 = wp::extract(var_279, var_391);
                var_393 = wp::mul(var_390, var_392);
                var_395 = wp::extract(var_284, var_394);
                var_396 = wp::mul(var_382, var_395);
                var_397 = wp::add(var_393, var_396);
                var_399 = wp::float64(var_398);
                var_400 = wp::sub(var_399, var_386);
                var_402 = wp::extract(var_289, var_401);
                var_403 = wp::mul(var_400, var_402);
                var_404 = wp::sub(var_397, var_403);
                var_406 = wp::extract(var_294, var_405);
                var_407 = wp::mul(var_386, var_406);
                var_408 = wp::sub(var_404, var_407);
                // (wp.float64(1.0) - u) * ea0[1] + u * ea1[1] - (wp.float64(1.0) - v) * eb0[1] - v * eb1[1],       <L 192>
                var_410 = wp::float64(var_409);
                var_411 = wp::sub(var_410, var_382);
                var_413 = wp::extract(var_279, var_412);
                var_414 = wp::mul(var_411, var_413);
                var_416 = wp::extract(var_284, var_415);
                var_417 = wp::mul(var_382, var_416);
                var_418 = wp::add(var_414, var_417);
                var_420 = wp::float64(var_419);
                var_421 = wp::sub(var_420, var_386);
                var_423 = wp::extract(var_289, var_422);
                var_424 = wp::mul(var_421, var_423);
                var_425 = wp::sub(var_418, var_424);
                var_427 = wp::extract(var_294, var_426);
                var_428 = wp::mul(var_386, var_427);
                var_429 = wp::sub(var_425, var_428);
                // (wp.float64(1.0) - u) * ea0[2] + u * ea1[2] - (wp.float64(1.0) - v) * eb0[2] - v * eb1[2],       <L 193>
                var_431 = wp::float64(var_430);
                var_432 = wp::sub(var_431, var_382);
                var_434 = wp::extract(var_279, var_433);
                var_435 = wp::mul(var_432, var_434);
                var_437 = wp::extract(var_284, var_436);
                var_438 = wp::mul(var_382, var_437);
                var_439 = wp::add(var_435, var_438);
                var_441 = wp::float64(var_440);
                var_442 = wp::sub(var_441, var_386);
                var_444 = wp::extract(var_289, var_443);
                var_445 = wp::mul(var_442, var_444);
                var_446 = wp::sub(var_439, var_445);
                var_448 = wp::extract(var_294, var_447);
                var_449 = wp::mul(var_386, var_448);
                var_450 = wp::sub(var_446, var_449);
                var_451 = wp::vec_t<3,wp::float64>(var_408, var_429, var_450);
                // hat_xIcoord = wp.vec3d(                                                        <L 195>
                // (wp.float64(1.0) - u) * hat_ea0[0] + u * hat_ea1[0] - (wp.float64(1.0) - v) * hat_eb0[0] - v * hat_eb1[0],       <L 196>
                var_453 = wp::float64(var_452);
                var_454 = wp::sub(var_453, var_382);
                var_456 = wp::extract(var_363, var_455);
                var_457 = wp::mul(var_454, var_456);
                var_459 = wp::extract(var_368, var_458);
                var_460 = wp::mul(var_382, var_459);
                var_461 = wp::add(var_457, var_460);
                var_463 = wp::float64(var_462);
                var_464 = wp::sub(var_463, var_386);
                var_466 = wp::extract(var_373, var_465);
                var_467 = wp::mul(var_464, var_466);
                var_468 = wp::sub(var_461, var_467);
                var_470 = wp::extract(var_378, var_469);
                var_471 = wp::mul(var_386, var_470);
                var_472 = wp::sub(var_468, var_471);
                // (wp.float64(1.0) - u) * hat_ea0[1] + u * hat_ea1[1] - (wp.float64(1.0) - v) * hat_eb0[1] - v * hat_eb1[1],       <L 197>
                var_474 = wp::float64(var_473);
                var_475 = wp::sub(var_474, var_382);
                var_477 = wp::extract(var_363, var_476);
                var_478 = wp::mul(var_475, var_477);
                var_480 = wp::extract(var_368, var_479);
                var_481 = wp::mul(var_382, var_480);
                var_482 = wp::add(var_478, var_481);
                var_484 = wp::float64(var_483);
                var_485 = wp::sub(var_484, var_386);
                var_487 = wp::extract(var_373, var_486);
                var_488 = wp::mul(var_485, var_487);
                var_489 = wp::sub(var_482, var_488);
                var_491 = wp::extract(var_378, var_490);
                var_492 = wp::mul(var_386, var_491);
                var_493 = wp::sub(var_489, var_492);
                // (wp.float64(1.0) - u) * hat_ea0[2] + u * hat_ea1[2] - (wp.float64(1.0) - v) * hat_eb0[2] - v * hat_eb1[2],       <L 198>
                var_495 = wp::float64(var_494);
                var_496 = wp::sub(var_495, var_382);
                var_498 = wp::extract(var_363, var_497);
                var_499 = wp::mul(var_496, var_498);
                var_501 = wp::extract(var_368, var_500);
                var_502 = wp::mul(var_382, var_501);
                var_503 = wp::add(var_499, var_502);
                var_505 = wp::float64(var_504);
                var_506 = wp::sub(var_505, var_386);
                var_508 = wp::extract(var_373, var_507);
                var_509 = wp::mul(var_506, var_508);
                var_510 = wp::sub(var_503, var_509);
                var_512 = wp::extract(var_378, var_511);
                var_513 = wp::mul(var_386, var_512);
                var_514 = wp::sub(var_510, var_513);
                var_515 = wp::vec_t<3,wp::float64>(var_472, var_493, var_514);
                // coeff = get_combined_coulomb_friction(mu_body[bodyI[idx]], mu_body[bodyJ[idx]]) * c_lambda[idx] * scale       <L 200>
                var_516 = wp::address(var_bodyI, var_0);
                var_518 = wp::load(var_516);
                var_517 = wp::address(var_mu_body, var_518);
                var_519 = wp::address(var_bodyJ, var_0);
                var_521 = wp::load(var_519);
                var_520 = wp::address(var_mu_body, var_521);
                var_523 = wp::load(var_517);
                var_524 = wp::load(var_520);
                var_522 = get_combined_coulomb_friction_0(var_523, var_524);
                var_525 = wp::address(var_c_lambda, var_0);
                var_527 = wp::load(var_525);
                var_526 = wp::mul(var_522, var_527);
                var_528 = wp::mul(var_526, var_scale);
                // grad_f = friction.friction_gradient(normal[idx], xIcoord, hat_xIcoord, hat_h, epsv, coeff)       <L 201>
                var_529 = wp::address(var_normal, var_0);
                var_531 = wp::load(var_529);
                var_530 = friction_gradient_0(var_531, var_451, var_515, var_hat_h, var_epsv, var_528);
                // grad_b = vec12d(                                                               <L 202>
                // grad_f[0] * (wp.float64(1.0) - u),                                             <L 203>
                var_533 = wp::extract(var_530, var_532);
                var_535 = wp::float64(var_534);
                var_536 = wp::sub(var_535, var_382);
                var_537 = wp::mul(var_533, var_536);
                // grad_f[1] * (wp.float64(1.0) - u),                                             <L 204>
                var_539 = wp::extract(var_530, var_538);
                var_541 = wp::float64(var_540);
                var_542 = wp::sub(var_541, var_382);
                var_543 = wp::mul(var_539, var_542);
                // grad_f[2] * (wp.float64(1.0) - u),                                             <L 205>
                var_545 = wp::extract(var_530, var_544);
                var_547 = wp::float64(var_546);
                var_548 = wp::sub(var_547, var_382);
                var_549 = wp::mul(var_545, var_548);
                // grad_f[0] * u,                                                                 <L 206>
                var_551 = wp::extract(var_530, var_550);
                var_552 = wp::mul(var_551, var_382);
                // grad_f[1] * u,                                                                 <L 207>
                var_554 = wp::extract(var_530, var_553);
                var_555 = wp::mul(var_554, var_382);
                // grad_f[2] * u,                                                                 <L 208>
                var_557 = wp::extract(var_530, var_556);
                var_558 = wp::mul(var_557, var_382);
                // grad_f[0] * -(wp.float64(1.0) - v),                                            <L 209>
                var_560 = wp::extract(var_530, var_559);
                var_562 = wp::float64(var_561);
                var_563 = wp::sub(var_562, var_386);
                var_564 = wp::neg(var_563);
                var_565 = wp::mul(var_560, var_564);
                // grad_f[1] * -(wp.float64(1.0) - v),                                            <L 210>
                var_567 = wp::extract(var_530, var_566);
                var_569 = wp::float64(var_568);
                var_570 = wp::sub(var_569, var_386);
                var_571 = wp::neg(var_570);
                var_572 = wp::mul(var_567, var_571);
                // grad_f[2] * -(wp.float64(1.0) - v),                                            <L 211>
                var_574 = wp::extract(var_530, var_573);
                var_576 = wp::float64(var_575);
                var_577 = wp::sub(var_576, var_386);
                var_578 = wp::neg(var_577);
                var_579 = wp::mul(var_574, var_578);
                // grad_f[0] * -v,                                                                <L 212>
                var_581 = wp::extract(var_530, var_580);
                var_582 = wp::neg(var_386);
                var_583 = wp::mul(var_581, var_582);
                // grad_f[1] * -v,                                                                <L 213>
                var_585 = wp::extract(var_530, var_584);
                var_586 = wp::neg(var_386);
                var_587 = wp::mul(var_585, var_586);
                // grad_f[2] * -v,                                                                <L 214>
                var_589 = wp::extract(var_530, var_588);
                var_590 = wp::neg(var_386);
                var_591 = wp::mul(var_589, var_590);
                var_592 = wp::vec_t<12,wp::float64>({var_537, var_543, var_549, var_552, var_555, var_558, var_565, var_572, var_579, var_583, var_587, var_591});
            }
            var_593 = wp::where(var_327, var_358, var_592);
            var_594 = wp::where(var_327, var_119, var_382);
            var_595 = wp::where(var_327, var_123, var_386);
            var_596 = wp::where(var_327, var_134, var_451);
            var_597 = wp::where(var_327, var_144, var_515);
            var_598 = wp::where(var_327, var_167, var_528);
            var_599 = wp::where(var_327, var_169, var_530);
        }
        var_600 = wp::where(var_12, var_233, var_593);
        var_601 = wp::where(var_12, var_234, var_255);
        var_602 = wp::where(var_12, var_93, var_258);
        var_603 = wp::where(var_12, var_98, var_298);
        var_604 = wp::where(var_12, var_105, var_326);
        var_605 = wp::where(var_12, var_114, var_359);
        var_606 = wp::where(var_12, var_119, var_594);
        var_607 = wp::where(var_12, var_123, var_595);
        var_608 = wp::where(var_12, var_134, var_596);
        var_609 = wp::where(var_12, var_144, var_597);
        var_610 = wp::where(var_12, var_167, var_598);
        var_611 = wp::where(var_12, var_169, var_599);
        // for vI in range(4):                                                                    <L 221>
        // grad_b_I = wp.vec3d(grad_b[vI * 3 + 0], grad_b[vI * 3 + 1], grad_b[vI * 3 + 2])        <L 222>
        var_614 = wp::mul(var_612, var_613);
        var_616 = wp::add(var_614, var_615);
        var_617 = wp::extract(var_600, var_616);
        var_619 = wp::mul(var_612, var_618);
        var_621 = wp::add(var_619, var_620);
        var_622 = wp::extract(var_600, var_621);
        var_624 = wp::mul(var_612, var_623);
        var_626 = wp::add(var_624, var_625);
        var_627 = wp::extract(var_600, var_626);
        var_628 = wp::vec_t<3,wp::float64>(var_617, var_622, var_627);
        // wp.atomic_add(gradient_x, vInd[vI], grad_b_I)                                          <L 223>
        var_629 = wp::extract(var_601, var_612);
        var_630 = wp::atomic_add(var_gradient_x, var_629, var_628);
        // grad_b_I = wp.vec3d(grad_b[vI * 3 + 0], grad_b[vI * 3 + 1], grad_b[vI * 3 + 2])        <L 222>
        var_633 = wp::mul(var_631, var_632);
        var_635 = wp::add(var_633, var_634);
        var_636 = wp::extract(var_600, var_635);
        var_638 = wp::mul(var_631, var_637);
        var_640 = wp::add(var_638, var_639);
        var_641 = wp::extract(var_600, var_640);
        var_643 = wp::mul(var_631, var_642);
        var_645 = wp::add(var_643, var_644);
        var_646 = wp::extract(var_600, var_645);
        var_647 = wp::vec_t<3,wp::float64>(var_636, var_641, var_646);
        // wp.atomic_add(gradient_x, vInd[vI], grad_b_I)                                          <L 223>
        var_648 = wp::extract(var_601, var_631);
        var_649 = wp::atomic_add(var_gradient_x, var_648, var_647);
        // grad_b_I = wp.vec3d(grad_b[vI * 3 + 0], grad_b[vI * 3 + 1], grad_b[vI * 3 + 2])        <L 222>
        var_652 = wp::mul(var_650, var_651);
        var_654 = wp::add(var_652, var_653);
        var_655 = wp::extract(var_600, var_654);
        var_657 = wp::mul(var_650, var_656);
        var_659 = wp::add(var_657, var_658);
        var_660 = wp::extract(var_600, var_659);
        var_662 = wp::mul(var_650, var_661);
        var_664 = wp::add(var_662, var_663);
        var_665 = wp::extract(var_600, var_664);
        var_666 = wp::vec_t<3,wp::float64>(var_655, var_660, var_665);
        // wp.atomic_add(gradient_x, vInd[vI], grad_b_I)                                          <L 223>
        var_667 = wp::extract(var_601, var_650);
        var_668 = wp::atomic_add(var_gradient_x, var_667, var_666);
        // grad_b_I = wp.vec3d(grad_b[vI * 3 + 0], grad_b[vI * 3 + 1], grad_b[vI * 3 + 2])        <L 222>
        var_671 = wp::mul(var_669, var_670);
        var_673 = wp::add(var_671, var_672);
        var_674 = wp::extract(var_600, var_673);
        var_676 = wp::mul(var_669, var_675);
        var_678 = wp::add(var_676, var_677);
        var_679 = wp::extract(var_600, var_678);
        var_681 = wp::mul(var_669, var_680);
        var_683 = wp::add(var_681, var_682);
        var_684 = wp::extract(var_600, var_683);
        var_685 = wp::vec_t<3,wp::float64>(var_674, var_679, var_684);
        // wp.atomic_add(gradient_x, vInd[vI], grad_b_I)                                          <L 223>
        var_686 = wp::extract(var_601, var_669);
        var_687 = wp::atomic_add(var_gradient_x, var_686, var_685);
    }
}



extern "C" __global__ void grad_IPC_collisions_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3,wp::float64>> var_gradient_x,
    wp::array_t<wp::float64> var_c_lambda,
    wp::array_t<wp::vec_t<2,wp::float64>> var_closest_points,
    wp::array_t<wp::vec_t<3,wp::float64>> var_normal,
    wp::array_t<wp::int32> var_nodeI,
    wp::array_t<wp::int32> var_nodeJ,
    wp::array_t<wp::int32> var_bodyI,
    wp::array_t<wp::int32> var_bodyJ,
    wp::array_t<wp::int32> var_collision_type,
    wp::array_t<wp::vec_t<3,wp::float64>> var_x,
    wp::array_t<wp::int32> var_surf_vi,
    wp::array_t<wp::vec_t<3,wp::float64>> var_hat_x,
    wp::array_t<wp::vec_t<3,wp::float64>> var_X,
    wp::array_t<wp::float64> var_mu_body,
    wp::array_t<wp::float64> var_node_xi,
    wp::array_t<wp::float64> var_edge_xi,
    wp::array_t<wp::float64> var_face_xi,
    wp::array_t<wp::float64> var_node_area,
    wp::array_t<wp::float64> var_edge_area,
    wp::array_t<wp::vec_t<2,wp::int32>> var_edge,
    wp::array_t<wp::vec_t<3,wp::int32>> var_face,
    wp::float64 var_dhat,
    wp::float64 var_hat_h,
    wp::float64 var_kappa,
    wp::float64 var_int_w_PTEE,
    wp::float64 var_scale,
    wp::float64 var_epsv,
    wp::int32 var_E,
    wp::array_t<wp::int32> var_node2env,
    wp::array_t<wp::int32> var_env_states,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_gradient_x,
    wp::array_t<wp::float64> adj_c_lambda,
    wp::array_t<wp::vec_t<2,wp::float64>> adj_closest_points,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_normal,
    wp::array_t<wp::int32> adj_nodeI,
    wp::array_t<wp::int32> adj_nodeJ,
    wp::array_t<wp::int32> adj_bodyI,
    wp::array_t<wp::int32> adj_bodyJ,
    wp::array_t<wp::int32> adj_collision_type,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_x,
    wp::array_t<wp::int32> adj_surf_vi,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_hat_x,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_X,
    wp::array_t<wp::float64> adj_mu_body,
    wp::array_t<wp::float64> adj_node_xi,
    wp::array_t<wp::float64> adj_edge_xi,
    wp::array_t<wp::float64> adj_face_xi,
    wp::array_t<wp::float64> adj_node_area,
    wp::array_t<wp::float64> adj_edge_area,
    wp::array_t<wp::vec_t<2,wp::int32>> adj_edge,
    wp::array_t<wp::vec_t<3,wp::int32>> adj_face,
    wp::float64 adj_dhat,
    wp::float64 adj_hat_h,
    wp::float64 adj_kappa,
    wp::float64 adj_int_w_PTEE,
    wp::float64 adj_scale,
    wp::float64 adj_epsv,
    wp::int32 adj_E,
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
        const wp::int32 var_1 = 12;
        wp::vec_t<12,wp::float64> var_2;
        wp::vec_t<4,wp::int32> var_3;
        wp::int32* var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        const wp::int32 var_9 = 1;
        bool var_10;
        wp::int32 var_11;
        bool var_12;
        wp::int32* var_13;
        bool var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32* var_21;
        bool var_22;
        wp::int32 var_23;
        wp::int32* var_24;
        wp::int32* var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::vec_t<3,wp::int32>* var_29;
        wp::vec_t<3,wp::int32> var_30;
        wp::vec_t<3,wp::int32> var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        wp::int32* var_35;
        wp::int32* var_36;
        wp::int32 var_37;
        const wp::int32 var_38 = 1;
        bool var_39;
        wp::int32 var_40;
        wp::int32* var_41;
        wp::int32* var_42;
        wp::int32 var_43;
        const wp::int32 var_44 = 2;
        bool var_45;
        wp::int32 var_46;
        bool var_47;
        const wp::int32 var_48 = 0;
        wp::int32 var_49;
        const wp::int32 var_50 = 1;
        wp::int32 var_51;
        const wp::int32 var_52 = 2;
        wp::int32 var_53;
        wp::vec_t<4,wp::int32> var_54;
        wp::vec_t<3,wp::float64>* var_55;
        wp::vec_t<3,wp::float64> var_56;
        wp::vec_t<3,wp::float64> var_57;
        const wp::int32 var_58 = 0;
        wp::int32 var_59;
        wp::vec_t<3,wp::float64>* var_60;
        wp::vec_t<3,wp::float64> var_61;
        wp::vec_t<3,wp::float64> var_62;
        const wp::int32 var_63 = 1;
        wp::int32 var_64;
        wp::vec_t<3,wp::float64>* var_65;
        wp::vec_t<3,wp::float64> var_66;
        wp::vec_t<3,wp::float64> var_67;
        const wp::int32 var_68 = 2;
        wp::int32 var_69;
        wp::vec_t<3,wp::float64>* var_70;
        wp::vec_t<3,wp::float64> var_71;
        wp::vec_t<3,wp::float64> var_72;
        wp::vec_t<3,wp::float64>* var_73;
        wp::vec_t<3,wp::float64> var_74;
        wp::vec_t<3,wp::float64> var_75;
        const wp::int32 var_76 = 0;
        wp::int32 var_77;
        wp::vec_t<3,wp::float64>* var_78;
        wp::vec_t<3,wp::float64> var_79;
        wp::vec_t<3,wp::float64> var_80;
        const wp::int32 var_81 = 1;
        wp::int32 var_82;
        wp::vec_t<3,wp::float64>* var_83;
        wp::vec_t<3,wp::float64> var_84;
        wp::vec_t<3,wp::float64> var_85;
        const wp::int32 var_86 = 2;
        wp::int32 var_87;
        wp::vec_t<3,wp::float64>* var_88;
        wp::vec_t<3,wp::float64> var_89;
        wp::vec_t<3,wp::float64> var_90;
        wp::float64* var_91;
        wp::float64* var_92;
        wp::float64 var_93;
        wp::float64 var_94;
        wp::float64 var_95;
        wp::float64 var_96;
        wp::float64 var_97;
        wp::float64 var_98;
        wp::vec_t<12,wp::float64> var_99;
        wp::float64 var_100;
        const wp::float32 var_101 = 2.0;
        wp::float64 var_102;
        wp::float64 var_103;
        wp::float64 var_104;
        wp::float64 var_105;
        const wp::int32 var_106 = 0;
        bool var_107;
        wp::float64 var_108;
        wp::float64* var_109;
        wp::float64 var_110;
        wp::float64 var_111;
        wp::float64 var_112;
        wp::float64 var_113;
        wp::float64 var_114;
        wp::vec_t<12,wp::float64> var_115;
        wp::vec_t<12,wp::float64> var_116;
        const wp::int32 var_117 = 0;
        wp::vec_t<2,wp::float64>* var_118;
        wp::float64 var_119;
        wp::vec_t<2,wp::float64> var_120;
        const wp::int32 var_121 = 1;
        wp::vec_t<2,wp::float64>* var_122;
        wp::float64 var_123;
        wp::vec_t<2,wp::float64> var_124;
        const wp::float32 var_125 = 1.0;
        wp::float64 var_126;
        wp::float64 var_127;
        wp::float64 var_128;
        wp::vec_t<3,wp::float64> var_129;
        wp::vec_t<3,wp::float64> var_130;
        wp::vec_t<3,wp::float64> var_131;
        wp::vec_t<3,wp::float64> var_132;
        wp::vec_t<3,wp::float64> var_133;
        wp::vec_t<3,wp::float64> var_134;
        const wp::float32 var_135 = 1.0;
        wp::float64 var_136;
        wp::float64 var_137;
        wp::float64 var_138;
        wp::vec_t<3,wp::float64> var_139;
        wp::vec_t<3,wp::float64> var_140;
        wp::vec_t<3,wp::float64> var_141;
        wp::vec_t<3,wp::float64> var_142;
        wp::vec_t<3,wp::float64> var_143;
        wp::vec_t<3,wp::float64> var_144;
        const wp::float32 var_145 = 1.0;
        wp::float64 var_146;
        const wp::float32 var_147 = 1.0;
        wp::float64 var_148;
        wp::float64 var_149;
        wp::float64 var_150;
        wp::float64 var_151;
        wp::float64 var_152;
        wp::float64 var_153;
        wp::vec_t<4,wp::float64> var_154;
        wp::int32* var_155;
        wp::float64* var_156;
        wp::int32 var_157;
        wp::int32* var_158;
        wp::float64* var_159;
        wp::int32 var_160;
        wp::float64 var_161;
        wp::float64 var_162;
        wp::float64 var_163;
        wp::float64* var_164;
        wp::float64 var_165;
        wp::float64 var_166;
        wp::float64 var_167;
        wp::vec_t<3,wp::float64>* var_168;
        wp::vec_t<3,wp::float64> var_169;
        wp::vec_t<3,wp::float64> var_170;
        const wp::int32 var_171 = 0;
        wp::float64 var_172;
        const wp::int32 var_173 = 0;
        wp::float64 var_174;
        wp::float64 var_175;
        const wp::int32 var_176 = 1;
        wp::float64 var_177;
        const wp::int32 var_178 = 0;
        wp::float64 var_179;
        wp::float64 var_180;
        const wp::int32 var_181 = 2;
        wp::float64 var_182;
        const wp::int32 var_183 = 0;
        wp::float64 var_184;
        wp::float64 var_185;
        const wp::int32 var_186 = 0;
        wp::float64 var_187;
        const wp::int32 var_188 = 1;
        wp::float64 var_189;
        wp::float64 var_190;
        const wp::int32 var_191 = 1;
        wp::float64 var_192;
        const wp::int32 var_193 = 1;
        wp::float64 var_194;
        wp::float64 var_195;
        const wp::int32 var_196 = 2;
        wp::float64 var_197;
        const wp::int32 var_198 = 1;
        wp::float64 var_199;
        wp::float64 var_200;
        const wp::int32 var_201 = 0;
        wp::float64 var_202;
        const wp::int32 var_203 = 2;
        wp::float64 var_204;
        wp::float64 var_205;
        const wp::int32 var_206 = 1;
        wp::float64 var_207;
        const wp::int32 var_208 = 2;
        wp::float64 var_209;
        wp::float64 var_210;
        const wp::int32 var_211 = 2;
        wp::float64 var_212;
        const wp::int32 var_213 = 2;
        wp::float64 var_214;
        wp::float64 var_215;
        const wp::int32 var_216 = 0;
        wp::float64 var_217;
        const wp::int32 var_218 = 3;
        wp::float64 var_219;
        wp::float64 var_220;
        const wp::int32 var_221 = 1;
        wp::float64 var_222;
        const wp::int32 var_223 = 3;
        wp::float64 var_224;
        wp::float64 var_225;
        const wp::int32 var_226 = 2;
        wp::float64 var_227;
        const wp::int32 var_228 = 3;
        wp::float64 var_229;
        wp::float64 var_230;
        wp::vec_t<12,wp::float64> var_231;
        wp::vec_t<12,wp::float64> var_232;
        wp::vec_t<12,wp::float64> var_233;
        wp::vec_t<4,wp::int32> var_234;
        wp::int32* var_235;
        wp::int32 var_236;
        wp::int32 var_237;
        wp::int32* var_238;
        wp::int32 var_239;
        wp::int32 var_240;
        wp::vec_t<2,wp::int32>* var_241;
        wp::vec_t<2,wp::int32> var_242;
        wp::vec_t<2,wp::int32> var_243;
        wp::vec_t<2,wp::int32>* var_244;
        wp::vec_t<2,wp::int32> var_245;
        wp::vec_t<2,wp::int32> var_246;
        const wp::int32 var_247 = 0;
        wp::int32 var_248;
        const wp::int32 var_249 = 1;
        wp::int32 var_250;
        const wp::int32 var_251 = 0;
        wp::int32 var_252;
        const wp::int32 var_253 = 1;
        wp::int32 var_254;
        wp::vec_t<4,wp::int32> var_255;
        wp::float64* var_256;
        wp::float64* var_257;
        wp::float64 var_258;
        wp::float64 var_259;
        wp::float64 var_260;
        const wp::int32 var_261 = 0;
        wp::int32 var_262;
        wp::int32* var_263;
        wp::int32* var_264;
        wp::int32 var_265;
        bool var_266;
        wp::int32 var_267;
        const wp::int32 var_268 = 0;
        wp::int32 var_269;
        wp::int32* var_270;
        wp::int32* var_271;
        wp::int32 var_272;
        bool var_273;
        wp::int32 var_274;
        bool var_275;
        const wp::int32 var_276 = 0;
        wp::int32 var_277;
        wp::vec_t<3,wp::float64>* var_278;
        wp::vec_t<3,wp::float64> var_279;
        wp::vec_t<3,wp::float64> var_280;
        const wp::int32 var_281 = 1;
        wp::int32 var_282;
        wp::vec_t<3,wp::float64>* var_283;
        wp::vec_t<3,wp::float64> var_284;
        wp::vec_t<3,wp::float64> var_285;
        const wp::int32 var_286 = 0;
        wp::int32 var_287;
        wp::vec_t<3,wp::float64>* var_288;
        wp::vec_t<3,wp::float64> var_289;
        wp::vec_t<3,wp::float64> var_290;
        const wp::int32 var_291 = 1;
        wp::int32 var_292;
        wp::vec_t<3,wp::float64>* var_293;
        wp::vec_t<3,wp::float64> var_294;
        wp::vec_t<3,wp::float64> var_295;
        wp::float64 var_296;
        wp::float64 var_297;
        wp::float64 var_298;
        const wp::int32 var_299 = 0;
        wp::int32 var_300;
        wp::vec_t<3,wp::float64>* var_301;
        wp::vec_t<3,wp::float64> var_302;
        wp::vec_t<3,wp::float64> var_303;
        const wp::int32 var_304 = 1;
        wp::int32 var_305;
        wp::vec_t<3,wp::float64>* var_306;
        wp::vec_t<3,wp::float64> var_307;
        wp::vec_t<3,wp::float64> var_308;
        const wp::int32 var_309 = 0;
        wp::int32 var_310;
        wp::vec_t<3,wp::float64>* var_311;
        wp::vec_t<3,wp::float64> var_312;
        wp::vec_t<3,wp::float64> var_313;
        const wp::int32 var_314 = 1;
        wp::int32 var_315;
        wp::vec_t<3,wp::float64>* var_316;
        wp::vec_t<3,wp::float64> var_317;
        wp::vec_t<3,wp::float64> var_318;
        wp::float64 var_319;
        wp::float64 var_320;
        wp::float64 var_321;
        const wp::float32 var_322 = 2.0;
        wp::float64 var_323;
        wp::float64 var_324;
        wp::float64 var_325;
        wp::float64 var_326;
        bool var_327;
        wp::float64 var_328;
        wp::float64* var_329;
        wp::float64* var_330;
        wp::float64 var_331;
        wp::float64 var_332;
        wp::float64 var_333;
        wp::float64 var_334;
        wp::float64 var_335;
        wp::float64 var_336;
        wp::float64 var_337;
        wp::vec_t<12,wp::float64> var_338;
        wp::vec_t<12,wp::float64> var_339;
        const wp::float32 var_340 = 1.0;
        wp::float64 var_341;
        bool var_342;
        wp::float64 var_343;
        wp::float64* var_344;
        wp::float64* var_345;
        wp::float64 var_346;
        wp::float64 var_347;
        wp::float64 var_348;
        wp::float64 var_349;
        wp::float64 var_350;
        wp::float64 var_351;
        wp::float64 var_352;
        wp::vec_t<12,wp::float64> var_353;
        wp::vec_t<12,wp::float64> var_354;
        wp::vec_t<12,wp::float64> var_355;
        wp::vec_t<12,wp::float64> var_356;
        wp::vec_t<12,wp::float64> var_357;
        wp::vec_t<12,wp::float64> var_358;
        wp::float64 var_359;
        const wp::int32 var_360 = 0;
        wp::int32 var_361;
        wp::vec_t<3,wp::float64>* var_362;
        wp::vec_t<3,wp::float64> var_363;
        wp::vec_t<3,wp::float64> var_364;
        const wp::int32 var_365 = 1;
        wp::int32 var_366;
        wp::vec_t<3,wp::float64>* var_367;
        wp::vec_t<3,wp::float64> var_368;
        wp::vec_t<3,wp::float64> var_369;
        const wp::int32 var_370 = 0;
        wp::int32 var_371;
        wp::vec_t<3,wp::float64>* var_372;
        wp::vec_t<3,wp::float64> var_373;
        wp::vec_t<3,wp::float64> var_374;
        const wp::int32 var_375 = 1;
        wp::int32 var_376;
        wp::vec_t<3,wp::float64>* var_377;
        wp::vec_t<3,wp::float64> var_378;
        wp::vec_t<3,wp::float64> var_379;
        const wp::int32 var_380 = 0;
        wp::vec_t<2,wp::float64>* var_381;
        wp::float64 var_382;
        wp::vec_t<2,wp::float64> var_383;
        const wp::int32 var_384 = 1;
        wp::vec_t<2,wp::float64>* var_385;
        wp::float64 var_386;
        wp::vec_t<2,wp::float64> var_387;
        const wp::float32 var_388 = 1.0;
        wp::float64 var_389;
        wp::float64 var_390;
        const wp::int32 var_391 = 0;
        wp::float64 var_392;
        wp::float64 var_393;
        const wp::int32 var_394 = 0;
        wp::float64 var_395;
        wp::float64 var_396;
        wp::float64 var_397;
        const wp::float32 var_398 = 1.0;
        wp::float64 var_399;
        wp::float64 var_400;
        const wp::int32 var_401 = 0;
        wp::float64 var_402;
        wp::float64 var_403;
        wp::float64 var_404;
        const wp::int32 var_405 = 0;
        wp::float64 var_406;
        wp::float64 var_407;
        wp::float64 var_408;
        const wp::float32 var_409 = 1.0;
        wp::float64 var_410;
        wp::float64 var_411;
        const wp::int32 var_412 = 1;
        wp::float64 var_413;
        wp::float64 var_414;
        const wp::int32 var_415 = 1;
        wp::float64 var_416;
        wp::float64 var_417;
        wp::float64 var_418;
        const wp::float32 var_419 = 1.0;
        wp::float64 var_420;
        wp::float64 var_421;
        const wp::int32 var_422 = 1;
        wp::float64 var_423;
        wp::float64 var_424;
        wp::float64 var_425;
        const wp::int32 var_426 = 1;
        wp::float64 var_427;
        wp::float64 var_428;
        wp::float64 var_429;
        const wp::float32 var_430 = 1.0;
        wp::float64 var_431;
        wp::float64 var_432;
        const wp::int32 var_433 = 2;
        wp::float64 var_434;
        wp::float64 var_435;
        const wp::int32 var_436 = 2;
        wp::float64 var_437;
        wp::float64 var_438;
        wp::float64 var_439;
        const wp::float32 var_440 = 1.0;
        wp::float64 var_441;
        wp::float64 var_442;
        const wp::int32 var_443 = 2;
        wp::float64 var_444;
        wp::float64 var_445;
        wp::float64 var_446;
        const wp::int32 var_447 = 2;
        wp::float64 var_448;
        wp::float64 var_449;
        wp::float64 var_450;
        wp::vec_t<3,wp::float64> var_451;
        const wp::float32 var_452 = 1.0;
        wp::float64 var_453;
        wp::float64 var_454;
        const wp::int32 var_455 = 0;
        wp::float64 var_456;
        wp::float64 var_457;
        const wp::int32 var_458 = 0;
        wp::float64 var_459;
        wp::float64 var_460;
        wp::float64 var_461;
        const wp::float32 var_462 = 1.0;
        wp::float64 var_463;
        wp::float64 var_464;
        const wp::int32 var_465 = 0;
        wp::float64 var_466;
        wp::float64 var_467;
        wp::float64 var_468;
        const wp::int32 var_469 = 0;
        wp::float64 var_470;
        wp::float64 var_471;
        wp::float64 var_472;
        const wp::float32 var_473 = 1.0;
        wp::float64 var_474;
        wp::float64 var_475;
        const wp::int32 var_476 = 1;
        wp::float64 var_477;
        wp::float64 var_478;
        const wp::int32 var_479 = 1;
        wp::float64 var_480;
        wp::float64 var_481;
        wp::float64 var_482;
        const wp::float32 var_483 = 1.0;
        wp::float64 var_484;
        wp::float64 var_485;
        const wp::int32 var_486 = 1;
        wp::float64 var_487;
        wp::float64 var_488;
        wp::float64 var_489;
        const wp::int32 var_490 = 1;
        wp::float64 var_491;
        wp::float64 var_492;
        wp::float64 var_493;
        const wp::float32 var_494 = 1.0;
        wp::float64 var_495;
        wp::float64 var_496;
        const wp::int32 var_497 = 2;
        wp::float64 var_498;
        wp::float64 var_499;
        const wp::int32 var_500 = 2;
        wp::float64 var_501;
        wp::float64 var_502;
        wp::float64 var_503;
        const wp::float32 var_504 = 1.0;
        wp::float64 var_505;
        wp::float64 var_506;
        const wp::int32 var_507 = 2;
        wp::float64 var_508;
        wp::float64 var_509;
        wp::float64 var_510;
        const wp::int32 var_511 = 2;
        wp::float64 var_512;
        wp::float64 var_513;
        wp::float64 var_514;
        wp::vec_t<3,wp::float64> var_515;
        wp::int32* var_516;
        wp::float64* var_517;
        wp::int32 var_518;
        wp::int32* var_519;
        wp::float64* var_520;
        wp::int32 var_521;
        wp::float64 var_522;
        wp::float64 var_523;
        wp::float64 var_524;
        wp::float64* var_525;
        wp::float64 var_526;
        wp::float64 var_527;
        wp::float64 var_528;
        wp::vec_t<3,wp::float64>* var_529;
        wp::vec_t<3,wp::float64> var_530;
        wp::vec_t<3,wp::float64> var_531;
        const wp::int32 var_532 = 0;
        wp::float64 var_533;
        const wp::float32 var_534 = 1.0;
        wp::float64 var_535;
        wp::float64 var_536;
        wp::float64 var_537;
        const wp::int32 var_538 = 1;
        wp::float64 var_539;
        const wp::float32 var_540 = 1.0;
        wp::float64 var_541;
        wp::float64 var_542;
        wp::float64 var_543;
        const wp::int32 var_544 = 2;
        wp::float64 var_545;
        const wp::float32 var_546 = 1.0;
        wp::float64 var_547;
        wp::float64 var_548;
        wp::float64 var_549;
        const wp::int32 var_550 = 0;
        wp::float64 var_551;
        wp::float64 var_552;
        const wp::int32 var_553 = 1;
        wp::float64 var_554;
        wp::float64 var_555;
        const wp::int32 var_556 = 2;
        wp::float64 var_557;
        wp::float64 var_558;
        const wp::int32 var_559 = 0;
        wp::float64 var_560;
        const wp::float32 var_561 = 1.0;
        wp::float64 var_562;
        wp::float64 var_563;
        wp::float64 var_564;
        wp::float64 var_565;
        const wp::int32 var_566 = 1;
        wp::float64 var_567;
        const wp::float32 var_568 = 1.0;
        wp::float64 var_569;
        wp::float64 var_570;
        wp::float64 var_571;
        wp::float64 var_572;
        const wp::int32 var_573 = 2;
        wp::float64 var_574;
        const wp::float32 var_575 = 1.0;
        wp::float64 var_576;
        wp::float64 var_577;
        wp::float64 var_578;
        wp::float64 var_579;
        const wp::int32 var_580 = 0;
        wp::float64 var_581;
        wp::float64 var_582;
        wp::float64 var_583;
        const wp::int32 var_584 = 1;
        wp::float64 var_585;
        wp::float64 var_586;
        wp::float64 var_587;
        const wp::int32 var_588 = 2;
        wp::float64 var_589;
        wp::float64 var_590;
        wp::float64 var_591;
        wp::vec_t<12,wp::float64> var_592;
        wp::vec_t<12,wp::float64> var_593;
        wp::float64 var_594;
        wp::float64 var_595;
        wp::vec_t<3,wp::float64> var_596;
        wp::vec_t<3,wp::float64> var_597;
        wp::float64 var_598;
        wp::vec_t<3,wp::float64> var_599;
        wp::vec_t<12,wp::float64> var_600;
        wp::vec_t<4,wp::int32> var_601;
        wp::float64 var_602;
        wp::float64 var_603;
        wp::float64 var_604;
        wp::float64 var_605;
        wp::float64 var_606;
        wp::float64 var_607;
        wp::vec_t<3,wp::float64> var_608;
        wp::vec_t<3,wp::float64> var_609;
        wp::float64 var_610;
        wp::vec_t<3,wp::float64> var_611;
        const wp::int32 var_612 = 0;
        const wp::int32 var_613 = 3;
        wp::int32 var_614;
        const wp::int32 var_615 = 0;
        wp::int32 var_616;
        wp::float64 var_617;
        const wp::int32 var_618 = 3;
        wp::int32 var_619;
        const wp::int32 var_620 = 1;
        wp::int32 var_621;
        wp::float64 var_622;
        const wp::int32 var_623 = 3;
        wp::int32 var_624;
        const wp::int32 var_625 = 2;
        wp::int32 var_626;
        wp::float64 var_627;
        wp::vec_t<3,wp::float64> var_628;
        wp::int32 var_629;
        wp::vec_t<3,wp::float64> var_630;
        const wp::int32 var_631 = 1;
        const wp::int32 var_632 = 3;
        wp::int32 var_633;
        const wp::int32 var_634 = 0;
        wp::int32 var_635;
        wp::float64 var_636;
        const wp::int32 var_637 = 3;
        wp::int32 var_638;
        const wp::int32 var_639 = 1;
        wp::int32 var_640;
        wp::float64 var_641;
        const wp::int32 var_642 = 3;
        wp::int32 var_643;
        const wp::int32 var_644 = 2;
        wp::int32 var_645;
        wp::float64 var_646;
        wp::vec_t<3,wp::float64> var_647;
        wp::int32 var_648;
        wp::vec_t<3,wp::float64> var_649;
        const wp::int32 var_650 = 2;
        const wp::int32 var_651 = 3;
        wp::int32 var_652;
        const wp::int32 var_653 = 0;
        wp::int32 var_654;
        wp::float64 var_655;
        const wp::int32 var_656 = 3;
        wp::int32 var_657;
        const wp::int32 var_658 = 1;
        wp::int32 var_659;
        wp::float64 var_660;
        const wp::int32 var_661 = 3;
        wp::int32 var_662;
        const wp::int32 var_663 = 2;
        wp::int32 var_664;
        wp::float64 var_665;
        wp::vec_t<3,wp::float64> var_666;
        wp::int32 var_667;
        wp::vec_t<3,wp::float64> var_668;
        const wp::int32 var_669 = 3;
        const wp::int32 var_670 = 3;
        wp::int32 var_671;
        const wp::int32 var_672 = 0;
        wp::int32 var_673;
        wp::float64 var_674;
        const wp::int32 var_675 = 3;
        wp::int32 var_676;
        const wp::int32 var_677 = 1;
        wp::int32 var_678;
        wp::float64 var_679;
        const wp::int32 var_680 = 3;
        wp::int32 var_681;
        const wp::int32 var_682 = 2;
        wp::int32 var_683;
        wp::float64 var_684;
        wp::vec_t<3,wp::float64> var_685;
        wp::int32 var_686;
        wp::vec_t<3,wp::float64> var_687;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::vec_t<12,wp::float64> adj_2 = {};
        wp::vec_t<4,wp::int32> adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        bool adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        bool adj_10 = {};
        wp::int32 adj_11 = {};
        bool adj_12 = {};
        wp::int32 adj_13 = {};
        bool adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        bool adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::vec_t<3,wp::int32> adj_29 = {};
        wp::vec_t<3,wp::int32> adj_30 = {};
        wp::vec_t<3,wp::int32> adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::int32 adj_38 = {};
        bool adj_39 = {};
        wp::int32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::int32 adj_42 = {};
        wp::int32 adj_43 = {};
        wp::int32 adj_44 = {};
        bool adj_45 = {};
        wp::int32 adj_46 = {};
        bool adj_47 = {};
        wp::int32 adj_48 = {};
        wp::int32 adj_49 = {};
        wp::int32 adj_50 = {};
        wp::int32 adj_51 = {};
        wp::int32 adj_52 = {};
        wp::int32 adj_53 = {};
        wp::vec_t<4,wp::int32> adj_54 = {};
        wp::vec_t<3,wp::float64> adj_55 = {};
        wp::vec_t<3,wp::float64> adj_56 = {};
        wp::vec_t<3,wp::float64> adj_57 = {};
        wp::int32 adj_58 = {};
        wp::int32 adj_59 = {};
        wp::vec_t<3,wp::float64> adj_60 = {};
        wp::vec_t<3,wp::float64> adj_61 = {};
        wp::vec_t<3,wp::float64> adj_62 = {};
        wp::int32 adj_63 = {};
        wp::int32 adj_64 = {};
        wp::vec_t<3,wp::float64> adj_65 = {};
        wp::vec_t<3,wp::float64> adj_66 = {};
        wp::vec_t<3,wp::float64> adj_67 = {};
        wp::int32 adj_68 = {};
        wp::int32 adj_69 = {};
        wp::vec_t<3,wp::float64> adj_70 = {};
        wp::vec_t<3,wp::float64> adj_71 = {};
        wp::vec_t<3,wp::float64> adj_72 = {};
        wp::vec_t<3,wp::float64> adj_73 = {};
        wp::vec_t<3,wp::float64> adj_74 = {};
        wp::vec_t<3,wp::float64> adj_75 = {};
        wp::int32 adj_76 = {};
        wp::int32 adj_77 = {};
        wp::vec_t<3,wp::float64> adj_78 = {};
        wp::vec_t<3,wp::float64> adj_79 = {};
        wp::vec_t<3,wp::float64> adj_80 = {};
        wp::int32 adj_81 = {};
        wp::int32 adj_82 = {};
        wp::vec_t<3,wp::float64> adj_83 = {};
        wp::vec_t<3,wp::float64> adj_84 = {};
        wp::vec_t<3,wp::float64> adj_85 = {};
        wp::int32 adj_86 = {};
        wp::int32 adj_87 = {};
        wp::vec_t<3,wp::float64> adj_88 = {};
        wp::vec_t<3,wp::float64> adj_89 = {};
        wp::vec_t<3,wp::float64> adj_90 = {};
        wp::float64 adj_91 = {};
        wp::float64 adj_92 = {};
        wp::float64 adj_93 = {};
        wp::float64 adj_94 = {};
        wp::float64 adj_95 = {};
        wp::float64 adj_96 = {};
        wp::float64 adj_97 = {};
        wp::float64 adj_98 = {};
        wp::vec_t<12,wp::float64> adj_99 = {};
        wp::float64 adj_100 = {};
        wp::float32 adj_101 = {};
        wp::float64 adj_102 = {};
        wp::float64 adj_103 = {};
        wp::float64 adj_104 = {};
        wp::float64 adj_105 = {};
        wp::int32 adj_106 = {};
        bool adj_107 = {};
        wp::float64 adj_108 = {};
        wp::float64 adj_109 = {};
        wp::float64 adj_110 = {};
        wp::float64 adj_111 = {};
        wp::float64 adj_112 = {};
        wp::float64 adj_113 = {};
        wp::float64 adj_114 = {};
        wp::vec_t<12,wp::float64> adj_115 = {};
        wp::vec_t<12,wp::float64> adj_116 = {};
        wp::int32 adj_117 = {};
        wp::vec_t<2,wp::float64> adj_118 = {};
        wp::float64 adj_119 = {};
        wp::vec_t<2,wp::float64> adj_120 = {};
        wp::int32 adj_121 = {};
        wp::vec_t<2,wp::float64> adj_122 = {};
        wp::float64 adj_123 = {};
        wp::vec_t<2,wp::float64> adj_124 = {};
        wp::float32 adj_125 = {};
        wp::float64 adj_126 = {};
        wp::float64 adj_127 = {};
        wp::float64 adj_128 = {};
        wp::vec_t<3,wp::float64> adj_129 = {};
        wp::vec_t<3,wp::float64> adj_130 = {};
        wp::vec_t<3,wp::float64> adj_131 = {};
        wp::vec_t<3,wp::float64> adj_132 = {};
        wp::vec_t<3,wp::float64> adj_133 = {};
        wp::vec_t<3,wp::float64> adj_134 = {};
        wp::float32 adj_135 = {};
        wp::float64 adj_136 = {};
        wp::float64 adj_137 = {};
        wp::float64 adj_138 = {};
        wp::vec_t<3,wp::float64> adj_139 = {};
        wp::vec_t<3,wp::float64> adj_140 = {};
        wp::vec_t<3,wp::float64> adj_141 = {};
        wp::vec_t<3,wp::float64> adj_142 = {};
        wp::vec_t<3,wp::float64> adj_143 = {};
        wp::vec_t<3,wp::float64> adj_144 = {};
        wp::float32 adj_145 = {};
        wp::float64 adj_146 = {};
        wp::float32 adj_147 = {};
        wp::float64 adj_148 = {};
        wp::float64 adj_149 = {};
        wp::float64 adj_150 = {};
        wp::float64 adj_151 = {};
        wp::float64 adj_152 = {};
        wp::float64 adj_153 = {};
        wp::vec_t<4,wp::float64> adj_154 = {};
        wp::int32 adj_155 = {};
        wp::float64 adj_156 = {};
        wp::int32 adj_157 = {};
        wp::int32 adj_158 = {};
        wp::float64 adj_159 = {};
        wp::int32 adj_160 = {};
        wp::float64 adj_161 = {};
        wp::float64 adj_162 = {};
        wp::float64 adj_163 = {};
        wp::float64 adj_164 = {};
        wp::float64 adj_165 = {};
        wp::float64 adj_166 = {};
        wp::float64 adj_167 = {};
        wp::vec_t<3,wp::float64> adj_168 = {};
        wp::vec_t<3,wp::float64> adj_169 = {};
        wp::vec_t<3,wp::float64> adj_170 = {};
        wp::int32 adj_171 = {};
        wp::float64 adj_172 = {};
        wp::int32 adj_173 = {};
        wp::float64 adj_174 = {};
        wp::float64 adj_175 = {};
        wp::int32 adj_176 = {};
        wp::float64 adj_177 = {};
        wp::int32 adj_178 = {};
        wp::float64 adj_179 = {};
        wp::float64 adj_180 = {};
        wp::int32 adj_181 = {};
        wp::float64 adj_182 = {};
        wp::int32 adj_183 = {};
        wp::float64 adj_184 = {};
        wp::float64 adj_185 = {};
        wp::int32 adj_186 = {};
        wp::float64 adj_187 = {};
        wp::int32 adj_188 = {};
        wp::float64 adj_189 = {};
        wp::float64 adj_190 = {};
        wp::int32 adj_191 = {};
        wp::float64 adj_192 = {};
        wp::int32 adj_193 = {};
        wp::float64 adj_194 = {};
        wp::float64 adj_195 = {};
        wp::int32 adj_196 = {};
        wp::float64 adj_197 = {};
        wp::int32 adj_198 = {};
        wp::float64 adj_199 = {};
        wp::float64 adj_200 = {};
        wp::int32 adj_201 = {};
        wp::float64 adj_202 = {};
        wp::int32 adj_203 = {};
        wp::float64 adj_204 = {};
        wp::float64 adj_205 = {};
        wp::int32 adj_206 = {};
        wp::float64 adj_207 = {};
        wp::int32 adj_208 = {};
        wp::float64 adj_209 = {};
        wp::float64 adj_210 = {};
        wp::int32 adj_211 = {};
        wp::float64 adj_212 = {};
        wp::int32 adj_213 = {};
        wp::float64 adj_214 = {};
        wp::float64 adj_215 = {};
        wp::int32 adj_216 = {};
        wp::float64 adj_217 = {};
        wp::int32 adj_218 = {};
        wp::float64 adj_219 = {};
        wp::float64 adj_220 = {};
        wp::int32 adj_221 = {};
        wp::float64 adj_222 = {};
        wp::int32 adj_223 = {};
        wp::float64 adj_224 = {};
        wp::float64 adj_225 = {};
        wp::int32 adj_226 = {};
        wp::float64 adj_227 = {};
        wp::int32 adj_228 = {};
        wp::float64 adj_229 = {};
        wp::float64 adj_230 = {};
        wp::vec_t<12,wp::float64> adj_231 = {};
        wp::vec_t<12,wp::float64> adj_232 = {};
        wp::vec_t<12,wp::float64> adj_233 = {};
        wp::vec_t<4,wp::int32> adj_234 = {};
        wp::int32 adj_235 = {};
        wp::int32 adj_236 = {};
        wp::int32 adj_237 = {};
        wp::int32 adj_238 = {};
        wp::int32 adj_239 = {};
        wp::int32 adj_240 = {};
        wp::vec_t<2,wp::int32> adj_241 = {};
        wp::vec_t<2,wp::int32> adj_242 = {};
        wp::vec_t<2,wp::int32> adj_243 = {};
        wp::vec_t<2,wp::int32> adj_244 = {};
        wp::vec_t<2,wp::int32> adj_245 = {};
        wp::vec_t<2,wp::int32> adj_246 = {};
        wp::int32 adj_247 = {};
        wp::int32 adj_248 = {};
        wp::int32 adj_249 = {};
        wp::int32 adj_250 = {};
        wp::int32 adj_251 = {};
        wp::int32 adj_252 = {};
        wp::int32 adj_253 = {};
        wp::int32 adj_254 = {};
        wp::vec_t<4,wp::int32> adj_255 = {};
        wp::float64 adj_256 = {};
        wp::float64 adj_257 = {};
        wp::float64 adj_258 = {};
        wp::float64 adj_259 = {};
        wp::float64 adj_260 = {};
        wp::int32 adj_261 = {};
        wp::int32 adj_262 = {};
        wp::int32 adj_263 = {};
        wp::int32 adj_264 = {};
        wp::int32 adj_265 = {};
        bool adj_266 = {};
        wp::int32 adj_267 = {};
        wp::int32 adj_268 = {};
        wp::int32 adj_269 = {};
        wp::int32 adj_270 = {};
        wp::int32 adj_271 = {};
        wp::int32 adj_272 = {};
        bool adj_273 = {};
        wp::int32 adj_274 = {};
        bool adj_275 = {};
        wp::int32 adj_276 = {};
        wp::int32 adj_277 = {};
        wp::vec_t<3,wp::float64> adj_278 = {};
        wp::vec_t<3,wp::float64> adj_279 = {};
        wp::vec_t<3,wp::float64> adj_280 = {};
        wp::int32 adj_281 = {};
        wp::int32 adj_282 = {};
        wp::vec_t<3,wp::float64> adj_283 = {};
        wp::vec_t<3,wp::float64> adj_284 = {};
        wp::vec_t<3,wp::float64> adj_285 = {};
        wp::int32 adj_286 = {};
        wp::int32 adj_287 = {};
        wp::vec_t<3,wp::float64> adj_288 = {};
        wp::vec_t<3,wp::float64> adj_289 = {};
        wp::vec_t<3,wp::float64> adj_290 = {};
        wp::int32 adj_291 = {};
        wp::int32 adj_292 = {};
        wp::vec_t<3,wp::float64> adj_293 = {};
        wp::vec_t<3,wp::float64> adj_294 = {};
        wp::vec_t<3,wp::float64> adj_295 = {};
        wp::float64 adj_296 = {};
        wp::float64 adj_297 = {};
        wp::float64 adj_298 = {};
        wp::int32 adj_299 = {};
        wp::int32 adj_300 = {};
        wp::vec_t<3,wp::float64> adj_301 = {};
        wp::vec_t<3,wp::float64> adj_302 = {};
        wp::vec_t<3,wp::float64> adj_303 = {};
        wp::int32 adj_304 = {};
        wp::int32 adj_305 = {};
        wp::vec_t<3,wp::float64> adj_306 = {};
        wp::vec_t<3,wp::float64> adj_307 = {};
        wp::vec_t<3,wp::float64> adj_308 = {};
        wp::int32 adj_309 = {};
        wp::int32 adj_310 = {};
        wp::vec_t<3,wp::float64> adj_311 = {};
        wp::vec_t<3,wp::float64> adj_312 = {};
        wp::vec_t<3,wp::float64> adj_313 = {};
        wp::int32 adj_314 = {};
        wp::int32 adj_315 = {};
        wp::vec_t<3,wp::float64> adj_316 = {};
        wp::vec_t<3,wp::float64> adj_317 = {};
        wp::vec_t<3,wp::float64> adj_318 = {};
        wp::float64 adj_319 = {};
        wp::float64 adj_320 = {};
        wp::float64 adj_321 = {};
        wp::float32 adj_322 = {};
        wp::float64 adj_323 = {};
        wp::float64 adj_324 = {};
        wp::float64 adj_325 = {};
        wp::float64 adj_326 = {};
        bool adj_327 = {};
        wp::float64 adj_328 = {};
        wp::float64 adj_329 = {};
        wp::float64 adj_330 = {};
        wp::float64 adj_331 = {};
        wp::float64 adj_332 = {};
        wp::float64 adj_333 = {};
        wp::float64 adj_334 = {};
        wp::float64 adj_335 = {};
        wp::float64 adj_336 = {};
        wp::float64 adj_337 = {};
        wp::vec_t<12,wp::float64> adj_338 = {};
        wp::vec_t<12,wp::float64> adj_339 = {};
        wp::float32 adj_340 = {};
        wp::float64 adj_341 = {};
        bool adj_342 = {};
        wp::float64 adj_343 = {};
        wp::float64 adj_344 = {};
        wp::float64 adj_345 = {};
        wp::float64 adj_346 = {};
        wp::float64 adj_347 = {};
        wp::float64 adj_348 = {};
        wp::float64 adj_349 = {};
        wp::float64 adj_350 = {};
        wp::float64 adj_351 = {};
        wp::float64 adj_352 = {};
        wp::vec_t<12,wp::float64> adj_353 = {};
        wp::vec_t<12,wp::float64> adj_354 = {};
        wp::vec_t<12,wp::float64> adj_355 = {};
        wp::vec_t<12,wp::float64> adj_356 = {};
        wp::vec_t<12,wp::float64> adj_357 = {};
        wp::vec_t<12,wp::float64> adj_358 = {};
        wp::float64 adj_359 = {};
        wp::int32 adj_360 = {};
        wp::int32 adj_361 = {};
        wp::vec_t<3,wp::float64> adj_362 = {};
        wp::vec_t<3,wp::float64> adj_363 = {};
        wp::vec_t<3,wp::float64> adj_364 = {};
        wp::int32 adj_365 = {};
        wp::int32 adj_366 = {};
        wp::vec_t<3,wp::float64> adj_367 = {};
        wp::vec_t<3,wp::float64> adj_368 = {};
        wp::vec_t<3,wp::float64> adj_369 = {};
        wp::int32 adj_370 = {};
        wp::int32 adj_371 = {};
        wp::vec_t<3,wp::float64> adj_372 = {};
        wp::vec_t<3,wp::float64> adj_373 = {};
        wp::vec_t<3,wp::float64> adj_374 = {};
        wp::int32 adj_375 = {};
        wp::int32 adj_376 = {};
        wp::vec_t<3,wp::float64> adj_377 = {};
        wp::vec_t<3,wp::float64> adj_378 = {};
        wp::vec_t<3,wp::float64> adj_379 = {};
        wp::int32 adj_380 = {};
        wp::vec_t<2,wp::float64> adj_381 = {};
        wp::float64 adj_382 = {};
        wp::vec_t<2,wp::float64> adj_383 = {};
        wp::int32 adj_384 = {};
        wp::vec_t<2,wp::float64> adj_385 = {};
        wp::float64 adj_386 = {};
        wp::vec_t<2,wp::float64> adj_387 = {};
        wp::float32 adj_388 = {};
        wp::float64 adj_389 = {};
        wp::float64 adj_390 = {};
        wp::int32 adj_391 = {};
        wp::float64 adj_392 = {};
        wp::float64 adj_393 = {};
        wp::int32 adj_394 = {};
        wp::float64 adj_395 = {};
        wp::float64 adj_396 = {};
        wp::float64 adj_397 = {};
        wp::float32 adj_398 = {};
        wp::float64 adj_399 = {};
        wp::float64 adj_400 = {};
        wp::int32 adj_401 = {};
        wp::float64 adj_402 = {};
        wp::float64 adj_403 = {};
        wp::float64 adj_404 = {};
        wp::int32 adj_405 = {};
        wp::float64 adj_406 = {};
        wp::float64 adj_407 = {};
        wp::float64 adj_408 = {};
        wp::float32 adj_409 = {};
        wp::float64 adj_410 = {};
        wp::float64 adj_411 = {};
        wp::int32 adj_412 = {};
        wp::float64 adj_413 = {};
        wp::float64 adj_414 = {};
        wp::int32 adj_415 = {};
        wp::float64 adj_416 = {};
        wp::float64 adj_417 = {};
        wp::float64 adj_418 = {};
        wp::float32 adj_419 = {};
        wp::float64 adj_420 = {};
        wp::float64 adj_421 = {};
        wp::int32 adj_422 = {};
        wp::float64 adj_423 = {};
        wp::float64 adj_424 = {};
        wp::float64 adj_425 = {};
        wp::int32 adj_426 = {};
        wp::float64 adj_427 = {};
        wp::float64 adj_428 = {};
        wp::float64 adj_429 = {};
        wp::float32 adj_430 = {};
        wp::float64 adj_431 = {};
        wp::float64 adj_432 = {};
        wp::int32 adj_433 = {};
        wp::float64 adj_434 = {};
        wp::float64 adj_435 = {};
        wp::int32 adj_436 = {};
        wp::float64 adj_437 = {};
        wp::float64 adj_438 = {};
        wp::float64 adj_439 = {};
        wp::float32 adj_440 = {};
        wp::float64 adj_441 = {};
        wp::float64 adj_442 = {};
        wp::int32 adj_443 = {};
        wp::float64 adj_444 = {};
        wp::float64 adj_445 = {};
        wp::float64 adj_446 = {};
        wp::int32 adj_447 = {};
        wp::float64 adj_448 = {};
        wp::float64 adj_449 = {};
        wp::float64 adj_450 = {};
        wp::vec_t<3,wp::float64> adj_451 = {};
        wp::float32 adj_452 = {};
        wp::float64 adj_453 = {};
        wp::float64 adj_454 = {};
        wp::int32 adj_455 = {};
        wp::float64 adj_456 = {};
        wp::float64 adj_457 = {};
        wp::int32 adj_458 = {};
        wp::float64 adj_459 = {};
        wp::float64 adj_460 = {};
        wp::float64 adj_461 = {};
        wp::float32 adj_462 = {};
        wp::float64 adj_463 = {};
        wp::float64 adj_464 = {};
        wp::int32 adj_465 = {};
        wp::float64 adj_466 = {};
        wp::float64 adj_467 = {};
        wp::float64 adj_468 = {};
        wp::int32 adj_469 = {};
        wp::float64 adj_470 = {};
        wp::float64 adj_471 = {};
        wp::float64 adj_472 = {};
        wp::float32 adj_473 = {};
        wp::float64 adj_474 = {};
        wp::float64 adj_475 = {};
        wp::int32 adj_476 = {};
        wp::float64 adj_477 = {};
        wp::float64 adj_478 = {};
        wp::int32 adj_479 = {};
        wp::float64 adj_480 = {};
        wp::float64 adj_481 = {};
        wp::float64 adj_482 = {};
        wp::float32 adj_483 = {};
        wp::float64 adj_484 = {};
        wp::float64 adj_485 = {};
        wp::int32 adj_486 = {};
        wp::float64 adj_487 = {};
        wp::float64 adj_488 = {};
        wp::float64 adj_489 = {};
        wp::int32 adj_490 = {};
        wp::float64 adj_491 = {};
        wp::float64 adj_492 = {};
        wp::float64 adj_493 = {};
        wp::float32 adj_494 = {};
        wp::float64 adj_495 = {};
        wp::float64 adj_496 = {};
        wp::int32 adj_497 = {};
        wp::float64 adj_498 = {};
        wp::float64 adj_499 = {};
        wp::int32 adj_500 = {};
        wp::float64 adj_501 = {};
        wp::float64 adj_502 = {};
        wp::float64 adj_503 = {};
        wp::float32 adj_504 = {};
        wp::float64 adj_505 = {};
        wp::float64 adj_506 = {};
        wp::int32 adj_507 = {};
        wp::float64 adj_508 = {};
        wp::float64 adj_509 = {};
        wp::float64 adj_510 = {};
        wp::int32 adj_511 = {};
        wp::float64 adj_512 = {};
        wp::float64 adj_513 = {};
        wp::float64 adj_514 = {};
        wp::vec_t<3,wp::float64> adj_515 = {};
        wp::int32 adj_516 = {};
        wp::float64 adj_517 = {};
        wp::int32 adj_518 = {};
        wp::int32 adj_519 = {};
        wp::float64 adj_520 = {};
        wp::int32 adj_521 = {};
        wp::float64 adj_522 = {};
        wp::float64 adj_523 = {};
        wp::float64 adj_524 = {};
        wp::float64 adj_525 = {};
        wp::float64 adj_526 = {};
        wp::float64 adj_527 = {};
        wp::float64 adj_528 = {};
        wp::vec_t<3,wp::float64> adj_529 = {};
        wp::vec_t<3,wp::float64> adj_530 = {};
        wp::vec_t<3,wp::float64> adj_531 = {};
        wp::int32 adj_532 = {};
        wp::float64 adj_533 = {};
        wp::float32 adj_534 = {};
        wp::float64 adj_535 = {};
        wp::float64 adj_536 = {};
        wp::float64 adj_537 = {};
        wp::int32 adj_538 = {};
        wp::float64 adj_539 = {};
        wp::float32 adj_540 = {};
        wp::float64 adj_541 = {};
        wp::float64 adj_542 = {};
        wp::float64 adj_543 = {};
        wp::int32 adj_544 = {};
        wp::float64 adj_545 = {};
        wp::float32 adj_546 = {};
        wp::float64 adj_547 = {};
        wp::float64 adj_548 = {};
        wp::float64 adj_549 = {};
        wp::int32 adj_550 = {};
        wp::float64 adj_551 = {};
        wp::float64 adj_552 = {};
        wp::int32 adj_553 = {};
        wp::float64 adj_554 = {};
        wp::float64 adj_555 = {};
        wp::int32 adj_556 = {};
        wp::float64 adj_557 = {};
        wp::float64 adj_558 = {};
        wp::int32 adj_559 = {};
        wp::float64 adj_560 = {};
        wp::float32 adj_561 = {};
        wp::float64 adj_562 = {};
        wp::float64 adj_563 = {};
        wp::float64 adj_564 = {};
        wp::float64 adj_565 = {};
        wp::int32 adj_566 = {};
        wp::float64 adj_567 = {};
        wp::float32 adj_568 = {};
        wp::float64 adj_569 = {};
        wp::float64 adj_570 = {};
        wp::float64 adj_571 = {};
        wp::float64 adj_572 = {};
        wp::int32 adj_573 = {};
        wp::float64 adj_574 = {};
        wp::float32 adj_575 = {};
        wp::float64 adj_576 = {};
        wp::float64 adj_577 = {};
        wp::float64 adj_578 = {};
        wp::float64 adj_579 = {};
        wp::int32 adj_580 = {};
        wp::float64 adj_581 = {};
        wp::float64 adj_582 = {};
        wp::float64 adj_583 = {};
        wp::int32 adj_584 = {};
        wp::float64 adj_585 = {};
        wp::float64 adj_586 = {};
        wp::float64 adj_587 = {};
        wp::int32 adj_588 = {};
        wp::float64 adj_589 = {};
        wp::float64 adj_590 = {};
        wp::float64 adj_591 = {};
        wp::vec_t<12,wp::float64> adj_592 = {};
        wp::vec_t<12,wp::float64> adj_593 = {};
        wp::float64 adj_594 = {};
        wp::float64 adj_595 = {};
        wp::vec_t<3,wp::float64> adj_596 = {};
        wp::vec_t<3,wp::float64> adj_597 = {};
        wp::float64 adj_598 = {};
        wp::vec_t<3,wp::float64> adj_599 = {};
        wp::vec_t<12,wp::float64> adj_600 = {};
        wp::vec_t<4,wp::int32> adj_601 = {};
        wp::float64 adj_602 = {};
        wp::float64 adj_603 = {};
        wp::float64 adj_604 = {};
        wp::float64 adj_605 = {};
        wp::float64 adj_606 = {};
        wp::float64 adj_607 = {};
        wp::vec_t<3,wp::float64> adj_608 = {};
        wp::vec_t<3,wp::float64> adj_609 = {};
        wp::float64 adj_610 = {};
        wp::vec_t<3,wp::float64> adj_611 = {};
        wp::int32 adj_612 = {};
        wp::int32 adj_613 = {};
        wp::int32 adj_614 = {};
        wp::int32 adj_615 = {};
        wp::int32 adj_616 = {};
        wp::float64 adj_617 = {};
        wp::int32 adj_618 = {};
        wp::int32 adj_619 = {};
        wp::int32 adj_620 = {};
        wp::int32 adj_621 = {};
        wp::float64 adj_622 = {};
        wp::int32 adj_623 = {};
        wp::int32 adj_624 = {};
        wp::int32 adj_625 = {};
        wp::int32 adj_626 = {};
        wp::float64 adj_627 = {};
        wp::vec_t<3,wp::float64> adj_628 = {};
        wp::int32 adj_629 = {};
        wp::vec_t<3,wp::float64> adj_630 = {};
        wp::int32 adj_631 = {};
        wp::int32 adj_632 = {};
        wp::int32 adj_633 = {};
        wp::int32 adj_634 = {};
        wp::int32 adj_635 = {};
        wp::float64 adj_636 = {};
        wp::int32 adj_637 = {};
        wp::int32 adj_638 = {};
        wp::int32 adj_639 = {};
        wp::int32 adj_640 = {};
        wp::float64 adj_641 = {};
        wp::int32 adj_642 = {};
        wp::int32 adj_643 = {};
        wp::int32 adj_644 = {};
        wp::int32 adj_645 = {};
        wp::float64 adj_646 = {};
        wp::vec_t<3,wp::float64> adj_647 = {};
        wp::int32 adj_648 = {};
        wp::vec_t<3,wp::float64> adj_649 = {};
        wp::int32 adj_650 = {};
        wp::int32 adj_651 = {};
        wp::int32 adj_652 = {};
        wp::int32 adj_653 = {};
        wp::int32 adj_654 = {};
        wp::float64 adj_655 = {};
        wp::int32 adj_656 = {};
        wp::int32 adj_657 = {};
        wp::int32 adj_658 = {};
        wp::int32 adj_659 = {};
        wp::float64 adj_660 = {};
        wp::int32 adj_661 = {};
        wp::int32 adj_662 = {};
        wp::int32 adj_663 = {};
        wp::int32 adj_664 = {};
        wp::float64 adj_665 = {};
        wp::vec_t<3,wp::float64> adj_666 = {};
        wp::int32 adj_667 = {};
        wp::vec_t<3,wp::float64> adj_668 = {};
        wp::int32 adj_669 = {};
        wp::int32 adj_670 = {};
        wp::int32 adj_671 = {};
        wp::int32 adj_672 = {};
        wp::int32 adj_673 = {};
        wp::float64 adj_674 = {};
        wp::int32 adj_675 = {};
        wp::int32 adj_676 = {};
        wp::int32 adj_677 = {};
        wp::int32 adj_678 = {};
        wp::float64 adj_679 = {};
        wp::int32 adj_680 = {};
        wp::int32 adj_681 = {};
        wp::int32 adj_682 = {};
        wp::int32 adj_683 = {};
        wp::float64 adj_684 = {};
        wp::vec_t<3,wp::float64> adj_685 = {};
        wp::int32 adj_686 = {};
        wp::vec_t<3,wp::float64> adj_687 = {};
        //---------
        // forward
        // def grad_IPC_collisions(                                                               <L 65>
        // idx = wp.tid()                                                                         <L 97>
        var_0 = builtin_tid1d();
        // grad_b = wp.vec(length=12, dtype=wp.float64)                                           <L 99>
        var_2 = wp::vec_t<12,wp::float64>();
        // vInd = wp.vec4i()                                                                      <L 100>
        var_3 = wp::vec_t<4,wp::int32>();
        // if collision_type[idx] == TRIANGLE_POINT_PAIR or collision_type[idx] == POINT_TRIANGLE_PAIR:       <L 101>
        var_4 = wp::address(var_collision_type, var_0);
        var_7 = wp::load(var_4);
        var_6 = (var_7 == var_5);
        var_8 = wp::address(var_collision_type, var_0);
        var_11 = wp::load(var_8);
        var_10 = (var_11 == var_9);
        var_12 = var_6 || var_10;
        if (var_12) {
            // svi = wp.select(collision_type[idx] == POINT_TRIANGLE_PAIR, nodeJ[idx], nodeI[idx])       <L 102>
            var_13 = wp::address(var_collision_type, var_0);
            var_15 = wp::load(var_13);
            var_14 = (var_15 == var_9);
            var_16 = wp::address(var_nodeJ, var_0);
            var_17 = wp::address(var_nodeI, var_0);
            var_19 = wp::load(var_16);
            var_20 = wp::load(var_17);
            var_18 = wp::select(var_14, var_19, var_20);
            // faceJ = wp.select(collision_type[idx] == POINT_TRIANGLE_PAIR, nodeI[idx], nodeJ[idx])       <L 103>
            var_21 = wp::address(var_collision_type, var_0);
            var_23 = wp::load(var_21);
            var_22 = (var_23 == var_9);
            var_24 = wp::address(var_nodeI, var_0);
            var_25 = wp::address(var_nodeJ, var_0);
            var_27 = wp::load(var_24);
            var_28 = wp::load(var_25);
            var_26 = wp::select(var_22, var_27, var_28);
            // faceVInd = face[faceJ]                                                             <L 104>
            var_29 = wp::address(var_face, var_26);
            var_31 = wp::load(var_29);
            var_30 = wp::copy(var_31);
            // xI = surf_vi[svi]                                                                  <L 105>
            var_32 = wp::address(var_surf_vi, var_18);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // if (env_states[node2env[xI]] == ENV_STATE_INVALID) or (env_states[node2env[xI]] == ENV_STATE_NEWTON_SOLVED):       <L 106>
            var_35 = wp::address(var_node2env, var_33);
            var_37 = wp::load(var_35);
            var_36 = wp::address(var_env_states, var_37);
            var_40 = wp::load(var_36);
            var_39 = (var_40 == var_38);
            var_41 = wp::address(var_node2env, var_33);
            var_43 = wp::load(var_41);
            var_42 = wp::address(var_env_states, var_43);
            var_46 = wp::load(var_42);
            var_45 = (var_46 == var_44);
            var_47 = var_39 || var_45;
            if (var_47) {
                // return                                                                         <L 107>
                goto label0;
            }
            // vInd = wp.vec4i(xI, faceVInd[0], faceVInd[1], faceVInd[2])                         <L 108>
            var_49 = wp::extract(var_30, var_48);
            var_51 = wp::extract(var_30, var_50);
            var_53 = wp::extract(var_30, var_52);
            var_54 = wp::vec_t<4,wp::int32>(var_33, var_49, var_51, var_53);
            // p = x[xI]                                                                          <L 109>
            var_55 = wp::address(var_x, var_33);
            var_57 = wp::load(var_55);
            var_56 = wp::copy(var_57);
            // t0 = x[faceVInd[0]]                                                                <L 110>
            var_59 = wp::extract(var_30, var_58);
            var_60 = wp::address(var_x, var_59);
            var_62 = wp::load(var_60);
            var_61 = wp::copy(var_62);
            // t1 = x[faceVInd[1]]                                                                <L 111>
            var_64 = wp::extract(var_30, var_63);
            var_65 = wp::address(var_x, var_64);
            var_67 = wp::load(var_65);
            var_66 = wp::copy(var_67);
            // t2 = x[faceVInd[2]]                                                                <L 112>
            var_69 = wp::extract(var_30, var_68);
            var_70 = wp::address(var_x, var_69);
            var_72 = wp::load(var_70);
            var_71 = wp::copy(var_72);
            // hat_p = hat_x[xI]                                                                  <L 113>
            var_73 = wp::address(var_hat_x, var_33);
            var_75 = wp::load(var_73);
            var_74 = wp::copy(var_75);
            // hat_t0 = hat_x[faceVInd[0]]                                                        <L 114>
            var_77 = wp::extract(var_30, var_76);
            var_78 = wp::address(var_hat_x, var_77);
            var_80 = wp::load(var_78);
            var_79 = wp::copy(var_80);
            // hat_t1 = hat_x[faceVInd[1]]                                                        <L 115>
            var_82 = wp::extract(var_30, var_81);
            var_83 = wp::address(var_hat_x, var_82);
            var_85 = wp::load(var_83);
            var_84 = wp::copy(var_85);
            // hat_t2 = hat_x[faceVInd[2]]                                                        <L 116>
            var_87 = wp::extract(var_30, var_86);
            var_88 = wp::address(var_hat_x, var_87);
            var_90 = wp::load(var_88);
            var_89 = wp::copy(var_90);
            // xi = node_xi[svi] + face_xi[faceJ]                                                 <L 117>
            var_91 = wp::address(var_node_xi, var_18);
            var_92 = wp::address(var_face_xi, var_26);
            var_94 = wp::load(var_91);
            var_95 = wp::load(var_92);
            var_93 = wp::add(var_94, var_95);
            // dsq = point_triangle_distance(p, t0, t1, t2) - xi * xi                             <L 118>
            var_96 = point_triangle_distance_0(var_56, var_61, var_66, var_71);
            var_97 = wp::mul(var_93, var_93);
            var_98 = wp::sub(var_96, var_97);
            // grad_b = point_triangle_distance_gradient(p, t0, t1, t2)                           <L 119>
            var_99 = point_triangle_distance_gradient_0(var_56, var_61, var_66, var_71);
            // activeGap2 = dhat * dhat + wp.float64(2.0) * dhat * xi                             <L 122>
            var_100 = wp::mul(var_dhat, var_dhat);
            var_102 = wp::float64(var_101);
            var_103 = wp::mul(var_102, var_dhat);
            var_104 = wp::mul(var_103, var_93);
            var_105 = wp::add(var_100, var_104);
            // if E == BARRIER:                                                                   <L 123>
            var_107 = (var_E == var_106);
            if (var_107) {
                // db_div_dd = scale * int_w_PTEE * node_area[svi] * dhat * barrier.barrier_gradient(dsq, activeGap2, kappa)       <L 124>
                var_108 = wp::mul(var_scale, var_int_w_PTEE);
                var_109 = wp::address(var_node_area, var_18);
                var_111 = wp::load(var_109);
                var_110 = wp::mul(var_108, var_111);
                var_112 = wp::mul(var_110, var_dhat);
                var_113 = barrier_gradient_0(var_98, var_105, var_kappa);
                var_114 = wp::mul(var_112, var_113);
                // grad_b *= db_div_dd                                                            <L 125>
                var_115 = wp::mul(var_99, var_114);
            }
            var_116 = wp::where(var_107, var_115, var_99);
            if (!var_107) {
                // u = closest_points[idx][0]                                                     <L 127>
                var_118 = wp::address(var_closest_points, var_0);
                var_120 = wp::load(var_118);
                var_119 = wp::extract(var_120, var_117);
                // v = closest_points[idx][1]                                                     <L 128>
                var_122 = wp::address(var_closest_points, var_0);
                var_124 = wp::load(var_122);
                var_123 = wp::extract(var_124, var_121);
                // xIcoord = p - (wp.float64(1.0) - u - v) * t0 - u * t1 - v * t2                 <L 129>
                var_126 = wp::float64(var_125);
                var_127 = wp::sub(var_126, var_119);
                var_128 = wp::sub(var_127, var_123);
                var_129 = wp::mul(var_128, var_61);
                var_130 = wp::sub(var_56, var_129);
                var_131 = wp::mul(var_119, var_66);
                var_132 = wp::sub(var_130, var_131);
                var_133 = wp::mul(var_123, var_71);
                var_134 = wp::sub(var_132, var_133);
                // hat_xIcoord = hat_p - (wp.float64(1.0) - u - v) * hat_t0 - u * hat_t1 - v * hat_t2       <L 130>
                var_136 = wp::float64(var_135);
                var_137 = wp::sub(var_136, var_119);
                var_138 = wp::sub(var_137, var_123);
                var_139 = wp::mul(var_138, var_79);
                var_140 = wp::sub(var_74, var_139);
                var_141 = wp::mul(var_119, var_84);
                var_142 = wp::sub(var_140, var_141);
                var_143 = wp::mul(var_123, var_89);
                var_144 = wp::sub(var_142, var_143);
                // weights = wp.vec4d(wp.float64(1.0), -(wp.float64(1.0) - u - v), -u, -v)        <L 131>
                var_146 = wp::float64(var_145);
                var_148 = wp::float64(var_147);
                var_149 = wp::sub(var_148, var_119);
                var_150 = wp::sub(var_149, var_123);
                var_151 = wp::neg(var_150);
                var_152 = wp::neg(var_119);
                var_153 = wp::neg(var_123);
                var_154 = wp::vec_t<4,wp::float64>(var_146, var_151, var_152, var_153);
                // coeff = get_combined_coulomb_friction(mu_body[bodyI[idx]], mu_body[bodyJ[idx]]) * c_lambda[idx] * scale       <L 132>
                var_155 = wp::address(var_bodyI, var_0);
                var_157 = wp::load(var_155);
                var_156 = wp::address(var_mu_body, var_157);
                var_158 = wp::address(var_bodyJ, var_0);
                var_160 = wp::load(var_158);
                var_159 = wp::address(var_mu_body, var_160);
                var_162 = wp::load(var_156);
                var_163 = wp::load(var_159);
                var_161 = get_combined_coulomb_friction_0(var_162, var_163);
                var_164 = wp::address(var_c_lambda, var_0);
                var_166 = wp::load(var_164);
                var_165 = wp::mul(var_161, var_166);
                var_167 = wp::mul(var_165, var_scale);
                // grad_f = friction.friction_gradient(normal[idx], xIcoord, hat_xIcoord, hat_h, epsv, coeff)       <L 133>
                var_168 = wp::address(var_normal, var_0);
                var_170 = wp::load(var_168);
                var_169 = friction_gradient_0(var_170, var_134, var_144, var_hat_h, var_epsv, var_167);
                // grad_b = vec12d(                                                               <L 134>
                // grad_f[0] * weights[0],                                                        <L 135>
                var_172 = wp::extract(var_169, var_171);
                var_174 = wp::extract(var_154, var_173);
                var_175 = wp::mul(var_172, var_174);
                // grad_f[1] * weights[0],                                                        <L 136>
                var_177 = wp::extract(var_169, var_176);
                var_179 = wp::extract(var_154, var_178);
                var_180 = wp::mul(var_177, var_179);
                // grad_f[2] * weights[0],                                                        <L 137>
                var_182 = wp::extract(var_169, var_181);
                var_184 = wp::extract(var_154, var_183);
                var_185 = wp::mul(var_182, var_184);
                // grad_f[0] * weights[1],                                                        <L 138>
                var_187 = wp::extract(var_169, var_186);
                var_189 = wp::extract(var_154, var_188);
                var_190 = wp::mul(var_187, var_189);
                // grad_f[1] * weights[1],                                                        <L 139>
                var_192 = wp::extract(var_169, var_191);
                var_194 = wp::extract(var_154, var_193);
                var_195 = wp::mul(var_192, var_194);
                // grad_f[2] * weights[1],                                                        <L 140>
                var_197 = wp::extract(var_169, var_196);
                var_199 = wp::extract(var_154, var_198);
                var_200 = wp::mul(var_197, var_199);
                // grad_f[0] * weights[2],                                                        <L 141>
                var_202 = wp::extract(var_169, var_201);
                var_204 = wp::extract(var_154, var_203);
                var_205 = wp::mul(var_202, var_204);
                // grad_f[1] * weights[2],                                                        <L 142>
                var_207 = wp::extract(var_169, var_206);
                var_209 = wp::extract(var_154, var_208);
                var_210 = wp::mul(var_207, var_209);
                // grad_f[2] * weights[2],                                                        <L 143>
                var_212 = wp::extract(var_169, var_211);
                var_214 = wp::extract(var_154, var_213);
                var_215 = wp::mul(var_212, var_214);
                // grad_f[0] * weights[3],                                                        <L 144>
                var_217 = wp::extract(var_169, var_216);
                var_219 = wp::extract(var_154, var_218);
                var_220 = wp::mul(var_217, var_219);
                // grad_f[1] * weights[3],                                                        <L 145>
                var_222 = wp::extract(var_169, var_221);
                var_224 = wp::extract(var_154, var_223);
                var_225 = wp::mul(var_222, var_224);
                // grad_f[2] * weights[3],                                                        <L 146>
                var_227 = wp::extract(var_169, var_226);
                var_229 = wp::extract(var_154, var_228);
                var_230 = wp::mul(var_227, var_229);
                var_231 = wp::vec_t<12,wp::float64>({var_175, var_180, var_185, var_190, var_195, var_200, var_205, var_210, var_215, var_220, var_225, var_230});
            }
            var_232 = wp::where(var_107, var_116, var_231);
        }
        var_233 = wp::where(var_12, var_232, var_2);
        var_234 = wp::where(var_12, var_54, var_3);
        if (!var_12) {
            // edgeI = nodeI[idx]                                                                 <L 152>
            var_235 = wp::address(var_nodeI, var_0);
            var_237 = wp::load(var_235);
            var_236 = wp::copy(var_237);
            // edgeJ = nodeJ[idx]                                                                 <L 153>
            var_238 = wp::address(var_nodeJ, var_0);
            var_240 = wp::load(var_238);
            var_239 = wp::copy(var_240);
            // edgeVIndI = edge[edgeI]                                                            <L 154>
            var_241 = wp::address(var_edge, var_236);
            var_243 = wp::load(var_241);
            var_242 = wp::copy(var_243);
            // edgeVIndJ = edge[edgeJ]                                                            <L 155>
            var_244 = wp::address(var_edge, var_239);
            var_246 = wp::load(var_244);
            var_245 = wp::copy(var_246);
            // vInd = wp.vec4i(edgeVIndI[0], edgeVIndI[1], edgeVIndJ[0], edgeVIndJ[1])            <L 156>
            var_248 = wp::extract(var_242, var_247);
            var_250 = wp::extract(var_242, var_249);
            var_252 = wp::extract(var_245, var_251);
            var_254 = wp::extract(var_245, var_253);
            var_255 = wp::vec_t<4,wp::int32>(var_248, var_250, var_252, var_254);
            // xi = edge_xi[edgeI] + edge_xi[edgeJ]                                               <L 157>
            var_256 = wp::address(var_edge_xi, var_236);
            var_257 = wp::address(var_edge_xi, var_239);
            var_259 = wp::load(var_256);
            var_260 = wp::load(var_257);
            var_258 = wp::add(var_259, var_260);
            // if (env_states[node2env[edgeVIndI[0]]] == ENV_STATE_INVALID) or (env_states[node2env[edgeVIndI[0]]] == ENV_STATE_NEWTON_SOLVED):       <L 158>
            var_262 = wp::extract(var_242, var_261);
            var_263 = wp::address(var_node2env, var_262);
            var_265 = wp::load(var_263);
            var_264 = wp::address(var_env_states, var_265);
            var_267 = wp::load(var_264);
            var_266 = (var_267 == var_38);
            var_269 = wp::extract(var_242, var_268);
            var_270 = wp::address(var_node2env, var_269);
            var_272 = wp::load(var_270);
            var_271 = wp::address(var_env_states, var_272);
            var_274 = wp::load(var_271);
            var_273 = (var_274 == var_44);
            var_275 = var_266 || var_273;
            if (var_275) {
                // return                                                                         <L 159>
                goto label1;
            }
            // ea0 = x[edgeVIndI[0]]                                                              <L 160>
            var_277 = wp::extract(var_242, var_276);
            var_278 = wp::address(var_x, var_277);
            var_280 = wp::load(var_278);
            var_279 = wp::copy(var_280);
            // ea1 = x[edgeVIndI[1]]                                                              <L 161>
            var_282 = wp::extract(var_242, var_281);
            var_283 = wp::address(var_x, var_282);
            var_285 = wp::load(var_283);
            var_284 = wp::copy(var_285);
            // eb0 = x[edgeVIndJ[0]]                                                              <L 162>
            var_287 = wp::extract(var_245, var_286);
            var_288 = wp::address(var_x, var_287);
            var_290 = wp::load(var_288);
            var_289 = wp::copy(var_290);
            // eb1 = x[edgeVIndJ[1]]                                                              <L 163>
            var_292 = wp::extract(var_245, var_291);
            var_293 = wp::address(var_x, var_292);
            var_295 = wp::load(var_293);
            var_294 = wp::copy(var_295);
            // dsq = edge_edge_distance(ea0, ea1, eb0, eb1) - xi * xi                             <L 164>
            var_296 = edge_edge_distance_0(var_279, var_284, var_289, var_294);
            var_297 = wp::mul(var_258, var_258);
            var_298 = wp::sub(var_296, var_297);
            // EA0 = X[edgeVIndI[0]]                                                              <L 166>
            var_300 = wp::extract(var_242, var_299);
            var_301 = wp::address(var_X, var_300);
            var_303 = wp::load(var_301);
            var_302 = wp::copy(var_303);
            // EA1 = X[edgeVIndI[1]]                                                              <L 167>
            var_305 = wp::extract(var_242, var_304);
            var_306 = wp::address(var_X, var_305);
            var_308 = wp::load(var_306);
            var_307 = wp::copy(var_308);
            // EB0 = X[edgeVIndJ[0]]                                                              <L 168>
            var_310 = wp::extract(var_245, var_309);
            var_311 = wp::address(var_X, var_310);
            var_313 = wp::load(var_311);
            var_312 = wp::copy(var_313);
            // EB1 = X[edgeVIndJ[1]]                                                              <L 169>
            var_315 = wp::extract(var_245, var_314);
            var_316 = wp::address(var_X, var_315);
            var_318 = wp::load(var_316);
            var_317 = wp::copy(var_318);
            // eps_x = edge_edge_mollifier_threshold(EA0, EA1, EB0, EB1)                          <L 170>
            var_319 = edge_edge_mollifier_threshold_0(var_302, var_307, var_312, var_317);
            // m = edge_edge_mollifier(ea0, ea1, eb0, eb1, eps_x)                                 <L 171>
            var_320 = edge_edge_mollifier_0(var_279, var_284, var_289, var_294, var_319);
            // activeGap2 = dhat * dhat + wp.float64(2.0) * dhat * xi                             <L 173>
            var_321 = wp::mul(var_dhat, var_dhat);
            var_323 = wp::float64(var_322);
            var_324 = wp::mul(var_323, var_dhat);
            var_325 = wp::mul(var_324, var_258);
            var_326 = wp::add(var_321, var_325);
            // if E == BARRIER:                                                                   <L 175>
            var_327 = (var_E == var_106);
            if (var_327) {
                // db_div_dd = scale * int_w_PTEE * (edge_area[edgeI] + edge_area[edgeJ]) * dhat * barrier.barrier_gradient(dsq, activeGap2, kappa)       <L 176>
                var_328 = wp::mul(var_scale, var_int_w_PTEE);
                var_329 = wp::address(var_edge_area, var_236);
                var_330 = wp::address(var_edge_area, var_239);
                var_332 = wp::load(var_329);
                var_333 = wp::load(var_330);
                var_331 = wp::add(var_332, var_333);
                var_334 = wp::mul(var_328, var_331);
                var_335 = wp::mul(var_334, var_dhat);
                var_336 = barrier_gradient_0(var_298, var_326, var_kappa);
                var_337 = wp::mul(var_335, var_336);
                // grad_b = edge_edge_distance_gradient(ea0, ea1, eb0, eb1)                       <L 177>
                var_338 = edge_edge_distance_gradient_0(var_279, var_284, var_289, var_294);
                // grad_b *= db_div_dd                                                            <L 178>
                var_339 = wp::mul(var_338, var_337);
                // if m < wp.float64(1.0):                                                        <L 179>
                var_341 = wp::float64(var_340);
                var_342 = (var_320 < var_341);
                if (var_342) {
                    // b = scale * int_w_PTEE * (edge_area[edgeI] + edge_area[edgeJ]) * dhat * barrier.barrier(dsq, activeGap2, kappa)       <L 180>
                    var_343 = wp::mul(var_scale, var_int_w_PTEE);
                    var_344 = wp::address(var_edge_area, var_236);
                    var_345 = wp::address(var_edge_area, var_239);
                    var_347 = wp::load(var_344);
                    var_348 = wp::load(var_345);
                    var_346 = wp::add(var_347, var_348);
                    var_349 = wp::mul(var_343, var_346);
                    var_350 = wp::mul(var_349, var_dhat);
                    var_351 = barrier_0(var_298, var_326, var_kappa);
                    var_352 = wp::mul(var_350, var_351);
                    // grad_m = edge_edge_mollifier_gradient(ea0, ea1, eb0, eb1, eps_x)           <L 181>
                    var_353 = edge_edge_mollifier_gradient_0(var_279, var_284, var_289, var_294, var_319);
                    // grad_b = m * grad_b + grad_m * b                                           <L 182>
                    var_354 = wp::mul(var_320, var_339);
                    var_355 = wp::mul(var_353, var_352);
                    var_356 = wp::add(var_354, var_355);
                }
                var_357 = wp::where(var_342, var_356, var_339);
            }
            var_358 = wp::where(var_327, var_357, var_233);
            var_359 = wp::where(var_327, var_337, var_114);
            if (!var_327) {
                // hat_ea0 = hat_x[edgeVIndI[0]]                                                  <L 184>
                var_361 = wp::extract(var_242, var_360);
                var_362 = wp::address(var_hat_x, var_361);
                var_364 = wp::load(var_362);
                var_363 = wp::copy(var_364);
                // hat_ea1 = hat_x[edgeVIndI[1]]                                                  <L 185>
                var_366 = wp::extract(var_242, var_365);
                var_367 = wp::address(var_hat_x, var_366);
                var_369 = wp::load(var_367);
                var_368 = wp::copy(var_369);
                // hat_eb0 = hat_x[edgeVIndJ[0]]                                                  <L 186>
                var_371 = wp::extract(var_245, var_370);
                var_372 = wp::address(var_hat_x, var_371);
                var_374 = wp::load(var_372);
                var_373 = wp::copy(var_374);
                // hat_eb1 = hat_x[edgeVIndJ[1]]                                                  <L 187>
                var_376 = wp::extract(var_245, var_375);
                var_377 = wp::address(var_hat_x, var_376);
                var_379 = wp::load(var_377);
                var_378 = wp::copy(var_379);
                // u = closest_points[idx][0]                                                     <L 188>
                var_381 = wp::address(var_closest_points, var_0);
                var_383 = wp::load(var_381);
                var_382 = wp::extract(var_383, var_380);
                // v = closest_points[idx][1]                                                     <L 189>
                var_385 = wp::address(var_closest_points, var_0);
                var_387 = wp::load(var_385);
                var_386 = wp::extract(var_387, var_384);
                // xIcoord = wp.vec3d(                                                            <L 190>
                // (wp.float64(1.0) - u) * ea0[0] + u * ea1[0] - (wp.float64(1.0) - v) * eb0[0] - v * eb1[0],       <L 191>
                var_389 = wp::float64(var_388);
                var_390 = wp::sub(var_389, var_382);
                var_392 = wp::extract(var_279, var_391);
                var_393 = wp::mul(var_390, var_392);
                var_395 = wp::extract(var_284, var_394);
                var_396 = wp::mul(var_382, var_395);
                var_397 = wp::add(var_393, var_396);
                var_399 = wp::float64(var_398);
                var_400 = wp::sub(var_399, var_386);
                var_402 = wp::extract(var_289, var_401);
                var_403 = wp::mul(var_400, var_402);
                var_404 = wp::sub(var_397, var_403);
                var_406 = wp::extract(var_294, var_405);
                var_407 = wp::mul(var_386, var_406);
                var_408 = wp::sub(var_404, var_407);
                // (wp.float64(1.0) - u) * ea0[1] + u * ea1[1] - (wp.float64(1.0) - v) * eb0[1] - v * eb1[1],       <L 192>
                var_410 = wp::float64(var_409);
                var_411 = wp::sub(var_410, var_382);
                var_413 = wp::extract(var_279, var_412);
                var_414 = wp::mul(var_411, var_413);
                var_416 = wp::extract(var_284, var_415);
                var_417 = wp::mul(var_382, var_416);
                var_418 = wp::add(var_414, var_417);
                var_420 = wp::float64(var_419);
                var_421 = wp::sub(var_420, var_386);
                var_423 = wp::extract(var_289, var_422);
                var_424 = wp::mul(var_421, var_423);
                var_425 = wp::sub(var_418, var_424);
                var_427 = wp::extract(var_294, var_426);
                var_428 = wp::mul(var_386, var_427);
                var_429 = wp::sub(var_425, var_428);
                // (wp.float64(1.0) - u) * ea0[2] + u * ea1[2] - (wp.float64(1.0) - v) * eb0[2] - v * eb1[2],       <L 193>
                var_431 = wp::float64(var_430);
                var_432 = wp::sub(var_431, var_382);
                var_434 = wp::extract(var_279, var_433);
                var_435 = wp::mul(var_432, var_434);
                var_437 = wp::extract(var_284, var_436);
                var_438 = wp::mul(var_382, var_437);
                var_439 = wp::add(var_435, var_438);
                var_441 = wp::float64(var_440);
                var_442 = wp::sub(var_441, var_386);
                var_444 = wp::extract(var_289, var_443);
                var_445 = wp::mul(var_442, var_444);
                var_446 = wp::sub(var_439, var_445);
                var_448 = wp::extract(var_294, var_447);
                var_449 = wp::mul(var_386, var_448);
                var_450 = wp::sub(var_446, var_449);
                var_451 = wp::vec_t<3,wp::float64>(var_408, var_429, var_450);
                // hat_xIcoord = wp.vec3d(                                                        <L 195>
                // (wp.float64(1.0) - u) * hat_ea0[0] + u * hat_ea1[0] - (wp.float64(1.0) - v) * hat_eb0[0] - v * hat_eb1[0],       <L 196>
                var_453 = wp::float64(var_452);
                var_454 = wp::sub(var_453, var_382);
                var_456 = wp::extract(var_363, var_455);
                var_457 = wp::mul(var_454, var_456);
                var_459 = wp::extract(var_368, var_458);
                var_460 = wp::mul(var_382, var_459);
                var_461 = wp::add(var_457, var_460);
                var_463 = wp::float64(var_462);
                var_464 = wp::sub(var_463, var_386);
                var_466 = wp::extract(var_373, var_465);
                var_467 = wp::mul(var_464, var_466);
                var_468 = wp::sub(var_461, var_467);
                var_470 = wp::extract(var_378, var_469);
                var_471 = wp::mul(var_386, var_470);
                var_472 = wp::sub(var_468, var_471);
                // (wp.float64(1.0) - u) * hat_ea0[1] + u * hat_ea1[1] - (wp.float64(1.0) - v) * hat_eb0[1] - v * hat_eb1[1],       <L 197>
                var_474 = wp::float64(var_473);
                var_475 = wp::sub(var_474, var_382);
                var_477 = wp::extract(var_363, var_476);
                var_478 = wp::mul(var_475, var_477);
                var_480 = wp::extract(var_368, var_479);
                var_481 = wp::mul(var_382, var_480);
                var_482 = wp::add(var_478, var_481);
                var_484 = wp::float64(var_483);
                var_485 = wp::sub(var_484, var_386);
                var_487 = wp::extract(var_373, var_486);
                var_488 = wp::mul(var_485, var_487);
                var_489 = wp::sub(var_482, var_488);
                var_491 = wp::extract(var_378, var_490);
                var_492 = wp::mul(var_386, var_491);
                var_493 = wp::sub(var_489, var_492);
                // (wp.float64(1.0) - u) * hat_ea0[2] + u * hat_ea1[2] - (wp.float64(1.0) - v) * hat_eb0[2] - v * hat_eb1[2],       <L 198>
                var_495 = wp::float64(var_494);
                var_496 = wp::sub(var_495, var_382);
                var_498 = wp::extract(var_363, var_497);
                var_499 = wp::mul(var_496, var_498);
                var_501 = wp::extract(var_368, var_500);
                var_502 = wp::mul(var_382, var_501);
                var_503 = wp::add(var_499, var_502);
                var_505 = wp::float64(var_504);
                var_506 = wp::sub(var_505, var_386);
                var_508 = wp::extract(var_373, var_507);
                var_509 = wp::mul(var_506, var_508);
                var_510 = wp::sub(var_503, var_509);
                var_512 = wp::extract(var_378, var_511);
                var_513 = wp::mul(var_386, var_512);
                var_514 = wp::sub(var_510, var_513);
                var_515 = wp::vec_t<3,wp::float64>(var_472, var_493, var_514);
                // coeff = get_combined_coulomb_friction(mu_body[bodyI[idx]], mu_body[bodyJ[idx]]) * c_lambda[idx] * scale       <L 200>
                var_516 = wp::address(var_bodyI, var_0);
                var_518 = wp::load(var_516);
                var_517 = wp::address(var_mu_body, var_518);
                var_519 = wp::address(var_bodyJ, var_0);
                var_521 = wp::load(var_519);
                var_520 = wp::address(var_mu_body, var_521);
                var_523 = wp::load(var_517);
                var_524 = wp::load(var_520);
                var_522 = get_combined_coulomb_friction_0(var_523, var_524);
                var_525 = wp::address(var_c_lambda, var_0);
                var_527 = wp::load(var_525);
                var_526 = wp::mul(var_522, var_527);
                var_528 = wp::mul(var_526, var_scale);
                // grad_f = friction.friction_gradient(normal[idx], xIcoord, hat_xIcoord, hat_h, epsv, coeff)       <L 201>
                var_529 = wp::address(var_normal, var_0);
                var_531 = wp::load(var_529);
                var_530 = friction_gradient_0(var_531, var_451, var_515, var_hat_h, var_epsv, var_528);
                // grad_b = vec12d(                                                               <L 202>
                // grad_f[0] * (wp.float64(1.0) - u),                                             <L 203>
                var_533 = wp::extract(var_530, var_532);
                var_535 = wp::float64(var_534);
                var_536 = wp::sub(var_535, var_382);
                var_537 = wp::mul(var_533, var_536);
                // grad_f[1] * (wp.float64(1.0) - u),                                             <L 204>
                var_539 = wp::extract(var_530, var_538);
                var_541 = wp::float64(var_540);
                var_542 = wp::sub(var_541, var_382);
                var_543 = wp::mul(var_539, var_542);
                // grad_f[2] * (wp.float64(1.0) - u),                                             <L 205>
                var_545 = wp::extract(var_530, var_544);
                var_547 = wp::float64(var_546);
                var_548 = wp::sub(var_547, var_382);
                var_549 = wp::mul(var_545, var_548);
                // grad_f[0] * u,                                                                 <L 206>
                var_551 = wp::extract(var_530, var_550);
                var_552 = wp::mul(var_551, var_382);
                // grad_f[1] * u,                                                                 <L 207>
                var_554 = wp::extract(var_530, var_553);
                var_555 = wp::mul(var_554, var_382);
                // grad_f[2] * u,                                                                 <L 208>
                var_557 = wp::extract(var_530, var_556);
                var_558 = wp::mul(var_557, var_382);
                // grad_f[0] * -(wp.float64(1.0) - v),                                            <L 209>
                var_560 = wp::extract(var_530, var_559);
                var_562 = wp::float64(var_561);
                var_563 = wp::sub(var_562, var_386);
                var_564 = wp::neg(var_563);
                var_565 = wp::mul(var_560, var_564);
                // grad_f[1] * -(wp.float64(1.0) - v),                                            <L 210>
                var_567 = wp::extract(var_530, var_566);
                var_569 = wp::float64(var_568);
                var_570 = wp::sub(var_569, var_386);
                var_571 = wp::neg(var_570);
                var_572 = wp::mul(var_567, var_571);
                // grad_f[2] * -(wp.float64(1.0) - v),                                            <L 211>
                var_574 = wp::extract(var_530, var_573);
                var_576 = wp::float64(var_575);
                var_577 = wp::sub(var_576, var_386);
                var_578 = wp::neg(var_577);
                var_579 = wp::mul(var_574, var_578);
                // grad_f[0] * -v,                                                                <L 212>
                var_581 = wp::extract(var_530, var_580);
                var_582 = wp::neg(var_386);
                var_583 = wp::mul(var_581, var_582);
                // grad_f[1] * -v,                                                                <L 213>
                var_585 = wp::extract(var_530, var_584);
                var_586 = wp::neg(var_386);
                var_587 = wp::mul(var_585, var_586);
                // grad_f[2] * -v,                                                                <L 214>
                var_589 = wp::extract(var_530, var_588);
                var_590 = wp::neg(var_386);
                var_591 = wp::mul(var_589, var_590);
                var_592 = wp::vec_t<12,wp::float64>({var_537, var_543, var_549, var_552, var_555, var_558, var_565, var_572, var_579, var_583, var_587, var_591});
            }
            var_593 = wp::where(var_327, var_358, var_592);
            var_594 = wp::where(var_327, var_119, var_382);
            var_595 = wp::where(var_327, var_123, var_386);
            var_596 = wp::where(var_327, var_134, var_451);
            var_597 = wp::where(var_327, var_144, var_515);
            var_598 = wp::where(var_327, var_167, var_528);
            var_599 = wp::where(var_327, var_169, var_530);
        }
        var_600 = wp::where(var_12, var_233, var_593);
        var_601 = wp::where(var_12, var_234, var_255);
        var_602 = wp::where(var_12, var_93, var_258);
        var_603 = wp::where(var_12, var_98, var_298);
        var_604 = wp::where(var_12, var_105, var_326);
        var_605 = wp::where(var_12, var_114, var_359);
        var_606 = wp::where(var_12, var_119, var_594);
        var_607 = wp::where(var_12, var_123, var_595);
        var_608 = wp::where(var_12, var_134, var_596);
        var_609 = wp::where(var_12, var_144, var_597);
        var_610 = wp::where(var_12, var_167, var_598);
        var_611 = wp::where(var_12, var_169, var_599);
        // for vI in range(4):                                                                    <L 221>
        // grad_b_I = wp.vec3d(grad_b[vI * 3 + 0], grad_b[vI * 3 + 1], grad_b[vI * 3 + 2])        <L 222>
        var_614 = wp::mul(var_612, var_613);
        var_616 = wp::add(var_614, var_615);
        var_617 = wp::extract(var_600, var_616);
        var_619 = wp::mul(var_612, var_618);
        var_621 = wp::add(var_619, var_620);
        var_622 = wp::extract(var_600, var_621);
        var_624 = wp::mul(var_612, var_623);
        var_626 = wp::add(var_624, var_625);
        var_627 = wp::extract(var_600, var_626);
        var_628 = wp::vec_t<3,wp::float64>(var_617, var_622, var_627);
        // wp.atomic_add(gradient_x, vInd[vI], grad_b_I)                                          <L 223>
        var_629 = wp::extract(var_601, var_612);
        // var_630 = wp::atomic_add(var_gradient_x, var_629, var_628);
        // grad_b_I = wp.vec3d(grad_b[vI * 3 + 0], grad_b[vI * 3 + 1], grad_b[vI * 3 + 2])        <L 222>
        var_633 = wp::mul(var_631, var_632);
        var_635 = wp::add(var_633, var_634);
        var_636 = wp::extract(var_600, var_635);
        var_638 = wp::mul(var_631, var_637);
        var_640 = wp::add(var_638, var_639);
        var_641 = wp::extract(var_600, var_640);
        var_643 = wp::mul(var_631, var_642);
        var_645 = wp::add(var_643, var_644);
        var_646 = wp::extract(var_600, var_645);
        var_647 = wp::vec_t<3,wp::float64>(var_636, var_641, var_646);
        // wp.atomic_add(gradient_x, vInd[vI], grad_b_I)                                          <L 223>
        var_648 = wp::extract(var_601, var_631);
        // var_649 = wp::atomic_add(var_gradient_x, var_648, var_647);
        // grad_b_I = wp.vec3d(grad_b[vI * 3 + 0], grad_b[vI * 3 + 1], grad_b[vI * 3 + 2])        <L 222>
        var_652 = wp::mul(var_650, var_651);
        var_654 = wp::add(var_652, var_653);
        var_655 = wp::extract(var_600, var_654);
        var_657 = wp::mul(var_650, var_656);
        var_659 = wp::add(var_657, var_658);
        var_660 = wp::extract(var_600, var_659);
        var_662 = wp::mul(var_650, var_661);
        var_664 = wp::add(var_662, var_663);
        var_665 = wp::extract(var_600, var_664);
        var_666 = wp::vec_t<3,wp::float64>(var_655, var_660, var_665);
        // wp.atomic_add(gradient_x, vInd[vI], grad_b_I)                                          <L 223>
        var_667 = wp::extract(var_601, var_650);
        // var_668 = wp::atomic_add(var_gradient_x, var_667, var_666);
        // grad_b_I = wp.vec3d(grad_b[vI * 3 + 0], grad_b[vI * 3 + 1], grad_b[vI * 3 + 2])        <L 222>
        var_671 = wp::mul(var_669, var_670);
        var_673 = wp::add(var_671, var_672);
        var_674 = wp::extract(var_600, var_673);
        var_676 = wp::mul(var_669, var_675);
        var_678 = wp::add(var_676, var_677);
        var_679 = wp::extract(var_600, var_678);
        var_681 = wp::mul(var_669, var_680);
        var_683 = wp::add(var_681, var_682);
        var_684 = wp::extract(var_600, var_683);
        var_685 = wp::vec_t<3,wp::float64>(var_674, var_679, var_684);
        // wp.atomic_add(gradient_x, vInd[vI], grad_b_I)                                          <L 223>
        var_686 = wp::extract(var_601, var_669);
        // var_687 = wp::atomic_add(var_gradient_x, var_686, var_685);
        //---------
        // reverse
        wp::adj_atomic_add(var_gradient_x, var_686, var_685, adj_gradient_x, adj_686, adj_685, adj_687);
        wp::adj_extract(var_601, var_669, adj_601, adj_669, adj_686);
        // adj: wp.atomic_add(gradient_x, vInd[vI], grad_b_I)                                     <L 223>
        wp::adj_vec_t(var_674, var_679, var_684, adj_674, adj_679, adj_684, adj_685);
        wp::adj_extract(var_600, var_683, adj_600, adj_683, adj_684);
        wp::adj_add(var_681, var_682, adj_681, adj_682, adj_683);
        wp::adj_mul(var_669, var_680, adj_669, adj_680, adj_681);
        wp::adj_extract(var_600, var_678, adj_600, adj_678, adj_679);
        wp::adj_add(var_676, var_677, adj_676, adj_677, adj_678);
        wp::adj_mul(var_669, var_675, adj_669, adj_675, adj_676);
        wp::adj_extract(var_600, var_673, adj_600, adj_673, adj_674);
        wp::adj_add(var_671, var_672, adj_671, adj_672, adj_673);
        wp::adj_mul(var_669, var_670, adj_669, adj_670, adj_671);
        // adj: grad_b_I = wp.vec3d(grad_b[vI * 3 + 0], grad_b[vI * 3 + 1], grad_b[vI * 3 + 2])   <L 222>
        wp::adj_atomic_add(var_gradient_x, var_667, var_666, adj_gradient_x, adj_667, adj_666, adj_668);
        wp::adj_extract(var_601, var_650, adj_601, adj_650, adj_667);
        // adj: wp.atomic_add(gradient_x, vInd[vI], grad_b_I)                                     <L 223>
        wp::adj_vec_t(var_655, var_660, var_665, adj_655, adj_660, adj_665, adj_666);
        wp::adj_extract(var_600, var_664, adj_600, adj_664, adj_665);
        wp::adj_add(var_662, var_663, adj_662, adj_663, adj_664);
        wp::adj_mul(var_650, var_661, adj_650, adj_661, adj_662);
        wp::adj_extract(var_600, var_659, adj_600, adj_659, adj_660);
        wp::adj_add(var_657, var_658, adj_657, adj_658, adj_659);
        wp::adj_mul(var_650, var_656, adj_650, adj_656, adj_657);
        wp::adj_extract(var_600, var_654, adj_600, adj_654, adj_655);
        wp::adj_add(var_652, var_653, adj_652, adj_653, adj_654);
        wp::adj_mul(var_650, var_651, adj_650, adj_651, adj_652);
        // adj: grad_b_I = wp.vec3d(grad_b[vI * 3 + 0], grad_b[vI * 3 + 1], grad_b[vI * 3 + 2])   <L 222>
        wp::adj_atomic_add(var_gradient_x, var_648, var_647, adj_gradient_x, adj_648, adj_647, adj_649);
        wp::adj_extract(var_601, var_631, adj_601, adj_631, adj_648);
        // adj: wp.atomic_add(gradient_x, vInd[vI], grad_b_I)                                     <L 223>
        wp::adj_vec_t(var_636, var_641, var_646, adj_636, adj_641, adj_646, adj_647);
        wp::adj_extract(var_600, var_645, adj_600, adj_645, adj_646);
        wp::adj_add(var_643, var_644, adj_643, adj_644, adj_645);
        wp::adj_mul(var_631, var_642, adj_631, adj_642, adj_643);
        wp::adj_extract(var_600, var_640, adj_600, adj_640, adj_641);
        wp::adj_add(var_638, var_639, adj_638, adj_639, adj_640);
        wp::adj_mul(var_631, var_637, adj_631, adj_637, adj_638);
        wp::adj_extract(var_600, var_635, adj_600, adj_635, adj_636);
        wp::adj_add(var_633, var_634, adj_633, adj_634, adj_635);
        wp::adj_mul(var_631, var_632, adj_631, adj_632, adj_633);
        // adj: grad_b_I = wp.vec3d(grad_b[vI * 3 + 0], grad_b[vI * 3 + 1], grad_b[vI * 3 + 2])   <L 222>
        wp::adj_atomic_add(var_gradient_x, var_629, var_628, adj_gradient_x, adj_629, adj_628, adj_630);
        wp::adj_extract(var_601, var_612, adj_601, adj_612, adj_629);
        // adj: wp.atomic_add(gradient_x, vInd[vI], grad_b_I)                                     <L 223>
        wp::adj_vec_t(var_617, var_622, var_627, adj_617, adj_622, adj_627, adj_628);
        wp::adj_extract(var_600, var_626, adj_600, adj_626, adj_627);
        wp::adj_add(var_624, var_625, adj_624, adj_625, adj_626);
        wp::adj_mul(var_612, var_623, adj_612, adj_623, adj_624);
        wp::adj_extract(var_600, var_621, adj_600, adj_621, adj_622);
        wp::adj_add(var_619, var_620, adj_619, adj_620, adj_621);
        wp::adj_mul(var_612, var_618, adj_612, adj_618, adj_619);
        wp::adj_extract(var_600, var_616, adj_600, adj_616, adj_617);
        wp::adj_add(var_614, var_615, adj_614, adj_615, adj_616);
        wp::adj_mul(var_612, var_613, adj_612, adj_613, adj_614);
        // adj: grad_b_I = wp.vec3d(grad_b[vI * 3 + 0], grad_b[vI * 3 + 1], grad_b[vI * 3 + 2])   <L 222>
        // adj: for vI in range(4):                                                               <L 221>
        wp::adj_where(var_12, var_169, var_599, adj_12, adj_169, adj_599, adj_611);
        wp::adj_where(var_12, var_167, var_598, adj_12, adj_167, adj_598, adj_610);
        wp::adj_where(var_12, var_144, var_597, adj_12, adj_144, adj_597, adj_609);
        wp::adj_where(var_12, var_134, var_596, adj_12, adj_134, adj_596, adj_608);
        wp::adj_where(var_12, var_123, var_595, adj_12, adj_123, adj_595, adj_607);
        wp::adj_where(var_12, var_119, var_594, adj_12, adj_119, adj_594, adj_606);
        wp::adj_where(var_12, var_114, var_359, adj_12, adj_114, adj_359, adj_605);
        wp::adj_where(var_12, var_105, var_326, adj_12, adj_105, adj_326, adj_604);
        wp::adj_where(var_12, var_98, var_298, adj_12, adj_98, adj_298, adj_603);
        wp::adj_where(var_12, var_93, var_258, adj_12, adj_93, adj_258, adj_602);
        wp::adj_where(var_12, var_234, var_255, adj_12, adj_234, adj_255, adj_601);
        wp::adj_where(var_12, var_233, var_593, adj_12, adj_233, adj_593, adj_600);
        if (!var_12) {
            wp::adj_where(var_327, var_169, var_530, adj_327, adj_169, adj_530, adj_599);
            wp::adj_where(var_327, var_167, var_528, adj_327, adj_167, adj_528, adj_598);
            wp::adj_where(var_327, var_144, var_515, adj_327, adj_144, adj_515, adj_597);
            wp::adj_where(var_327, var_134, var_451, adj_327, adj_134, adj_451, adj_596);
            wp::adj_where(var_327, var_123, var_386, adj_327, adj_123, adj_386, adj_595);
            wp::adj_where(var_327, var_119, var_382, adj_327, adj_119, adj_382, adj_594);
            wp::adj_where(var_327, var_358, var_592, adj_327, adj_358, adj_592, adj_593);
            if (!var_327) {
                wp::adj_vec_t({var_537, var_543, var_549, var_552, var_555, var_558, var_565, var_572, var_579, var_583, var_587, var_591}, {&adj_537, &adj_543, &adj_549, &adj_552, &adj_555, &adj_558, &adj_565, &adj_572, &adj_579, &adj_583, &adj_587, &adj_591}, adj_592);
                wp::adj_mul(var_589, var_590, adj_589, adj_590, adj_591);
                wp::adj_neg(var_386, adj_386, adj_590);
                wp::adj_extract(var_530, var_588, adj_530, adj_588, adj_589);
                // adj: grad_f[2] * -v,                                                           <L 214>
                wp::adj_mul(var_585, var_586, adj_585, adj_586, adj_587);
                wp::adj_neg(var_386, adj_386, adj_586);
                wp::adj_extract(var_530, var_584, adj_530, adj_584, adj_585);
                // adj: grad_f[1] * -v,                                                           <L 213>
                wp::adj_mul(var_581, var_582, adj_581, adj_582, adj_583);
                wp::adj_neg(var_386, adj_386, adj_582);
                wp::adj_extract(var_530, var_580, adj_530, adj_580, adj_581);
                // adj: grad_f[0] * -v,                                                           <L 212>
                wp::adj_mul(var_574, var_578, adj_574, adj_578, adj_579);
                wp::adj_neg(var_577, adj_577, adj_578);
                wp::adj_sub(var_576, var_386, adj_576, adj_386, adj_577);
                wp::adj_float64(var_575, adj_575, adj_576);
                wp::adj_extract(var_530, var_573, adj_530, adj_573, adj_574);
                // adj: grad_f[2] * -(wp.float64(1.0) - v),                                       <L 211>
                wp::adj_mul(var_567, var_571, adj_567, adj_571, adj_572);
                wp::adj_neg(var_570, adj_570, adj_571);
                wp::adj_sub(var_569, var_386, adj_569, adj_386, adj_570);
                wp::adj_float64(var_568, adj_568, adj_569);
                wp::adj_extract(var_530, var_566, adj_530, adj_566, adj_567);
                // adj: grad_f[1] * -(wp.float64(1.0) - v),                                       <L 210>
                wp::adj_mul(var_560, var_564, adj_560, adj_564, adj_565);
                wp::adj_neg(var_563, adj_563, adj_564);
                wp::adj_sub(var_562, var_386, adj_562, adj_386, adj_563);
                wp::adj_float64(var_561, adj_561, adj_562);
                wp::adj_extract(var_530, var_559, adj_530, adj_559, adj_560);
                // adj: grad_f[0] * -(wp.float64(1.0) - v),                                       <L 209>
                wp::adj_mul(var_557, var_382, adj_557, adj_382, adj_558);
                wp::adj_extract(var_530, var_556, adj_530, adj_556, adj_557);
                // adj: grad_f[2] * u,                                                            <L 208>
                wp::adj_mul(var_554, var_382, adj_554, adj_382, adj_555);
                wp::adj_extract(var_530, var_553, adj_530, adj_553, adj_554);
                // adj: grad_f[1] * u,                                                            <L 207>
                wp::adj_mul(var_551, var_382, adj_551, adj_382, adj_552);
                wp::adj_extract(var_530, var_550, adj_530, adj_550, adj_551);
                // adj: grad_f[0] * u,                                                            <L 206>
                wp::adj_mul(var_545, var_548, adj_545, adj_548, adj_549);
                wp::adj_sub(var_547, var_382, adj_547, adj_382, adj_548);
                wp::adj_float64(var_546, adj_546, adj_547);
                wp::adj_extract(var_530, var_544, adj_530, adj_544, adj_545);
                // adj: grad_f[2] * (wp.float64(1.0) - u),                                        <L 205>
                wp::adj_mul(var_539, var_542, adj_539, adj_542, adj_543);
                wp::adj_sub(var_541, var_382, adj_541, adj_382, adj_542);
                wp::adj_float64(var_540, adj_540, adj_541);
                wp::adj_extract(var_530, var_538, adj_530, adj_538, adj_539);
                // adj: grad_f[1] * (wp.float64(1.0) - u),                                        <L 204>
                wp::adj_mul(var_533, var_536, adj_533, adj_536, adj_537);
                wp::adj_sub(var_535, var_382, adj_535, adj_382, adj_536);
                wp::adj_float64(var_534, adj_534, adj_535);
                wp::adj_extract(var_530, var_532, adj_530, adj_532, adj_533);
                // adj: grad_f[0] * (wp.float64(1.0) - u),                                        <L 203>
                // adj: grad_b = vec12d(                                                          <L 202>
                adj_friction_gradient_0(var_531, var_451, var_515, var_hat_h, var_epsv, var_528, adj_529, adj_451, adj_515, adj_hat_h, adj_epsv, adj_528, adj_530);
                wp::adj_load(var_529, adj_529, adj_531);
                wp::adj_address(var_normal, var_0, adj_normal, adj_0, adj_529);
                // adj: grad_f = friction.friction_gradient(normal[idx], xIcoord, hat_xIcoord, hat_h, epsv, coeff)  <L 201>
                wp::adj_mul(var_526, var_scale, adj_526, adj_scale, adj_528);
                wp::adj_mul(var_522, var_527, adj_522, adj_525, adj_526);
                wp::adj_load(var_525, adj_525, adj_527);
                wp::adj_address(var_c_lambda, var_0, adj_c_lambda, adj_0, adj_525);
                adj_get_combined_coulomb_friction_0(var_523, var_524, adj_517, adj_520, adj_522);
                wp::adj_load(var_520, adj_520, adj_524);
                wp::adj_load(var_517, adj_517, adj_523);
                wp::adj_address(var_mu_body, var_521, adj_mu_body, adj_519, adj_520);
                wp::adj_load(var_519, adj_519, adj_521);
                wp::adj_address(var_bodyJ, var_0, adj_bodyJ, adj_0, adj_519);
                wp::adj_address(var_mu_body, var_518, adj_mu_body, adj_516, adj_517);
                wp::adj_load(var_516, adj_516, adj_518);
                wp::adj_address(var_bodyI, var_0, adj_bodyI, adj_0, adj_516);
                // adj: coeff = get_combined_coulomb_friction(mu_body[bodyI[idx]], mu_body[bodyJ[idx]]) * c_lambda[idx] * scale  <L 200>
                wp::adj_vec_t(var_472, var_493, var_514, adj_472, adj_493, adj_514, adj_515);
                wp::adj_sub(var_510, var_513, adj_510, adj_513, adj_514);
                wp::adj_mul(var_386, var_512, adj_386, adj_512, adj_513);
                wp::adj_extract(var_378, var_511, adj_378, adj_511, adj_512);
                wp::adj_sub(var_503, var_509, adj_503, adj_509, adj_510);
                wp::adj_mul(var_506, var_508, adj_506, adj_508, adj_509);
                wp::adj_extract(var_373, var_507, adj_373, adj_507, adj_508);
                wp::adj_sub(var_505, var_386, adj_505, adj_386, adj_506);
                wp::adj_float64(var_504, adj_504, adj_505);
                wp::adj_add(var_499, var_502, adj_499, adj_502, adj_503);
                wp::adj_mul(var_382, var_501, adj_382, adj_501, adj_502);
                wp::adj_extract(var_368, var_500, adj_368, adj_500, adj_501);
                wp::adj_mul(var_496, var_498, adj_496, adj_498, adj_499);
                wp::adj_extract(var_363, var_497, adj_363, adj_497, adj_498);
                wp::adj_sub(var_495, var_382, adj_495, adj_382, adj_496);
                wp::adj_float64(var_494, adj_494, adj_495);
                // adj: (wp.float64(1.0) - u) * hat_ea0[2] + u * hat_ea1[2] - (wp.float64(1.0) - v) * hat_eb0[2] - v * hat_eb1[2],  <L 198>
                wp::adj_sub(var_489, var_492, adj_489, adj_492, adj_493);
                wp::adj_mul(var_386, var_491, adj_386, adj_491, adj_492);
                wp::adj_extract(var_378, var_490, adj_378, adj_490, adj_491);
                wp::adj_sub(var_482, var_488, adj_482, adj_488, adj_489);
                wp::adj_mul(var_485, var_487, adj_485, adj_487, adj_488);
                wp::adj_extract(var_373, var_486, adj_373, adj_486, adj_487);
                wp::adj_sub(var_484, var_386, adj_484, adj_386, adj_485);
                wp::adj_float64(var_483, adj_483, adj_484);
                wp::adj_add(var_478, var_481, adj_478, adj_481, adj_482);
                wp::adj_mul(var_382, var_480, adj_382, adj_480, adj_481);
                wp::adj_extract(var_368, var_479, adj_368, adj_479, adj_480);
                wp::adj_mul(var_475, var_477, adj_475, adj_477, adj_478);
                wp::adj_extract(var_363, var_476, adj_363, adj_476, adj_477);
                wp::adj_sub(var_474, var_382, adj_474, adj_382, adj_475);
                wp::adj_float64(var_473, adj_473, adj_474);
                // adj: (wp.float64(1.0) - u) * hat_ea0[1] + u * hat_ea1[1] - (wp.float64(1.0) - v) * hat_eb0[1] - v * hat_eb1[1],  <L 197>
                wp::adj_sub(var_468, var_471, adj_468, adj_471, adj_472);
                wp::adj_mul(var_386, var_470, adj_386, adj_470, adj_471);
                wp::adj_extract(var_378, var_469, adj_378, adj_469, adj_470);
                wp::adj_sub(var_461, var_467, adj_461, adj_467, adj_468);
                wp::adj_mul(var_464, var_466, adj_464, adj_466, adj_467);
                wp::adj_extract(var_373, var_465, adj_373, adj_465, adj_466);
                wp::adj_sub(var_463, var_386, adj_463, adj_386, adj_464);
                wp::adj_float64(var_462, adj_462, adj_463);
                wp::adj_add(var_457, var_460, adj_457, adj_460, adj_461);
                wp::adj_mul(var_382, var_459, adj_382, adj_459, adj_460);
                wp::adj_extract(var_368, var_458, adj_368, adj_458, adj_459);
                wp::adj_mul(var_454, var_456, adj_454, adj_456, adj_457);
                wp::adj_extract(var_363, var_455, adj_363, adj_455, adj_456);
                wp::adj_sub(var_453, var_382, adj_453, adj_382, adj_454);
                wp::adj_float64(var_452, adj_452, adj_453);
                // adj: (wp.float64(1.0) - u) * hat_ea0[0] + u * hat_ea1[0] - (wp.float64(1.0) - v) * hat_eb0[0] - v * hat_eb1[0],  <L 196>
                // adj: hat_xIcoord = wp.vec3d(                                                   <L 195>
                wp::adj_vec_t(var_408, var_429, var_450, adj_408, adj_429, adj_450, adj_451);
                wp::adj_sub(var_446, var_449, adj_446, adj_449, adj_450);
                wp::adj_mul(var_386, var_448, adj_386, adj_448, adj_449);
                wp::adj_extract(var_294, var_447, adj_294, adj_447, adj_448);
                wp::adj_sub(var_439, var_445, adj_439, adj_445, adj_446);
                wp::adj_mul(var_442, var_444, adj_442, adj_444, adj_445);
                wp::adj_extract(var_289, var_443, adj_289, adj_443, adj_444);
                wp::adj_sub(var_441, var_386, adj_441, adj_386, adj_442);
                wp::adj_float64(var_440, adj_440, adj_441);
                wp::adj_add(var_435, var_438, adj_435, adj_438, adj_439);
                wp::adj_mul(var_382, var_437, adj_382, adj_437, adj_438);
                wp::adj_extract(var_284, var_436, adj_284, adj_436, adj_437);
                wp::adj_mul(var_432, var_434, adj_432, adj_434, adj_435);
                wp::adj_extract(var_279, var_433, adj_279, adj_433, adj_434);
                wp::adj_sub(var_431, var_382, adj_431, adj_382, adj_432);
                wp::adj_float64(var_430, adj_430, adj_431);
                // adj: (wp.float64(1.0) - u) * ea0[2] + u * ea1[2] - (wp.float64(1.0) - v) * eb0[2] - v * eb1[2],  <L 193>
                wp::adj_sub(var_425, var_428, adj_425, adj_428, adj_429);
                wp::adj_mul(var_386, var_427, adj_386, adj_427, adj_428);
                wp::adj_extract(var_294, var_426, adj_294, adj_426, adj_427);
                wp::adj_sub(var_418, var_424, adj_418, adj_424, adj_425);
                wp::adj_mul(var_421, var_423, adj_421, adj_423, adj_424);
                wp::adj_extract(var_289, var_422, adj_289, adj_422, adj_423);
                wp::adj_sub(var_420, var_386, adj_420, adj_386, adj_421);
                wp::adj_float64(var_419, adj_419, adj_420);
                wp::adj_add(var_414, var_417, adj_414, adj_417, adj_418);
                wp::adj_mul(var_382, var_416, adj_382, adj_416, adj_417);
                wp::adj_extract(var_284, var_415, adj_284, adj_415, adj_416);
                wp::adj_mul(var_411, var_413, adj_411, adj_413, adj_414);
                wp::adj_extract(var_279, var_412, adj_279, adj_412, adj_413);
                wp::adj_sub(var_410, var_382, adj_410, adj_382, adj_411);
                wp::adj_float64(var_409, adj_409, adj_410);
                // adj: (wp.float64(1.0) - u) * ea0[1] + u * ea1[1] - (wp.float64(1.0) - v) * eb0[1] - v * eb1[1],  <L 192>
                wp::adj_sub(var_404, var_407, adj_404, adj_407, adj_408);
                wp::adj_mul(var_386, var_406, adj_386, adj_406, adj_407);
                wp::adj_extract(var_294, var_405, adj_294, adj_405, adj_406);
                wp::adj_sub(var_397, var_403, adj_397, adj_403, adj_404);
                wp::adj_mul(var_400, var_402, adj_400, adj_402, adj_403);
                wp::adj_extract(var_289, var_401, adj_289, adj_401, adj_402);
                wp::adj_sub(var_399, var_386, adj_399, adj_386, adj_400);
                wp::adj_float64(var_398, adj_398, adj_399);
                wp::adj_add(var_393, var_396, adj_393, adj_396, adj_397);
                wp::adj_mul(var_382, var_395, adj_382, adj_395, adj_396);
                wp::adj_extract(var_284, var_394, adj_284, adj_394, adj_395);
                wp::adj_mul(var_390, var_392, adj_390, adj_392, adj_393);
                wp::adj_extract(var_279, var_391, adj_279, adj_391, adj_392);
                wp::adj_sub(var_389, var_382, adj_389, adj_382, adj_390);
                wp::adj_float64(var_388, adj_388, adj_389);
                // adj: (wp.float64(1.0) - u) * ea0[0] + u * ea1[0] - (wp.float64(1.0) - v) * eb0[0] - v * eb1[0],  <L 191>
                // adj: xIcoord = wp.vec3d(                                                       <L 190>
                wp::adj_extract(var_387, var_384, adj_385, adj_384, adj_386);
                wp::adj_load(var_385, adj_385, adj_387);
                wp::adj_address(var_closest_points, var_0, adj_closest_points, adj_0, adj_385);
                // adj: v = closest_points[idx][1]                                                <L 189>
                wp::adj_extract(var_383, var_380, adj_381, adj_380, adj_382);
                wp::adj_load(var_381, adj_381, adj_383);
                wp::adj_address(var_closest_points, var_0, adj_closest_points, adj_0, adj_381);
                // adj: u = closest_points[idx][0]                                                <L 188>
                wp::adj_copy(var_379, adj_377, adj_378);
                wp::adj_load(var_377, adj_377, adj_379);
                wp::adj_address(var_hat_x, var_376, adj_hat_x, adj_376, adj_377);
                wp::adj_extract(var_245, var_375, adj_245, adj_375, adj_376);
                // adj: hat_eb1 = hat_x[edgeVIndJ[1]]                                             <L 187>
                wp::adj_copy(var_374, adj_372, adj_373);
                wp::adj_load(var_372, adj_372, adj_374);
                wp::adj_address(var_hat_x, var_371, adj_hat_x, adj_371, adj_372);
                wp::adj_extract(var_245, var_370, adj_245, adj_370, adj_371);
                // adj: hat_eb0 = hat_x[edgeVIndJ[0]]                                             <L 186>
                wp::adj_copy(var_369, adj_367, adj_368);
                wp::adj_load(var_367, adj_367, adj_369);
                wp::adj_address(var_hat_x, var_366, adj_hat_x, adj_366, adj_367);
                wp::adj_extract(var_242, var_365, adj_242, adj_365, adj_366);
                // adj: hat_ea1 = hat_x[edgeVIndI[1]]                                             <L 185>
                wp::adj_copy(var_364, adj_362, adj_363);
                wp::adj_load(var_362, adj_362, adj_364);
                wp::adj_address(var_hat_x, var_361, adj_hat_x, adj_361, adj_362);
                wp::adj_extract(var_242, var_360, adj_242, adj_360, adj_361);
                // adj: hat_ea0 = hat_x[edgeVIndI[0]]                                             <L 184>
            }
            wp::adj_where(var_327, var_337, var_114, adj_327, adj_337, adj_114, adj_359);
            wp::adj_where(var_327, var_357, var_233, adj_327, adj_357, adj_233, adj_358);
            if (var_327) {
                wp::adj_where(var_342, var_356, var_339, adj_342, adj_356, adj_339, adj_357);
                if (var_342) {
                    wp::adj_add(var_354, var_355, adj_354, adj_355, adj_356);
                    wp::adj_mul(var_353, var_352, adj_353, adj_352, adj_355);
                    wp::adj_mul(var_320, var_339, adj_320, adj_339, adj_354);
                    // adj: grad_b = m * grad_b + grad_m * b                                      <L 182>
                    adj_edge_edge_mollifier_gradient_0(var_279, var_284, var_289, var_294, var_319, adj_279, adj_284, adj_289, adj_294, adj_319, adj_353);
                    // adj: grad_m = edge_edge_mollifier_gradient(ea0, ea1, eb0, eb1, eps_x)      <L 181>
                    wp::adj_mul(var_350, var_351, adj_350, adj_351, adj_352);
                    adj_barrier_0(var_298, var_326, var_kappa, adj_298, adj_326, adj_kappa, adj_351);
                    wp::adj_mul(var_349, var_dhat, adj_349, adj_dhat, adj_350);
                    wp::adj_mul(var_343, var_346, adj_343, adj_346, adj_349);
                    wp::adj_add(var_347, var_348, adj_344, adj_345, adj_346);
                    wp::adj_load(var_345, adj_345, adj_348);
                    wp::adj_load(var_344, adj_344, adj_347);
                    wp::adj_address(var_edge_area, var_239, adj_edge_area, adj_239, adj_345);
                    wp::adj_address(var_edge_area, var_236, adj_edge_area, adj_236, adj_344);
                    wp::adj_mul(var_scale, var_int_w_PTEE, adj_scale, adj_int_w_PTEE, adj_343);
                    // adj: b = scale * int_w_PTEE * (edge_area[edgeI] + edge_area[edgeJ]) * dhat * barrier.barrier(dsq, activeGap2, kappa)  <L 180>
                }
                wp::adj_float64(var_340, adj_340, adj_341);
                // adj: if m < wp.float64(1.0):                                                   <L 179>
                wp::adj_mul(var_338, var_337, adj_338, adj_337, adj_339);
                // adj: grad_b *= db_div_dd                                                       <L 178>
                adj_edge_edge_distance_gradient_0(var_279, var_284, var_289, var_294, adj_279, adj_284, adj_289, adj_294, adj_338);
                // adj: grad_b = edge_edge_distance_gradient(ea0, ea1, eb0, eb1)                  <L 177>
                wp::adj_mul(var_335, var_336, adj_335, adj_336, adj_337);
                adj_barrier_gradient_0(var_298, var_326, var_kappa, adj_298, adj_326, adj_kappa, adj_336);
                wp::adj_mul(var_334, var_dhat, adj_334, adj_dhat, adj_335);
                wp::adj_mul(var_328, var_331, adj_328, adj_331, adj_334);
                wp::adj_add(var_332, var_333, adj_329, adj_330, adj_331);
                wp::adj_load(var_330, adj_330, adj_333);
                wp::adj_load(var_329, adj_329, adj_332);
                wp::adj_address(var_edge_area, var_239, adj_edge_area, adj_239, adj_330);
                wp::adj_address(var_edge_area, var_236, adj_edge_area, adj_236, adj_329);
                wp::adj_mul(var_scale, var_int_w_PTEE, adj_scale, adj_int_w_PTEE, adj_328);
                // adj: db_div_dd = scale * int_w_PTEE * (edge_area[edgeI] + edge_area[edgeJ]) * dhat * barrier.barrier_gradient(dsq, activeGap2, kappa)  <L 176>
            }
            // adj: if E == BARRIER:                                                              <L 175>
            wp::adj_add(var_321, var_325, adj_321, adj_325, adj_326);
            wp::adj_mul(var_324, var_258, adj_324, adj_258, adj_325);
            wp::adj_mul(var_323, var_dhat, adj_323, adj_dhat, adj_324);
            wp::adj_float64(var_322, adj_322, adj_323);
            wp::adj_mul(var_dhat, var_dhat, adj_dhat, adj_dhat, adj_321);
            // adj: activeGap2 = dhat * dhat + wp.float64(2.0) * dhat * xi                        <L 173>
            adj_edge_edge_mollifier_0(var_279, var_284, var_289, var_294, var_319, adj_279, adj_284, adj_289, adj_294, adj_319, adj_320);
            // adj: m = edge_edge_mollifier(ea0, ea1, eb0, eb1, eps_x)                            <L 171>
            adj_edge_edge_mollifier_threshold_0(var_302, var_307, var_312, var_317, adj_302, adj_307, adj_312, adj_317, adj_319);
            // adj: eps_x = edge_edge_mollifier_threshold(EA0, EA1, EB0, EB1)                     <L 170>
            wp::adj_copy(var_318, adj_316, adj_317);
            wp::adj_load(var_316, adj_316, adj_318);
            wp::adj_address(var_X, var_315, adj_X, adj_315, adj_316);
            wp::adj_extract(var_245, var_314, adj_245, adj_314, adj_315);
            // adj: EB1 = X[edgeVIndJ[1]]                                                         <L 169>
            wp::adj_copy(var_313, adj_311, adj_312);
            wp::adj_load(var_311, adj_311, adj_313);
            wp::adj_address(var_X, var_310, adj_X, adj_310, adj_311);
            wp::adj_extract(var_245, var_309, adj_245, adj_309, adj_310);
            // adj: EB0 = X[edgeVIndJ[0]]                                                         <L 168>
            wp::adj_copy(var_308, adj_306, adj_307);
            wp::adj_load(var_306, adj_306, adj_308);
            wp::adj_address(var_X, var_305, adj_X, adj_305, adj_306);
            wp::adj_extract(var_242, var_304, adj_242, adj_304, adj_305);
            // adj: EA1 = X[edgeVIndI[1]]                                                         <L 167>
            wp::adj_copy(var_303, adj_301, adj_302);
            wp::adj_load(var_301, adj_301, adj_303);
            wp::adj_address(var_X, var_300, adj_X, adj_300, adj_301);
            wp::adj_extract(var_242, var_299, adj_242, adj_299, adj_300);
            // adj: EA0 = X[edgeVIndI[0]]                                                         <L 166>
            wp::adj_sub(var_296, var_297, adj_296, adj_297, adj_298);
            wp::adj_mul(var_258, var_258, adj_258, adj_258, adj_297);
            adj_edge_edge_distance_0(var_279, var_284, var_289, var_294, adj_279, adj_284, adj_289, adj_294, adj_296);
            // adj: dsq = edge_edge_distance(ea0, ea1, eb0, eb1) - xi * xi                        <L 164>
            wp::adj_copy(var_295, adj_293, adj_294);
            wp::adj_load(var_293, adj_293, adj_295);
            wp::adj_address(var_x, var_292, adj_x, adj_292, adj_293);
            wp::adj_extract(var_245, var_291, adj_245, adj_291, adj_292);
            // adj: eb1 = x[edgeVIndJ[1]]                                                         <L 163>
            wp::adj_copy(var_290, adj_288, adj_289);
            wp::adj_load(var_288, adj_288, adj_290);
            wp::adj_address(var_x, var_287, adj_x, adj_287, adj_288);
            wp::adj_extract(var_245, var_286, adj_245, adj_286, adj_287);
            // adj: eb0 = x[edgeVIndJ[0]]                                                         <L 162>
            wp::adj_copy(var_285, adj_283, adj_284);
            wp::adj_load(var_283, adj_283, adj_285);
            wp::adj_address(var_x, var_282, adj_x, adj_282, adj_283);
            wp::adj_extract(var_242, var_281, adj_242, adj_281, adj_282);
            // adj: ea1 = x[edgeVIndI[1]]                                                         <L 161>
            wp::adj_copy(var_280, adj_278, adj_279);
            wp::adj_load(var_278, adj_278, adj_280);
            wp::adj_address(var_x, var_277, adj_x, adj_277, adj_278);
            wp::adj_extract(var_242, var_276, adj_242, adj_276, adj_277);
            // adj: ea0 = x[edgeVIndI[0]]                                                         <L 160>
            if (var_275) {
                label1:;
                // adj: return                                                                    <L 159>
            }
            wp::adj_load(var_271, adj_271, adj_274);
            wp::adj_address(var_env_states, var_272, adj_env_states, adj_270, adj_271);
            wp::adj_load(var_270, adj_270, adj_272);
            wp::adj_address(var_node2env, var_269, adj_node2env, adj_269, adj_270);
            wp::adj_extract(var_242, var_268, adj_242, adj_268, adj_269);
            wp::adj_load(var_264, adj_264, adj_267);
            wp::adj_address(var_env_states, var_265, adj_env_states, adj_263, adj_264);
            wp::adj_load(var_263, adj_263, adj_265);
            wp::adj_address(var_node2env, var_262, adj_node2env, adj_262, adj_263);
            wp::adj_extract(var_242, var_261, adj_242, adj_261, adj_262);
            // adj: if (env_states[node2env[edgeVIndI[0]]] == ENV_STATE_INVALID) or (env_states[node2env[edgeVIndI[0]]] == ENV_STATE_NEWTON_SOLVED):  <L 158>
            wp::adj_add(var_259, var_260, adj_256, adj_257, adj_258);
            wp::adj_load(var_257, adj_257, adj_260);
            wp::adj_load(var_256, adj_256, adj_259);
            wp::adj_address(var_edge_xi, var_239, adj_edge_xi, adj_239, adj_257);
            wp::adj_address(var_edge_xi, var_236, adj_edge_xi, adj_236, adj_256);
            // adj: xi = edge_xi[edgeI] + edge_xi[edgeJ]                                          <L 157>
            wp::adj_vec_t(var_248, var_250, var_252, var_254, adj_248, adj_250, adj_252, adj_254, adj_255);
            wp::adj_extract(var_245, var_253, adj_245, adj_253, adj_254);
            wp::adj_extract(var_245, var_251, adj_245, adj_251, adj_252);
            wp::adj_extract(var_242, var_249, adj_242, adj_249, adj_250);
            wp::adj_extract(var_242, var_247, adj_242, adj_247, adj_248);
            // adj: vInd = wp.vec4i(edgeVIndI[0], edgeVIndI[1], edgeVIndJ[0], edgeVIndJ[1])       <L 156>
            wp::adj_copy(var_246, adj_244, adj_245);
            wp::adj_load(var_244, adj_244, adj_246);
            wp::adj_address(var_edge, var_239, adj_edge, adj_239, adj_244);
            // adj: edgeVIndJ = edge[edgeJ]                                                       <L 155>
            wp::adj_copy(var_243, adj_241, adj_242);
            wp::adj_load(var_241, adj_241, adj_243);
            wp::adj_address(var_edge, var_236, adj_edge, adj_236, adj_241);
            // adj: edgeVIndI = edge[edgeI]                                                       <L 154>
            wp::adj_copy(var_240, adj_238, adj_239);
            wp::adj_load(var_238, adj_238, adj_240);
            wp::adj_address(var_nodeJ, var_0, adj_nodeJ, adj_0, adj_238);
            // adj: edgeJ = nodeJ[idx]                                                            <L 153>
            wp::adj_copy(var_237, adj_235, adj_236);
            wp::adj_load(var_235, adj_235, adj_237);
            wp::adj_address(var_nodeI, var_0, adj_nodeI, adj_0, adj_235);
            // adj: edgeI = nodeI[idx]                                                            <L 152>
        }
        wp::adj_where(var_12, var_54, var_3, adj_12, adj_54, adj_3, adj_234);
        wp::adj_where(var_12, var_232, var_2, adj_12, adj_232, adj_2, adj_233);
        if (var_12) {
            wp::adj_where(var_107, var_116, var_231, adj_107, adj_116, adj_231, adj_232);
            if (!var_107) {
                wp::adj_vec_t({var_175, var_180, var_185, var_190, var_195, var_200, var_205, var_210, var_215, var_220, var_225, var_230}, {&adj_175, &adj_180, &adj_185, &adj_190, &adj_195, &adj_200, &adj_205, &adj_210, &adj_215, &adj_220, &adj_225, &adj_230}, adj_231);
                wp::adj_mul(var_227, var_229, adj_227, adj_229, adj_230);
                wp::adj_extract(var_154, var_228, adj_154, adj_228, adj_229);
                wp::adj_extract(var_169, var_226, adj_169, adj_226, adj_227);
                // adj: grad_f[2] * weights[3],                                                   <L 146>
                wp::adj_mul(var_222, var_224, adj_222, adj_224, adj_225);
                wp::adj_extract(var_154, var_223, adj_154, adj_223, adj_224);
                wp::adj_extract(var_169, var_221, adj_169, adj_221, adj_222);
                // adj: grad_f[1] * weights[3],                                                   <L 145>
                wp::adj_mul(var_217, var_219, adj_217, adj_219, adj_220);
                wp::adj_extract(var_154, var_218, adj_154, adj_218, adj_219);
                wp::adj_extract(var_169, var_216, adj_169, adj_216, adj_217);
                // adj: grad_f[0] * weights[3],                                                   <L 144>
                wp::adj_mul(var_212, var_214, adj_212, adj_214, adj_215);
                wp::adj_extract(var_154, var_213, adj_154, adj_213, adj_214);
                wp::adj_extract(var_169, var_211, adj_169, adj_211, adj_212);
                // adj: grad_f[2] * weights[2],                                                   <L 143>
                wp::adj_mul(var_207, var_209, adj_207, adj_209, adj_210);
                wp::adj_extract(var_154, var_208, adj_154, adj_208, adj_209);
                wp::adj_extract(var_169, var_206, adj_169, adj_206, adj_207);
                // adj: grad_f[1] * weights[2],                                                   <L 142>
                wp::adj_mul(var_202, var_204, adj_202, adj_204, adj_205);
                wp::adj_extract(var_154, var_203, adj_154, adj_203, adj_204);
                wp::adj_extract(var_169, var_201, adj_169, adj_201, adj_202);
                // adj: grad_f[0] * weights[2],                                                   <L 141>
                wp::adj_mul(var_197, var_199, adj_197, adj_199, adj_200);
                wp::adj_extract(var_154, var_198, adj_154, adj_198, adj_199);
                wp::adj_extract(var_169, var_196, adj_169, adj_196, adj_197);
                // adj: grad_f[2] * weights[1],                                                   <L 140>
                wp::adj_mul(var_192, var_194, adj_192, adj_194, adj_195);
                wp::adj_extract(var_154, var_193, adj_154, adj_193, adj_194);
                wp::adj_extract(var_169, var_191, adj_169, adj_191, adj_192);
                // adj: grad_f[1] * weights[1],                                                   <L 139>
                wp::adj_mul(var_187, var_189, adj_187, adj_189, adj_190);
                wp::adj_extract(var_154, var_188, adj_154, adj_188, adj_189);
                wp::adj_extract(var_169, var_186, adj_169, adj_186, adj_187);
                // adj: grad_f[0] * weights[1],                                                   <L 138>
                wp::adj_mul(var_182, var_184, adj_182, adj_184, adj_185);
                wp::adj_extract(var_154, var_183, adj_154, adj_183, adj_184);
                wp::adj_extract(var_169, var_181, adj_169, adj_181, adj_182);
                // adj: grad_f[2] * weights[0],                                                   <L 137>
                wp::adj_mul(var_177, var_179, adj_177, adj_179, adj_180);
                wp::adj_extract(var_154, var_178, adj_154, adj_178, adj_179);
                wp::adj_extract(var_169, var_176, adj_169, adj_176, adj_177);
                // adj: grad_f[1] * weights[0],                                                   <L 136>
                wp::adj_mul(var_172, var_174, adj_172, adj_174, adj_175);
                wp::adj_extract(var_154, var_173, adj_154, adj_173, adj_174);
                wp::adj_extract(var_169, var_171, adj_169, adj_171, adj_172);
                // adj: grad_f[0] * weights[0],                                                   <L 135>
                // adj: grad_b = vec12d(                                                          <L 134>
                adj_friction_gradient_0(var_170, var_134, var_144, var_hat_h, var_epsv, var_167, adj_168, adj_134, adj_144, adj_hat_h, adj_epsv, adj_167, adj_169);
                wp::adj_load(var_168, adj_168, adj_170);
                wp::adj_address(var_normal, var_0, adj_normal, adj_0, adj_168);
                // adj: grad_f = friction.friction_gradient(normal[idx], xIcoord, hat_xIcoord, hat_h, epsv, coeff)  <L 133>
                wp::adj_mul(var_165, var_scale, adj_165, adj_scale, adj_167);
                wp::adj_mul(var_161, var_166, adj_161, adj_164, adj_165);
                wp::adj_load(var_164, adj_164, adj_166);
                wp::adj_address(var_c_lambda, var_0, adj_c_lambda, adj_0, adj_164);
                adj_get_combined_coulomb_friction_0(var_162, var_163, adj_156, adj_159, adj_161);
                wp::adj_load(var_159, adj_159, adj_163);
                wp::adj_load(var_156, adj_156, adj_162);
                wp::adj_address(var_mu_body, var_160, adj_mu_body, adj_158, adj_159);
                wp::adj_load(var_158, adj_158, adj_160);
                wp::adj_address(var_bodyJ, var_0, adj_bodyJ, adj_0, adj_158);
                wp::adj_address(var_mu_body, var_157, adj_mu_body, adj_155, adj_156);
                wp::adj_load(var_155, adj_155, adj_157);
                wp::adj_address(var_bodyI, var_0, adj_bodyI, adj_0, adj_155);
                // adj: coeff = get_combined_coulomb_friction(mu_body[bodyI[idx]], mu_body[bodyJ[idx]]) * c_lambda[idx] * scale  <L 132>
                wp::adj_vec_t(var_146, var_151, var_152, var_153, adj_146, adj_151, adj_152, adj_153, adj_154);
                wp::adj_neg(var_123, adj_123, adj_153);
                wp::adj_neg(var_119, adj_119, adj_152);
                wp::adj_neg(var_150, adj_150, adj_151);
                wp::adj_sub(var_149, var_123, adj_149, adj_123, adj_150);
                wp::adj_sub(var_148, var_119, adj_148, adj_119, adj_149);
                wp::adj_float64(var_147, adj_147, adj_148);
                wp::adj_float64(var_145, adj_145, adj_146);
                // adj: weights = wp.vec4d(wp.float64(1.0), -(wp.float64(1.0) - u - v), -u, -v)   <L 131>
                wp::adj_sub(var_142, var_143, adj_142, adj_143, adj_144);
                wp::adj_mul(var_123, var_89, adj_123, adj_89, adj_143);
                wp::adj_sub(var_140, var_141, adj_140, adj_141, adj_142);
                wp::adj_mul(var_119, var_84, adj_119, adj_84, adj_141);
                wp::adj_sub(var_74, var_139, adj_74, adj_139, adj_140);
                wp::adj_mul(var_138, var_79, adj_138, adj_79, adj_139);
                wp::adj_sub(var_137, var_123, adj_137, adj_123, adj_138);
                wp::adj_sub(var_136, var_119, adj_136, adj_119, adj_137);
                wp::adj_float64(var_135, adj_135, adj_136);
                // adj: hat_xIcoord = hat_p - (wp.float64(1.0) - u - v) * hat_t0 - u * hat_t1 - v * hat_t2  <L 130>
                wp::adj_sub(var_132, var_133, adj_132, adj_133, adj_134);
                wp::adj_mul(var_123, var_71, adj_123, adj_71, adj_133);
                wp::adj_sub(var_130, var_131, adj_130, adj_131, adj_132);
                wp::adj_mul(var_119, var_66, adj_119, adj_66, adj_131);
                wp::adj_sub(var_56, var_129, adj_56, adj_129, adj_130);
                wp::adj_mul(var_128, var_61, adj_128, adj_61, adj_129);
                wp::adj_sub(var_127, var_123, adj_127, adj_123, adj_128);
                wp::adj_sub(var_126, var_119, adj_126, adj_119, adj_127);
                wp::adj_float64(var_125, adj_125, adj_126);
                // adj: xIcoord = p - (wp.float64(1.0) - u - v) * t0 - u * t1 - v * t2            <L 129>
                wp::adj_extract(var_124, var_121, adj_122, adj_121, adj_123);
                wp::adj_load(var_122, adj_122, adj_124);
                wp::adj_address(var_closest_points, var_0, adj_closest_points, adj_0, adj_122);
                // adj: v = closest_points[idx][1]                                                <L 128>
                wp::adj_extract(var_120, var_117, adj_118, adj_117, adj_119);
                wp::adj_load(var_118, adj_118, adj_120);
                wp::adj_address(var_closest_points, var_0, adj_closest_points, adj_0, adj_118);
                // adj: u = closest_points[idx][0]                                                <L 127>
            }
            wp::adj_where(var_107, var_115, var_99, adj_107, adj_115, adj_99, adj_116);
            if (var_107) {
                wp::adj_mul(var_99, var_114, adj_99, adj_114, adj_115);
                // adj: grad_b *= db_div_dd                                                       <L 125>
                wp::adj_mul(var_112, var_113, adj_112, adj_113, adj_114);
                adj_barrier_gradient_0(var_98, var_105, var_kappa, adj_98, adj_105, adj_kappa, adj_113);
                wp::adj_mul(var_110, var_dhat, adj_110, adj_dhat, adj_112);
                wp::adj_mul(var_108, var_111, adj_108, adj_109, adj_110);
                wp::adj_load(var_109, adj_109, adj_111);
                wp::adj_address(var_node_area, var_18, adj_node_area, adj_18, adj_109);
                wp::adj_mul(var_scale, var_int_w_PTEE, adj_scale, adj_int_w_PTEE, adj_108);
                // adj: db_div_dd = scale * int_w_PTEE * node_area[svi] * dhat * barrier.barrier_gradient(dsq, activeGap2, kappa)  <L 124>
            }
            // adj: if E == BARRIER:                                                              <L 123>
            wp::adj_add(var_100, var_104, adj_100, adj_104, adj_105);
            wp::adj_mul(var_103, var_93, adj_103, adj_93, adj_104);
            wp::adj_mul(var_102, var_dhat, adj_102, adj_dhat, adj_103);
            wp::adj_float64(var_101, adj_101, adj_102);
            wp::adj_mul(var_dhat, var_dhat, adj_dhat, adj_dhat, adj_100);
            // adj: activeGap2 = dhat * dhat + wp.float64(2.0) * dhat * xi                        <L 122>
            adj_point_triangle_distance_gradient_0(var_56, var_61, var_66, var_71, adj_56, adj_61, adj_66, adj_71, adj_99);
            // adj: grad_b = point_triangle_distance_gradient(p, t0, t1, t2)                      <L 119>
            wp::adj_sub(var_96, var_97, adj_96, adj_97, adj_98);
            wp::adj_mul(var_93, var_93, adj_93, adj_93, adj_97);
            adj_point_triangle_distance_0(var_56, var_61, var_66, var_71, adj_56, adj_61, adj_66, adj_71, adj_96);
            // adj: dsq = point_triangle_distance(p, t0, t1, t2) - xi * xi                        <L 118>
            wp::adj_add(var_94, var_95, adj_91, adj_92, adj_93);
            wp::adj_load(var_92, adj_92, adj_95);
            wp::adj_load(var_91, adj_91, adj_94);
            wp::adj_address(var_face_xi, var_26, adj_face_xi, adj_26, adj_92);
            wp::adj_address(var_node_xi, var_18, adj_node_xi, adj_18, adj_91);
            // adj: xi = node_xi[svi] + face_xi[faceJ]                                            <L 117>
            wp::adj_copy(var_90, adj_88, adj_89);
            wp::adj_load(var_88, adj_88, adj_90);
            wp::adj_address(var_hat_x, var_87, adj_hat_x, adj_87, adj_88);
            wp::adj_extract(var_30, var_86, adj_30, adj_86, adj_87);
            // adj: hat_t2 = hat_x[faceVInd[2]]                                                   <L 116>
            wp::adj_copy(var_85, adj_83, adj_84);
            wp::adj_load(var_83, adj_83, adj_85);
            wp::adj_address(var_hat_x, var_82, adj_hat_x, adj_82, adj_83);
            wp::adj_extract(var_30, var_81, adj_30, adj_81, adj_82);
            // adj: hat_t1 = hat_x[faceVInd[1]]                                                   <L 115>
            wp::adj_copy(var_80, adj_78, adj_79);
            wp::adj_load(var_78, adj_78, adj_80);
            wp::adj_address(var_hat_x, var_77, adj_hat_x, adj_77, adj_78);
            wp::adj_extract(var_30, var_76, adj_30, adj_76, adj_77);
            // adj: hat_t0 = hat_x[faceVInd[0]]                                                   <L 114>
            wp::adj_copy(var_75, adj_73, adj_74);
            wp::adj_load(var_73, adj_73, adj_75);
            wp::adj_address(var_hat_x, var_33, adj_hat_x, adj_33, adj_73);
            // adj: hat_p = hat_x[xI]                                                             <L 113>
            wp::adj_copy(var_72, adj_70, adj_71);
            wp::adj_load(var_70, adj_70, adj_72);
            wp::adj_address(var_x, var_69, adj_x, adj_69, adj_70);
            wp::adj_extract(var_30, var_68, adj_30, adj_68, adj_69);
            // adj: t2 = x[faceVInd[2]]                                                           <L 112>
            wp::adj_copy(var_67, adj_65, adj_66);
            wp::adj_load(var_65, adj_65, adj_67);
            wp::adj_address(var_x, var_64, adj_x, adj_64, adj_65);
            wp::adj_extract(var_30, var_63, adj_30, adj_63, adj_64);
            // adj: t1 = x[faceVInd[1]]                                                           <L 111>
            wp::adj_copy(var_62, adj_60, adj_61);
            wp::adj_load(var_60, adj_60, adj_62);
            wp::adj_address(var_x, var_59, adj_x, adj_59, adj_60);
            wp::adj_extract(var_30, var_58, adj_30, adj_58, adj_59);
            // adj: t0 = x[faceVInd[0]]                                                           <L 110>
            wp::adj_copy(var_57, adj_55, adj_56);
            wp::adj_load(var_55, adj_55, adj_57);
            wp::adj_address(var_x, var_33, adj_x, adj_33, adj_55);
            // adj: p = x[xI]                                                                     <L 109>
            wp::adj_vec_t(var_33, var_49, var_51, var_53, adj_33, adj_49, adj_51, adj_53, adj_54);
            wp::adj_extract(var_30, var_52, adj_30, adj_52, adj_53);
            wp::adj_extract(var_30, var_50, adj_30, adj_50, adj_51);
            wp::adj_extract(var_30, var_48, adj_30, adj_48, adj_49);
            // adj: vInd = wp.vec4i(xI, faceVInd[0], faceVInd[1], faceVInd[2])                    <L 108>
            if (var_47) {
                label0:;
                // adj: return                                                                    <L 107>
            }
            wp::adj_load(var_42, adj_42, adj_46);
            wp::adj_address(var_env_states, var_43, adj_env_states, adj_41, adj_42);
            wp::adj_load(var_41, adj_41, adj_43);
            wp::adj_address(var_node2env, var_33, adj_node2env, adj_33, adj_41);
            wp::adj_load(var_36, adj_36, adj_40);
            wp::adj_address(var_env_states, var_37, adj_env_states, adj_35, adj_36);
            wp::adj_load(var_35, adj_35, adj_37);
            wp::adj_address(var_node2env, var_33, adj_node2env, adj_33, adj_35);
            // adj: if (env_states[node2env[xI]] == ENV_STATE_INVALID) or (env_states[node2env[xI]] == ENV_STATE_NEWTON_SOLVED):  <L 106>
            wp::adj_copy(var_34, adj_32, adj_33);
            wp::adj_load(var_32, adj_32, adj_34);
            wp::adj_address(var_surf_vi, var_18, adj_surf_vi, adj_18, adj_32);
            // adj: xI = surf_vi[svi]                                                             <L 105>
            wp::adj_copy(var_31, adj_29, adj_30);
            wp::adj_load(var_29, adj_29, adj_31);
            wp::adj_address(var_face, var_26, adj_face, adj_26, adj_29);
            // adj: faceVInd = face[faceJ]                                                        <L 104>
            wp::adj_select(var_22, var_27, var_28, adj_22, adj_24, adj_25, adj_26);
            wp::adj_load(var_25, adj_25, adj_28);
            wp::adj_load(var_24, adj_24, adj_27);
            wp::adj_address(var_nodeJ, var_0, adj_nodeJ, adj_0, adj_25);
            wp::adj_address(var_nodeI, var_0, adj_nodeI, adj_0, adj_24);
            wp::adj_load(var_21, adj_21, adj_23);
            wp::adj_address(var_collision_type, var_0, adj_collision_type, adj_0, adj_21);
            // adj: faceJ = wp.select(collision_type[idx] == POINT_TRIANGLE_PAIR, nodeI[idx], nodeJ[idx])  <L 103>
            wp::adj_select(var_14, var_19, var_20, adj_14, adj_16, adj_17, adj_18);
            wp::adj_load(var_17, adj_17, adj_20);
            wp::adj_load(var_16, adj_16, adj_19);
            wp::adj_address(var_nodeI, var_0, adj_nodeI, adj_0, adj_17);
            wp::adj_address(var_nodeJ, var_0, adj_nodeJ, adj_0, adj_16);
            wp::adj_load(var_13, adj_13, adj_15);
            wp::adj_address(var_collision_type, var_0, adj_collision_type, adj_0, adj_13);
            // adj: svi = wp.select(collision_type[idx] == POINT_TRIANGLE_PAIR, nodeJ[idx], nodeI[idx])  <L 102>
        }
        wp::adj_load(var_8, adj_8, adj_11);
        wp::adj_address(var_collision_type, var_0, adj_collision_type, adj_0, adj_8);
        wp::adj_load(var_4, adj_4, adj_7);
        wp::adj_address(var_collision_type, var_0, adj_collision_type, adj_0, adj_4);
        // adj: if collision_type[idx] == TRIANGLE_POINT_PAIR or collision_type[idx] == POINT_TRIANGLE_PAIR:  <L 101>
        // adj: vInd = wp.vec4i()                                                                 <L 100>
        // adj: grad_b = wp.vec(length=12, dtype=wp.float64)                                      <L 99>
        // adj: idx = wp.tid()                                                                    <L 97>
        // adj: def grad_IPC_collisions(                                                          <L 65>
        continue;
    }
}

