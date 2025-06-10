
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



extern "C" __global__ void dist_IPC_hs_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float64> var_energy_x,
    wp::array_t<wp::int32> var_hs_node,
    wp::array_t<wp::int32> var_hs_ground,
    wp::array_t<wp::vec_t<3,wp::float64>> var_x,
    wp::array_t<wp::int32> var_surf_vi,
    wp::array_t<wp::float64> var_node_xi,
    wp::array_t<wp::vec_t<3,wp::float64>> var_half_space_n,
    wp::array_t<wp::vec_t<3,wp::float64>> var_half_space_o)
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
        wp::int32 var_11;
        wp::int32 var_12;
        wp::vec_t<3,wp::float64>* var_13;
        wp::vec_t<3,wp::float64> var_14;
        wp::vec_t<3,wp::float64> var_15;
        wp::vec_t<3,wp::float64>* var_16;
        wp::vec_t<3,wp::float64> var_17;
        wp::vec_t<3,wp::float64> var_18;
        wp::vec_t<3,wp::float64>* var_19;
        wp::vec_t<3,wp::float64> var_20;
        wp::vec_t<3,wp::float64> var_21;
        wp::vec_t<3,wp::float64> var_22;
        wp::float64 var_23;
        wp::float64 var_24;
        //---------
        // forward
        // def dist_IPC_hs(                                                                       <L 18>
        // idx = wp.tid()                                                                         <L 28>
        var_0 = builtin_tid1d();
        // svi = hs_node[idx]                                                                     <L 29>
        var_1 = wp::address(var_hs_node, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // xi = node_xi[svi]                                                                      <L 30>
        var_4 = wp::address(var_node_xi, var_2);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // xI = surf_vi[svi]                                                                      <L 31>
        var_7 = wp::address(var_surf_vi, var_2);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // hs = hs_ground[idx]                                                                    <L 32>
        var_10 = wp::address(var_hs_ground, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // n = half_space_n[hs]                                                                   <L 33>
        var_13 = wp::address(var_half_space_n, var_11);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // o = half_space_o[hs]                                                                   <L 34>
        var_16 = wp::address(var_half_space_o, var_11);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // xIcoord = x[xI]                                                                        <L 35>
        var_19 = wp::address(var_x, var_8);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // d = wp.dot(xIcoord - o, n) - xi                                                        <L 36>
        var_22 = wp::sub(var_20, var_17);
        var_23 = wp::dot(var_22, var_14);
        var_24 = wp::sub(var_23, var_5);
        // energy_x[idx] = d                                                                      <L 37>
        wp::array_store(var_energy_x, var_0, var_24);
    }
}



extern "C" __global__ void dist_IPC_hs_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float64> var_energy_x,
    wp::array_t<wp::int32> var_hs_node,
    wp::array_t<wp::int32> var_hs_ground,
    wp::array_t<wp::vec_t<3,wp::float64>> var_x,
    wp::array_t<wp::int32> var_surf_vi,
    wp::array_t<wp::float64> var_node_xi,
    wp::array_t<wp::vec_t<3,wp::float64>> var_half_space_n,
    wp::array_t<wp::vec_t<3,wp::float64>> var_half_space_o,
    wp::array_t<wp::float64> adj_energy_x,
    wp::array_t<wp::int32> adj_hs_node,
    wp::array_t<wp::int32> adj_hs_ground,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_x,
    wp::array_t<wp::int32> adj_surf_vi,
    wp::array_t<wp::float64> adj_node_xi,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_half_space_n,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_half_space_o)
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
        wp::int32 var_11;
        wp::int32 var_12;
        wp::vec_t<3,wp::float64>* var_13;
        wp::vec_t<3,wp::float64> var_14;
        wp::vec_t<3,wp::float64> var_15;
        wp::vec_t<3,wp::float64>* var_16;
        wp::vec_t<3,wp::float64> var_17;
        wp::vec_t<3,wp::float64> var_18;
        wp::vec_t<3,wp::float64>* var_19;
        wp::vec_t<3,wp::float64> var_20;
        wp::vec_t<3,wp::float64> var_21;
        wp::vec_t<3,wp::float64> var_22;
        wp::float64 var_23;
        wp::float64 var_24;
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
        wp::vec_t<3,wp::float64> adj_13 = {};
        wp::vec_t<3,wp::float64> adj_14 = {};
        wp::vec_t<3,wp::float64> adj_15 = {};
        wp::vec_t<3,wp::float64> adj_16 = {};
        wp::vec_t<3,wp::float64> adj_17 = {};
        wp::vec_t<3,wp::float64> adj_18 = {};
        wp::vec_t<3,wp::float64> adj_19 = {};
        wp::vec_t<3,wp::float64> adj_20 = {};
        wp::vec_t<3,wp::float64> adj_21 = {};
        wp::vec_t<3,wp::float64> adj_22 = {};
        wp::float64 adj_23 = {};
        wp::float64 adj_24 = {};
        //---------
        // forward
        // def dist_IPC_hs(                                                                       <L 18>
        // idx = wp.tid()                                                                         <L 28>
        var_0 = builtin_tid1d();
        // svi = hs_node[idx]                                                                     <L 29>
        var_1 = wp::address(var_hs_node, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // xi = node_xi[svi]                                                                      <L 30>
        var_4 = wp::address(var_node_xi, var_2);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // xI = surf_vi[svi]                                                                      <L 31>
        var_7 = wp::address(var_surf_vi, var_2);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // hs = hs_ground[idx]                                                                    <L 32>
        var_10 = wp::address(var_hs_ground, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // n = half_space_n[hs]                                                                   <L 33>
        var_13 = wp::address(var_half_space_n, var_11);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // o = half_space_o[hs]                                                                   <L 34>
        var_16 = wp::address(var_half_space_o, var_11);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // xIcoord = x[xI]                                                                        <L 35>
        var_19 = wp::address(var_x, var_8);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // d = wp.dot(xIcoord - o, n) - xi                                                        <L 36>
        var_22 = wp::sub(var_20, var_17);
        var_23 = wp::dot(var_22, var_14);
        var_24 = wp::sub(var_23, var_5);
        // energy_x[idx] = d                                                                      <L 37>
        // wp::array_store(var_energy_x, var_0, var_24);
        //---------
        // reverse
        wp::adj_array_store(var_energy_x, var_0, var_24, adj_energy_x, adj_0, adj_24);
        // adj: energy_x[idx] = d                                                                 <L 37>
        wp::adj_sub(var_23, var_5, adj_23, adj_5, adj_24);
        wp::adj_dot(var_22, var_14, adj_22, adj_14, adj_23);
        wp::adj_sub(var_20, var_17, adj_20, adj_17, adj_22);
        // adj: d = wp.dot(xIcoord - o, n) - xi                                                   <L 36>
        wp::adj_copy(var_21, adj_19, adj_20);
        wp::adj_load(var_19, adj_19, adj_21);
        wp::adj_address(var_x, var_8, adj_x, adj_8, adj_19);
        // adj: xIcoord = x[xI]                                                                   <L 35>
        wp::adj_copy(var_18, adj_16, adj_17);
        wp::adj_load(var_16, adj_16, adj_18);
        wp::adj_address(var_half_space_o, var_11, adj_half_space_o, adj_11, adj_16);
        // adj: o = half_space_o[hs]                                                              <L 34>
        wp::adj_copy(var_15, adj_13, adj_14);
        wp::adj_load(var_13, adj_13, adj_15);
        wp::adj_address(var_half_space_n, var_11, adj_half_space_n, adj_11, adj_13);
        // adj: n = half_space_n[hs]                                                              <L 33>
        wp::adj_copy(var_12, adj_10, adj_11);
        wp::adj_load(var_10, adj_10, adj_12);
        wp::adj_address(var_hs_ground, var_0, adj_hs_ground, adj_0, adj_10);
        // adj: hs = hs_ground[idx]                                                               <L 32>
        wp::adj_copy(var_9, adj_7, adj_8);
        wp::adj_load(var_7, adj_7, adj_9);
        wp::adj_address(var_surf_vi, var_2, adj_surf_vi, adj_2, adj_7);
        // adj: xI = surf_vi[svi]                                                                 <L 31>
        wp::adj_copy(var_6, adj_4, adj_5);
        wp::adj_load(var_4, adj_4, adj_6);
        wp::adj_address(var_node_xi, var_2, adj_node_xi, adj_2, adj_4);
        // adj: xi = node_xi[svi]                                                                 <L 30>
        wp::adj_copy(var_3, adj_1, adj_2);
        wp::adj_load(var_1, adj_1, adj_3);
        wp::adj_address(var_hs_node, var_0, adj_hs_node, adj_0, adj_1);
        // adj: svi = hs_node[idx]                                                                <L 29>
        // adj: idx = wp.tid()                                                                    <L 28>
        // adj: def dist_IPC_hs(                                                                  <L 18>
        continue;
    }
}



extern "C" __global__ void dist_IPC_collisions_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float64> var_energy_x,
    wp::array_t<wp::int32> var_nodeI,
    wp::array_t<wp::int32> var_nodeJ,
    wp::array_t<wp::int32> var_collision_type,
    wp::array_t<wp::vec_t<3,wp::float64>> var_x,
    wp::array_t<wp::int32> var_surf_vi,
    wp::array_t<wp::float64> var_node_xi,
    wp::array_t<wp::float64> var_edge_xi,
    wp::array_t<wp::float64> var_face_xi,
    wp::array_t<wp::vec_t<2,wp::int32>> var_edge,
    wp::array_t<wp::vec_t<3,wp::int32>> var_face)
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
        const wp::int32 var_2 = 0;
        bool var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        const wp::int32 var_6 = 1;
        bool var_7;
        wp::int32 var_8;
        bool var_9;
        wp::int32* var_10;
        bool var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::int32* var_18;
        bool var_19;
        wp::int32 var_20;
        wp::int32* var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::vec_t<3,wp::int32>* var_26;
        wp::vec_t<3,wp::int32> var_27;
        wp::vec_t<3,wp::int32> var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        wp::vec_t<3,wp::float64>* var_32;
        wp::vec_t<3,wp::float64> var_33;
        wp::vec_t<3,wp::float64> var_34;
        const wp::int32 var_35 = 0;
        wp::int32 var_36;
        wp::vec_t<3,wp::float64>* var_37;
        wp::vec_t<3,wp::float64> var_38;
        wp::vec_t<3,wp::float64> var_39;
        const wp::int32 var_40 = 1;
        wp::int32 var_41;
        wp::vec_t<3,wp::float64>* var_42;
        wp::vec_t<3,wp::float64> var_43;
        wp::vec_t<3,wp::float64> var_44;
        const wp::int32 var_45 = 2;
        wp::int32 var_46;
        wp::vec_t<3,wp::float64>* var_47;
        wp::vec_t<3,wp::float64> var_48;
        wp::vec_t<3,wp::float64> var_49;
        wp::float64* var_50;
        wp::float64* var_51;
        wp::float64 var_52;
        wp::float64 var_53;
        wp::float64 var_54;
        wp::float64 var_55;
        wp::float64 var_56;
        wp::float64 var_57;
        wp::float64 var_58;
        wp::int32* var_59;
        wp::int32 var_60;
        wp::int32 var_61;
        wp::int32* var_62;
        wp::int32 var_63;
        wp::int32 var_64;
        wp::vec_t<2,wp::int32>* var_65;
        wp::vec_t<2,wp::int32> var_66;
        wp::vec_t<2,wp::int32> var_67;
        wp::vec_t<2,wp::int32>* var_68;
        wp::vec_t<2,wp::int32> var_69;
        wp::vec_t<2,wp::int32> var_70;
        wp::float64* var_71;
        wp::float64* var_72;
        wp::float64 var_73;
        wp::float64 var_74;
        wp::float64 var_75;
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
        const wp::int32 var_86 = 0;
        wp::int32 var_87;
        wp::vec_t<3,wp::float64>* var_88;
        wp::vec_t<3,wp::float64> var_89;
        wp::vec_t<3,wp::float64> var_90;
        const wp::int32 var_91 = 1;
        wp::int32 var_92;
        wp::vec_t<3,wp::float64>* var_93;
        wp::vec_t<3,wp::float64> var_94;
        wp::vec_t<3,wp::float64> var_95;
        wp::float64 var_96;
        wp::float64 var_97;
        wp::float64 var_98;
        wp::float64 var_99;
        wp::float64 var_100;
        wp::float64 var_101;
        //---------
        // forward
        // def dist_IPC_collisions(                                                               <L 41>
        // idx = wp.tid()                                                                         <L 54>
        var_0 = builtin_tid1d();
        // if collision_type[idx] == TRIANGLE_POINT_PAIR or collision_type[idx] == POINT_TRIANGLE_PAIR:       <L 56>
        var_1 = wp::address(var_collision_type, var_0);
        var_4 = wp::load(var_1);
        var_3 = (var_4 == var_2);
        var_5 = wp::address(var_collision_type, var_0);
        var_8 = wp::load(var_5);
        var_7 = (var_8 == var_6);
        var_9 = var_3 || var_7;
        if (var_9) {
            // svi = wp.select(collision_type[idx] == POINT_TRIANGLE_PAIR, nodeJ[idx], nodeI[idx])       <L 57>
            var_10 = wp::address(var_collision_type, var_0);
            var_12 = wp::load(var_10);
            var_11 = (var_12 == var_6);
            var_13 = wp::address(var_nodeJ, var_0);
            var_14 = wp::address(var_nodeI, var_0);
            var_16 = wp::load(var_13);
            var_17 = wp::load(var_14);
            var_15 = wp::select(var_11, var_16, var_17);
            // faceJ = wp.select(collision_type[idx] == POINT_TRIANGLE_PAIR, nodeI[idx], nodeJ[idx])       <L 58>
            var_18 = wp::address(var_collision_type, var_0);
            var_20 = wp::load(var_18);
            var_19 = (var_20 == var_6);
            var_21 = wp::address(var_nodeI, var_0);
            var_22 = wp::address(var_nodeJ, var_0);
            var_24 = wp::load(var_21);
            var_25 = wp::load(var_22);
            var_23 = wp::select(var_19, var_24, var_25);
            // faceVInd = face[faceJ]                                                             <L 59>
            var_26 = wp::address(var_face, var_23);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // xI = surf_vi[svi]                                                                  <L 60>
            var_29 = wp::address(var_surf_vi, var_15);
            var_31 = wp::load(var_29);
            var_30 = wp::copy(var_31);
            // p = x[xI]                                                                          <L 61>
            var_32 = wp::address(var_x, var_30);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // t0 = x[faceVInd[0]]                                                                <L 62>
            var_36 = wp::extract(var_27, var_35);
            var_37 = wp::address(var_x, var_36);
            var_39 = wp::load(var_37);
            var_38 = wp::copy(var_39);
            // t1 = x[faceVInd[1]]                                                                <L 63>
            var_41 = wp::extract(var_27, var_40);
            var_42 = wp::address(var_x, var_41);
            var_44 = wp::load(var_42);
            var_43 = wp::copy(var_44);
            // t2 = x[faceVInd[2]]                                                                <L 64>
            var_46 = wp::extract(var_27, var_45);
            var_47 = wp::address(var_x, var_46);
            var_49 = wp::load(var_47);
            var_48 = wp::copy(var_49);
            // xi = node_xi[svi] + face_xi[faceJ]                                                 <L 65>
            var_50 = wp::address(var_node_xi, var_15);
            var_51 = wp::address(var_face_xi, var_23);
            var_53 = wp::load(var_50);
            var_54 = wp::load(var_51);
            var_52 = wp::add(var_53, var_54);
            // dsq = point_triangle_distance(p, t0, t1, t2) - xi * xi                             <L 66>
            var_55 = point_triangle_distance_0(var_33, var_38, var_43, var_48);
            var_56 = wp::mul(var_52, var_52);
            var_57 = wp::sub(var_55, var_56);
            // energy_x[idx] = wp.sqrt(dsq)                                                       <L 67>
            var_58 = wp::sqrt(var_57);
            wp::array_store(var_energy_x, var_0, var_58);
        }
        if (!var_9) {
            // edgeI = nodeI[idx]                                                                 <L 69>
            var_59 = wp::address(var_nodeI, var_0);
            var_61 = wp::load(var_59);
            var_60 = wp::copy(var_61);
            // edgeJ = nodeJ[idx]                                                                 <L 70>
            var_62 = wp::address(var_nodeJ, var_0);
            var_64 = wp::load(var_62);
            var_63 = wp::copy(var_64);
            // edgeVIndI = edge[edgeI]                                                            <L 71>
            var_65 = wp::address(var_edge, var_60);
            var_67 = wp::load(var_65);
            var_66 = wp::copy(var_67);
            // edgeVIndJ = edge[edgeJ]                                                            <L 72>
            var_68 = wp::address(var_edge, var_63);
            var_70 = wp::load(var_68);
            var_69 = wp::copy(var_70);
            // xi = edge_xi[edgeI] + edge_xi[edgeJ]                                               <L 73>
            var_71 = wp::address(var_edge_xi, var_60);
            var_72 = wp::address(var_edge_xi, var_63);
            var_74 = wp::load(var_71);
            var_75 = wp::load(var_72);
            var_73 = wp::add(var_74, var_75);
            // ea0 = x[edgeVIndI[0]]                                                              <L 74>
            var_77 = wp::extract(var_66, var_76);
            var_78 = wp::address(var_x, var_77);
            var_80 = wp::load(var_78);
            var_79 = wp::copy(var_80);
            // ea1 = x[edgeVIndI[1]]                                                              <L 75>
            var_82 = wp::extract(var_66, var_81);
            var_83 = wp::address(var_x, var_82);
            var_85 = wp::load(var_83);
            var_84 = wp::copy(var_85);
            // eb0 = x[edgeVIndJ[0]]                                                              <L 76>
            var_87 = wp::extract(var_69, var_86);
            var_88 = wp::address(var_x, var_87);
            var_90 = wp::load(var_88);
            var_89 = wp::copy(var_90);
            // eb1 = x[edgeVIndJ[1]]                                                              <L 77>
            var_92 = wp::extract(var_69, var_91);
            var_93 = wp::address(var_x, var_92);
            var_95 = wp::load(var_93);
            var_94 = wp::copy(var_95);
            // dsq = edge_edge_distance(ea0, ea1, eb0, eb1) - xi * xi                             <L 78>
            var_96 = edge_edge_distance_0(var_79, var_84, var_89, var_94);
            var_97 = wp::mul(var_73, var_73);
            var_98 = wp::sub(var_96, var_97);
            // energy_x[idx] = wp.sqrt(dsq)                                                       <L 79>
            var_99 = wp::sqrt(var_98);
            wp::array_store(var_energy_x, var_0, var_99);
        }
        var_100 = wp::where(var_9, var_52, var_73);
        var_101 = wp::where(var_9, var_57, var_98);
    }
}



extern "C" __global__ void dist_IPC_collisions_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float64> var_energy_x,
    wp::array_t<wp::int32> var_nodeI,
    wp::array_t<wp::int32> var_nodeJ,
    wp::array_t<wp::int32> var_collision_type,
    wp::array_t<wp::vec_t<3,wp::float64>> var_x,
    wp::array_t<wp::int32> var_surf_vi,
    wp::array_t<wp::float64> var_node_xi,
    wp::array_t<wp::float64> var_edge_xi,
    wp::array_t<wp::float64> var_face_xi,
    wp::array_t<wp::vec_t<2,wp::int32>> var_edge,
    wp::array_t<wp::vec_t<3,wp::int32>> var_face,
    wp::array_t<wp::float64> adj_energy_x,
    wp::array_t<wp::int32> adj_nodeI,
    wp::array_t<wp::int32> adj_nodeJ,
    wp::array_t<wp::int32> adj_collision_type,
    wp::array_t<wp::vec_t<3,wp::float64>> adj_x,
    wp::array_t<wp::int32> adj_surf_vi,
    wp::array_t<wp::float64> adj_node_xi,
    wp::array_t<wp::float64> adj_edge_xi,
    wp::array_t<wp::float64> adj_face_xi,
    wp::array_t<wp::vec_t<2,wp::int32>> adj_edge,
    wp::array_t<wp::vec_t<3,wp::int32>> adj_face)
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
        const wp::int32 var_2 = 0;
        bool var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        const wp::int32 var_6 = 1;
        bool var_7;
        wp::int32 var_8;
        bool var_9;
        wp::int32* var_10;
        bool var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::int32* var_18;
        bool var_19;
        wp::int32 var_20;
        wp::int32* var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::vec_t<3,wp::int32>* var_26;
        wp::vec_t<3,wp::int32> var_27;
        wp::vec_t<3,wp::int32> var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        wp::vec_t<3,wp::float64>* var_32;
        wp::vec_t<3,wp::float64> var_33;
        wp::vec_t<3,wp::float64> var_34;
        const wp::int32 var_35 = 0;
        wp::int32 var_36;
        wp::vec_t<3,wp::float64>* var_37;
        wp::vec_t<3,wp::float64> var_38;
        wp::vec_t<3,wp::float64> var_39;
        const wp::int32 var_40 = 1;
        wp::int32 var_41;
        wp::vec_t<3,wp::float64>* var_42;
        wp::vec_t<3,wp::float64> var_43;
        wp::vec_t<3,wp::float64> var_44;
        const wp::int32 var_45 = 2;
        wp::int32 var_46;
        wp::vec_t<3,wp::float64>* var_47;
        wp::vec_t<3,wp::float64> var_48;
        wp::vec_t<3,wp::float64> var_49;
        wp::float64* var_50;
        wp::float64* var_51;
        wp::float64 var_52;
        wp::float64 var_53;
        wp::float64 var_54;
        wp::float64 var_55;
        wp::float64 var_56;
        wp::float64 var_57;
        wp::float64 var_58;
        wp::int32* var_59;
        wp::int32 var_60;
        wp::int32 var_61;
        wp::int32* var_62;
        wp::int32 var_63;
        wp::int32 var_64;
        wp::vec_t<2,wp::int32>* var_65;
        wp::vec_t<2,wp::int32> var_66;
        wp::vec_t<2,wp::int32> var_67;
        wp::vec_t<2,wp::int32>* var_68;
        wp::vec_t<2,wp::int32> var_69;
        wp::vec_t<2,wp::int32> var_70;
        wp::float64* var_71;
        wp::float64* var_72;
        wp::float64 var_73;
        wp::float64 var_74;
        wp::float64 var_75;
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
        const wp::int32 var_86 = 0;
        wp::int32 var_87;
        wp::vec_t<3,wp::float64>* var_88;
        wp::vec_t<3,wp::float64> var_89;
        wp::vec_t<3,wp::float64> var_90;
        const wp::int32 var_91 = 1;
        wp::int32 var_92;
        wp::vec_t<3,wp::float64>* var_93;
        wp::vec_t<3,wp::float64> var_94;
        wp::vec_t<3,wp::float64> var_95;
        wp::float64 var_96;
        wp::float64 var_97;
        wp::float64 var_98;
        wp::float64 var_99;
        wp::float64 var_100;
        wp::float64 var_101;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        bool adj_7 = {};
        wp::int32 adj_8 = {};
        bool adj_9 = {};
        wp::int32 adj_10 = {};
        bool adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        bool adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::vec_t<3,wp::int32> adj_26 = {};
        wp::vec_t<3,wp::int32> adj_27 = {};
        wp::vec_t<3,wp::int32> adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::vec_t<3,wp::float64> adj_32 = {};
        wp::vec_t<3,wp::float64> adj_33 = {};
        wp::vec_t<3,wp::float64> adj_34 = {};
        wp::int32 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::vec_t<3,wp::float64> adj_37 = {};
        wp::vec_t<3,wp::float64> adj_38 = {};
        wp::vec_t<3,wp::float64> adj_39 = {};
        wp::int32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::vec_t<3,wp::float64> adj_42 = {};
        wp::vec_t<3,wp::float64> adj_43 = {};
        wp::vec_t<3,wp::float64> adj_44 = {};
        wp::int32 adj_45 = {};
        wp::int32 adj_46 = {};
        wp::vec_t<3,wp::float64> adj_47 = {};
        wp::vec_t<3,wp::float64> adj_48 = {};
        wp::vec_t<3,wp::float64> adj_49 = {};
        wp::float64 adj_50 = {};
        wp::float64 adj_51 = {};
        wp::float64 adj_52 = {};
        wp::float64 adj_53 = {};
        wp::float64 adj_54 = {};
        wp::float64 adj_55 = {};
        wp::float64 adj_56 = {};
        wp::float64 adj_57 = {};
        wp::float64 adj_58 = {};
        wp::int32 adj_59 = {};
        wp::int32 adj_60 = {};
        wp::int32 adj_61 = {};
        wp::int32 adj_62 = {};
        wp::int32 adj_63 = {};
        wp::int32 adj_64 = {};
        wp::vec_t<2,wp::int32> adj_65 = {};
        wp::vec_t<2,wp::int32> adj_66 = {};
        wp::vec_t<2,wp::int32> adj_67 = {};
        wp::vec_t<2,wp::int32> adj_68 = {};
        wp::vec_t<2,wp::int32> adj_69 = {};
        wp::vec_t<2,wp::int32> adj_70 = {};
        wp::float64 adj_71 = {};
        wp::float64 adj_72 = {};
        wp::float64 adj_73 = {};
        wp::float64 adj_74 = {};
        wp::float64 adj_75 = {};
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
        wp::int32 adj_91 = {};
        wp::int32 adj_92 = {};
        wp::vec_t<3,wp::float64> adj_93 = {};
        wp::vec_t<3,wp::float64> adj_94 = {};
        wp::vec_t<3,wp::float64> adj_95 = {};
        wp::float64 adj_96 = {};
        wp::float64 adj_97 = {};
        wp::float64 adj_98 = {};
        wp::float64 adj_99 = {};
        wp::float64 adj_100 = {};
        wp::float64 adj_101 = {};
        //---------
        // forward
        // def dist_IPC_collisions(                                                               <L 41>
        // idx = wp.tid()                                                                         <L 54>
        var_0 = builtin_tid1d();
        // if collision_type[idx] == TRIANGLE_POINT_PAIR or collision_type[idx] == POINT_TRIANGLE_PAIR:       <L 56>
        var_1 = wp::address(var_collision_type, var_0);
        var_4 = wp::load(var_1);
        var_3 = (var_4 == var_2);
        var_5 = wp::address(var_collision_type, var_0);
        var_8 = wp::load(var_5);
        var_7 = (var_8 == var_6);
        var_9 = var_3 || var_7;
        if (var_9) {
            // svi = wp.select(collision_type[idx] == POINT_TRIANGLE_PAIR, nodeJ[idx], nodeI[idx])       <L 57>
            var_10 = wp::address(var_collision_type, var_0);
            var_12 = wp::load(var_10);
            var_11 = (var_12 == var_6);
            var_13 = wp::address(var_nodeJ, var_0);
            var_14 = wp::address(var_nodeI, var_0);
            var_16 = wp::load(var_13);
            var_17 = wp::load(var_14);
            var_15 = wp::select(var_11, var_16, var_17);
            // faceJ = wp.select(collision_type[idx] == POINT_TRIANGLE_PAIR, nodeI[idx], nodeJ[idx])       <L 58>
            var_18 = wp::address(var_collision_type, var_0);
            var_20 = wp::load(var_18);
            var_19 = (var_20 == var_6);
            var_21 = wp::address(var_nodeI, var_0);
            var_22 = wp::address(var_nodeJ, var_0);
            var_24 = wp::load(var_21);
            var_25 = wp::load(var_22);
            var_23 = wp::select(var_19, var_24, var_25);
            // faceVInd = face[faceJ]                                                             <L 59>
            var_26 = wp::address(var_face, var_23);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // xI = surf_vi[svi]                                                                  <L 60>
            var_29 = wp::address(var_surf_vi, var_15);
            var_31 = wp::load(var_29);
            var_30 = wp::copy(var_31);
            // p = x[xI]                                                                          <L 61>
            var_32 = wp::address(var_x, var_30);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // t0 = x[faceVInd[0]]                                                                <L 62>
            var_36 = wp::extract(var_27, var_35);
            var_37 = wp::address(var_x, var_36);
            var_39 = wp::load(var_37);
            var_38 = wp::copy(var_39);
            // t1 = x[faceVInd[1]]                                                                <L 63>
            var_41 = wp::extract(var_27, var_40);
            var_42 = wp::address(var_x, var_41);
            var_44 = wp::load(var_42);
            var_43 = wp::copy(var_44);
            // t2 = x[faceVInd[2]]                                                                <L 64>
            var_46 = wp::extract(var_27, var_45);
            var_47 = wp::address(var_x, var_46);
            var_49 = wp::load(var_47);
            var_48 = wp::copy(var_49);
            // xi = node_xi[svi] + face_xi[faceJ]                                                 <L 65>
            var_50 = wp::address(var_node_xi, var_15);
            var_51 = wp::address(var_face_xi, var_23);
            var_53 = wp::load(var_50);
            var_54 = wp::load(var_51);
            var_52 = wp::add(var_53, var_54);
            // dsq = point_triangle_distance(p, t0, t1, t2) - xi * xi                             <L 66>
            var_55 = point_triangle_distance_0(var_33, var_38, var_43, var_48);
            var_56 = wp::mul(var_52, var_52);
            var_57 = wp::sub(var_55, var_56);
            // energy_x[idx] = wp.sqrt(dsq)                                                       <L 67>
            var_58 = wp::sqrt(var_57);
            // wp::array_store(var_energy_x, var_0, var_58);
        }
        if (!var_9) {
            // edgeI = nodeI[idx]                                                                 <L 69>
            var_59 = wp::address(var_nodeI, var_0);
            var_61 = wp::load(var_59);
            var_60 = wp::copy(var_61);
            // edgeJ = nodeJ[idx]                                                                 <L 70>
            var_62 = wp::address(var_nodeJ, var_0);
            var_64 = wp::load(var_62);
            var_63 = wp::copy(var_64);
            // edgeVIndI = edge[edgeI]                                                            <L 71>
            var_65 = wp::address(var_edge, var_60);
            var_67 = wp::load(var_65);
            var_66 = wp::copy(var_67);
            // edgeVIndJ = edge[edgeJ]                                                            <L 72>
            var_68 = wp::address(var_edge, var_63);
            var_70 = wp::load(var_68);
            var_69 = wp::copy(var_70);
            // xi = edge_xi[edgeI] + edge_xi[edgeJ]                                               <L 73>
            var_71 = wp::address(var_edge_xi, var_60);
            var_72 = wp::address(var_edge_xi, var_63);
            var_74 = wp::load(var_71);
            var_75 = wp::load(var_72);
            var_73 = wp::add(var_74, var_75);
            // ea0 = x[edgeVIndI[0]]                                                              <L 74>
            var_77 = wp::extract(var_66, var_76);
            var_78 = wp::address(var_x, var_77);
            var_80 = wp::load(var_78);
            var_79 = wp::copy(var_80);
            // ea1 = x[edgeVIndI[1]]                                                              <L 75>
            var_82 = wp::extract(var_66, var_81);
            var_83 = wp::address(var_x, var_82);
            var_85 = wp::load(var_83);
            var_84 = wp::copy(var_85);
            // eb0 = x[edgeVIndJ[0]]                                                              <L 76>
            var_87 = wp::extract(var_69, var_86);
            var_88 = wp::address(var_x, var_87);
            var_90 = wp::load(var_88);
            var_89 = wp::copy(var_90);
            // eb1 = x[edgeVIndJ[1]]                                                              <L 77>
            var_92 = wp::extract(var_69, var_91);
            var_93 = wp::address(var_x, var_92);
            var_95 = wp::load(var_93);
            var_94 = wp::copy(var_95);
            // dsq = edge_edge_distance(ea0, ea1, eb0, eb1) - xi * xi                             <L 78>
            var_96 = edge_edge_distance_0(var_79, var_84, var_89, var_94);
            var_97 = wp::mul(var_73, var_73);
            var_98 = wp::sub(var_96, var_97);
            // energy_x[idx] = wp.sqrt(dsq)                                                       <L 79>
            var_99 = wp::sqrt(var_98);
            // wp::array_store(var_energy_x, var_0, var_99);
        }
        var_100 = wp::where(var_9, var_52, var_73);
        var_101 = wp::where(var_9, var_57, var_98);
        //---------
        // reverse
        wp::adj_where(var_9, var_57, var_98, adj_9, adj_57, adj_98, adj_101);
        wp::adj_where(var_9, var_52, var_73, adj_9, adj_52, adj_73, adj_100);
        if (!var_9) {
            wp::adj_array_store(var_energy_x, var_0, var_99, adj_energy_x, adj_0, adj_99);
            wp::adj_sqrt(var_98, var_99, adj_98, adj_99);
            // adj: energy_x[idx] = wp.sqrt(dsq)                                                  <L 79>
            wp::adj_sub(var_96, var_97, adj_96, adj_97, adj_98);
            wp::adj_mul(var_73, var_73, adj_73, adj_73, adj_97);
            adj_edge_edge_distance_0(var_79, var_84, var_89, var_94, adj_79, adj_84, adj_89, adj_94, adj_96);
            // adj: dsq = edge_edge_distance(ea0, ea1, eb0, eb1) - xi * xi                        <L 78>
            wp::adj_copy(var_95, adj_93, adj_94);
            wp::adj_load(var_93, adj_93, adj_95);
            wp::adj_address(var_x, var_92, adj_x, adj_92, adj_93);
            wp::adj_extract(var_69, var_91, adj_69, adj_91, adj_92);
            // adj: eb1 = x[edgeVIndJ[1]]                                                         <L 77>
            wp::adj_copy(var_90, adj_88, adj_89);
            wp::adj_load(var_88, adj_88, adj_90);
            wp::adj_address(var_x, var_87, adj_x, adj_87, adj_88);
            wp::adj_extract(var_69, var_86, adj_69, adj_86, adj_87);
            // adj: eb0 = x[edgeVIndJ[0]]                                                         <L 76>
            wp::adj_copy(var_85, adj_83, adj_84);
            wp::adj_load(var_83, adj_83, adj_85);
            wp::adj_address(var_x, var_82, adj_x, adj_82, adj_83);
            wp::adj_extract(var_66, var_81, adj_66, adj_81, adj_82);
            // adj: ea1 = x[edgeVIndI[1]]                                                         <L 75>
            wp::adj_copy(var_80, adj_78, adj_79);
            wp::adj_load(var_78, adj_78, adj_80);
            wp::adj_address(var_x, var_77, adj_x, adj_77, adj_78);
            wp::adj_extract(var_66, var_76, adj_66, adj_76, adj_77);
            // adj: ea0 = x[edgeVIndI[0]]                                                         <L 74>
            wp::adj_add(var_74, var_75, adj_71, adj_72, adj_73);
            wp::adj_load(var_72, adj_72, adj_75);
            wp::adj_load(var_71, adj_71, adj_74);
            wp::adj_address(var_edge_xi, var_63, adj_edge_xi, adj_63, adj_72);
            wp::adj_address(var_edge_xi, var_60, adj_edge_xi, adj_60, adj_71);
            // adj: xi = edge_xi[edgeI] + edge_xi[edgeJ]                                          <L 73>
            wp::adj_copy(var_70, adj_68, adj_69);
            wp::adj_load(var_68, adj_68, adj_70);
            wp::adj_address(var_edge, var_63, adj_edge, adj_63, adj_68);
            // adj: edgeVIndJ = edge[edgeJ]                                                       <L 72>
            wp::adj_copy(var_67, adj_65, adj_66);
            wp::adj_load(var_65, adj_65, adj_67);
            wp::adj_address(var_edge, var_60, adj_edge, adj_60, adj_65);
            // adj: edgeVIndI = edge[edgeI]                                                       <L 71>
            wp::adj_copy(var_64, adj_62, adj_63);
            wp::adj_load(var_62, adj_62, adj_64);
            wp::adj_address(var_nodeJ, var_0, adj_nodeJ, adj_0, adj_62);
            // adj: edgeJ = nodeJ[idx]                                                            <L 70>
            wp::adj_copy(var_61, adj_59, adj_60);
            wp::adj_load(var_59, adj_59, adj_61);
            wp::adj_address(var_nodeI, var_0, adj_nodeI, adj_0, adj_59);
            // adj: edgeI = nodeI[idx]                                                            <L 69>
        }
        if (var_9) {
            wp::adj_array_store(var_energy_x, var_0, var_58, adj_energy_x, adj_0, adj_58);
            wp::adj_sqrt(var_57, var_58, adj_57, adj_58);
            // adj: energy_x[idx] = wp.sqrt(dsq)                                                  <L 67>
            wp::adj_sub(var_55, var_56, adj_55, adj_56, adj_57);
            wp::adj_mul(var_52, var_52, adj_52, adj_52, adj_56);
            adj_point_triangle_distance_0(var_33, var_38, var_43, var_48, adj_33, adj_38, adj_43, adj_48, adj_55);
            // adj: dsq = point_triangle_distance(p, t0, t1, t2) - xi * xi                        <L 66>
            wp::adj_add(var_53, var_54, adj_50, adj_51, adj_52);
            wp::adj_load(var_51, adj_51, adj_54);
            wp::adj_load(var_50, adj_50, adj_53);
            wp::adj_address(var_face_xi, var_23, adj_face_xi, adj_23, adj_51);
            wp::adj_address(var_node_xi, var_15, adj_node_xi, adj_15, adj_50);
            // adj: xi = node_xi[svi] + face_xi[faceJ]                                            <L 65>
            wp::adj_copy(var_49, adj_47, adj_48);
            wp::adj_load(var_47, adj_47, adj_49);
            wp::adj_address(var_x, var_46, adj_x, adj_46, adj_47);
            wp::adj_extract(var_27, var_45, adj_27, adj_45, adj_46);
            // adj: t2 = x[faceVInd[2]]                                                           <L 64>
            wp::adj_copy(var_44, adj_42, adj_43);
            wp::adj_load(var_42, adj_42, adj_44);
            wp::adj_address(var_x, var_41, adj_x, adj_41, adj_42);
            wp::adj_extract(var_27, var_40, adj_27, adj_40, adj_41);
            // adj: t1 = x[faceVInd[1]]                                                           <L 63>
            wp::adj_copy(var_39, adj_37, adj_38);
            wp::adj_load(var_37, adj_37, adj_39);
            wp::adj_address(var_x, var_36, adj_x, adj_36, adj_37);
            wp::adj_extract(var_27, var_35, adj_27, adj_35, adj_36);
            // adj: t0 = x[faceVInd[0]]                                                           <L 62>
            wp::adj_copy(var_34, adj_32, adj_33);
            wp::adj_load(var_32, adj_32, adj_34);
            wp::adj_address(var_x, var_30, adj_x, adj_30, adj_32);
            // adj: p = x[xI]                                                                     <L 61>
            wp::adj_copy(var_31, adj_29, adj_30);
            wp::adj_load(var_29, adj_29, adj_31);
            wp::adj_address(var_surf_vi, var_15, adj_surf_vi, adj_15, adj_29);
            // adj: xI = surf_vi[svi]                                                             <L 60>
            wp::adj_copy(var_28, adj_26, adj_27);
            wp::adj_load(var_26, adj_26, adj_28);
            wp::adj_address(var_face, var_23, adj_face, adj_23, adj_26);
            // adj: faceVInd = face[faceJ]                                                        <L 59>
            wp::adj_select(var_19, var_24, var_25, adj_19, adj_21, adj_22, adj_23);
            wp::adj_load(var_22, adj_22, adj_25);
            wp::adj_load(var_21, adj_21, adj_24);
            wp::adj_address(var_nodeJ, var_0, adj_nodeJ, adj_0, adj_22);
            wp::adj_address(var_nodeI, var_0, adj_nodeI, adj_0, adj_21);
            wp::adj_load(var_18, adj_18, adj_20);
            wp::adj_address(var_collision_type, var_0, adj_collision_type, adj_0, adj_18);
            // adj: faceJ = wp.select(collision_type[idx] == POINT_TRIANGLE_PAIR, nodeI[idx], nodeJ[idx])  <L 58>
            wp::adj_select(var_11, var_16, var_17, adj_11, adj_13, adj_14, adj_15);
            wp::adj_load(var_14, adj_14, adj_17);
            wp::adj_load(var_13, adj_13, adj_16);
            wp::adj_address(var_nodeI, var_0, adj_nodeI, adj_0, adj_14);
            wp::adj_address(var_nodeJ, var_0, adj_nodeJ, adj_0, adj_13);
            wp::adj_load(var_10, adj_10, adj_12);
            wp::adj_address(var_collision_type, var_0, adj_collision_type, adj_0, adj_10);
            // adj: svi = wp.select(collision_type[idx] == POINT_TRIANGLE_PAIR, nodeJ[idx], nodeI[idx])  <L 57>
        }
        wp::adj_load(var_5, adj_5, adj_8);
        wp::adj_address(var_collision_type, var_0, adj_collision_type, adj_0, adj_5);
        wp::adj_load(var_1, adj_1, adj_4);
        wp::adj_address(var_collision_type, var_0, adj_collision_type, adj_0, adj_1);
        // adj: if collision_type[idx] == TRIANGLE_POINT_PAIR or collision_type[idx] == POINT_TRIANGLE_PAIR:  <L 56>
        // adj: idx = wp.tid()                                                                    <L 54>
        // adj: def dist_IPC_collisions(                                                          <L 41>
        continue;
    }
}

