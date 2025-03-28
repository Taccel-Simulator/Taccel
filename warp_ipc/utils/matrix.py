import torch
import numpy as np
from numpy.typing import NDArray
import warp as wp
import warp.sparse as wps
import inspect
from icecream import ic
from typing import Tuple, List
ENABLE_COO_OFB = wp.constant(True)

@wp.struct
class COOMatrix3x3:
    rows: wp.array(dtype=wp.int32)
    cols: wp.array(dtype=wp.int32)
    vals: wp.array(dtype=wp.mat33d)
    capacity: wp.int32
    size: wp.int32
    n_rows: wp.int32
    n_cols: wp.int32

def COOMatrix3x3_init(mat: COOMatrix3x3, device: str, size: int=1, n_rows: int=0, n_cols: int=0) -> None:
    mat.rows = wp.zeros(size, dtype=wp.int32, device=device)
    mat.cols = wp.zeros(size, dtype=wp.int32, device=device)
    mat.vals = wp.zeros(size, dtype=wp.mat33d, device=device)
    mat.size = wp.int32(size)
    mat.capacity = size
    mat.n_rows = n_rows
    mat.n_cols = n_cols

def COOMatrix3x3_resize(mat: COOMatrix3x3, new_size: int):
    if new_size > mat.capacity:
        new_rows = wp.zeros(new_size, dtype=wp.int32, device=mat.rows.device)
        new_cols = wp.zeros(new_size, dtype=wp.int32, device=mat.cols.device)
        new_vals = wp.zeros(new_size, dtype=wp.mat33d, device=mat.vals.device)
        wp.copy(new_rows, mat.rows)
        wp.copy(new_cols, mat.cols)
        wp.copy(new_vals, mat.vals)
        mat.rows = new_rows
        mat.cols = new_cols
        mat.vals = new_vals
        mat.size = new_size
        mat.capacity = new_size
    else:
        mat.size = new_size

def COOMatrix3x3_zero_(mat: COOMatrix3x3):
    mat.rows.zero_()
    mat.cols.zero_()
    mat.vals.zero_()

def COOMatrix3x3_zero_val_(mat: COOMatrix3x3):
    mat.vals.zero_()

def COOMatrix3x3_clear(mat: COOMatrix3x3):
    mat.size = 0
    COOMatrix3x3_zero_(mat)

def COOMatrix3x3_clear_dense(mat: COOMatrix3x3) -> NDArray:
    dense = np.zeros((mat.n_rows * 3, mat.n_cols * 3), dtype=np.float64)
    rows_np = mat.rows.numpy()
    cols_np = mat.cols.numpy()
    vals_np = mat.vals.numpy()
    for (row, col, val) in zip(rows_np, cols_np, vals_np):
        dense[row * 3:row * 3 + 3, col * 3:col * 3 + 3] += val
    return dense

def COOMatrix3x3_to_torch_triplets(mat: COOMatrix3x3) -> Tuple[NDArray, NDArray, NDArray]:
    torch.cuda.empty_cache()
    return (wp.to_torch(mat.rows)[:mat.size], wp.to_torch(mat.cols)[:mat.size], wp.to_torch(mat.vals)[:mat.size])

def COOMatrix3x3_to_numpy_triplets(mat: COOMatrix3x3) -> Tuple[NDArray, NDArray, NDArray]:
    return (mat.rows.numpy()[:mat.size], mat.cols.numpy()[:mat.size], mat.vals.numpy()[:mat.size])

def bsr_from_coos_inplace(coos: List[COOMatrix3x3], sm: wps.BsrMatrix, device):
    all_rows_list = []
    all_cols_list = []
    all_vals_list = []
    for coo in coos:
        if coo.size == 0:
            continue
        coo: COOMatrix3x3
        (rows, cols, vals) = COOMatrix3x3_to_torch_triplets(coo)
        all_rows_list.append(rows)
        all_cols_list.append(cols)
        all_vals_list.append(vals)
    all_rows = wp.from_torch(torch.cat(all_rows_list, axis=0).contiguous(), dtype=wp.int32).to(device)
    all_cols = wp.from_torch(torch.cat(all_cols_list, axis=0).contiguous(), dtype=wp.int32).to(device)
    all_vals = wp.from_torch(torch.cat(all_vals_list, axis=0).contiguous(), dtype=wp.float64).to(device)
    wps.bsr_set_from_triplets(sm, all_rows, all_cols, all_vals)

def bsr_from_coos(*coos, device) -> wps.BsrMatrix:
    assert len(coos) > 0
    _coo: COOMatrix3x3 = coos[0]
    sm = wps.bsr_zeros(coos, _coo.n_rows, _coo.n_cols, block_type=wp.mat(shape=(3, 3), dtype=wp.float64), device=device)
    bsr_from_coos_inplace(coos, sm, device)
    return sm

def coos_to_numpy_triplets(*coos) -> Tuple[NDArray, NDArray, NDArray]:
    all_rows = []
    all_cols = []
    all_vals = []
    for coo in coos:
        coo: COOMatrix3x3
        if coo.size == 0:
            continue
        (rows, cols, vals) = COOMatrix3x3_to_numpy_triplets(coo)
        all_rows.append(rows)
        all_cols.append(cols)
        all_vals.append(vals)
    return (np.concatenate(all_rows, axis=0), np.concatenate(all_cols, axis=0), np.concatenate(all_vals, axis=0))

@wp.func
def COOMatrix3x3_atomic_add(mat: COOMatrix3x3, block_index: wp.int32, v: wp.mat33d):
    pass

@wp.func
def COOMatrix3x3_atomic_add_with_inds(mat: COOMatrix3x3, block_index: wp.int32, i: wp.int32, j: wp.int32, v: wp.mat33d):
    pass

@wp.func
def COOMatrix3x3_set_val(mat: COOMatrix3x3, block_index: wp.int32, v: wp.mat33d):
    pass

@wp.func
def COOMatrix3x3_insert(mat: COOMatrix3x3, block_index: wp.int32, i: wp.int32, j: wp.int32, v: wp.mat33d):
    pass

@wp.func
def COOMatrix3x3_insert_place(mat: COOMatrix3x3, block_index: wp.int32, i: wp.int32, j: wp.int32):
    pass

@wp.kernel
def test_insert_COO3x3(mat: COOMatrix3x3):
    pass
if __name__ == '__main__':
    wp.init()
    mat = COOMatrix3x3()
    COOMatrix3x3_init(mat, 'cuda:0', 4, 3, 3)
    ic(mat.size)
    wp.launch(test_insert_COO3x3, dim=2, inputs=[mat])
    ic(COOMatrix3x3_clear_dense(mat))
    ic(mat.cols)
    COOMatrix3x3_resize(mat, 20)