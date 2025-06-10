import numpy as np
from icecream import ic
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csc_matrix, vstack
from scipy.sparse.linalg import inv as sparse_inv
ZERO_THRESH = 1e-12

def zeros_by_thresh(arr: NDArray):
    arr[np.abs(arr) < ZERO_THRESH] = 0

def gauss_eliminate(arr: NDArray):
    """
    inplace row major gauss elimination
    return: eliminated array, non-major column indices, rank
    """
    assert len(arr.shape) == 2
    (n_rows, n_cols) = arr.shape
    cols_is_not_major = np.zeros((n_cols,), dtype=np.int32)
    col_index = 0
    row_index = 0
    while col_index < n_cols and row_index < n_rows:
        major_row = np.argmax(np.abs(arr[row_index:, col_index])) + row_index
        current_row = arr[row_index, :].copy()
        arr[row_index, :] = arr[major_row, :]
        arr[major_row, :] = current_row
        major_elem = arr[row_index, col_index]
        if np.isclose(major_elem, 0):
            cols_is_not_major[col_index] = 1
            col_index += 1
            continue
        coef = arr[row_index + 1:, col_index] / major_elem
        arr[row_index + 1:, col_index:] -= np.outer(coef, arr[row_index, col_index:])
        row_index += 1
        col_index += 1
        zeros_by_thresh(arr)
    cols_is_not_major[col_index:] = 1
    return (arr, np.nonzero(cols_is_not_major)[0], row_index)
if __name__ == '__main__':
    test_arr = np.random.randn(3, 6)
    test_arr[:, 1] = test_arr[:, 0] * 2
    test_arr[2, :] = test_arr[0, :] + test_arr[1, :]
    ic(test_arr)
    (arr, non_major_col_inds, rank) = gauss_eliminate(test_arr)
    ic(arr, non_major_col_inds, rank)
    arr = arr[:rank]
    ic(arr)
    sparse_U_upper_rows = csc_matrix(arr)
    data = np.ones(len(non_major_col_inds), dtype=np.float64)
    coords = np.stack([np.arange(len(non_major_col_inds)), non_major_col_inds], axis=0)
    sparse_U_lower_rows = coo_matrix((data, coords))
    sparse_U = vstack([sparse_U_upper_rows, sparse_U_lower_rows]).tocsc()
    ic(sparse_U.toarray())
    sparse_V = sparse_inv(sparse_U)
    VU = (sparse_V @ sparse_U).toarray()
    zeros_by_thresh(VU)
    ic(sparse_V, VU)
    ic(sparse_V.indptr, sparse_V.indices, sparse_V.data)
    bsr_V = sparse_V.tobsr(blocksize=(3, 3))
    ic(bsr_V, bsr_V.indptr, bsr_V.indices, bsr_V.data, bsr_V.toarray() - sparse_V.toarray())
    ic(bsr_V.shape)