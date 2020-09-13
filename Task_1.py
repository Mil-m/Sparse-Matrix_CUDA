import cupy as cp
import cudf

# we don't use numpy package in the current module
import numpy as np
import scipy
import math

from cuml.linear_model import LinearRegression


def filter_cells(sparse_gpu_array, min_genes, max_genes, rows_per_batch=10000, barcodes=None):
    """
    Filter cells that have genes greater than a max number of genes or less than
    a minimum number of genes.
    Parameters
    ----------
    sparse_gpu_array : cupy.sparse.csr_matrix of shape (n_cells, n_genes)
        CSR matrix to filter
    min_genes : int
        Lower bound on number of genes to keep
    max_genes : int
        Upper bound on number of genes to keep
    rows_per_batch : int
        Batch size to use for filtering. This can be adjusted for performance
        to trade-off memory use.
    barcodes : series
        cudf series containing cell barcodes.
    Returns
    -------
    filtered : scipy.sparse.csr_matrix of shape (n_cells, n_genes)
        Matrix on host with filtered cells
    barcodes : If barcodes are provided, also returns a series of
        filtered barcodes.
    """

    n_batches = math.ceil(sparse_gpu_array.shape[0] / rows_per_batch)
    filtered_list = []
    barcodes_batch = None
    for batch in range(n_batches):
        # 'rows_per_batch' instead 'batch_size' value
        batch_size = rows_per_batch
        start_idx = batch * batch_size
        # 'start_idx' instead 'batch * batch_size' value
        stop_idx = min(batch * batch_size + batch_size, sparse_gpu_array.shape[0])
        arr_batch = sparse_gpu_array[start_idx:stop_idx]
        if barcodes is not None:
            barcodes_batch = barcodes[start_idx:stop_idx]
        filtered_list.append(_filter_cells(arr_batch,
                                            min_genes=min_genes,
                                            max_genes=max_genes,
                                            barcodes=barcodes_batch))

    if barcodes is None:
        return scipy.sparse.vstack(filtered_list)
    else:
        filtered_data = [x[0] for x in filtered_list]
        filtered_barcodes = [x[1] for x in filtered_list]
        # we can already use 'cudf.concat(filtered_barcodes)' in the return
        filtered_barcodes = cudf.concat(filtered_barcodes)
        return scipy.sparse.vstack(filtered_data), filtered_barcodes.reset_index(drop=True)


def _filter_cells(sparse_gpu_array, min_genes, max_genes, barcodes=None):
    degrees = cp.diff(sparse_gpu_array.indptr)
    # 1-dimensional array that's why we don't need in .ravel() method
    query = ((min_genes <= degrees) & (degrees <= max_genes)).ravel()
    # .get() method is necessary for cupy.sparse matrix and don't exist for scipy.sparse matrix from example
    query = query.get()
    if barcodes is None:
        return sparse_gpu_array.get()[query]
    else:
        return sparse_gpu_array.get()[query], barcodes[query]
