import cupy as cp
import cudf
import scipy
import math


def filter_cells(sparse_gpu_array, min_genes, max_genes, rows_per_batch=10000, barcodes=None):
    """
    Filter cells that have genes greater than a max number of genes or less than a minimum number of genes.

    :param sparse_gpu_array: cupy.sparse.csr_matrix, CSR matrix to filter
    :param min_genes: int, lower bound on number of genes to keep
    :param max_genes: int, upper bound on number of genes to keep
    :param rows_per_batch: int, batch size to use for filtering
    :param barcodes: cudf.core.series.Series, series with cell barcodes

    :return: filtered: scipy.sparse.csr_matrix, matrix on host with filtered cells of shape (n_cells, n_genes)
             barcodes: cudf.core.series.Series, If barcodes are provided, also returns a series of filtered barcodes.
    """

    def _filter_cells(sparse_gpu_array, min_genes, max_genes, barcodes=None):
        degrees = cp.diff(sparse_gpu_array.indptr)
        query = ((min_genes <= degrees) & (degrees <= max_genes))
        query = query.get()
        if barcodes is None:
            return sparse_gpu_array.get()[query]
        else:
            return sparse_gpu_array.get()[query], barcodes[query]

    n_batches = math.ceil(sparse_gpu_array.shape[0] / rows_per_batch)
    filtered_list = []
    barcodes_batch = None
    for i in range(n_batches):
        start_idx = i * rows_per_batch
        stop_idx = min(start_idx + rows_per_batch, sparse_gpu_array.shape[0])
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
        return scipy.sparse.vstack(filtered_data), cudf.concat(filtered_barcodes).reset_index(drop=True)
