import scipy
import math
import cupy as cp
import cudf
import numpy as np
from collections import Counter


def filter_genes(matrix, min_counts, min_cells, max_counts, max_cells, rows_per_batch=10000):
    """
    Creation dictionary with segments by values and by cells counts filtered by sparse gene expressions matrix

    :param matrix: cupyx.scipy.sparse.csr.csr_matrix, current gene expressions CSR matrix for filtering
    :param min_counts: int, minimum expression values
    :param min_cells: int, minimum cells
    :param max_counts: int, maximum expression values
    :param max_cells: int, maximum cells
    :param rows_per_batch: int, offset in number of rows

    :return: common_genes_counter: dict, dictionary with segments by values and by cells counts
    """

    def filtering_values(row_data, common_genes_counter, min_counts, min_cells, max_counts, max_cells):
        """
        Filtering values by sparse gene expressions matrix
        :return: common_genes_counter: dict, dictionary with segments by values and by cells counts
        """

        common_genes_counter['segment_by_counts'] = cp.concatenate(
            [common_genes_counter['segment_by_counts'],
             row_data[(row_data >= min_counts).get() & (row_data <= max_counts).get()]]
        )

        counter = Counter(row_data.get())
        common_genes_counter['segment_by_cells'] = np.concatenate(
            [common_genes_counter['segment_by_cells'],
             [x for x, count in counter.items() if count >= min_cells and count <= max_cells]
             ]
        )

        return common_genes_counter

    common_genes_counter = {
        'segment_by_counts': cp.empty(0),
        'segment_by_cells': np.empty(0)
    }

    n_batches = math.ceil(matrix.shape[0] / rows_per_batch)

    for i in range(n_batches):
        start_idx = matrix.indptr[i * rows_per_batch]
        stop_idx = matrix.indptr[min(start_idx + rows_per_batch, matrix.shape[0])]
        row_data = matrix[start_idx:stop_idx].data

        common_genes_counter = filtering_values(
            row_data=row_data,
            common_genes_counter=common_genes_counter,
            min_counts=min_counts,
            min_cells=min_cells,
            max_counts=max_counts,
            max_cells=max_cells
        )

    return common_genes_counter
