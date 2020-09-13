1. Inefficient using of variables such as: 'batch_size', 'batch * batch_size', 'filtered_barcodes'
2. Inefficient using of '.ravel()' method for a 1-dimensional array
3. Returns could be more transparent for understanding (dictionary as an example)
4. Functions can be nested inside each other if the helper function is not used elsewhere in the module
5. Documentation not in PEP 257 style
