

import numpy as np
import math


def match_tensor_shape(data, tensor):
    tensor_shape = tf_to_np_shape(tensor.get_shape().as_list())
    if not compare_shapes(data.shape, tensor_shape):
        data = np.reshape(data, tensor_shape)
    return data

def tf_to_np_shape(tf_shape):
    """
    tf shapes use None for arbitrarily sized dimensions
    np shapes use -1 for arbitrarily sized dimensions

    Converts a tf shape to a np shape
    """
    return [x if x is not None else -1 for x in tf_shape]
    

def compare_shapes(shape_a, shape_b):
    if len(shape_a) != len(shape_b): return False
    shape_a = tf_to_np_shape(shape_a)
    shape_b = tf_to_np_shape(shape_b)
    for dim_a, dim_b in zip(shape_a, shape_b):
        if dim_a <= -1 or dim_b <= -1: continue
        if dim_a != dim_b: return False
    return True


def get_num_batches(dataset_size, batch_size):
    if batch_size is None or batch_size > dataset_size:
        return dataset_size
    elif batch_size <= 0:
        return 0
    else: 
        return int(math.ceil(dataset_size / float(batch_size)))


def batch_dataset(dataset, batch_size, include_progress = False):
    if batch_size is None: 
        yield dataset
    else:
        x, y = dataset 
        batch_total = get_num_batches(len(x), batch_size) 
        for batch_num in range(batch_total):
            batch_idx = batch_num * batch_size
            batch_x = x[batch_idx : batch_idx + batch_size]
            batch_y = y[batch_idx : batch_idx + batch_size]

            if include_progress:
                yield batch_x, batch_y, batch_num, batch_total
            else:
                yield batch_x, batch_y



