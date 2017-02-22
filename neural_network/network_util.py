

import numpy as np
import math
import tensorflow as tf
import sys

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



def make_per_class_eval_tensor(eval_func, net_output, exp_output, scope = None):
    """
    |net_output| and |exp_output| are placeholders
    |eval_func| should be the function used to make the eval_tensor
    """
    with tf.name_scope(scope, default_name = 'per_class_eval'):
        # Figure out number of classes
        # |exp_output| should have 2 dims, the second should be the number of classes
        exp_output.get_shape().assert_has_rank(2)
        num_classes = exp_output.get_shape().as_list()[1]
        
        print("Found {} Possible Classes".format(num_classes))

        filtered = []
        with tf.name_scope('class_filtering'):
            for class_idx in range(num_classes):    
                
                with tf.name_scope('class_{}'.format(class_idx)):
                    filter_mask = tf.equal(tf.argmax(exp_output, 1), tf.cast(class_idx, tf.int64))
                    class_net_output = tf.boolean_mask(net_output, filter_mask)
                    class_exp_output = tf.boolean_mask(exp_output, filter_mask)

                    filtered.append((class_net_output, class_exp_output))
            
        return tf.stack([eval_func(pred, exp) for pred, exp in filtered], name = 'per_class_eval')
     


def filter_by_classes(pred, exp, num_classes):
    filtered = []
    for i in range(num_classes):
        filtered.append(tuple(filter_by_class(pred, exp, i)))
    return filtered


def filter_by_class(pred, exp, class_idx):
    
    pred = np.asarray(pred) 
    exp = np.asarray(exp)
    filter_mask = np.equal(np.argmax(exp, 1), class_idx)
   
    filter_pred = pred[filter_mask] 
    filter_exp = exp[filter_mask]

    return filter_pred, filter_exp


