
import tensorflow as tf
import tensorflow.contrib as tfc

import numpy as np

import sys
import math
import random

class Network(object):
    def __init__(self, input_shape, layers):
        """
        For |layers| see:
            https://www.tensorflow.org/api_docs/python/contrib.layers/higher_level_ops_for_building_neural_network_layers_

        """
        self.input_shape = input_shape
        self.layers = layers
        
        if type(input_shape) is int:
            self.net_input_shape = [None, input_shape] 
        else:
            self.net_input_shape = (None,) + tuple(input_shape)
        
        self.net_input = tf.placeholder(tf.float32, shape = self.net_input_shape,
                                        name = "network_input_tensor")

        print("Constructing {} Layer Network".format(len(layers)))
        print("  {:35s} : {}".format("Input Shape", self.net_input.get_shape()))

        prev_layer_output = self.net_input

        for layer_num, layer in enumerate(layers):
            layer_type = layer.func.__name__
            layer_name = "layer_{:d}_{}".format(layer_num, layer_type)
            prev_layer_output = layer(inputs = prev_layer_output, scope = layer_name)

            layer_msg = "Layer {:d} ({}) Shape".format(layer_num, layer_type)
            print("  {:35s} : {}".format(layer_msg, prev_layer_output.get_shape()))
        print("")

        self.net_output = prev_layer_output
        self.exp_output = tf.placeholder(tf.float32, self.net_output.get_shape(),
                                         name = "loss_expected_output")
        
        self.eval_net_output = tf.placeholder(tf.float32, self.net_output.get_shape(),
                                              name = "eval_net_output")

        self.sess = None

        self.train_step = None
        self.global_step = None

    def fit(self, train_data, optimizer, loss,
            epochs, mb_size = None,
            evaluation_freq = None, evaluation_func = None, evaluation_fmt = None,
            validation_data = None, 
            test_data = None,
            gpu_mem_fraction = 0.6,
            shuffle_freq = None):

        """
        For |optimizer| see:
            https://www.tensorflow.org/api_docs/python/train/optimizers

        For |loss| see:
            https://www.tensorflow.org/api_docs/python/contrib.losses/other_functions_and_classes
            https://www.tensorflow.org/api_docs/python/nn/classification

        """
        
        # reshape given data
        train_data = self._reshape_dataset(train_data)
        validation_data = self._reshape_dataset(validation_data)
        test_data = self._reshape_dataset(test_data)
        
        agg_method = tf.AggregationMethod.DEFAULT
        #agg_method = tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
        #agg_method = tf.AggregationMethod.EXPERIMENTAL_TREE

        # setting up our loss tensor
        loss_tensor = loss(self.net_output, self.exp_output)
        self.global_step = tf.Variable(0, trainable = False, name = "net_global_step")
        self.train_step = optimizer.minimize(loss_tensor, global_step = self.global_step,
                                             aggregation_method=agg_method)
        
        if evaluation_func is not None:
            eval_tensor = evaluation_func(self.eval_net_output, self.exp_output)
        else:
            eval_tensor = loss(self.eval_net_output, self.exp_output)
        
        if evaluation_fmt is None: evaluation_fmt = ".5f"

        # initilize our session and our graph variables
        if self.sess is None:
            gpu_options = None
            if gpu_mem_fraction is not None:
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = gpu_mem_fraction)
            sess_config = tf.ConfigProto(gpu_options = gpu_options, 
                                         log_device_placement = False,
                                         allow_soft_placement = False)

            self.sess = tf.Session(config = sess_config)
            self.sess.run(tf.global_variables_initializer())
        
        for epoch in range(epochs):
            
            epoch_msg = "Training Epoch {:4d} / {:4d}".format(epoch, epochs)
            self._run_training_epoch(train_data, mb_size, verbose = True, verbose_prefix = epoch_msg)
            
            # check for mid-train evaluations
            if evaluation_freq is not None and epoch % evaluation_freq == 0: 
                print("\nMid-Train Evaluation")
                train_eval = self._evaluate(train_data, eval_tensor)
                print("  Training   : {:{}}".format(train_eval, evaluation_fmt))

                if validation_data is not None:
                    validation_eval = self._evaluate(validation_data, eval_tensor)
                    print("  Validation : {:{}}".format(validation_eval, evaluation_fmt))
                
                print("")
            
            if shuffle_freq is not None and epoch % shuffle_freq == 0:
                #print("\nShuffling Training Data")
                train_data = self._shuffle_dataset(train_data)

        
        # Perform final evaluations
        print("\nFinal Evaluation")
        train_eval = self._evaluate(train_data, eval_tensor)
        print("  Training   : {:{}}".format(train_eval, evaluation_fmt))
         
        if validation_data is not None:
            validation_eval = self._evaluate(validation_data, eval_tensor)
            print("  Validation : {:{}}".format(validation_eval, evaluation_fmt))

        if test_data is not None:
            test_eval = self._evaluate(test_data, eval_tensor)
            print("  Testing    : {:{}}".format(test_eval, evaluation_fmt))


    def _reshape_dataset(self, dataset):
        if dataset is None: return None
        x,y = dataset
        return match_tensor_shape(x, self.net_input), match_tensor_shape(y, self.net_output)

    def _shuffle_dataset(self, dataset):
        zipped_dataset = zip(*dataset) 
        random.shuffle(zipped_dataset)
        return list(zip(*zipped_dataset))

    def _run_training_epoch(self, train_data, mb_size = None, feed_dict_kwargs = dict(), 
                            verbose = False, verbose_prefix = None):
        train_x, train_y = train_data
        
        with self.sess.as_default():
            for mb_x, mb_y, mb_num, mb_total in batch_dataset(train_data, mb_size, True):
                
                if verbose:
                    prefix = ''
                    if verbose_prefix is not None:
                        prefix = verbose_prefix + "    "
                    
                    mb_msg = "Mini-Batch {:5d} / {:5d}".format(mb_num, mb_total)
                    sys.stdout.write("{}{}     \r".format(prefix, mb_msg))
                    sys.stdout.flush()

                feed_dict_kwargs[self.net_input] = mb_x
                feed_dict_kwargs[self.exp_output] = mb_y
                self.train_step.run(feed_dict=feed_dict_kwargs)
            
    
    def _evaluate(self, dataset, eval_tensor, chunk_size = 2000):
        eval_x, eval_y = dataset

        with self.sess.as_default():
            results = [] 
            for chunk_x, chunk_y, in batch_dataset(dataset, chunk_size):
                results.extend(self.net_output.eval(feed_dict={self.net_input : chunk_x}))
                
            return eval_tensor.eval(feed_dict={self.eval_net_output : results,
                                               self.exp_output : eval_y})


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



def batch_dataset(dataset, batch_size, include_progress = False):
    if batch_size is None: 
        yield dataset
    else:
        x, y = dataset 
        batch_total = int(math.ceil(len(x) / float(batch_size)))
        for batch_num in range(batch_total):
            batch_idx = batch_num * batch_size
            batch_x = x[batch_idx : batch_idx + batch_size]
            batch_y = y[batch_idx : batch_idx + batch_size]

            if include_progress:
                yield batch_x, batch_y, batch_num, batch_total
            else:
                yield batch_x, batch_y




