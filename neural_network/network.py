
import tensorflow as tf
import tensorflow.contrib as tfc
import tensorflow.contrib.layers as tfcl

import numpy as np

import sys
import math
import random
import os

from network_util import match_tensor_shape, batch_dataset, get_num_batches

from summary import NetworkSummary

class Network(object):
    def __init__(self, input_shape, layers, logdir = None, network_name = 'network'):
        """
        For |layers| see:
            https://www.tensorflow.org/api_docs/python/contrib.layers/higher_level_ops_for_building_neural_network_layers_

        """
        self.input_shape = input_shape
        self.layers = layers
        
        self.network_name = network_name

        self.logdir = logdir
        self.network_summary = NetworkSummary(logdir, max_queue = 3, flush_secs = 60)

        if type(input_shape) is int:
            self.net_input_shape = [None, input_shape] 
        else:
            self.net_input_shape = (None,) + tuple(input_shape)
        
        with tf.name_scope('net_input'):
            self.net_input = tf.placeholder(tf.float32, shape = self.net_input_shape,
                                            name = "network_input_tensor")

        print("\nConstructing {} Layer Network".format(len(layers)))
        print("  {:35s} : {}".format("Input Shape", self.net_input.get_shape()))

        prev_layer_output = self.net_input
        
        made_kernel_images = False

        for layer_num, layer in enumerate(layers):
            layer_type = layer.func.__name__
            layer_name = "layer_{:d}_{}".format(layer_num, layer_type)
            
            with tf.name_scope(layer_name) as layer_scope:
                prev_layer_output = layer(inputs = prev_layer_output, scope = layer_name)
                self.network_summary.add_layer_summary(layer_name, prev_layer_output, layer_scope)

            layer_msg = "Layer {:d} ({}) Shape".format(layer_num, layer_type)
            print("  {:35s} : {}".format(layer_msg, prev_layer_output.get_shape()))
        print("")
        
        with tf.name_scope('net_output') as output_scope:
            self.net_output = prev_layer_output
            self.network_summary.add_output_summary(self.net_output, scope = output_scope)

        self.exp_output = tf.placeholder(tf.float32, self.net_output.get_shape(),
                                         name = "loss_expected_output")
        
        self.eval_net_output = tf.placeholder(tf.float32, self.net_output.get_shape(),
                                              name = "eval_net_output")

        self.sess = None
        self.saver = None
        self.train_step = None
        self.global_step = None

    def fit(self, train_data, optimizer, loss,
            epochs, mb_size = None,
            evaluation_freq = None, evaluation_func = None, evaluation_fmt = None,
            validation_data = None, 
            test_data = None,
            gpu_mem_fraction = None,
            shuffle_freq = None,
            l1_reg_strength = 0.0,
            l2_reg_strength = 0.0,
            summaries_per_epoch = None,
            checkpoint_freq = None):

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

        if summaries_per_epoch <= 0:
            summaries_per_epoch = None

        agg_method = tf.AggregationMethod.DEFAULT
        #agg_method = tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
        #agg_method = tf.AggregationMethod.EXPERIMENTAL_TREE

        # setting up our loss tensor
         
        self.network_summary.add_input_summary(self.net_input, mb_size)

        with tf.name_scope("loss") as loss_scope:
            grad_loss = loss(self.net_output, self.exp_output)

            # setup regularization
            if l1_reg_strength > 0.0 or l2_reg_strength > 0.0:
                l1_reg = None
                if l1_reg_strength > 0.0:
                    l1_reg = tfcl.l1_regularizer(l1_reg_strength)
                
                l2_reg = None
                if l2_reg_strength > 0.0:
                    l2_reg = tfcl.l2_regularizer(l2_reg_strength)

                l1_l2_reg = tfcl.sum_regularizer((l1_reg, l2_reg))
                reg_penalty = tfcl.apply_regularization(l1_l2_reg, self._get_weight_variables())

                loss_tensor = grad_loss + reg_penalty
            else:
                loss_tensor = grad_loss 
            
            self.network_summary.add_loss_summary(loss_tensor, grad_loss, reg_penalty, loss_scope)

        self.network_summary.add_variable_summary()

        # setup train steps
        self.global_step = tf.Variable(0, trainable = False, name = "net_global_step")
    
        try:
            opt_name = optimizer.__class__.__name__
        except:
            opt_name = 'optimizer'
        
        with tf.name_scope(opt_name):
            self.train_step = optimizer.minimize(loss_tensor, global_step = self.global_step,
                                                 aggregation_method=agg_method)
        
        with tf.name_scope('evaluation') as eval_scope:
            if evaluation_func is not None:
                eval_tensor = evaluation_func(self.eval_net_output, self.exp_output)
            else:
                eval_tensor = loss(self.eval_net_output, self.exp_output)
            
            self.network_summary.add_eval_summary(eval_tensor, 'train', eval_scope)
            self.network_summary.add_eval_summary(eval_tensor, 'validation', eval_scope)
            self.network_summary.add_eval_summary(eval_tensor, 'test', eval_scope)

        if evaluation_fmt is None: evaluation_fmt = ".5f"

        # initilize our session and our graph variables
        if self.sess is None:
            gpu_options = None
            if gpu_mem_fraction is not None:
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = gpu_mem_fraction)
            sess_config = tf.ConfigProto(gpu_options = gpu_options, 
                                         log_device_placement = False,
                                         allow_soft_placement = False)
            
            self.saver = tf.train.Saver()
            self.sess = tf.Session(config = sess_config)
            self.sess.run(tf.global_variables_initializer())
        
        self.network_summary.add_graph(self.sess.graph)
        
        for epoch in range(epochs):
            
            epoch_msg = "Training Epoch {:4d} / {:4d}".format(epoch, epochs)
            self._run_training_epoch(train_data, mb_size, 
                                     summaries_per_epoch = summaries_per_epoch,
                                     verbose = True, verbose_prefix = epoch_msg)
            
            # check for mid-train evaluations
            if evaluation_freq is not None and epoch % evaluation_freq == 0: 
                print("\nMid-Train Evaluation")
                train_eval = self._evaluate(train_data, eval_tensor, name = 'train')
                print("  Training   : {:{}}".format(train_eval, evaluation_fmt))

                if validation_data is not None:
                    validation_eval = self._evaluate(validation_data, eval_tensor, name = 'validation')
                    print("  Validation : {:{}}".format(validation_eval, evaluation_fmt))
                
                print("")
            
            if checkpoint_freq is not None and epoch % checkpoint_freq == 0:
                print("Saving Mid-Train Checkpoint")
                self._save_checkpoint()

            if shuffle_freq is not None and epoch % shuffle_freq == 0:
                train_data = self._shuffle_dataset(train_data)
            
            self.network_summary.flush()
        
        print("Saving Final Checkpoint")
        self._save_checkpoint()

        # Perform final evaluations
        print("\nFinal Evaluation")
        train_eval = self._evaluate(train_data, eval_tensor, name = 'train')
        print("  Training   : {:{}}".format(train_eval, evaluation_fmt))
         
        if validation_data is not None:
            validation_eval = self._evaluate(validation_data, eval_tensor, name = 'validation')
            print("  Validation : {:{}}".format(validation_eval, evaluation_fmt))

        if test_data is not None:
            test_eval = self._evaluate(test_data, eval_tensor, name = 'test')
            print("  Testing    : {:{}}".format(test_eval, evaluation_fmt))
        
        self.network_summary.flush()

    def _get_weight_variables(self):    
        vars = tf.trainable_variables()
        return [v for v in vars if 'weight' in v.name]

    def _reshape_dataset(self, dataset):
        if dataset is None: return None
        x,y = dataset
        return match_tensor_shape(x, self.net_input), \
               match_tensor_shape(y, self.net_output)

    def _shuffle_dataset(self, dataset):
        zipped_dataset = zip(*dataset) 
        random.shuffle(zipped_dataset)
        return list(zip(*zipped_dataset))

    def _run_training_epoch(self, train_data, mb_size = None, feed_dict_kwargs = dict(),
                            summaries_per_epoch = None,
                            verbose = False, verbose_prefix = None):
        train_x, train_y = train_data
       
        mb_total = get_num_batches(len(train_x), mb_size)
        
        if summaries_per_epoch is None:
            summary_every = None
        elif summaries_per_epoch >= mb_total:
            summary_every = 1
        elif summaries_per_epoch == 1:
            summary_every = mb_total
        else:
            summary_every = int(math.ceil(mb_total / float(summaries_per_epoch)))

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
                
                fetches = [self.train_step]

                if (summary_every is not None) and (mb_num >= mb_total-1 or (mb_num+1) % summary_every == 0):
                #if (summary_every is not None) and (mb_num >= mb_total-1 or mb_num % summary_every == 0):
                    train_summary = self.network_summary.get_training_summary()
                    if train_summary is not None:
                        fetches.extend([train_summary, self.global_step]) 
                
                run_results = self.sess.run(fetches, feed_dict = feed_dict_kwargs)
                
                if len(run_results) > 1:
                    _, summary, step = run_results
                    self.network_summary.write(summary, step)


    def _evaluate(self, dataset, eval_tensor, chunk_size = 2000, name = 'eval'):
        eval_x, eval_y = dataset

        with self.sess.as_default():
            results = [] 
            for chunk_x, chunk_y, in batch_dataset(dataset, chunk_size):
                results.extend(self.net_output.eval(feed_dict={self.net_input : chunk_x}))

            feed_dict = {self.eval_net_output : results,
                         self.exp_output : eval_y}

            fetches = [eval_tensor]

            eval_summary = self.network_summary.get_evaluation_summary(name)
            if eval_summary is not None:
                fetches.extend([eval_summary, self.global_step])
            
            run_results = self.sess.run(fetches, feed_dict = feed_dict)

            if len(run_results) > 1:
                eval_results, summary, step = run_results
                self.network_summary.write(summary, step)
            else:
                eval_results = run_results
            
            return eval_results

    def _save_checkpoint(self):
        if self.sess is None:
            raise Exception("Cannot save checkpoint without an active session!")
        
        if self.saver is None:
            raise Exception("Cannot save checkpoint without a tf.train.Saver instance!")
        
        if self.logdir is not None:
            save_path = os.path.join(self.logdir, self.network_name) 
        else:
            save_path = os.path.join('./', self.network_name) 
        
        self.saver.save(self.sess, save_path, self.global_step)


