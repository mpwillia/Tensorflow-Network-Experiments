
import tensorflow as tf
import tensorflow.contrib as tfc
import tensorflow.contrib.layers as tfcl

print("Using Tensorflow Version: {}".format(tf.__version__))

import numpy as np

import sys
import math
import random
import os
from functools import partial 

from network_util import match_tensor_shape, batch_dataset, get_num_batches, \
    make_per_class_eval_tensor, print_eval_results, print_fit_results

from summary import NetworkSummary

from collections import namedtuple
EvalResults = namedtuple('EvalResults', ['overall', 'per_class'])
FitResults = namedtuple('FitResults', ['train', 'validation', 'test'])



class Network(object):
    def __init__(self, input_shape, layers, pred_act_fn = None, 
                 logdir = None, network_name = 'network'):
        """
        For |layers| see:
            https://www.tensorflow.org/api_docs/python/contrib.layers/higher_level_ops_for_building_neural_network_layers_

        """
        self.input_shape = input_shape
        self.layers = layers
        self.pred_act_fn = pred_act_fn
        
        self.network_name = network_name
        self.input_shape = input_shape
        
        
        self.sess = None
        self.saver = None
        self.train_step = None

        # setup global step counter
        self.global_step = tf.Variable(0, trainable = False, name = "net_global_step")
        
        # setup the network's summaries
        self.logdir = logdir
        self.network_summary = NetworkSummary(logdir, max_queue = 3, flush_secs = 60)
        
        # setup the network's input shape and input placeholder
        if type(input_shape) is int:
            self.net_input_shape = [None, input_shape] 
        else:
            self.net_input_shape = (None,) + tuple(input_shape)
        
        with tf.name_scope('net_input'):
            self.net_input = tf.placeholder(tf.float32, shape = self.net_input_shape,
                                            name = "network_input_tensor")

        print("\nConstructing {} Layer Network".format(len(layers)))
        print("  {:35s} : {}".format("Input Shape", self.net_input.get_shape()))

        
        self.using_dropout = False
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name = "dropout_keep_prob")
    
        # layer states are only applicable for recurrent layers
        self.layer_states = []

        made_kernel_images = False
        prev_layer_output = self.net_input
        for layer_num, layer in enumerate(layers):
            layer_type = layer.func.__name__
            layer_name = "layer_{:d}_{}".format(layer_num, layer_type)
            
            layer_kwargs = {'inputs' : prev_layer_output,
                            'scope' : layer_name}
            
            # handle dropout layers
            if 'dropout' in layer_type:
                self.using_dropout = True
                layer_kwargs['keep_prob'] = self.keep_prob

            with tf.name_scope(layer_name) as layer_scope:
                layer_output = layer(**layer_kwargs)
                
                try:
                    # check if the layer is recurrent, if so extract the state 
                    if len(layer_output) == 2:
                        prev_layer_output, state = layer_output
                    else:
                        prev_layer_output = layer_output[0]
                        state = None
                except:
                    prev_layer_output = layer_output
                    state = None
                
                self.layer_states.append(state)
                self.network_summary.add_layer_summary(layer_name, prev_layer_output, layer_scope)

            layer_msg = "Layer {:d} ({}) Shape".format(layer_num, layer_type)
            print("  {:35s} : {}".format(layer_msg, prev_layer_output.get_shape()))
        print("")
        
        with tf.name_scope('net_output') as output_scope:
            self.net_output = prev_layer_output
            self.network_summary.add_output_summary(self.net_output, scope = output_scope)

            if self.pred_act_fn is not None:
                self.pred_net_output = self.pred_act_fn(prev_layer_output)
            else:
                self.pred_net_output = prev_layer_output
            self.network_summary.add_output_summary(self.pred_net_output, scope = output_scope)

        self.exp_output = tf.placeholder(tf.float32, self.net_output.get_shape(),
                                         name = "loss_expected_output")
        
        self.eval_net_output = tf.placeholder(tf.float32, self.net_output.get_shape(),
                                              name = "eval_net_output")

    # Various Network Getters -------------------------------------------------
    def get_global_step(self):
        """
        Returns the current global step if the network has an active session,
            otherwise returns None
        """
        if self.sess is not None:
            return self.sess.run(self.global_step)
  
    def _get_weight_variables(self):    
        vars = tf.trainable_variables()
        return [v for v in vars if 'weight' in v.name]

    
    # Session Handling --------------------------------------------------------
    def init_session(self):
        """
        Initializes the network's tensorflow session along with initializing
        all tensorflow variables. Will also create a new tensorflow Saver instance
        for the network if needed.
        
        If a session has already been created when this method is called, for 
        example through loading a saved network, then all uninitialized variables 
        will be initialized.
        
        If a session has not yet been created then:
            - A new Saver instance will be created
            - A new session will be created
            - All variables will be initialized
        
        If a session already exists (through loading a saved network) then:
            - All uninitialized variables will be initialized

        """
        if self.sess is None:
            sess_config = tf.ConfigProto(
                                         log_device_placement = False,
                                         allow_soft_placement = False)
            
            self.saver = tf.train.Saver()
            self.sess = tf.Session(config = sess_config)
            self.sess.run(tf.global_variables_initializer())
        else: 
            list_of_variables = tf.global_variables()
            uninitialized_variables = list(tf.get_variable(name) for name in
                                       self.sess.run(tf.report_uninitialized_variables(list_of_variables)))
            self.sess.run(tf.initialize_variables(uninitialized_variables))


    def close(self):
        """
        Closes and deletes this Network's session.
        """
        if self.sess is not None:
            self.sess.close()
            self.sess = None
    

    # Network Prediction ------------------------------------------------------
    def predict(self, input_data, chunk_size = 500):
        """
        Makes predictions on the given input data. Returns the index of the output
        with the highest value for each item in the input data.
        
        Arguments:
            |input_data| the input data to make predictions on. 

        Optional:
            |chunk_size| the maximum number of items to process with one run of
                the network. All of the input data will be processed but it will
                be broken into the smaller chunks for better memory usage. If
                the chunk size is None then all of the input data will be processed
                in one run of the network.
                     
        Returns the index of the output with the highest value for each item in
            the input.
        """
        with self.sess.as_default():
            feed_dict = dict()
            if self.using_dropout:
                feed_dict[self.keep_prob] = 1.0

            results = [] 
            for chunk_x in batch_dataset(input_data, chunk_size, has_outputs = False):
                feed_dict[self.net_input] = chunk_x
                results.extend(self.pred_net_output.eval(feed_dict=feed_dict))
            return np.argmax(results, 1)


    def sample(self, input_data, temperature = 1.0, filter_zero = True, chunk_size = 500):
        """
        Samples the networks response to the given input data. Returns a randomly
        selected output index based on the network's predicted probability of each
        possible output for each item in the input data. In otherwords the probability
        of selecting a given output index is given by the network's predicted
        probabilities.

        Arguments:
            |input_data| the input data to make predictions on. 
        
        Optional:
            |temperature| the temperature value changes the distribution of the 
                network's predicted probabilities for each output. Accepts any
                nonzero float value. Defaults to 1.0. 
                
                A higher temperature value makes the resulting probability 
                distribution more evenly spread while a lower temperature value 
                makes the resulting probability distribution less evenly spread.
                There are four distinct effects that can be achieved with the 
                temperature value.
                 
                Temperature Effects (from low to high):
                    temp=0.0 - Has the same effect as calling predict() where the
                                output with the highest value will always be selected.

                    temp<1.0 - Higher probabilities are increased further while 
                                lower probabilities are decreased further. Results 
                                in less randomness in the output.

                    temp=1.0 - Has no effect on the network's predicted probabilities.

                    temp>1.0 - Higher probabilities are made smaller while lower
                                probabilities are made larger. Results in more
                                randomness in the output.

            |filter_zero| if True then when applying temperature to the predicted
                probabilities, probabilities of zero will be filtered out.

            |chunk_size| the maximum number of items to process with one run of
                the network. All of the input data will be processed but it will
                be broken into the smaller chunks for better memory usage. If
                the chunk size is None then all of the input data will be processed
                in one run of the network.

        """
        with self.sess.as_default():

            feed_dict = dict()
            if self.using_dropout:
                feed_dict[self.keep_prob] = 1.0

            results = [] 
            for chunk_x in batch_dataset(input_data, chunk_size, has_outputs = False):
                feed_dict[self.net_input] = chunk_x
                results.extend(self.pred_net_output.eval(feed_dict=feed_dict))
            results = np.asarray(results)

            if temperature <= 0.0:
                return np.argmax(results, 1)
            
            num_choices = results.shape[1] # (batch, outputs)

            if filter_zero:
                choices = np.arange(num_choices)
                def apply_temperature(results_1d):
                    non_zero = np.nonzero(results_1d) 
                    nz_results = results_1d[non_zero]
                    nz_choices = choices[non_zero]

                    probs = np.exp(np.log(nz_results) / temperature)
                    probs /= np.sum(probs)
                    
                    return np.random.choice(nz_choices, p = probs)
                
                return np.apply_along_axis(apply_temperature, 1, results)
            else:

                probs = np.exp(np.log(results) / temperature)
                probs /= np.sum(probs, 1)
                f = lambda p: np.random.choice(num_choices, p=p)
                return np.apply_along_axis(f, 1, probs)




    # Network Training --------------------------------------------------------
    def fit(self, train_data, optimizer, loss,
            epochs, mb_size = None,
            evaluation_freq = None, evaluation_func = None, evaluation_fmt = None,
            evaluation_target = None, max_step = None,
            per_class_evaluation = False,
            validation_data = None, 
            test_data = None,
            shuffle_freq = None,
            l1_reg_strength = 0.0,
            l2_reg_strength = 0.0,
            dropout_keep_prob = 1.0,
            summaries_per_epoch = None,
            save_checkpoints = False, checkpoint_freq = None,
            verbose = False):
        """
        For |optimizer| see:
            https://www.tensorflow.org/api_docs/python/train/optimizers

        For |loss| see:
            https://www.tensorflow.org/api_docs/python/contrib.losses/other_functions_and_classes
            https://www.tensorflow.org/api_docs/python/nn/classification

        """

        # reshape given data
        #train_data = self._reshape_dataset(train_data)
        #validation_data = self._reshape_dataset(validation_data)
        #test_data = self._reshape_dataset(test_data)


        train_feed_dict = dict()

        # handle dropout
        if self.using_dropout:
            train_feed_dict[self.keep_prob] = dropout_keep_prob

        if summaries_per_epoch <= 0:
            summaries_per_epoch = None

        self.network_summary.add_input_summary(self.net_input, mb_size)

        # setting up our loss tensor
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
                reg_penalty = None
                loss_tensor = grad_loss 
            
            self.network_summary.add_loss_summary(loss_tensor, grad_loss, reg_penalty, loss_scope)
        
        # adds a summary for all trainable variables
        self.network_summary.add_variable_summary()

        # setting up our optimizer 
        try:
            opt_name = optimizer.__class__.__name__
        except:
            opt_name = 'optimizer'
        
        with tf.name_scope(opt_name):
            self.train_step = optimizer.minimize(loss_tensor, global_step = self.global_step)
        

        # setting up our evaluation function and summaries
        if evaluation_func is None:
            evaluation_func = loss
        with tf.name_scope('evaluation') as eval_scope:
            
            # overall eval tensor
            eval_tensor = evaluation_func(self.eval_net_output, self.exp_output)
            
            self.network_summary.add_eval_summary(eval_tensor, 'train', eval_scope)
            self.network_summary.add_eval_summary(eval_tensor, 'validation', eval_scope)
            self.network_summary.add_eval_summary(eval_tensor, 'test', eval_scope)

        
        # setting up our per class evaluation function and summaries
        with tf.name_scope('per_class_evaluation') as per_class_eval_scope:
            # per class eval tensor
            per_class_evaluation = False,
            per_class_eval_tensor = None
            if per_class_evaluation:
                per_class_eval_tensor = make_per_class_eval_tensor(evaluation_func,
                                                                   self.eval_net_output,
                                                                   self.exp_output, 
                                                                   scope = per_class_eval_scope) 
                def add_per_class_summary(name):
                    self.network_summary.add_per_class_eval_summary(per_class_eval_tensor,
                                                                    max_val = 1.0,
                                                                    name = name,
                                                                    scope = per_class_eval_scope)
                add_per_class_summary('train')
                add_per_class_summary('validation')
                add_per_class_summary('test')
       

        # setting up the formating for printing the evaluation results
        if evaluation_fmt is None: evaluation_fmt = ".5f"
        
        # initialize our session
        self.init_session()
        
        # add a graph summary
        self.network_summary.add_graph(self.sess.graph)
        

        epoch_eval_results = []
        initial_step = self.get_global_step()

        for epoch in range(epochs):
            
            # execute our training epoch
            epoch_msg = "Training Epoch {:4d} / {:4d}".format(epoch, epochs)
            self._run_training_epoch(train_data, mb_size, 
                                     feed_dict_kwargs = train_feed_dict,
                                     summaries_per_epoch = summaries_per_epoch,
                                     verbose = True, 
                                     verbose_prefix = epoch_msg)
            
            # check for mid-train evaluations
            if evaluation_freq is not None and epoch % evaluation_freq == 0: 
                
                # evaluate on the training dataset
                if verbose > 1: print("\nMid-Train Evaluation")
                train_eval = self._evaluate(train_data, eval_tensor, per_class_eval_tensor,  name = 'train')

                
                # evaluate on the validation dataset
                if validation_data is not None:
                    validation_eval = self._evaluate(validation_data, eval_tensor, per_class_eval_tensor, name = 'validation')
                else:
                    validation_eval = None

                
                # check if we've met our early stopping evaluation target
                if evaluation_target:
                    if validation_eval is not None:
                        met_target = validation_eval.overall >= evaluation_target
                    else:
                        met_target = train_eval.overall >= evaluation_target
                else:
                    met_target = None
                
                # add the mid train evaluation results to our list
                epoch_fit_results = FitResults(train = train_eval, validation = validation_eval, test = None)
                epoch_eval_results.append(epoch_fit_results)
                
                # print the fit results
                if verbose > 1: print_fit_results(epoch_fit_results, evaluation_fmt)
                
                # break early if we've met our evaluation target
                if met_target is not None and met_target:
                    print("\n\nReached Evaluation Target of {}".format(evaluation_target))
                    break
            
            # break early if we've met our step target
            if max_step is not None and self.get_global_step() >= max_step:
                print("\n\nReached Max Step Target of {}".format(max_step))
                break

            if verbose > 1: print("")
            
            # save a checkpoint if needed
            if save_checkpoints and checkpoint_freq is not None and epoch % checkpoint_freq == 0:
                if verbose > 1: print("Saving Mid-Train Checkpoint")
                self._save_checkpoint()

            # shuffle the dataset if needed
            if shuffle_freq is not None and epoch % shuffle_freq == 0:
                train_data = self._shuffle_dataset(train_data)
            
            self.network_summary.flush()
        
        # report the number of training steps taken
        final_step = self.get_global_step()
        total_steps = final_step - initial_step 
        if verbose > 0:
            print("\nTrained for {:d} Steps".format(total_steps))
        
        # save the final checkpoint
        if save_checkpoints:
            if verbose > 1: print("Saving Final Checkpoint")
            self._save_checkpoint()
      
        if verbose == 1: print("")
        
        # Perform final fit result evaluations
        # final training evaluation
        if verbose > 1: print("Final Evaluation")
        train_eval = self._evaluate(train_data, eval_tensor, per_class_eval_tensor, name = 'train')
         
        # final validation evaluation
        if validation_data is not None:
            validation_eval = self._evaluate(validation_data, eval_tensor, per_class_eval_tensor, name = 'validation')
        else:
            validation_eval = None
        
        # final test evaluation
        if test_data is not None:
            test_eval = self._evaluate(test_data, eval_tensor, per_class_eval_tensor, name = 'test')
        else:
            test_eval = None
        
        # print and return the final fit results
        fit_results = FitResults(train = train_eval, validation = validation_eval, test = test_eval)
        if verbose > 1: print_fit_results(fit_results, evaluation_fmt)

        self.network_summary.flush()
        return fit_results



    # Single Network Training Epoch -------------------------------------------
    def _run_training_epoch(self, train_data, 
                            mb_size = None, 
                            feed_dict_kwargs = dict(),
                            summaries_per_epoch = None,
                            verbose = False, 
                            verbose_prefix = None):
        """
        Runs a single training epoch.

        Arguments:
            |train_data| the data to train on

        Optional:
            |mb_size| the size of the minibatches, if None then no minibatching
                will be done.
            |feed_dict_kwargs| any extra kwargs to be passed to the network.
            |summaries_per_epoch| how many summaries to produce for this epoch.
                The summaries will be evenly distributed across the minibatches.
                If None then no summaries will be made.
            |verbose| if True the progress information will be printed
            |verbose_prefix| extra information to append to the progress messages
                printed when the |verbose| argument is True.
        """
        train_x, train_y = train_data
       
        mb_total = get_num_batches(len(train_x), mb_size)
        
        # Compute when to generate summaries
        if summaries_per_epoch is None:
            summary_every = None
        elif summaries_per_epoch >= mb_total:
            summary_every = 1
        elif summaries_per_epoch == 1:
            summary_every = mb_total
        else:
            summary_every = int(math.ceil(mb_total / float(summaries_per_epoch)))

        with self.sess.as_default():
            # Iterate over the batches
            for mb_x, mb_y, mb_num, mb_total in self._batch_for_train(train_data, mb_size, True):

                if verbose:
                    # print progress message if verbose
                    prefix = ''
                    if verbose_prefix is not None:
                        prefix = verbose_prefix + "    "
                    
                    mb_msg = "Mini-Batch {:5d} / {:5d}".format(mb_num, mb_total)
                    sys.stdout.write("{}{}     \r".format(prefix, mb_msg))
                    sys.stdout.flush()

                feed_dict_kwargs[self.net_input] = mb_x
                feed_dict_kwargs[self.exp_output] = mb_y
                
                fetches = [self.train_step]
                
                # if this is a summary epoch then add those
                if (summary_every is not None) and (mb_num >= mb_total-1 or (mb_num+1) % summary_every == 0):
                    train_summary = self.network_summary.get_training_summary()
                    if train_summary is not None:
                        fetches.extend([train_summary, self.global_step]) 
                
                run_results = self.sess.run(fetches, feed_dict = feed_dict_kwargs)
                self._process_run_results(run_results)

    def _batch_for_train(self, dataset, batch_size, include_progress = False):
        """used to define batching specific to the training epochs"""
        return batch_dataset(dataset, batch_size, include_progress, True)
    

    # Network Performance Evaluation ------------------------------------------
    def _evaluate(self, dataset, eval_tensor, 
                  per_class_eval_tensor = None, 
                  chunk_size = 2000, 
                  name = 'eval'):
        """
        Evaluates the network's performance on the given dataset.

        Arguments:
            |dataset| the dataset to evaluate the network's performance on
            |eval_tensor| the tensor to use for evaluation. This tensor should
                accept the results of the network's predictions on the dataset
                and the expected outputs to produce a metric for how good the
                network's predictions are. For example, it could compute the 
                network's accuracy.

        Optional:
            |per_class_eval_tensor| if the network is performing classification
                then this tensor can be used to evaluate the network's performance
                for each class individually. This tensor accepts the same inputs
                as the |eval_tensor| but is expected to produce a vector of metrics
                where each element is the metric for each class.

            |chunk_size| the maximum number of items to process with one run of
                the network. All of the input data will be processed but it will
                be broken into the smaller chunks for better memory usage. If
                the chunk size is None then all of the input data will be processed
                in one run of the network.

            |name| gives a name for the evaluation being performed. Used for
                grouping like summaries together. For example, can be used to
                group evaluation of validation data seperately from the evaluation
                of the testing data.

        Returns the evaluation results as an EvalResults tuple.
        """

        eval_x, eval_y = dataset

        with self.sess.as_default():

            feed_dict = dict()
            if self.using_dropout:
                feed_dict[self.keep_prob] = 1.0

            results = [] 
            for chunk_x, chunk_y, in self._batch_for_eval(dataset, chunk_size):
                feed_dict[self.net_input] = chunk_x
                results.extend(self.net_output.eval(feed_dict=feed_dict))

            feed_dict = {self.eval_net_output : results,
                         self.exp_output : eval_y}

            fetches = [eval_tensor]

            if per_class_eval_tensor is not None:
                fetches.append(per_class_eval_tensor)

            non_summary_size = len(fetches)

            eval_summary = self.network_summary.get_evaluation_summary(name)
            if eval_summary is not None:
                fetches.extend([eval_summary, self.global_step])
            
            eval_results = self.sess.run(fetches, feed_dict = feed_dict)
            return self._process_eval_results(eval_results, non_summary_size)
 

    def _batch_for_eval(self, dataset, batch_size, include_progress = False):
        """used to define batching specific to the evaluation epochs"""
        return batch_dataset(dataset, batch_size, include_progress, True)

    
    
    # Result Handling (Training and Evaluation) -------------------------------
    def _process_eval_results(self, eval_results, non_summary_size = 1):
        """
        Takes the raw, unprocessed evaluation results and extracts out the
        relevant information as an EvalResults tuple.

        Arguments:
            |eval_results| the raw, unprocessed evaluation results

        Optional:
            |non_summary_size| the expected number of items not used for summaries

        Returns the processed EvalResults tuple
        """
        eval_results = self._process_run_results(eval_results, non_summary_size)
        
        if len(eval_results) == 1:
            return EvalResults(overall = eval_results[0])
        elif len(eval_results) == 2:
            return EvalResults(overall = eval_results[0], per_class = eval_results[1])
        else:
            raise ValueError("Don't know how to process eval_results with length {:d}!".format(len(eval_results)))
         

    def _process_run_results(self, run_results, non_summary_size = 1):
        """
        Takes the raw, unprocessed run results and extracts out the nonsummary
        information as a tuple.

        Arguments:
            |run_results| the raw, unprocessed run results

        Optional:
            |non_summary_size| the expected number of items not used for summaries

        Returns a tuple of the non-summary items from the run
        """
        if len(run_results) == non_summary_size + 2:
            summary, step = run_results[-2:]
            self.network_summary.write(summary, step)
        elif len(run_results) != non_summary_size:
            raise ValueError("Don't know how to process run_results with length {:d}!".format(len(run_results)))
        
        return tuple(run_results[:non_summary_size])




    # Dataset Utilities -------------------------------------------------------
    def _reshape_dataset(self, dataset):
        if dataset is None: return None
        x,y = dataset
        return match_tensor_shape(x, self.net_input), \
               match_tensor_shape(y, self.net_output)

    def _shuffle_dataset(self, dataset):
        zipped_dataset = zip(*dataset) 
        random.shuffle(zipped_dataset)
        return list(zip(*zipped_dataset))


    # Network Pickling and Variable I/O ---------------------------------------
    def __getstate__(self):
        odict = self.__dict__.copy()

        # Strip Tensorflow Content
        del odict['sess']
        del odict['saver']
        del odict['global_step']
        del odict['train_step']
        del odict['network_summary']
        del odict['exp_output']
        del odict['eval_net_output']
        del odict['net_input']
        del odict['net_output']
        del odict['pred_net_output']
        del odict['net_input_shape']
        del odict['layer_states']
        del odict['using_dropout']
        del odict['keep_prob']
        return odict

    def __setstate__(self, state):
        self.__init__(**state)


    def save_variables(self, path):
        """
        Saves the networks tensorflow variable states to the given filepath

        Arguments:
            |path| the filepath to save the tensorflow variable states to

        """
        if self.saver is None or self.sess is None:
            raise Exception("Cannot save variables without a session and saver")
        self.saver.save(self.sess, path)


    def load_variables(self, path):
        """
        Loads the networks tensorflow variable states from the given filepath

        Arguments:
            |path| the filepath to load the tensorflow variable states from

        """
        self.init_session()  
        self.saver.restore(self.sess, path)



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


