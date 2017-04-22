
import tensorflow as tf

import random
import numpy as np

from neural_network import Network 
from neural_network import layers
from neural_network.loss import softmax_cross_entropy_with_logits
from neural_network.evaluation import accuracy

from neural_network.network_util import print_fit_results 

def run_rnn_test():

    fixed_length = 16
    dataset = generate_dataset(num_samples = 2**fixed_length, fixed_length = fixed_length)

    #from pprint import pprint 
    #test = list(zip(*dataset))
    #print("Dataset Samples:")
    #pprint(test[:50])
    #print("\n\n")
    
    train, test = split_dataset(dataset)
    print_dataset_shape(train, 'Train')
    print_dataset_shape(test, 'Test')

    net = Network([fixed_length,1],
                  [layers.lstm(num_units=32),
                   layers.lstm(num_units=32),
                   layers.rnn_most_recent(),
                   layers.fully_connected(num_outputs=fixed_length+1, activation_fn = None)],
                  logdir = None,
                  network_name = 'basic_lstm_network')

    # Setup our training parameters
    opt = tf.train.AdamOptimizer()
    
    loss_func = softmax_cross_entropy_with_logits
    eval_func = accuracy

    epochs = 20
    mb_size = 128
    eval_freq = 1
    eval_fmt = '8.3%'
    verbose = True

    fit_results = net.fit(train, opt, loss_func, epochs, mb_size,
                          #evaluation_freq = eval_freq, 
                          evaluation_func = eval_func,
                          evaluation_fmt = eval_fmt,
                          shuffle_freq = 1,
                          test_data = test,
                          l2_reg_strength = 0.0001,
                          verbose = verbose)
 
    print_fit_results(fit_results, eval_fmt, 'Final Results')

    
    user_input_to_net(net, fixed_length)

def user_input_to_net(net, fixed_length):
    
    quit_keywords = {'q', 'quit', 'exit'}

    running = True
    print("\n\n")
    while running:
        user_input = raw_input("Enter a number: ").strip().lower()
        
        if user_input.lower() in quit_keywords:
            running = False
            print("Quitting...")
        else:
            try:
                val = int(user_input)
                
            except ValueError:
                print("Invalid Input\n")
                val = None
            
            if val is not None:
                bin_arr = int_to_bin_arr(val, fixed_length)
                num_ones = bin_arr.count(1)
                print("Entered: {0:0{1}b} [{2:d}]".format(val, fixed_length, val))
                
                bin_arr = np.asarray([wrap_elements(bin_arr)])
                net_pred = net.predict(bin_arr)[0]

                correct = net_pred == num_ones
                print("Number of Ones: {:d}  [Expected: {:d} | {}]".format(net_pred, num_ones, str(correct)))
                print("\n")


def print_dataset_shape(dataset, name = 'Dataset'):
    inputs, outputs = dataset
    print("{} Shapes: {}  -->  {}".format(name, inputs.shape, outputs.shape))


def split_dataset(dataset, train_p = 0.8):
    inputs, outputs = dataset
    num_train = int(len(outputs) * train_p)
    train = (inputs[:num_train], outputs[:num_train])
    test = (inputs[num_train:], outputs[num_train:])
    return train, test


def generate_dataset(num_samples = 10000, fixed_length = None, shuffle = True):
    
    values = list(range(num_samples)) 
    binary_arrs = [int_to_bin_arr(i, fixed_length) for i in values]

    extra_powers = [int_to_bin_arr(2**(i%8), fixed_length) for i in range(int(num_samples * 0.05))]
    extra_zeros = [int_to_bin_arr(0, fixed_length) for i in range(int(num_samples * 0.01))]
    print("Extra Zeros: {}".format(len(extra_zeros)))
    print("Extra Powers: {}".format(len(extra_powers)))

    binary_arrs.extend(extra_zeros)
    binary_arrs.extend(extra_powers)
    num_ones = [bin_arr.count(1) for bin_arr in binary_arrs]

    one_hot_unique_vals = fixed_length
    if one_hot_unique_vals is None:
        one_hot_unique_vals = max(num_ones)
    one_hot_unique_vals += 1

    one_hot_num_ones = [make_one_hot(n, one_hot_unique_vals) for n in num_ones]

    if shuffle:
        dataset = zip(binary_arrs, one_hot_num_ones)
        random.shuffle(dataset)
        x,y = tuple(zip(*dataset))
    else:
        x = binary_arrs
        y = one_hot_num_ones
    
    x = [wrap_elements(arr) for arr in x]
    return np.asarray(x), np.asarray(y)

def make_one_hot(val, num_unique_vals):
    one_hot = np.zeros(num_unique_vals)
    one_hot[val] = 1
    return one_hot

def wrap_elements(arr):
    return [[i] for i in arr]

def int_to_bin_arr(i, fixed_length = None):
    if fixed_length is None:
        bin_str = '{:b}'.format(i)
    else:
        bin_str = '{0:0{1}b}'.format(i, fixed_length)

    #return [map(int,s) for s in bin_str]
    return [int(s) for s in bin_str]


