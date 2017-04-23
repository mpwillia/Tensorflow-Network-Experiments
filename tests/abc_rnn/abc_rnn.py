
import numpy as np
import string
import random

from dataset_util import create_abc_dataset, encode_dataset, shuffle_dataset, split_dataset, print_dataset_shape

import tensorflow as tf
from neural_network import Network 
from neural_network import layers
from neural_network.loss import softmax_cross_entropy_with_logits
from neural_network.evaluation import accuracy
from neural_network.network_util import print_fit_results 


def run_test():
    
    name = 'abc_rnn_test'
    sequence_length = 3

    dataset, encoding = create_abc_dataset(None, seq_len = sequence_length, pos_inflation = 50, neg_to_pos = 10.0)
    dataset = encode_dataset(dataset, encoding)
    dataset = shuffle_dataset(dataset)
    train, test = split_dataset(dataset)

    print_dataset_shape(dataset)
    print_dataset_shape(train, "Training")
    print_dataset_shape(test, "Testing")
    
    net = create_and_train_network(train, test, sequence_length, encoding, name)
    
    user_input_to_net(net, sequence_length, encoding)


def user_input_to_net(net, seq_len, encoding):
    quit_keywords = {'q', 'quit', 'exit', 'bye', 'goodbye'}

    running = True
    print("\n\n")

    while running:
        try:
            user_input = raw_input("Enter {:d} Characters: ".format(seq_len))
            if user_input.lower() in quit_keywords:
                running = False
            elif len(user_input) < seq_len:
                print('Invalid number of characters!\n')
            elif user_input not in encoding:
                print('Input contains invalid characters!\n') 
            else:
                user_input = user_input[-seq_len:]
                net_pred = net.predict(np.asarray([encoding.encode(user_input)]))[0]
                pred = encoding.decode(net_pred)
                ordinal = ord(pred)
                 
                if pred == '\0':
                    pred = '"Not a Sequence"'
               
                print("Network Predicts: {} [Ord: {}  Encoded: {}]\n".format(pred, ordinal, net_pred))
        
        except KeyboardInterrupt, EOFError:
            running = False

    print("\nQuitting...")


def create_and_train_network(train, test, seq_len, encoding, name):
    net = Network([seq_len],
                  [layers.one_hot(depth=len(encoding)),
                   layers.lstm(num_units=32),
                   layers.rnn_most_recent(),
                   layers.fully_connected(num_outputs=len(encoding), activation_fn = None)],
                  pred_act_fn = tf.nn.softmax, 
                  logdir = None,
                  network_name = name)

    # Setup our training parameters
    opt = tf.train.AdamOptimizer(0.01)
    
    loss_func = softmax_cross_entropy_with_logits
    eval_func = accuracy

    epochs = 10
    mb_size = 128
    eval_freq = None
    eval_fmt = '8.3%'
    verbose = True

    fit_results = net.fit(train, opt, loss_func, epochs, mb_size,
                          evaluation_freq = eval_freq, 
                          evaluation_func = eval_func,
                          evaluation_fmt = eval_fmt,
                          shuffle_freq = 1,
                          test_data = test,
                          l2_reg_strength = 0.001,
                          verbose = verbose)
 
    print_fit_results(fit_results, eval_fmt, 'Final Results')
    
    return net














