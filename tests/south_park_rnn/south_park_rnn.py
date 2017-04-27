
import random
import dataset_util
import char_encoding
from char_encoding import Encoding
import numpy as np

import os
import sys

import tensorflow as tf
from neural_network import Network 
from neural_network import layers
from neural_network.loss import softmax_cross_entropy_with_logits
from neural_network.evaluation import accuracy
from neural_network.network_util import print_fit_results 

from neural_network.network_fileio import save, load

NETWORK_SAVE_DIR = "/home/mike/Development/Tensorflow-Network-Experiments/tests/south_park_rnn/saved_network"

def run_test():
    
    if not os.path.exists(NETWORK_SAVE_DIR):
        os.mkdir(NETWORK_SAVE_DIR)

    name = 'south_park_lstm_rnn'
    sample_length = 100
    dataset_size = None 

    dataset, encoding = load_and_prepare_dataset(dataset_size, sample_length, verbose = True)

    print(encoding)
    dataset_util.print_dataset_shape(dataset)

    train, test = dataset_util.split_dataset(dataset)
    
    dataset_util.print_dataset_shape(train, 'Training')
    dataset_util.print_dataset_shape(test, 'Testing')

    try:
        net = load(name, path = NETWORK_SAVE_DIR)
        user_input_to_net(net, sample_length, encoding)
    except IOError:
        net = create_and_train_net(train, test, encoding, sample_length, name)
        sys.setrecursionlimit(10**6)
        save(net, name, path = NETWORK_SAVE_DIR)

    



def user_input_to_net(net, sample_length, encoding):
    
    quit_keywords = {'q', 'quit', 'exit', 'bye', 'goodbye'}

    running = True
    print("\n\n")

    chat_history = ""
    while running:
        try:
            user_input = raw_input("User: ")
            chat_history += user_input 
            if user_input.lower() in quit_keywords:
                running = False
            else:
                result = generate_phrase(net, sample_length, encoding, chat_history)
                chat_history += result
        except KeyboardInterrupt, EOFError:
            running = False

    print("\nQuitting...")


def generate_phrase(net, sample_length, encoding, seed_string, char_temp_mult = 1.00):
    
    sys.stdout.write("\nNetwork: ")
    sys.stdout.flush()

    input_string = seed_string + '\n'
    output_string = ""

    temp = 0.5
    while len(output_string) == 0 or output_string[-1] != '\n':
        
        if sample_length is not None:
            input_string = fit_string_to_length(input_string + output_string, sample_length)
            
        encoded_input = np.asarray([encoding.encode(input_string)])
        net_output = net.sample(encoded_input, temperature = temp)[0]
        
        new_char = encoding.decode(net_output)
        output_string += new_char
        temp *= char_temp_mult

        sys.stdout.write(new_char)
        sys.stdout.flush()
    
    print("")
    print("Final Temp: {}".format(temp))
    return output_string



def fit_string_to_length(s, length):
    if len(s) > length:
        return s[-length:]
    elif len(s) < length:
        return "{0:>{1}}".format(s, length)
    else:
        s



def create_and_train_net(train, test, encoding, sample_length, name):
    net = Network([sample_length],
                  [layers.one_hot(depth=len(encoding)),
                   layers.lstm(num_units=256),
                   layers.dropout(),
                   layers.lstm(num_units=256),
                   layers.dropout(),
                   layers.lstm(num_units=256),
                   layers.dropout(),
                   layers.rnn_most_recent(),
                   layers.fully_connected(num_outputs=len(encoding), activation_fn = None)],
                  pred_act_fn = tf.nn.softmax, 
                  logdir = None,
                  network_name = name)

    # Setup our training parameters
    opt = tf.train.AdamOptimizer(0.006)
    
    loss_func = softmax_cross_entropy_with_logits
    eval_func = accuracy

    epochs = 10
    mb_size = 100
    eval_freq = 1
    eval_fmt = '8.3%'
    verbose = True
    keep_prob = 0.5

    fit_results = net.fit(train, opt, loss_func, epochs, mb_size,
                          evaluation_freq = eval_freq, 
                          evaluation_func = eval_func,
                          evaluation_fmt = eval_fmt,
                          shuffle_freq = 1,
                          test_data = test,
                          l2_reg_strength = 0.0001,
                          dropout_keep_prob = keep_prob,
                          verbose = verbose)
 
    print_fit_results(fit_results, eval_fmt, 'Final Results')
    
    return net





def load_and_prepare_dataset(max_lines = None, sample_length = None, verbose = False):
    
    if verbose: print("Loading Corpus")
    corpus = dataset_util.load_south_park_corpus()
    if verbose: print("Corpus Size: {}".format(len(corpus)))

    if verbose: print("\nCleaning Corpus")
    corpus = dataset_util.clean_corpus(corpus)
    if verbose: print("Cleaned Corpus Size: {}".format(len(corpus)))

    encoding = Encoding(corpus)

    if verbose: print("\nExpanding Corpus")
    corpus = dataset_util.random_line_concat_iterations(corpus, iters = 2, concat_p = 0.20)
    if verbose: print("Expanded Corpus Size: {}".format(len(corpus)))
        
    if max_lines is not None:
        corpus = corpus[:max_lines]
    
    if verbose: print("\nConverting Corpus to Trainable Dataset")
    inputs, outputs = dataset_util.convert_to_dataset(corpus, sample_length) 
    if verbose: print("Dataset Size: {}".format(len(inputs)))

    if verbose: print("\nEncoding Dataset")
    inputs = np.asarray([encoding.encode_str(s) for s in inputs])
    outputs = np.asarray([encoding.encode_one_hot(c) for c in outputs])

    if verbose: print("Shuffling Dataset")
    dataset = dataset_util.shuffle_dataset((inputs, outputs))
    
    return dataset, encoding






