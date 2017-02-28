
import tensorflow as tf
from neural_network.loss import softmax_cross_entropy_with_logits
from neural_network.evaluation import accuracy

from  dataset_util import filter_dataset, split_dataset

def run_all_number_baseline(dataset, net):
    return run_baseline(dataset, net) 

def run_baseline(dataset, net, filter_labels = None):
    if filter_labels is not None: 
        dataset = filter_dataset(mnist, filter_labels)

    train_data = split_dataset(dataset.train)
    val_data = split_dataset(dataset.validation)
    test_data = split_dataset(dataset.test)

    # Setup our training parameters
    opt = tf.train.AdamOptimizer(0.001)
    
    loss_func = softmax_cross_entropy_with_logits
    eval_func = accuracy

    epochs = 5
    mb_size = 128
    eval_freq = None
    per_class_eval = True
    sums_per_epoch = 10
    verbose = True
     
    # Fit the network to our data
    fit_results = net.fit(train_data, opt, loss_func, epochs, mb_size, 
            evaluation_freq = eval_freq, evaluation_func = eval_func,
            evaluation_fmt = '8.3%', per_class_evaluation = per_class_eval,
            validation_data = val_data, 
            test_data = test_data, 
            shuffle_freq = 1,
            l2_reg_strength = 0.0001,
            summaries_per_epoch = sums_per_epoch,
            verbose = verbose)
    
    return net, fit_results

