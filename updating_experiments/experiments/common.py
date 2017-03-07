import tensorflow as tf
from neural_network import Network 
from neural_network import layers
from neural_network.loss import softmax_cross_entropy_with_logits
from neural_network.evaluation import accuracy

from ..updatable_network import UpdatableNetwork 


def _handle_kwargs(given, default):
    merged = {k:v for k,v in default.iteritems()} 
    for k,v in given.iteritems():
        merged[k] = v
    return merged


def get_mnist_fit_params():
    # Setup our training parameters
    opt = tf.train.AdamOptimizer(0.0001)
    
    loss_func = softmax_cross_entropy_with_logits
    eval_func = accuracy

    return {'optimizer' : opt,
            'loss' : loss_func,
            'mb_size' : 128,
            'evaluation_freq' : None,
            'evaluation_func' : eval_func,
            'evaluation_fmt'  : '8.3%',
            'per_class_evaluation' : True,
            'shuffle_freq' : 1,
            'l2_reg_strength' : 0.0001,
            'verbose' : True}


def fit_net(net, epochs, train_data, val_data, test_data, **kwargs):
    
    kwargs['epochs'] = epochs
    kwargs['train_data'] = train_data
    kwargs['validation_data'] = val_data
    kwargs['test_data'] = test_data

    fit_results = net.fit(**kwargs)

    # Setup our training parameters
    #opt = tf.train.AdamOptimizer(0.0001)
    #
    #loss_func = softmax_cross_entropy_with_logits
    #eval_func = accuracy

    #fit_results = net.fit(train_data, opt, loss_func, epochs, 128, 
    #        evaluation_freq = None, evaluation_func = eval_func,
    #        evaluation_fmt = '8.3%', per_class_evaluation = True,
    #        validation_data = val_data, 
    #        test_data = test_data, 
    #        shuffle_freq = 1,
    #        l2_reg_strength = 0.0001,
    #        summaries_per_epoch = None,
    #        verbose = verbose)
    
    return net, fit_results


def load_mnist_network(updatable = False, **kwargs):

    if load_mnist_network.prev_net is not None:
        return load_mnist_network.prev_net

    if updatable:
        cls = UpdatableNetwork 
        name = 'updatable_mnist_network'
    else:
        cls = Network
        name = 'mnist_network'

    # Setup the network
    net = cls([28,28,1], 
                  [layers.convolution2d(num_outputs=32, kernel_size=5),
                   layers.convolution2d(num_outputs=64, kernel_size=3),
                   layers.max_pool2d(),
                   layers.flatten(),
                   layers.fully_connected(num_outputs=1000, activation_fn=tf.nn.relu),
                   layers.fully_connected(num_outputs=10, activation_fn=None)],
                  logdir = None,
                  network_name = name, **kwargs)

    load_mnist_network.prev_net = net
    return net

load_mnist_network.prev_net = None


