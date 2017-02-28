
import tensorflow as tf

from neural_network import Network 
from neural_network import layers
from neural_network.loss import softmax_cross_entropy_with_logits
from neural_network.evaluation import accuracy


from neural_network.network_util import print_fit_results 

#LOG_DIR="/media/mike/Main Storage/tensorflow-logs/mnist_test_logdir"
LOG_DIR=None

def run_mnist_test():
    mnist = load_mnist()
    
    # setup our data
    def split_dataset(dataset):
        return dataset.images, dataset.labels

    train_data = split_dataset(mnist.train)
    val_data = split_dataset(mnist.validation)
    test_data = split_dataset(mnist.test)
   
    # Setup the network
    net = Network([28,28,1], 
                  [layers.convolution2d(num_outputs=32, kernel_size=5),
                   layers.convolution2d(num_outputs=64, kernel_size=3),
                   layers.max_pool2d(),
                   layers.flatten(),
                   layers.fully_connected(num_outputs=1000, activation_fn=tf.nn.relu),
                   layers.fully_connected(num_outputs=10, activation_fn=None)],
                  logdir = LOG_DIR,
                  network_name = 'mnist_network')
    
    # Setup our training parameters
    opt = tf.train.AdamOptimizer(0.001)
    
    loss_func = softmax_cross_entropy_with_logits
    eval_func = accuracy

    epochs = 2
    mb_size = 512
    eval_freq = None
    eval_fmt = '8.3%'
    per_class_eval = True
    sums_per_epoch = 10
    checkpoint_freq = None
    save_checkpoints = False
    verbose = True
    
    # Fit the network to our data
    fit_results = net.fit(train_data, opt, loss_func, epochs, mb_size, 
            evaluation_freq = eval_freq, evaluation_func = eval_func,
            evaluation_fmt = eval_fmt, per_class_evaluation = per_class_eval,
            validation_data = val_data, 
            test_data = test_data, 
            shuffle_freq = 1,
            l2_reg_strength = 0.0001,
            summaries_per_epoch = sums_per_epoch,
            save_checkpoints = save_checkpoints,
            checkpoint_freq = checkpoint_freq,
            verbose = verbose)
     
    print_fit_results(fit_results, eval_fmt, 'Final Results')

def load_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets("MNIST_data/", one_hot=True)

