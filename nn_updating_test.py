
import numpy as np
import tensorflow as tf

from neural_network import Network 
from neural_network import layers
from neural_network.loss import softmax_cross_entropy_with_logits
from neural_network.evaluation import accuracy

from collections import namedtuple

Dataset = namedtuple('Dataset', ['images', 'labels'])
Datasets = namedtuple('Datasets', ['train', 'validation', 'test'])

def main():
    mnist = load_mnist()
    
    #run_all_number_baseline(mnist)
    #run_number_baseline(mnist, filter_labels = [2,5,6])
    run_drip_test(mnist)

def run_drip_test(mnist = None, net = None):
    if mnist is None: mnist = load_mnist()
    if net is None: net = load_mnist_network()
    
    val_data = split_dataset(mnist.validation)
    test_data = split_dataset(mnist.test)

    # Setup our training parameters
    opt = tf.train.AdamOptimizer(0.0001)
    
    loss_func = softmax_cross_entropy_with_logits
    eval_func = accuracy

    epochs = 1
    mb_size = 64
    eval_freq = None
    per_class_eval = True
    sums_per_epoch = 10   
    
    def run_fit(net, train_data):
        # Fit the network to our data
        net.fit(train_data, opt, loss_func, epochs, mb_size, 
                evaluation_freq = eval_freq, evaluation_func = eval_func,
                evaluation_fmt = '8.3%', per_class_evaluation = per_class_eval,
                validation_data = val_data, 
                test_data = test_data, 
                shuffle_freq = 1,
                l2_reg_strength = 0.0001,
                summaries_per_epoch = sums_per_epoch)
   
    for label in range(10):
        print("Training for just label {}".format(label))
        label_trn_data = filter_dataset(mnist.train, [label])
        #if label > 1:
        #    a = zip(*label_trn_data)
        #    a,b = zip(*a[:129])
        #    label_trn_data = Dataset(np.asarray(a), np.asarray(b))

        run_fit(net, label_trn_data)
    
    print("Finished")

def run_number_baseline(mnist = None, net = None, filter_labels = None):
    if mnist is None: mnist = load_mnist()
    if net is None: net = load_mnist_network()
    if filter_labels is not None: mnist = filter_dataset(mnist, filter_labels)

    train_data = split_dataset(mnist.train)
    val_data = split_dataset(mnist.validation)
    test_data = split_dataset(mnist.test)

    # Setup our training parameters
    opt = tf.train.AdamOptimizer(0.001)
    
    loss_func = softmax_cross_entropy_with_logits
    eval_func = accuracy

    epochs = 25
    mb_size = 128
    eval_freq = 1
    per_class_eval = True
    sums_per_epoch = 10
     
    # Fit the network to our data
    net.fit(train_data, opt, loss_func, epochs, mb_size, 
            evaluation_freq = eval_freq, evaluation_func = eval_func,
            evaluation_fmt = '8.3%', per_class_evaluation = per_class_eval,
            validation_data = val_data, 
            test_data = test_data, 
            shuffle_freq = 1,
            l2_reg_strength = 0.0001,
            summaries_per_epoch = sums_per_epoch)
    
    return net

def setup_drip_datasets(datasets):
    
    trn_by_label = []
    for label in range(10):
        trn_by_label.append(filter_dataset(datasets.train, [label])) 

    val = datasets.validation
    tst = datasets.test
    
    return Datasets(trn_by_label)

def filter_datasets(datasets, labels = None):
    if labels is None: return None
    labels = set(labels)

    trn_filtered = filter_dataset(datasets.train)
    val_filtered = filter_dataset(datasets.validation)
    tst_filtered = filter_dataset(datasets.test)
    
    return Datasets(trn_filtered, val_filtered, tst_filtered)

def filter_dataset(dataset, labels):
    labels = set(labels)
    dataset_images, dataset_ohl = split_dataset(dataset)
    dataset_labels = one_hot_to_label(dataset_ohl)
    dataset_zipped = list(zip(dataset_images, dataset_ohl, dataset_labels))
    dataset_filtered = [(img, ohl) for img, ohl, label in dataset_zipped if label in labels] 
    filt_images, filt_ohl = zip(*dataset_filtered)
    return Dataset(np.asarray(filt_images), np.asarray(filt_ohl))

def one_hot_to_label(one_hot_encoding):
    return np.argmax(one_hot_encoding, axis = 1)

def load_mnist_network():
    # Setup the network
    net = Network([28,28,1], 
                  [layers.convolution2d(num_outputs=32, kernel_size=5),
                   layers.convolution2d(num_outputs=64, kernel_size=3),
                   layers.max_pool2d(),
                   layers.flatten(),
                   layers.fully_connected(num_outputs=1000, activation_fn=tf.nn.relu),
                   layers.fully_connected(num_outputs=10, activation_fn=None)],
                  logdir = None,
                  network_name = 'mnist_network')
    return net

def split_dataset(dataset):
    return dataset.images, dataset.labels

def load_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets("MNIST_data/", one_hot=True)

if __name__ == "__main__":
    main()
