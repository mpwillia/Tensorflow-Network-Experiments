
import tensorflow as tf
import tensorflow.contrib as tfc
import tensorflow.contrib.layers as tfcl
from neural_network import Network 
from neural_network import layers

def main():
    print("Loading")
    mnist = load_mnist()
    
    check_mnist(mnist)

    #create_fc_mnist_network(mnist)
    create_conv_mnist_network(mnist)


def create_conv_mnist_network(mnist = None):
    if mnist is None: mnist = load_mnist()

    mnist_image_size = 784 
    
    net = Network([28,28,1], 
                  [layers.convolution2d(num_outputs=32, kernel_size=5),
                   layers.convolution2d(num_outputs=64, kernel_size=3),
                   layers.max_pool2d(),
                   layers.flatten(),
                   layers.fully_connected(num_outputs=1000, activation_fn=tf.nn.relu),
                   layers.fully_connected(num_outputs=10, activation_fn=None)])

    opt = tf.train.AdamOptimizer(0.001)
    
    def split_dataset(dataset):
        return dataset.images, dataset.labels

    train_data = split_dataset(mnist.train)
    val_data = split_dataset(mnist.validation)
    test_data = split_dataset(mnist.test)

    loss = lambda logits, labels : tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    accuracy = lambda net_output, exp_output : tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net_output, 1), tf.argmax(exp_output, 1)), tf.float32))

    epochs = 100
    mb_size = 256
    eval_freq = 10

    net.fit(train_data, opt, loss, epochs, mb_size, 
            evaluation_freq = eval_freq, evaluation_func = accuracy,
            evaluation_fmt = '8.3%',
            validation_data = val_data, 
            test_data = test_data, 
            shuffle_freq = 1)
    
    #net.test(train_data)


def create_fc_mnist_network(mnist = None):
    
    if mnist is None: mnist = load_mnist()

    mnist_image_size = 784 
    
    net = Network(mnist_image_size, 
                  [layers.fully_connected(num_outputs=1000, activation_fn=tf.nn.sigmoid),
                   layers.fully_connected(num_outputs=10, activation_fn=None)])

    opt = tf.train.GradientDescentOptimizer(0.25)
    
    def split_dataset(dataset):
        return dataset.images, dataset.labels

    train_data = split_dataset(mnist.train)
    validation_data = split_dataset(mnist.validation)
    test_data = split_dataset(mnist.test)

    loss = lambda logits, labels : tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    
    
    accuracy = lambda net_output, exp_output : tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net_output, 1), tf.argmax(exp_output, 1)), tf.float32))


    epochs = 100
    mb_size = 1
    evaluation_freq = 10

    net.fit(train_data, opt, loss, epochs, mb_size, evaluation_freq, accuracy,
            validation_data, test_data)


def check_mnist(mnist = None):
    if mnist is None: mnist = load_mnist()
    
    def dataset_info(dataset):
        images = dataset.images
        labels = dataset.labels
        return "Images {:13s} |  Labels {:13s}".format(str(images.shape), str(labels.shape))

    print("\nChecking MNIST")
    print("  Training   : {}".format(dataset_info(mnist.train)))
    print("  Validation : {}".format(dataset_info(mnist.validation)))
    print("  Testing    : {}".format(dataset_info(mnist.test)))
    print("")


def load_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets("MNIST_data/", one_hot=True)


if __name__ == "__main__":
    main()

