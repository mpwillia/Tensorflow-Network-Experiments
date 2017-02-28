
import tensorflow as tf
from neural_network.loss import softmax_cross_entropy_with_logits
from neural_network.evaluation import accuracy

from dataset_util import filter_dataset, split_dataset
from dataset_util import Dataset, Datasets

def run_drip_baseline(mnist, net):
    
    #mnist = setup_drip_datasets(mnist)

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
    verbose = True
    
    def run_fit(net, train_data):
        # Fit the network to our data
        return net.fit(train_data, opt, loss_func, epochs, mb_size, 
                evaluation_freq = eval_freq, evaluation_func = eval_func,
                evaluation_fmt = '8.3%', per_class_evaluation = per_class_eval,
                validation_data = val_data, 
                test_data = test_data, 
                shuffle_freq = 1,
                l2_reg_strength = 0.0001,
                summaries_per_epoch = sums_per_epoch,
                verbose = verbose)
   
    for label in range(10):
        print("Training for just label {}".format(label))
        label_trn_data = filter_dataset(mnist.train, [label])
        #if label > 1:
        #    a = zip(*label_trn_data)
        #    a,b = zip(*a[:129])
        #    label_trn_data = Dataset(np.asarray(a), np.asarray(b))

        label_fit_results = run_fit(net, label_trn_data)

    
    print("Finished")
    return label_fit_results


def setup_drip_datasets(datasets):
    
    trn_by_label = []
    for label in range(10):
        trn_by_label.append(filter_dataset(datasets.train, [label])) 

    val = datasets.validation
    tst = datasets.test
    
    return Dataset(trn_by_label)


