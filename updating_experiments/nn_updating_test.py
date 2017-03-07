
import numpy as np
import tensorflow as tf

#from common import load_mnist, load_mnist_network

from neural_network.network_util import print_fit_results 

from experiments import baselines, tests
from experiments import run_experiments, run_trials

#from neural_network import Network 
#from neural_network import layers
#from neural_network.loss import softmax_cross_entropy_with_logits
#from neural_network.evaluation import accuracy

def load_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets("MNIST_data/", one_hot=True)

def main():
    mnist = load_mnist()
    
    #run_baselines(mnist)
    #run_tests(mnist)
    #run_tests_extremes(mnist)
    
    #run_trials(mnist, 3, [(baselines.simple, {'epochs' : 1}),
    #                      (baselines.simple, {'epochs' : 2})])

    run_trials(mnist, 1, [(tests.recall, {'recall_p' : 0.50, 'eval_target' : 0.989, 'max_step' : 4080*2})])

    #baselines.simple.run(mnist, 1)
    #baselines.simple.run(mnist, 1)
    #baselines.simple.run(mnist, 1)
    #baselines.ideal_holdout.run(mnist)
    #baselines.naive_holdout.run(mnist)

    #tests.recall.run(mnist, 0.1)

    #mnist_net = load_mnist_network(False)
    
    #run_simple_baseline(mnist_data, mnist_net)
    #run_drip_baseline(mnist_data, mnist_net)
    #run_holdout_baseline(mnist_data, mnist_net)
    #run_naive_holdout_baseline(mnist_data, mnist_net)


    #run_holdout_test(mnist_data, 0.0)
    #run_holdout_test(mnist_data, 0.00001)
    #run_holdout_test(mnist_data, 0.01)
    #run_holdout_test(mnist_data, 0.05)
    #run_holdout_test(mnist_data, 0.1)
    #run_holdout_test(mnist_data, 0.25)
    #run_holdout_test(mnist_data, 0.5)
    #run_holdout_test(mnist_data, 0.75)
    #run_holdout_test(mnist_data, 0.9)
    #run_holdout_test(mnist_data, 0.99999)
    #run_drip_basic_test(mnist_data, mnist_net)

    #run_number_baseline(mnist, filter_labels = [2,5,6])
    #run_drip_test(mnist)

def run_baselines(data):
    run_experiments(data, [(baselines.simple, {'epochs' : 5}), 
                            (baselines.simple, {'epochs' : 10}),
                            #baselines.ideal_holdout,
                            baselines.naive_holdout
                            ],
                    True)

def run_tests(data):
    run_experiments(data, [(tests.recall, {'recall_p' : 0.0}), 
                           (tests.recall, {'recall_p' : 0.0001}), 
                           (tests.recall, {'recall_p' : 0.025}), 
                           (tests.recall, {'recall_p' : 0.05}), 
                           (tests.recall, {'recall_p' : 0.10}), 
                           (tests.recall, {'recall_p' : 0.20}), 
                           (tests.recall, {'recall_p' : 0.30}), 
                           (tests.recall, {'recall_p' : 0.40}), 
                           (tests.recall, {'recall_p' : 0.50}), 
                           (tests.recall, {'recall_p' : 0.60}), 
                           (tests.recall, {'recall_p' : 0.70}), 
                           (tests.recall, {'recall_p' : 0.80}), 
                           (tests.recall, {'recall_p' : 0.90}), 
                           (tests.recall, {'recall_p' : 0.95}), 
                           (tests.recall, {'recall_p' : 0.975}), 
                           (tests.recall, {'recall_p' : 0.9999}), 
                           ],
                    True)

def run_tests_extremes(data):
    run_experiments(data, [(tests.recall, {'recall_p' : 0.0}), 
                           (tests.recall, {'recall_p' : 0.0001}), 
                           (tests.recall, {'recall_p' : 0.9999}), 
                           ],
                    True)
if __name__ == "__main__":
    main()

