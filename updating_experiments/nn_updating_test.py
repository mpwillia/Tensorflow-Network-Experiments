
import numpy as np
import tensorflow as tf

from common import load_mnist, load_mnist_network

from neural_network.network_util import print_fit_results 

from experiments import baselines, tests

#from neural_network import Network 
#from neural_network import layers
#from neural_network.loss import softmax_cross_entropy_with_logits
#from neural_network.evaluation import accuracy


def main():
    mnist = load_mnist()
    
    #baselines.simple.run(mnist)
    #baselines.ideal_holdout.run(mnist)
    #baselines.naive_holdout.run(mnist)

    tests.recall.run(mnist, 0.1)

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



def run_simple_baseline(data, net):
    simple_results = baselines.simple.run(data, net)
    print_fit_results(simple_results, '8.3%', 'Simple Baseline')

def run_holdout_baseline(data, net):
    initial_results, holdout_results = baselines.holdout.run(data, net)
    print_fit_results(initial_results, '8.3%', 'Holdout Initial')
    print_fit_results(holdout_results, '8.3%', 'Holdout Final')

def run_naive_holdout_baseline(data, net):
    initial_results, holdout_results = baselines.naive_holdout.run(data, net)
    print_fit_results(initial_results, '8.3%', 'Holdout Initial')
    print_fit_results(holdout_results, '8.3%', 'Holdout Final')

def run_drip_baseline(data, net):
    drip_results = baselines.drip.run(data, net)
    print_fit_results(drip_results, '8.3%', 'Drip Baseline')

def run_holdout_test(data, recall_p):
    initial_results, holdout_results = tests.holdout.run(data, recall_p)
    print("Holdout Test Results with Recall of {:7.2%}".format(recall_p))
    print_fit_results(initial_results, '8.3%', 'Holdout Initial')
    print_fit_results(holdout_results, '8.3%', 'Holdout Final')

def run_drip_basic_test(data, net):
    drip_results = tests.basic_drip.run(data, net)
    print_fit_results(drip_results, '8.3%', 'Drip Basic Test')

if __name__ == "__main__":
    main()

