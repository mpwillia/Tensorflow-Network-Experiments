
import numpy as np
import tensorflow as tf

from common import load_mnist, load_mnist_network

from neural_network.network_util import print_fit_results 

import baselines
import tests

#from neural_network import Network 
#from neural_network import layers
#from neural_network.loss import softmax_cross_entropy_with_logits
#from neural_network.evaluation import accuracy


def main():
    mnist_data = load_mnist()
    mnist_net = load_mnist_network(True)
    
    #run_simple_baseline(mnist_data, mnist_net)
    #run_drip_baseline(mnist_data, mnist_net)
    run_holdout_baseline(mnist_data, mnist_net)
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

def run_drip_baseline(data, net):
    drip_results = baselines.drip.run(data, net)
    print_fit_results(drip_results, '8.3%', 'Drip Baseline')

def run_drip_basic_test(data, net):
    drip_results = tests.basic_drip.run(data, net)
    print_fit_results(drip_results, '8.3%', 'Drip Basic Test')

if __name__ == "__main__":
    main()

