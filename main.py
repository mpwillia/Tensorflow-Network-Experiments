

from tests import mnist_test, num_ones_test, south_park_rnn

from updating_experiments import nn_updating_test
import saving_network_test

def main():
    #mnist_test.run_mnist_test() 
    #nn_updating_test.main() 
    #saving_network_test.run_test()
    #num_ones_test.run_rnn_test()
    south_park_rnn.run_test()

if __name__ == "__main__":
    main()

