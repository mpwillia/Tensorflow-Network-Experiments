
from ..common import load_mnist_network, fit_net, get_mnist_fit_params
from ...dataset_util import filter_dataset, split_dataset, sample_dataset
from ..experiment import print_results 

def run(dataset):
    
    print("Running Ideal Holdout Baseline")
    net = load_mnist_network()

    holdout_labels = set([7])
    initial_labels = set(range(10)) - holdout_labels

    initial_train = filter_dataset(dataset.train, initial_labels)
    holdout_train = filter_dataset(dataset.train, holdout_labels)
    holdout_train = sample_dataset(holdout_train, 500)

    val_data = split_dataset(dataset.validation)
    test_data = split_dataset(dataset.test)
    
    # initial train -----------------------------------------------------------
    initial_epochs = 5
    holdout_epochs = 5
    verbose = True
    kwargs = get_mnist_fit_params()
    kwargs['verbose'] = verbose
    
    print("Initial Train")
    net, initial_results = fit_net(net, initial_epochs, initial_train, val_data, test_data, **kwargs)
    

    # holdout train -----------------------------------------------------------
    print("Holdout Train")
    net, final_results = fit_net(net, holdout_epochs, holdout_train, val_data, test_data, **kwargs)

    print_results(final_results, 'Ideal Holdout Baseline Results')

    return initial_results, final_results

