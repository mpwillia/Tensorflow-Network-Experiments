
from ..common import load_mnist_network, fit_net, get_mnist_fit_params
from ...dataset_util import filter_dataset, split_dataset, sample_dataset
from ..experiments import print_results, ExperimentResult

def run(dataset):
    baseline_name = "Naive Holdout Baseline"
    print("Running {}".format(baseline_name))
    net = load_mnist_network()

    holdout_labels = set([7])
    initial_labels = set(range(10)) - holdout_labels

    initial_train = filter_dataset(dataset.train, initial_labels)
    holdout_train = filter_dataset(dataset.train, set(range(10)))

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
    total_steps = net.get_global_step()

    net.close()
    print_results(final_results, baseline_name)
    exp_results = ExperimentResult(baseline_name, final_results, total_steps)

    return exp_results
    #return initial_results, final_results

