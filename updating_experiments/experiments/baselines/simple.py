
from ..common import load_mnist_network, fit_net, get_mnist_fit_params
from ...dataset_util import filter_dataset, split_dataset
from ..experiments import print_results, ExperimentResult


def run(dataset, epochs = 5, filter_labels = None):
    baseline_name = "Simple {:d} Epoch Baseline".format(epochs)
    print("Running {}".format(baseline_name))
    print("  Filter Labels : {}".format(str(filter_labels)))

    net = load_mnist_network()

    if filter_labels is not None: 
        dataset = filter_dataset(mnist, filter_labels)

    train_data = split_dataset(dataset.train)
    val_data = split_dataset(dataset.validation)
    test_data = split_dataset(dataset.test)
    
    verbose = True
    kwargs = get_mnist_fit_params()
    kwargs['verbose'] = verbose

    net, fit_results = fit_net(net, epochs, train_data, val_data, test_data, **kwargs)
    total_steps = net.get_global_step()
    
    net.close()
    print_results(fit_results, baseline_name)
    exp_results = ExperimentResult(baseline_name, fit_results, total_steps)
    return exp_results 

