
from ..common import fit_net, get_mnist_fit_params
from ..dataset_util import filter_dataset, split_dataset

def run_all_number_baseline(dataset, net):
    return run(dataset, net) 

def run(dataset, net, filter_labels = None):
    if filter_labels is not None: 
        dataset = filter_dataset(mnist, filter_labels)

    train_data = split_dataset(dataset.train)
    val_data = split_dataset(dataset.validation)
    test_data = split_dataset(dataset.test)
    

    epochs = 10
    verbose = True
    kwargs = get_mnist_fit_params()
    kwargs['verbose'] = verbose

    net, fit_results = fit_net(net, epochs, train_data, val_data, test_data, **kwargs)
    return fit_results

