
from ..common import fit_net, get_mnist_fit_params, load_mnist_network
from ...dataset_util import filter_dataset, split_dataset, sample_dataset
from ..experiments import print_results, ExperimentResult

def run(dataset, recall_p, eval_target = None, max_step = None):
    test_name = "Recall {:7.2%} Test".format(recall_p)
    print("Running {}".format(test_name))
    net = load_mnist_network(True, recall_p = recall_p)
    net.set_recall_p(recall_p)

    holdout_labels = set([7])
    initial_labels = set(range(10)) - holdout_labels

    initial_train = filter_dataset(dataset.train, initial_labels)
    holdout_train = filter_dataset(dataset.train, holdout_labels)
    #holdout_train = sample_dataset(holdout_train, 500)

    val_data = split_dataset(dataset.validation)
    test_data = split_dataset(dataset.test)
    
    # initial train -----------------------------------------------------------
    initial_epochs = 5
    holdout_epochs = 100
    verbose = True
    kwargs = get_mnist_fit_params()
    kwargs['verbose'] = verbose
    kwargs['evaluation_target'] = eval_target
    kwargs['max_step'] = max_step
    print("Initial Train")
    net, initial_results = fit_net(net, initial_epochs, initial_train, val_data, test_data, **kwargs)
    
    # holdout train -----------------------------------------------------------
    print("Holdout Train")
    kwargs['evaluation_freq'] = 1
    net, final_results = fit_net(net, holdout_epochs, holdout_train, val_data, test_data, **kwargs)
    total_steps = net.get_global_step()

    net.close()
    print_results(final_results, test_name)
    exp_results = ExperimentResult(test_name, final_results, total_steps)

    return exp_results
    #return initial_results, final_results

