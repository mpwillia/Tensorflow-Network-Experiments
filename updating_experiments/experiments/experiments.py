
from neural_network.network_util import print_fit_results 
from collections import namedtuple

ExperimentResult = namedtuple('ExperimentResult', ['name', 'results', 'steps'])

def print_results(results, name):
    print_fit_results(results, '8.3%', name)


def run_trials(data, num_trials, experiments, as_csv = True):
    
    all_trial_results = []
    for trial_num in range(num_trials):
        print("")
        print("="*80)
        print("Trial {:d} of {:d}".format(trial_num, num_trials))
        all_trial_results.append(run_experiments(data, experiments, as_csv)) 
    
     
    print("")
    print("="*80)
    print("Trials Complete")

    for trial_num, trial_results in enumerate(all_trial_results):
        print("Trial {:d} Results".format(trial_num))
        for exp_result in trial_results:
            print_exp_result(exp_result, as_csv)

        print("\n")
    print("\n")
    
def print_exp_result(exp_result, as_csv = False):
    name, results, steps = exp_result
    if not as_csv: 
        msg = "{}\nTotal Steps : {:d}".format(name, steps)
        print_fit_results(results, '8.3%', msg)
        print("")
    else:
        trn_acc = results.train.overall 
        val_acc = results.validation.overall 
        tst_acc = results.test.overall 
        
        line_fmt = "\"{:s}\",{:d},{:f},{:f},{:f}"
        print(line_fmt.format(name, steps, trn_acc, val_acc, tst_acc))

def run_experiments(data, experiments, as_csv = False):
    
    all_exp_results = []
    for num, experiment in enumerate(experiments):
        print("\nRunning Experiment {:d} / {:d}".format(num, len(experiments)))
        if type(experiment) in (tuple, list, set):
            exp_results = experiment[0].run(data, **experiment[1])
        else:
            exp_results = experiment.run(data)
        all_exp_results.append(exp_results)
    
    print("")
    print("="*80)
    print("All Experiment Results\n")
    for exp_result in all_exp_results:
        print_exp_result(exp_result, as_csv)
    
    return all_exp_results
