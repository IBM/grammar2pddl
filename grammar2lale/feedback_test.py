import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import json
from grammar2lale.grammar2lale import Grammar2Lale
from sklearn.datasets import load_iris, load_boston, load_diabetes, load_breast_cancer, load_linnerud
from grammar2lale.pipeline_optimizer import PipelineOptimizer
from grammar2lale.fairness_optimizer import FairnessOptimizer
from grammar2lale.auc_optimizer import AucOptimizer
from grammar2lale.custom_optimizer import CustomOptimizer
from lale.pretty_print import to_string
import statistics
import math
import datetime
import aif360.datasets
import random
import pandas as pd
import numpy as np
import time as time
from grammar2lale.baselines import get_baseline_planned_pipelines

crdate = datetime.datetime.today().strftime("%b_%d_%Y_%Hh_%Mm_%Ss")

def get_data_from_file(filename, label_col_name):
    df = pd.read_csv(filename)
    y_orig = df[label_col_name]
    label_dict = {l: idx for idx, l in enumerate(sorted(y_orig.unique().tolist()))}
    print('Labels mapping:\n' + str(label_dict))
    y = np.array([ label_dict[l] for l in y_orig ])
    X = df.drop([label_col_name], axis='columns').values

    print('#' * 30)
    print('X: ' + str(X.shape))
    print('y: ' + str(y.shape))
    print('#' * 30)
    return (X, y)

def get_data_from_aif360(dataset_func):
    aif360_data = dataset_func()
    X = aif360_data.features
    y = aif360_data.labels.ravel()
    print('#' * 30)
    print('X: ' + str(X.shape))
    print('y: ' + str(y.shape))
    print('#' * 30)
    return (X, y)


datasets = {
    "iris": (lambda: load_iris(return_X_y=True)),
    "diabetes": (lambda: load_diabetes(return_X_y=True)),
    "breast_cancer": (lambda: load_breast_cancer(return_X_y=True)),
    "linnerud": (lambda: load_linnerud(return_X_y=True)),
    "creditg": (lambda: aif360.datasets.GermanDataset()),
    "adult": (lambda: aif360.datasets.AdultDataset()),
    "creditg_c": (lambda: get_data_from_aif360(aif360.datasets.GermanDataset)),
    "adult_c": (lambda: get_data_from_aif360(aif360.datasets.AdultDataset)),
    "local": (lambda f, l: get_data_from_file(f, l)),
}


def set_unseen_costs(mainObj, args, iterationIndex=None, iterationCount=None):
    crcost = args.unseen_cost[0]
    if args.cost_strategy == 'decay':
        if iterationIndex is not None and iterationCount is not None:
            crcost = math.floor(np.interp(iterationIndex, [1, iterationCount], args.unseen_cost))
    mainObj.set_unseen_cost(crcost)


optimizer_choices = ['standard', 'fairness', 'auc_roc', 'custom_acc', 'custom_auroc' ]

def get_pipeline_description(pipeline):
    try:
        resstr = to_string(pipeline['trained_pipeline'])
        return resstr
    except:
        return pipeline['pipeline']


def get_and_print_results(trained_pipelines, planning_time, args=None, print_json_to_stdout=False):
    printable_pipelines = [{
        'id': p['id'],
        'pipeline': p['pipeline'],
        'trained_pipeline': get_pipeline_description(p),
        'accuracy': p['best_accuracy'],
        'optimization_duration': p['opt_duration']
    } for p in trained_pipelines]
    final_dict = dict()
    final_dict['pipelines'] = printable_pipelines
    final_dict['planning_time'] = planning_time
    if args is not None:
        final_dict['env'] = vars(args)
    print("==========================================================================")
    resstr = json.dumps(final_dict, indent=2)
    if print_json_to_stdout:
        print(resstr)
    accuracies = [p['accuracy'] for p in printable_pipelines]
    if len(accuracies) > 1:
        print("Min=" + str(min(accuracies)))
        print("Max=" + str(max(accuracies)))
        print("Mean=" + str(statistics.mean(accuracies)))
        print("Median=" + str(statistics.median(accuracies)))
        print("Stdev=" + str(statistics.stdev(accuracies)))
    elif len(accuracies) == 1:
        print("Single point: %s" % accuracies[0])
    print("==========================================================================")
    return resstr


def get_optimizer(args, dataset, max_opt_runtime=None):
    ds = None
    split_dnames = dataset.split(':')
    assert len(split_dnames) >= 1
    if len(split_dnames) > 1:
        assert len(split_dnames) == 3
        assert split_dnames[0] == 'local'
        dname = 'local'
    else:
        dname = dataset
    if dname in datasets:
        if dname == 'local':
            assert split_dnames[1] != ""
            assert split_dnames[2] != ""
            ds = datasets['local'](split_dnames[1], split_dnames[2])
        else:
            ds = datasets[dataset]()
    else:
        print("Did not find dataset " + dataset + ", using IRIS")
        ds = datasets['iris']()
    if args.optimizer == 'fairness':
        return FairnessOptimizer(data=ds, evals=args.num_evals)
    elif args.optimizer == 'auc_roc':
        return AucOptimizer(data=ds, evals=args.num_evals)
    elif args.optimizer == 'custom_acc':
        return CustomOptimizer(data=ds, scorer='balanced_accuracy', evals=args.num_evals, val_frac=0.3, max_runtime=max_opt_runtime)
    elif args.optimizer == 'custom_auroc':
        return CustomOptimizer(data=ds, scorer='roc_auc', evals=args.num_evals, val_frac=0.3, max_runtime=max_opt_runtime)
    else:
        return PipelineOptimizer(data=ds, evals=args.num_evals)

def get_dname(dataset):
    split_dnames = dataset.split(':')
    if len(split_dnames) == 1:
        return dataset
    else:
        assert len(split_dnames) == 3
        assert split_dnames[0] == 'local'
        dname = split_dnames[1].split('/')[-1].replace('.csv', '')
        return dname


def baselines_eval(args, cdate, dataset, max_opt_time, filename_base='baseline'):
    baselines = get_baseline_planned_pipelines()
    for idx, baseline in enumerate(baselines):
        opt = get_optimizer(args, dataset, max_opt_runtime=max_opt_time)
        assert opt.EVALS == args.num_evals
        # Jointly optimize directly for fixed pipeline structure
        opt.EVALS = args.number_of_plans * args.num_evals
        try:
            trained_pipeline = opt.evaluate_pipeline(baseline)
        except SystemExit:
            print('Optimization exited!')

        try:
            csv_filename = (
                filename_base + '_' + str(idx + 1) +
                '_' + get_dname(dataset) + '.csv'
            )
            opt.print_eval_history(
                os.path.join(args.out, cdate, csv_filename)
            )
        except BaseException as e:
            print(e)


def all_plans_at_once(args, cdate, dataset, filename_base='nofeedback'):
    prefix = filename_base + '_' + get_dname(dataset)
    filename = prefix + '.json'
    mainObj = Grammar2Lale(grammar_file=args.grammar_file)
    set_unseen_costs(mainObj, args)
    # Initializing optimizer before calling planner since the optimizer
    # is keeping track of the start time for the whole process (COULD BE IMPROVED)
    optimizer = get_optimizer(args, dataset)
    pipelines_step1 = mainObj.get_plans(num_pipelines=args.number_of_plans)
    trained_pipelines_step1, dropped_step1 = optimizer.evaluate_and_train_pipelines(pipelines_step1)
    print("Results for all plans in one shot")
    resstr = get_and_print_results(trained_pipelines_step1, mainObj.last_planner_time, args)
    with open(os.path.join(args.out, cdate, filename), "w") as f:
        f.write(resstr)
    try:
        csv_filename = filename_base + '_' + get_dname(dataset) + '.csv'
        optimizer.print_eval_history(
            os.path.join(args.out, cdate, csv_filename)
        )
    except BaseException as e:
        pass
    # also write htn and softgoal task
    mainObj.writeHTN(os.path.join(args.out, cdate), prefix)
    mainObj.writeLatestTask(os.path.join(args.out, cdate), prefix)

def dump_planning_tasks(args):
    mainObj = Grammar2Lale(grammar_file=args.grammar_file)
    set_unseen_costs(mainObj, args)
    if args.num_evals == 0:
        mainObj.get_plans(num_pipelines=args.number_of_plans)
        return
    all_constraints = mainObj.get_selectable_constraints()
    num_constraints = len(all_constraints)
    print("Total number of constraints is %s" % num_constraints)
    ## Hack: using args.num_evals as the number of constraints
    for j in range(10):
        c = random.sample(all_constraints, args.num_evals)
        #print("Dumping and solving for constraints: %s" %  str(c))
        mainObj.get_plans(num_pipelines=args.number_of_plans, constraints=c)

def plans_with_feedback(args, cdate, dataset, filename_base='withfeedback'):
    NO_ITERATIONS = math.ceil(args.number_of_plans / args.plans_per_round)
    PLANS_TO_PIPELINES_MULT = 1
    NO_PLANS = args.plans_per_round * PLANS_TO_PIPELINES_MULT
    PLANS_NO_MULT = 2
    mainObj = Grammar2Lale(grammar_file=args.grammar_file)
    optimizer = get_optimizer(args, dataset)
    plan_ids = []
    idx = 0

    prefix = filename_base + '_' + get_dname(dataset)
    mainObj.writeHTN(os.path.join(args.out, cdate), prefix)

    if args.dedupe_strategy == 1:
        NUMBER_OF_PIPELINES = args.number_of_plans
        MAX_NUMBER_OF_PLANS = NUMBER_OF_PIPELINES * PLANS_TO_PIPELINES_MULT
        while len(plan_ids) < NUMBER_OF_PIPELINES:
            set_unseen_costs(mainObj, args, idx, NO_ITERATIONS)
            curr_no_plans = NO_PLANS
            pipelines_step2 = []
            while len(pipelines_step2) == 0 and curr_no_plans <= MAX_NUMBER_OF_PLANS:
                pipelines_step2_prefilter = mainObj.get_plans(num_pipelines=curr_no_plans)
                pipelines_step2 = [p for p in pipelines_step2_prefilter if p['id'] not in plan_ids]
                curr_no_plans = math.ceil(curr_no_plans * PLANS_NO_MULT)

            if curr_no_plans > MAX_NUMBER_OF_PLANS:
                print("Obtaining the maximal allowed number of plans did not produce new pipelines")
                break
            print("Had " + str(len(pipelines_step2_prefilter)) + ", retaining " + str(len(pipelines_step2)) + " original ones")
            trained_pipelines_step2, dropped_step2 = optimizer.evaluate_and_train_pipelines(pipelines_step2)
            print("Results for round " + str(idx + 1))
            resstr = get_and_print_results(trained_pipelines_step2, mainObj.last_planner_time, args)
            feedback_step2 = optimizer.get_feedback(trained_pipelines_step2)
            mainObj.feedback(feedback_step2)
            prefix = filename_base + '_' + get_dname(dataset) + '_' + str(idx + 1)
            filename = prefix + '.json'
            with open(os.path.join(args.out, cdate, filename), "w") as f:
                f.write(resstr)
            mainObj.writeLatestTask(os.path.join(args.out, cdate), prefix)
            idx = idx + 1
            plan_ids.extend([p['id'] for p in pipelines_step2])
            print('%i/%i plans processed' % (len(plan_ids), args.number_of_plans))

    elif args.dedupe_strategy == 2:
        while len(plan_ids) < args.number_of_plans:
            set_unseen_costs(mainObj, args, idx, NO_ITERATIONS)
            pipelines_step2_prefilter = mainObj.get_plans(num_pipelines=(idx + 1) * NO_PLANS)
            pipelines_step2 = []
            unique_found = 0
            for p in pipelines_step2_prefilter:
                if p['id'] not in plan_ids:
                    pipelines_step2.append(p)
                    unique_found += 1
                if unique_found == NO_PLANS: break
            assert unique_found == NO_PLANS
            print(
                "Had " + str(len(pipelines_step2_prefilter)) + ", retaining "
                + str(len(pipelines_step2)) + " original ones"
            )
            trained_pipelines_step2, dropped_step2 = optimizer.evaluate_and_train_pipelines(pipelines_step2)
            print("Results for round " + str(idx + 1))
            resstr = get_and_print_results(trained_pipelines_step2, mainObj.last_planner_time, args)
            feedback_step2 = optimizer.get_feedback(trained_pipelines_step2)
            mainObj.feedback(feedback_step2)
            prefix = filename_base + '_' + get_dname(dataset) + '_' + str(idx + 1)
            filename = prefix + '.json'
            with open(os.path.join(args.out, cdate, filename), "w") as f:
                f.write(resstr)
            mainObj.writeLatestTask(os.path.join(args.out, cdate), prefix)
            idx = idx + 1
            plan_ids.extend([p['id'] for p in pipelines_step2])
            print('%i/%i plans processed' % (len(plan_ids), args.number_of_plans))
    else:
        raise Exception(
            'Unknown strategy for handling de-duplication specified'
            ' (specified: %i, expected: 1 or 2)'
            % (args.dedupe_strategy)
        )

    try:
        csv_filename = filename_base + '_' + get_dname(dataset) + '.csv'
        optimizer.print_eval_history(
            os.path.join(args.out, cdate, csv_filename)
        )
    except BaseException as e:
        pass

def process_dataset(ts, args, dataset_key):
    print("*************************************************************************")
    print("*************************************************************************")
    print("NOW WORKING ON DATASET: " + dataset_key)
    print("*************************************************************************")
    print("*************************************************************************")
    if args.dump_tasks:
        dump_planning_tasks(args)
    else:
        baseline_time = 0.
        if (
            (args.expt_mode == 'all') or
            (args.expt_mode == 'no-baseline') or
            (args.expt_mode == 'feedback')
        ):
            start_time = time.time()
            plans_with_feedback(args, ts, dataset_key)
            runtime = time.time() - start_time
            print('Planner with feedback execution time: %g' % runtime)
            if baseline_time < runtime: baseline_time = runtime
        if (
            (args.expt_mode == 'all') or
            (args.expt_mode == 'no-baseline') or
            (args.expt_mode == 'one-shot')
        ):
            start_time = time.time()
            all_plans_at_once(args, ts, dataset_key)
            runtime = time.time() - start_time
            print('Planner one-shot execution time: %g' % runtime)
            if baseline_time < runtime: baseline_time = runtime
        if args.expt_mode == 'all':
            baselines_eval(args, ts, dataset_key, baseline_time)

from joblib import Parallel, delayed
from contextlib import redirect_stdout
from io import StringIO
def main(args):
    global crdate
    if args.n_jobs == 1:
        for dataset_key in args.dataset:
            process_dataset(crdate, args, dataset_key)
    else:
        def process_dataset_no_log(ts, a, dk):
            exec_log = StringIO()
            with redirect_stdout(exec_log):
                process_dataset(ts, a, dk)
            return exec_log.getvalue()

        dfunc = lambda dk: process_dataset_no_log(ts=crdate, a=args, dk=dk)
        np = min(args.n_jobs, len(args.dataset))
        print('Processing %i datasets with %i parallel processes' % (len(args.dataset), np))
        ret = Parallel(n_jobs=np)(
            delayed(dfunc)(dataset_key) for dataset_key in args.dataset
        )
        print('!' * 40)
        print('All datasets processed .. logs following')
        print('!' * 40)
        for r in ret:
            print(r)


expt_modes = ['no-baseline', 'all', 'one-shot', 'feedback']
cost_strategies = ['fixed', 'decay']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--grammar-file", help="Grammar to parse", default=os.environ.get('GRAMMAR_FILE', None))
    parser.add_argument("--number-of-plans", help="The overall number of pipelines", type=int, default=int(os.environ.get('NUMBER_OF_PLANS', '1000')))
    parser.add_argument("--constraints", help="Constraints to guide search", nargs='+', default=os.environ.get('CONSTRAINTS', None))
    parser.add_argument("--plans-per-round", help="Number of plans per round", type=int, default=int(os.environ.get('PLANS_PER_ROUND', 50)))
    parser.add_argument("--dataset", help="Datasets to use (comma-separated list), one of " + str(datasets.keys()) + "; if 'local' is selected, it should be specified as --dataset=local:<path-to-file>:<label-column-name>", default=os.environ.get('DATASETS', 'iris').split(','), nargs='+')
    parser.add_argument("--out", help="Directory where to write output", default=os.environ.get('OUT', '/tmp/out'))
    parser.add_argument("--num-evals", help="Number of evaluations in the optimizer", type=int, default=int(os.environ.get('NUM_EVALS', 20)))
    parser.add_argument("--optimizer", help="Optimizer, choice of " + str(optimizer_choices), choices=optimizer_choices, default=os.environ.get('OPTIMIZER', 'standard'))
    parser.add_argument(
        "--dedupe_strategy", help="Strategy to handle de-duplication: 1 for MK, 2 for PR", default=int(os.environ.get('DEDUPE', '2')), type=int
    )
    parser.add_argument("--dump_tasks", help="Whether to dump planning tasks", default=False, type=bool)
    parser.add_argument("--expt_mode", help="Experiment mode, choice of " + str(expt_modes), choices=expt_modes, default=os.environ.get('MODE', 'no-baseline'))
    parser.add_argument("--n_jobs", help="Number of datasets to process in parallel", type=int, default=int(os.environ.get('N_JOBS', '1')))
    parser.add_argument("--cost-strategy", help="The cost strategy for previously unseen primitives, choice of " + str(cost_strategies), choices=cost_strategies, default=os.environ.get('COST_STRATEGY', 'fixed'))
    parser.add_argument("--unseen-cost", help="Cost for unseen primitives. It is either a single fixed value for the fixed strategy, or a start/end pair of values for the decay strategy", nargs='+', default=[int(x) for x in os.environ.get('UNSEEN_COST', '30').split(',')])
    args = parser.parse_args()
    from os import cpu_count
    print('# CPUs:' + str(cpu_count()))
    assert args.n_jobs >= 1 and args.n_jobs <= cpu_count()

    path = os.path.join(args.out, crdate)
    if not os.path.isdir(path):
        os.makedirs(path)

    main(args)
