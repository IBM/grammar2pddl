from lale.lib.sklearn import *
from lale.lib.xgboost import *
from lale.lib.lightgbm import *
from lale.lib.lale import ConcatFeatures as Concat
from lale.lib.lale import NoOp
from lale.pretty_print import to_string
import lale.helpers
import lale.search.op2hp
import hyperopt
import statistics
import numpy as np
import time as time
import pandas as pd
from grammar2lale.abstract_optimizer import APipelineOptimizer
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split
import traceback
import sys
import warnings
import multiprocessing

class CustomOptimizer(APipelineOptimizer):
    EVALS = 20

    def __init__(
            self, data, scorer='accuracy', evals=20, val_frac=0.2,
            max_runtime=None
    ):
        self.X, self.y = data
        self.EVALS = evals
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(
            self.X, self.y, test_size=val_frac, stratify=self.y, random_state=5489
        )
        self.scorer = get_scorer(scorer)
        self.start_time = time.time()
        self.eval_history = {
            'loss' : [],
            'score': [],
            'time_from_start' : [],
        }
        self.max_runtime = max_runtime
        print("Running optimization for " + str(self.max_runtime) + " seconds")

    def evaluate(self, trained_model):
        return self.scorer(trained_model, self.X, self.y)

    def search(self, planned_pipeline):
        # eval_idx = 0
        def point_to_trained(search_point):
            trainable = lale.helpers.create_instance_from_hyperopt_search_space(
                planned_pipeline, search_point
            )
            trained = trainable.fit(self.train_X, self.train_y)
            return trained
        def objective(search_point):
            current_time = time.time()
            if (self.max_runtime != None) and ((current_time - self.start_time) > self.max_runtime) :
                # If crossed runtime, don't even evaluate since we wish to stop but can't exit hyperopt
                print('RAN OUT OF TIME')
                sys.exit(0)

            loss = None
            score = None
            try:
                with warnings.catch_warnings(record=True) as w:
                    trained = point_to_trained(search_point)
                    score = self.evaluate(trained)
                loss = 1.0 - score
            except BaseException as e:
                loss = 100
                score = 0
            eval_time = time.time()
            # eval_idx += 1
            # print('Eval %i: %g' % (eval_idx, loss))
            self.eval_history['loss'].append(loss)
            self.eval_history['score'].append(score)
            self.eval_history['time_from_start'].append(eval_time - self.start_time)
            return {'loss': loss, 'status': hyperopt.STATUS_OK }

        search_space = lale.search.op2hp.hyperopt_search_space(planned_pipeline)
        trials = hyperopt.Trials()
        # If we want to do multiple runs and aggregate performance, this should be unset
        rstate = np.random.RandomState(5489)
        hyperopt.fmin(objective, search_space, hyperopt.tpe.suggest, self.EVALS, trials, rstate)
        best_point = hyperopt.space_eval(search_space, trials.argmin)
        result = lale.helpers.create_instance_from_hyperopt_search_space(
            planned_pipeline, best_point
        )
        best_loss = np.min(trials.losses())
        return result, 1.0 - best_loss

    def evaluate_pipeline(self, pipeline):
        if 'lale_pipeline' not in pipeline:
            return pipeline
        print("Starting to optimize " + pipeline['pipeline'])
        start_time = time.time()
        trainable_pipeline = None
        best_score = 0.0

        try:
            trainable_pipeline, best_score = self.search(pipeline['lale_pipeline'])
        except Exception as e:
            print("EXCEPTION OCCURRED: " + str(e))
            traceback.print_exc()

        end_time = time.time()
        print("Completed optimization for " + pipeline['pipeline'])
        tlp = pipeline.copy()
        tlp.update({
            'trained_pipeline': trainable_pipeline,
            'best_accuracy': best_score,
            'opt_duration': (end_time-start_time)
        })
        return tlp

    def get_eval_history(self) :
        return pd.DataFrame.from_dict(self.eval_history)

    def print_eval_history(self, out_filename) :
        self.get_eval_history().to_csv(out_filename, header=True, index=False)
        print('Evaluation history metrics + time saved in ' + out_filename)
