from lale.lib.sklearn import *
from lale.lib.xgboost import *
from lale.lib.lightgbm import *
#from lale.lib.lale import KeepNumbers, KeepNonNumbers
from lale.lib.lale import ConcatFeatures as Concat
from lale.lib.lale import NoOp
from lale.pretty_print import to_string
from lale.lib.lale import Hyperopt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np
import time as time
from abstract_optimizer import APipelineOptimizer


# Translates pipelines to LALE and uses hyperopt to train them and obtain feedback
class PipelineOptimizer(APipelineOptimizer):
    REGRESSION = True
    EVALS = 20

    def __init__(self, data=None, regression=True, evals=20):
        self.data = data if data is not None else load_iris(return_X_y=True)
        self.REGRESSION = regression
        self.EVALS = evals
        self.X, self.y = self.data

    def evaluate_pipeline(self, pipeline):
        if 'lale_pipeline' not in pipeline:
            return pipeline
        print("Starting to optimize " + pipeline['pipeline'])
        start_time = time.time()
        opt_scorer = 'r2' if self.REGRESSION else 'accuracy'
        opt = Hyperopt(
            estimator=pipeline['lale_pipeline'],
            max_evals=self.EVALS,
            scoring=opt_scorer
        )
        trained_pipeline = None
        best_accuracy = 0

        try:
            trained_pipeline = opt.fit(self.X, self.y)
            print('Fit completed.')
            predictions = trained_pipeline.predict(self.X)
            print('Predict completed.')
#            best_accuracy = -np.min(opt.get_trials().losses())
            best_accuracy = accuracy_score(self.y, [round(pred) for pred in predictions])
            print('Best accuracy: ' + str(best_accuracy))
        except Exception as e:
            print("EXCEPTION OCCURRED: " + str(e))

        end_time = time.time()
        print("Completed optimization for " + pipeline['pipeline'])
        tlp = pipeline.copy()
        tlp.update({
            'trained_pipeline': trained_pipeline,
            'best_accuracy': best_accuracy,
            'opt_duration': (end_time-start_time)
        })
        return tlp

