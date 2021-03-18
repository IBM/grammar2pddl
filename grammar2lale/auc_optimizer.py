from lale.lib.sklearn import *
from lale.lib.xgboost import *
from lale.lib.lightgbm import *
from lale.lib.lale import ConcatFeatures as Concat
from lale.lib.lale import NoOp
from lale.pretty_print import to_string
from sklearn.datasets import load_iris
import lale.helpers
import lale.search.op2hp
import hyperopt
import statistics
import numpy as np
import time as time
import pandas as pd
from grammar2lale.abstract_optimizer import APipelineOptimizer
import aif360.datasets
import aif360.metrics
from sklearn.metrics import roc_auc_score
import traceback

class AucOptimizer(APipelineOptimizer):
    EVALS = 20

    def __init__(self, data=None, evals=20):
        self.data = data if data is not None else aif360.datasets.GermanDataset()
        if isinstance(self.data, tuple):
            print(self.data)
            
        self.EVALS = evals
        self.trainds, self.testds = self.data.split([0.7], shuffle=True, seed=42)

    @staticmethod
    def to_dataframes(aifds):
        X = pd.DataFrame(aifds.features, columns=aifds.feature_names)
        y = pd.Series(aifds.labels.ravel(), name=aifds.label_names[0])
        return X, y

    def evaluate(self, model, dataset):
        X, y = AucOptimizer.to_dataframes(dataset)
        predictions = model.predict_proba(X)[:,1]
        accuracy = roc_auc_score(y.values, predictions)
        return accuracy

    def metrics_to_loss(self, metrics):
        base_loss = 1.0 - metrics
        return base_loss

    def search(self, trainable_pipeline, dataset):
        search_test_ds = dataset.split(3, shuffle=True, seed=42)
        test_df = [AucOptimizer.to_dataframes(ds) for ds in search_test_ds]
        train_df = [(pd.concat([Xy[0] for j, Xy in enumerate(test_df) if j != i]),
                    pd.concat([Xy[1] for j, Xy in enumerate(test_df) if i != j]))
                    for i in range(len(test_df))]
        def point_to_trained(search_point, train_X, train_y):
            trainable = lale.helpers.create_instance_from_hyperopt_search_space(trainable_pipeline, search_point)
            trained = trainable.fit(train_X, train_y)
            return trained
        def objective(search_point):
            losses = []
            for i in range(len(search_test_ds)):
                try:
                    trained = point_to_trained(search_point, *train_df[i])
                except BaseException as e:
                    losses.append(100)
                else:
                    metrics = self.evaluate(trained, search_test_ds[i])
                    losses.append(self.metrics_to_loss(metrics))
            loss = statistics.mean(losses)
            return {'loss': loss, 'status': hyperopt.STATUS_OK }
        search_space = lale.search.op2hp.hyperopt_search_space(trainable_pipeline)
        trials = hyperopt.Trials()
        rstate = np.random.RandomState(42)
        hyperopt.fmin(objective, search_space, hyperopt.tpe.suggest, self.EVALS, trials, rstate)
        best_point = hyperopt.space_eval(search_space, trials.argmin)
        result = point_to_trained(best_point, *AucOptimizer.to_dataframes(dataset))
        return result

    def evaluate_pipeline(self, pipeline):
        if 'lale_pipeline' not in pipeline:
            return pipeline
        print("Starting to optimize " + pipeline['pipeline'])
        start_time = time.time()
        trained_pipeline = None
        best_accuracy = 0;

        try:
            trained_pipeline = self.search(pipeline['lale_pipeline'], self.trainds)
            metrics = self.evaluate(trained_pipeline, self.testds)
            best_accuracy = metrics
        except Exception as e:
            print("EXCEPTION OCCURRED: " + str(e))
            traceback.print_exc()

        end_time = time.time()
        print("Completed optimization for " + pipeline['pipeline'])
        tlp = pipeline.copy()
        tlp.update({'trained_pipeline': trained_pipeline, 'best_accuracy': best_accuracy, 'opt_duration': (end_time-start_time)})
        return tlp
