from lale.lib.sklearn import *
from lale.lib.xgboost import *
from lale.lib.lightgbm import *
#from lale.lib.lale import KeepNumbers, KeepNonNumbers
from lale.lib.lale import ConcatFeatures as Concat
from lale.lib.lale import NoOp
from lale.pretty_print import to_string
from abc import *

class APipelineOptimizer(ABC):
    def to_lale_pipeline(self, pipeline):
        try:
            lale_pipeline = eval(pipeline['pipeline'], globals())
            lp = pipeline.copy()
            lp.update({'lale_pipeline': lale_pipeline})
            return lp
        except Exception as e:
            print('Cannot eval ' + pipeline['pipeline'] + ": " + str(e))
            return pipeline

    @abstractmethod
    def evaluate_pipeline(self, pipeline):
        pass

    def evaluate_and_train_pipelines(self, pipelines):
        trained = []
        dropped = []
        for i, p in enumerate(pipelines):
            print('Plan %i/%i' % (i + 1, len(pipelines)))
            lp = self.to_lale_pipeline(p)
            elp = self.evaluate_pipeline(lp)
            if 'trained_pipeline' not in elp:
                dropped.append(elp)
            else:
                trained.append(elp)
        return trained, dropped

    def get_feedback(self, trained_pipelines):
        feedback = {}
        for tp in trained_pipelines:
            feedback[tp['id']] = tp['best_accuracy']
        return feedback

    def project_trained_pipelines(self, trained_pipelines):
        return [{'id': p['id'],
                 'pipeline': to_string(p['trained_pipeline']),
                 'score': p['best_accuracy']} for p in trained_pipelines]

