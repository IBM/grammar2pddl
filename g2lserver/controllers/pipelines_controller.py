import connexion
import sys

from g2lserver.models.pipeline_feedback import PipelineFeedback  # noqa: E501
from g2lserver.models.pipeline_feedback_results import PipelineFeedbackResults  # noqa: E501
from g2lserver.models.pipeline_generation_params import PipelineGenerationParams  # noqa: E501
from g2lserver.models.pipelines import Pipelines  # noqa: E501
from g2lserver.models.pipelines_inner import PipelinesInner
from g2lserver.GrammarFS import GrammarFS
from grammar2lale.pipeline_optimizer import PipelineOptimizer
from sklearn.datasets import load_iris

def feedback(grammarId, body):  # noqa: E501
    """Provide feedback for previously generated pipelines

     # noqa: E501

    :param grammarId: Grammar we are providing feedback for
    :type grammarId: str
    :param body: Pipeline feedback
    :type body: dict | bytes

    :rtype: PipelineFeedbackResults
    """
#    if connexion.request.is_json:
#        body = PipelineFeedback.from_dict(connexion.request.get_json())  # noqa: E501

    if not GrammarFS.getInstance().has_grammar(grammarId):
        return 'Cannot find grammar with ID ' + grammarId, 404
    gobj = GrammarFS.getInstance().get_grammar_object(grammarId)
    try:
        acclow = body.get('accuracy_low')
        acchigh = body.get('accuracy_high')
        print("Got here", file=sys.stderr)
        crfeed = {}
        for f in body.get('feedback'):
            crfeed[f['id']] = f['accuracy']
        (successIDs, failedIDs) = gobj.feedback(crfeed, acclow, acchigh)
        print("Got here", file=sys.stderr)
        return PipelineFeedbackResults.from_dict({'valid_pipelines': successIDs, 'invalid_pipelines': failedIDs})
    except Exception as e:
        print(e, file=sys.stderr)
        return "Error occurred " + str(e), 500


def get_pipelines(grammarId, body):  # noqa: E501
    """Get the next set of pipelines

     # noqa: E501

    :param grammarId: Grammar to draw pipelines from
    :type grammarId: str
    :param body: Pipeline generation parameters
    :type body: dict | bytes

    :rtype: Pipelines
    """
#    if connexion.request.is_json:
#        body = PipelineGenerationParams.from_dict(connexion.request.get_json())  # noqa: E501

    if not GrammarFS.getInstance().has_grammar(grammarId):
        return 'Cannot find grammar with ID ' + grammarId, 404

    num_pipelines = body['number_of_pipelines'] if ('number_of_pipelines' in body) else 10
    constraints = body['constraints'] if 'constraints' in body else []

    if num_pipelines <= 0:
        return "Invalid number of pipelines requested", 405

    gobj = GrammarFS.getInstance().get_grammar_object(grammarId)
    try:
        pipelines = gobj.get_plans(num_pipelines=num_pipelines, constraints=constraints)
        inner_pipelines = [PipelinesInner.from_dict({k: p[k] for k in ('id', 'pipeline', 'score')}) for p in pipelines]
        ret_pipelines = Pipelines.from_dict(dikt=inner_pipelines)
        return ret_pipelines
    except Exception as e:
        print(e, file=sys.stderr)
        return "Error occurred " + str(e), 500


def train_pipelines(grammarId, body):  # noqa: E501
    """Gets a number of pipelines and self-trains the parameters. Scores returned are accuracy scores from the training.

     # noqa: E501

    :param grammarId: Grammar to draw pipelines from
    :type grammarId: str
    :param body: Pipeline generation parameters
    :type body: dict | bytes

    :rtype: Pipelines
    """
#    if connexion.request.is_json:
#        body = PipelineGenerationParams.from_dict(connexion.request.get_json())  # noqa: E501

    if not GrammarFS.getInstance().has_grammar(grammarId):
        return 'Cannot find grammar with ID ' + grammarId, 404

    num_pipelines = body['number_of_pipelines'] if ('number_of_pipelines' in body) else 10
    constraints = body['constraints'] if 'constraints' in body else []

    if num_pipelines <= 0:
        return "Invalid number of pipelines requested", 405

    gobj = GrammarFS.getInstance().get_grammar_object(grammarId)
    try:
        pipelines = gobj.get_plans(num_pipelines=num_pipelines, constraints=constraints)
        print("Got here", file=sys.stderr)
        opt = PipelineOptimizer(load_iris(return_X_y=True))
        print("Got here", file=sys.stderr)
        (trained_pipelines, dropped_pipelines) = opt.evaluate_and_train_pipelines(pipelines)
        print("Got here", file=sys.stderr)
        feedback = opt.get_feedback(trained_pipelines)
        print("Got here", file=sys.stderr)
        gobj.feedback(feedback)
        print("Got here", file=sys.stderr)
        projected_pipelines = opt.project_trained_pipelines(trained_pipelines)
        print("Got here", file=sys.stderr)
        print(projected_pipelines, file=sys.stderr)
        print("Got here", file=sys.stderr)
        inner_pipelines = [PipelinesInner.from_dict(p) for p in projected_pipelines]
        print("Got here", file=sys.stderr)
        ret_pipelines = Pipelines.from_dict(dikt=inner_pipelines)
        return ret_pipelines
    except Exception as e:
        print(e, file=sys.stderr)
        return "Error occurred " + str(e), 500
