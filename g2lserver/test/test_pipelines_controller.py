# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from g2lserver.models.pipeline_feedback import PipelineFeedback  # noqa: E501
from g2lserver.models.pipeline_feedback_results import PipelineFeedbackResults  # noqa: E501
from g2lserver.models.pipeline_generation_params import PipelineGenerationParams  # noqa: E501
from g2lserver.models.pipelines import Pipelines  # noqa: E501
from g2lserver.test import BaseTestCase


class TestPipelinesController(BaseTestCase):
    """PipelinesController integration test stubs"""

    def test_feedback(self):
        """Test case for feedback

        Provide feedback for previously generated pipelines
        """
        body = PipelineFeedback()
        response = self.client.open(
            '//feedback/{grammarId}'.format(grammarId='grammarId_example'),
            method='POST',
            data=json.dumps(body),
            content_type='application/json')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_pipelines(self):
        """Test case for get_pipelines

        Get the next set of pipelines
        """
        body = PipelineGenerationParams()
        response = self.client.open(
            '//pipelines/{grammarId}'.format(grammarId='grammarId_example'),
            method='POST',
            data=json.dumps(body),
            content_type='application/json')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_train_pipelines(self):
        """Test case for train_pipelines

        Gets a number of pipelines and self-trains the parameters. Scores returned are accuracy scores from the training.
        """
        body = PipelineGenerationParams()
        response = self.client.open(
            '//trained-pipelines/{grammarId}'.format(grammarId='grammarId_example'),
            method='POST',
            data=json.dumps(body),
            content_type='application/json')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
