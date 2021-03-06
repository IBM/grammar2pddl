# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from g2lserver.models.base_model_ import Model
from g2lserver import util


class PipelineGenerationParams(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    def __init__(self, number_of_pipelines: int=None, constraints: List[str]=None):  # noqa: E501
        """PipelineGenerationParams - a model defined in Swagger

        :param number_of_pipelines: The number_of_pipelines of this PipelineGenerationParams.  # noqa: E501
        :type number_of_pipelines: int
        :param constraints: The constraints of this PipelineGenerationParams.  # noqa: E501
        :type constraints: List[str]
        """
        self.swagger_types = {
            'number_of_pipelines': int,
            'constraints': List[str]
        }

        self.attribute_map = {
            'number_of_pipelines': 'numberOfPipelines',
            'constraints': 'constraints'
        }

        self._number_of_pipelines = number_of_pipelines
        self._constraints = constraints

    @classmethod
    def from_dict(cls, dikt) -> 'PipelineGenerationParams':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The PipelineGenerationParams of this PipelineGenerationParams.  # noqa: E501
        :rtype: PipelineGenerationParams
        """
        return util.deserialize_model(dikt, cls)

    @property
    def number_of_pipelines(self) -> int:
        """Gets the number_of_pipelines of this PipelineGenerationParams.


        :return: The number_of_pipelines of this PipelineGenerationParams.
        :rtype: int
        """
        return self._number_of_pipelines

    @number_of_pipelines.setter
    def number_of_pipelines(self, number_of_pipelines: int):
        """Sets the number_of_pipelines of this PipelineGenerationParams.


        :param number_of_pipelines: The number_of_pipelines of this PipelineGenerationParams.
        :type number_of_pipelines: int
        """

        self._number_of_pipelines = number_of_pipelines

    @property
    def constraints(self) -> List[str]:
        """Gets the constraints of this PipelineGenerationParams.


        :return: The constraints of this PipelineGenerationParams.
        :rtype: List[str]
        """
        return self._constraints

    @constraints.setter
    def constraints(self, constraints: List[str]):
        """Sets the constraints of this PipelineGenerationParams.


        :param constraints: The constraints of this PipelineGenerationParams.
        :type constraints: List[str]
        """

        self._constraints = constraints
