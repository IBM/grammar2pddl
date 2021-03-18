# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from g2lserver.models.base_model_ import Model
from g2lserver import util


class Grammar(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    def __init__(self, grammar: str=None):  # noqa: E501
        """Grammar - a model defined in Swagger

        :param grammar: The grammar of this Grammar.  # noqa: E501
        :type grammar: str
        """
        self.swagger_types = {
            'grammar': str
        }

        self.attribute_map = {
            'grammar': 'grammar'
        }

        self._grammar = grammar

    @classmethod
    def from_dict(cls, dikt) -> 'Grammar':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The Grammar of this Grammar.  # noqa: E501
        :rtype: Grammar
        """
        return util.deserialize_model(dikt, cls)

    @property
    def grammar(self) -> str:
        """Gets the grammar of this Grammar.


        :return: The grammar of this Grammar.
        :rtype: str
        """
        return self._grammar

    @grammar.setter
    def grammar(self, grammar: str):
        """Sets the grammar of this Grammar.


        :param grammar: The grammar of this Grammar.
        :type grammar: str
        """
        if grammar is None:
            raise ValueError("Invalid value for `grammar`, must not be `None`")  # noqa: E501

        self._grammar = grammar