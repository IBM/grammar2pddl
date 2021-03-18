# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from g2lserver.models.grammar import Grammar  # noqa: E501
from g2lserver.models.grammar_id import GrammarID  # noqa: E501
from g2lserver.test import BaseTestCase


class TestGrammarController(BaseTestCase):
    """GrammarController integration test stubs"""

    def test_add_grammar(self):
        """Test case for add_grammar

        Add a new grammar
        """
        body = Grammar()
        response = self.client.open(
            '//grammar',
            method='POST',
            data=json.dumps(body),
            content_type='application/json')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_delete_grammar(self):
        """Test case for delete_grammar

        Delete an existing grammar
        """
        response = self.client.open(
            '//grammar/{grammarId}'.format(grammarId='grammarId_example'),
            method='DELETE')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
