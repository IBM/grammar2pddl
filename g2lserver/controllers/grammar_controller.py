import connexion
import sys

from g2lserver.models.grammar import Grammar  # noqa: E501
from g2lserver.models.grammar_id import GrammarID  # noqa: E501
from g2lserver.GrammarFS import GrammarFS
from grammar2lale.grammar2lale import Grammar2Lale

def add_grammar(body):  # noqa: E501
    """Add a new grammar

    Provide a new grammar to the service and receive an unique identifier # noqa: E501

    :param body: Grammar that needs to be added
    :type body: dict | bytes

    :rtype: GrammarID
    """
#    if connexion.request.is_json:
#            body = Grammar.from_dict(connexion.request.get_json())  # noqa: E501
    # try storing the grammar
    try:
        grammar_id = GrammarFS.getInstance().store_grammar(body['grammar'])
    except Exception as e:
        print(e, file=sys.stderr)
        return '', 405

    return GrammarID(id=grammar_id)


def delete_grammar(grammarId):  # noqa: E501
    """Delete an existing grammar

     # noqa: E501

    :param grammarId: Grammar to be deleted
    :type grammarId: str

    :rtype: None
    """
    try:
        GrammarFS.getInstance().delete_grammar(grammarId)
        return '', 200
    except Exception as e:
        print(e, file=sys.stderr)
        return str(e), 404
