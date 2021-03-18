import os
import shutil
import hashlib
from datetime import date
from grammar2lale.grammar2lale import Grammar2Lale


class GrammarFS:
    __instance = None

    def __mkdir(self):
        crdate = date.today()
        self.dirname = os.path.join(os.getcwd(), 'grammar_' + str(crdate.year) + "_" + str(crdate.month) + "_" + str(crdate.day))


    def __init__(self):
        if GrammarFS.__instance is not None:
            raise Exception("Can only have one instance of this class")
        self.__mkdir()
        if os.path.exists(self.dirname):
            if os.path.isdir(self.dirname):
                shutil.rmtree(self.dirname)
            else:
                os.remove(self.dirname)
        os.makedirs(self.dirname)
        self.grammars = {}
        GrammarFS.__instance = self


    @staticmethod
    def getInstance():
        if GrammarFS.__instance is None:
            GrammarFS()
        return GrammarFS.__instance


    def __grammar_name(self, grammar_id):
        return os.path.join(self.dirname, grammar_id + '.txt')


    def has_grammar(self, grammar_id):
        return grammar_id in self.grammars


    def get_grammar_object(self, grammar_id):
        return self.grammars[grammar_id]


    def store_grammar(self, grammar_text):
        grammar_id = hashlib.md5(grammar_text.encode('utf-8')).hexdigest()
        if self.has_grammar(grammar_id):
            raise Exception("Grammar already exists, with id " + grammar_id)
        fname = self.__grammar_name(grammar_id)
        with open(fname, "w") as f:
            f.write(grammar_text)
        self.grammars[grammar_id] = Grammar2Lale(grammar_file=fname)
        return grammar_id


    def delete_grammar(self, grammar_id):
        if not self.has_grammar(grammar_id):
            raise Exception("Grammar with id " + grammar_id + " not found")
        gobj = self.grammars.pop(grammar_id)
        del gobj
        fname = self.__grammar_name(grammar_id)
        os.remove(fname)

