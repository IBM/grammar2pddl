#! /usr/bin/env python

import os
import sys

try:
    # Python 3.x
    from builtins import open as file_open
except ImportError:
    # Python 2.x
    from codecs import open as file_open


class ParseError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return self.value

class PDDL(object): 
    def __init__(self, d):
        self.d = d

    def dump_PDDL_domain(self, recreate=False):
        if 'domain' not in self.d or recreate:
            self.d['domain'] = self._nested_list_to_pddl(self.d['domain_parsed'])

        return self.d['domain']

    def dump_PDDL_problem(self, recreate=False):
        if 'problem' not in self.d or recreate:
            self.d['problem'] = self._nested_list_to_pddl(self.d['problem_parsed'])

        return self.d['problem']

    def add_cost_support(self, mapping, cost_dict, default_htn_cost=1, default_not_mentioned_cost=0):
        ret = {}
        ret['problem_parsed'] = self._add_cost_support_problem(self.d['problem_parsed'])
        dom = self._add_cost_support_domain(self.d['domain_parsed'])
        action_names = self._get_action_names(dom)
        action_costs = {}
        for action_name, cost in cost_dict.items():
            translated_name = self._pddl_action_name(mapping, action_name)
            action_costs[translated_name] = cost

        ## Going over the operators and setting the costs according to the provided one
        ## By default htn operators get the cost 1 (better than 0), while other not mentioned
        # operators get the cost of 0 
        for action_name, ind in action_names.items():
            if action_name in action_costs:
                cost = action_costs[action_name]
            elif "htn" in action_name:
                cost = default_htn_cost
            else:
                cost = default_not_mentioned_cost

            dom[ind] = self._add_cost_to_action(dom[ind], cost)
            
        ret['domain_parsed'] = dom 

        return PDDL(ret)

    def add_soft_goals_support(self, mapping, rules_mapping, penalties_dict):
        ## Cost support is assumed
        ret = {}
        prob = self._add_cost_support_problem(self.d['problem_parsed'])
        dom = self._add_cost_support_domain(self.d['domain_parsed'])

        ## Going over the operators in penalties_dict and
        # 1. If a rule:
        #   1.1 Adding a predicate for that rule
        #   1.2 Adding the predicate to the effects of the corresponding operators
        # 2. Adding a corresponding predicate to the goal
        # 3. Adding an action with penalty cost that achieves that predicate 

        goals = []
        predicates_index = self._get_predicates_index(dom)
        action_names = self._get_action_names(dom)
#        print(action_names)
        for action_name, cost in penalties_dict.items():
            predicate_name = ""
            if action_name in rules_mapping:
                predicate_name = action_name
                dom[predicates_index].append([predicate_name])
                actions = rules_mapping[action_name]
                for action in actions:
                    htn_action_name = self._pddl_action_name(mapping, action)

                    ind = action_names[htn_action_name]
                    dom[ind] = self._add_effect_to_action(dom[ind], predicate_name)
            else:
                assert(action_name in mapping)
                predicate_name = self._pddl_action_name(mapping, action_name)
            goals.append(predicate_name)
            dom.append(self._add_forgo_action(predicate_name, cost))
            
        ret['domain_parsed'] = dom 
        ret['problem_parsed'] = self._add_to_goal(prob, goals)

        return PDDL(ret)


    def _get_predicates_index(self, input):
        i = 0
        while i < len(input):
            e = input[i]
            if ':predicates' in e[0]:
                return i
            i += 1
        return None

    def _pddl_action_name(self, mapping, action_name):
        if action_name in mapping:
            return mapping[action_name].lower()
        return "htn_" + action_name.lower()

    def _add_to_goal(self, input, predicates):
        ret = input[:]
        i = 0
        while i < len(ret):
            e = ret[i]
            if ':goal' in e[0]:
                break
            i += 1

        assert(i < len(ret))
        ## Check that the goal has two elements, second of which is the "and" list
        # otherwise, create it
        assert(len(ret[i]) == 2)
        assert(type(ret[i][1]) == list)
        if not "and" == ret[i][1][0]:
            g = ret[i][1][:]
            ret[i][1] = []
            ret[i][1].append("and")
            ret[i][1].append(g)

        for p in predicates:
            # adding to the 'and'
            ret[i][1].append([p])
        return ret

    def _add_forgo_action(self, action_name, cost):
        return [':action', "__forgo_"+action_name, 
                ':parameters', [], 
                ':precondition', [], 
                ':effect', ['and', [action_name], ['increase', ['total-cost'], cost]]]

    def _is_increase_total_cost_effect(self, e):
        return 'increase' in e[0] and 'total-cost' in e[1]

    def _get_total_cost_eff(self, cost):
        return ['increase', ['total-cost'], cost]


    def _add_atom_to_effect(self, effect, atom):
        ret = effect[:]
        for e in ret:
            if e == atom:
                return ret
        ret.append([atom])
        return ret

    def _add_effect_to_action(self, action, atom):
        ret = action[:]
        i = 0
        while i < len(ret):
            e = ret[i]
            if ':effect' in e:
                break
            i += 1

        if i == len(ret):
            # Should not happen!!
            print(ret)
        # The next element is the list of effects
        # Overwrite the existing total-cost
        ret[i+1] = self._add_atom_to_effect(ret[i+1], atom)

        return ret

    def _add_cost_to_effect(self, effect, cost):
        ret = effect[:]
        i = 0
        while i < len(ret):
            e = ret[i]
            if self._is_increase_total_cost_effect(e):
                ret[i] = self._get_total_cost_eff(cost)
                break
            i += 1
        if i == len(ret):
            ret.append(self._get_total_cost_eff(cost))

        return ret

    def _add_cost_to_action(self, action, cost):
        ret = action[:]
        i = 0
        while i < len(ret):
            e = ret[i]
            if ':effect' in e:
                break
            i += 1
        
        if i == len(ret):
            # Should not happen!!
            print(ret)
        # The next element is the list of effects
        # Overwrite the existing total-cost
        ret[i+1] = self._add_cost_to_effect(ret[i+1], cost)

        return ret

    def _add_cost_support_domain(self, raw):
        raw2 = self._add_action_costs_requirement_to_domain(raw)
        return self._add_total_cost_to_domain(raw2)

    def _add_cost_support_problem(self, raw):
        return self._add_total_cost_to_problem(raw)

    def _nl_check(self, e):
        if type(e) == int:
            return " "
        if ':precondition' in e or ':effect' in e or ':parameters' in e:
            return "\n"
        return " "

    def _nested_list_to_pddl_rec(self, input, i):
        s=''
        x = ' '
        for e in input:
            if type(e)==list:
                if i < 3:
                    s+="\n"+i*4*x
                else:
                    s+= " "
                s+= "(" 
                s+= self._nested_list_to_pddl_rec(e, i+1)
                s+= ")"
            else:
                s+= self._nl_check(e) + str(e)
        return s

    def _nested_list_to_pddl(self, input):
        ret = "(" + self._nested_list_to_pddl_rec(input, 0) + "\n)"
        return ret.replace("( ", "(")

    def _get_action_names(self, input):
        ret = {}
        i = 0
        while i < len(input):
            e = input[i]
            if type(e)==list:
                if ":action" in e[0]:
                    ret[e[1]] = i
            i += 1
        return ret

    def _add_action_costs_requirement_to_domain(self, input):
        ret = input[:]
        i = 0
        while i < len(ret):
            e = ret[i]
            if type(e)==list:
                if ":requirements" in e[0]:
                    ## Adding ":action-costs", if needed
                    ret[i] = self._add_action_costs_requirement(e)
                    return ret
                if ":action" in e[0]:
                    break
            i += 1
        ## if reached actions before requirements
        if i < len(ret):
            ret.insert(2, [':requirements', ':action-costs'])
        return ret

    def _add_total_cost_to_domain(self, input):
        ## Check if (:functions exists. If not, add 
        # (:functions (total-cost) - number
        # )
        # Otherwise, add to functions (total-cost) - number

        ret = input[:]
        i = 0
        while i < len(ret):
            e = ret[i]
            if type(e)==list:
                if ":functions" in e[0]:
                    ret[i] = self._add_total_cost_to_functions_if_needed(e)
                    return ret
                if ":action" in e[0]:
                    break
            i += 1

        ## if reached actions before functions
        if i < len(ret):
            ret.insert(i, [':functions', ['total-cost'], '-', 'number'])
        return ret

    def _is_total_cost_in_functions_list(self, e):
        ## Checking if one of the elements is "total-cost"
        for t in e[1:]:
            if type(t)==list and "total-cost" in t[0]:
                return True
        return False

    def _is_total_cost_in_predicate_list(self, e):
        ## Checking if one of the elements is "total-cost"
        for t in e[1:]:
            if type(t)==list and "total-cost" in t[1]:
                return True
        return False

    def _add_total_cost_to_functions_if_needed(self, e):
        if self._is_total_cost_in_functions_list(e):
            return e
        return e + [['total-cost'], '-', 'number']

    def _add_total_cost_to_init_if_needed(self, e):
        if self._is_total_cost_in_predicate_list(e):
            return e
        return e + [['=', ['total-cost'], '0']]


    def _add_action_costs_requirement(self, e):
        for l in e:
            if  ':action-costs' in l:
                return e
        return e + [':action-costs']

    def _add_total_cost_to_problem(self, input):
        ## Adding (= (total-cost) 0) to the initial state, if not there
        ## Adding (:metric minimize (total-cost)) to the top list
        ret = input[:]
        i = 0
        while i < len(ret):
            e = ret[i]
            if type(e)==list:
                if ":init" in e[0]:
                    ret[i] = self._add_total_cost_to_init_if_needed(e)
                if ":metric" in e[0]:
                    break
            i += 1

        ## if reached actions before functions
        if i == len(ret):
            # no metric
            ret.append([':metric', 'minimize', ['total-cost']])
        return ret


class PDDLGenerator(object): 
    # Basic functions for parsing PDDL (Lisp) files.
    def parse_nested_list(self, input_file):
        tokens = self.tokenize(input_file)
        next_token = next(tokens)
        if next_token != "(":
            raise ParseError("Expected '(', got %s." % next_token)
        result = list(self.parse_list_aux(tokens))
        for tok in tokens:  # Check that generator is exhausted.
            raise ParseError("Unexpected token: %s." % tok)
        return result

    def tokenize(self, input):
        for line in input:
            line = line.split(";", 1)[0]  # Strip comments.
            try:
                line.encode("ascii")
            except UnicodeEncodeError:
                raise ParseError("Non-ASCII character outside comment: %s" %
                                line[0:-1])
            line = line.replace("(", " ( ").replace(")", " ) ").replace("?", " ?")
            for token in line.split():
                yield token.lower()

    def parse_list_aux(self, tokenstream):
        # Leading "(" has already been swallowed.
        while True:
            try:
                token = next(tokenstream)
            except StopIteration:
                raise ParseError("Missing ')'")
            if token == ")":
                return
            elif token == "(":
                yield list(self.parse_list_aux(tokenstream))
            else:
                yield token


    def make_call(self, command, local_folder, enable_output=False):
        import os
        if (sys.version_info > (3, 0)):
            import subprocess
        else:
            import subprocess32 as subprocess

        if enable_output:
            subprocess.check_call(command, cwd=local_folder)
        else:
            FNULL = open(os.devnull, 'w')
            subprocess.check_call(command, cwd=local_folder,stdout=FNULL, stderr=subprocess.STDOUT)


    HTN2PDDL_COMMAND="run_translator.sh"

    def translate_to_PDDL(self, HTN_domain, HTN_problem, enable_planners_output=False):

        ## Assumption: the HTN domain and problem files have predefined names
        ## Running translation to PDDL and readign the created PDDL task
        ## htntranslate -t ordered -i 20 -p .htn.pddl -l "(start)"  domain.hpddl pfile.hpddl
        HTN_domain_file = "/tmp/domain.htn"
        HTN_problem_file = "/tmp/problem.htn"
        PDDL_domain_file = "/tmp/domain.htn.pddl"
        PDDL_problem_file = "/tmp/problem.htn.pddl"

        with open(HTN_domain_file, "w") as f:
            f.write(HTN_domain)
        with open(HTN_problem_file, "w") as f:
            f.write(HTN_problem)

        # sh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
        # sh_path = os.path.abspath(os.path.join(sh_path, self.HTN2PDDL_COMMAND))
        # command = [sh_path, HTN_domain_file, HTN_problem_file]
        command = ["hpddl2pddl", "-t", "ordered", "-i", "20", "-p", ".htn.pddl", "-l", '(start)', HTN_domain_file, HTN_problem_file]
        try:
#            print("Running: " + str(command))
            self.make_call(command, '/tmp/', enable_output=enable_planners_output)
        except:
            raise
        ret = {}
        # reading the created PDDL files
        with open(PDDL_domain_file, "r") as d:
            PDDL_domain = d.read()
            ret['domain'] = PDDL_domain


        with open(PDDL_problem_file, "r") as p:
            PDDL_problem = p.read()
            ret['problem'] = PDDL_problem

        ret['domain_parsed'] = self.parse_nested_list(file_open(PDDL_domain_file, encoding='ISO-8859-1'))
        ret['problem_parsed'] = self.parse_nested_list(file_open(PDDL_problem_file, encoding='ISO-8859-1'))

        return PDDL(ret)

    def read(self, PDDL_domain_file, PDDL_problem_file):
        ret = {}
        # reading the created PDDL files
        with open(PDDL_domain_file, "r") as d:
            PDDL_domain = d.read()
            ret['domain'] = PDDL_domain

        with open(PDDL_problem_file, "r") as p:
            PDDL_problem = p.read()
            ret['problem'] = PDDL_problem

        ret['domain_parsed'] = self.parse_nested_list(file_open(PDDL_domain_file, encoding='ISO-8859-1'))
        ret['problem_parsed'] = self.parse_nested_list(file_open(PDDL_problem_file, encoding='ISO-8859-1'))

        return PDDL(ret)

