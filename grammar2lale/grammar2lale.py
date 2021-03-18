#! /usr/bin/env python

from htn import HTN
from pddl import PDDL, PDDLGenerator
from planner import *
from lark import Token
import tempfile
import os
import hashlib
import statistics
import math
import re
import argparse

class Grammar2Lale:

    MIN_COST = 50
    MAX_COST = 101
    DEFAULT_UNSEEN_COST = 50
    CONSTRAINT_PENALTY = 5 * MAX_COST
    SELECTABLE_CONSTRAINT_TOKEN_IDS = ['EMPTYARGNAME', 'CCNAME', 'SINGLEQUOTES']
    last_planner_time = 0


    def __init__(self, grammar_text=None, grammar_file=None, domain_file=None, problem_file=None):
        self.domain_file = domain_file
        self.problem_file = problem_file
        self.has_temp_file = False
        if grammar_file is None:
            grammar_f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", prefix="grammar", delete=False)
            self.grammar_filepath = grammar_f.name
            grammar_f.write(grammar_text)
            grammar_f.close()
            self.has_temp_file = True
        else:
            self.grammar_filepath = grammar_file

        print("Generating HTN specification from grammar")
        self.htn = HTN(self.grammar_filepath)
        print("Printing HTN domain")
        self.htn_domain = self.htn.generate_domain()
        self.htn_problem = self.htn.generate_problem()

        self.pddl_gen = PDDLGenerator()
        if self.domain_file is not None and self.problem_file is not None:
            self.base_task = self.pddl_gen.read(self.domain_file, self.problem_file)
        else:
            self.base_task = self.pddl_gen.translate_to_PDDL(self.htn_domain, self.htn_problem, enable_planners_output=True)

        self.costs = {}
        self.returned_plans = {}
        self.full_feedback = {}
        self._nix_syntactic_costs()


    def __del__(self):
        if self.has_temp_file:
            os.remove(self.grammar_filepath)


    def set_unseen_cost(self, cost):
        print('Setting unseen cost to ' + str(cost))
        self.DEFAULT_UNSEEN_COST = cost


    def _bypass_htn_translation(self, domain_file, problem_file):
        self.domain_file = domain_file
        self.problem_file = problem_file


    # transforms the constraints (strings) given into penalties
    def _map_constraints_to_penalties(self, constraints):
        # to determine the penalties for constraints, we need the keys in self.mapping
        if constraints is None:
            return {}
        penalties = {}
        toklist = list(self.htn.mapping.keys())
        toklist.extend(list(self.htn.rules_mapping.keys()))
        for token in toklist:
            stok = str(token)
            if stok in constraints:
                penalties[token] = self.CONSTRAINT_PENALTY
            else:
                # Not sure this part is actually needed, leaving it here just in case
                stok = stok.strip('\'')
                if stok in constraints:
                    penalties[token] = self.CONSTRAINT_PENALTY
        return penalties


    # creates PDDL task
    def create_pddl_task(self, np=10, constraints=[]):
        print("Generating PDDL description...")
        self.base_task = self.pddl_gen.translate_to_PDDL(self.htn_domain, self.htn_problem)
        adapted_costs_task = self.base_task.add_cost_support(self.htn.mapping, self.costs, default_htn_cost=1, default_not_mentioned_cost=self.DEFAULT_UNSEEN_COST)
        penalties = self._map_constraints_to_penalties(constraints)
        softgoal_task = adapted_costs_task.add_soft_goals_support(self.htn.mapping, self.htn.rules_mapping, penalties)
        task = {}
        task['domain'] = softgoal_task.dump_PDDL_domain(recreate=True)
        task['problem'] = softgoal_task.dump_PDDL_problem(recreate=True)
        task['numplans'] = max(int(np * 1.5), 50)
        self.last_task = task
        self.last_constraints = constraints


    def run_pddl_planner(self):
        print("Running the planner...")
        time, retobj = run_planner(self.last_task)
        self.last_planner_time = time
        self.last_planner_object = retobj
        print("Plans returned after " + str(time) + " seconds.")


    def translate_to_pipelines(self, num_pipelines):
        print("Translating plans to LALE pipelines.")
        inv_mapping = self.htn.invert_mapping()
        allplans = [(plan_to_pipeline(plan, inv_mapping), plan_actions(plan), plan_tokens(plan, inv_mapping), plan['cost']) for plan in self.last_planner_object['plans']]
        selected_plans = {}
        retplans = []
        for (planstr, actions, tokens, plancost) in allplans:
            crplan = {}
            crplan['id'] = hashlib.md5(planstr.encode('utf-8')).hexdigest()
            crplan['pipeline'] = planstr
            crplan['actions'] = actions
            crplan['tokens'] = tokens
            crplan['score'] = plancost
            if crplan['id'] in selected_plans:
                continue
            selected_plans[crplan['id']] = crplan
            retplans.append(crplan)
        for plan in retplans[0:num_pipelines]:
            self.returned_plans[plan['id']] = plan
        return retplans[0:num_pipelines]


    # computes pipelines given the constraints
    def get_plans(self, num_pipelines=10, constraints=[]):
        self.create_pddl_task(num_pipelines, constraints)
        print("Obtaining " + str(num_pipelines) + " plans with constraints " + str(constraints))
        self.run_pddl_planner()
        return self.translate_to_pipelines(num_pipelines)


    # returns a list of selectable constraints that the user can pass to obtain pipelines
    def get_selectable_constraints(self):
        constraints = []
        toklist = list(self.htn.mapping.keys())
        for token in toklist:
            if not isinstance(token, Token):
                continue
            if token.type in self.SELECTABLE_CONSTRAINT_TOKEN_IDS:
                constraints.append(str(token))
        rulelist = list(self.htn.rules_mapping.keys())
        for token in rulelist:
            constraints.append(str(token))
        return list(set(constraints))

    def _nix_syntactic_costs(self):
        for token in self.htn.mapping:
            if not re.match('^[_a-zA-Z]', str(token)):
                self.costs[token] = 0


    # it recomputes costs for actions given all the feedback so far
    def _recompute_costs(self, min_score=None, max_score=None):
        # we have to find the feedback for every single action
        token_feedback = {}
        for pid in self.full_feedback:
            if pid not in self.returned_plans:
                continue
            for token in self.returned_plans[pid]['tokens']:
                if token not in token_feedback:
                    token_feedback[token] = []
                token_feedback[token].extend(self.full_feedback[pid])
        all_scores = [item for sublist in list(token_feedback.values()) for item in sublist]
        minscore = min(all_scores) if min_score is None else min_score
        maxscore = max(all_scores) if max_score is None else max_score
        if minscore == maxscore:
            return
        self.costs.clear()
        for token in token_feedback:
            factor = (statistics.mean(token_feedback[token]) - minscore) / (maxscore - minscore)
            self.costs[token] = math.floor(self.MIN_COST + (1. - factor) * (self.MAX_COST - self.MIN_COST))
        self._nix_syntactic_costs()


    # feedback_scores should be a dictionary of pipeline IDs to a floating point number
    def feedback(self, feedback_scores, min_score=None, max_score=None):
        success_ids = []
        failed_ids = []
        for pid in feedback_scores:
            if pid in self.returned_plans:
                success_ids.append(pid)
                if pid in self.full_feedback:
                    self.full_feedback[pid].append(feedback_scores[pid])
                else:
                    self.full_feedback[pid] = [ feedback_scores[pid] ]
            else:
                failed_ids.append(pid)
        self._recompute_costs(min_score, max_score)
        return (success_ids, failed_ids)


    def writeHTN(self, path, prefix):
        try:
            with open(os.path.join(path, 'domain_' + prefix + '.hpddl'), "w") as f:
                f.write(self.htn_domain)
            with open(os.path.join(path, 'problem_' + prefix + '.hpddl'), "w") as f:
                f.write(self.htn_problem)
        except BaseException as e:
            print('Failed to write HTN domain/problem to disk: ' + str(e))


    def writeLatestTask(self, path, prefix):
        try:
            with open(os.path.join(path, 'domain_' + prefix + '.pddl'), "w") as f:
                if self.last_constraints is not None:
                    f.write(';Constraints: ' + str(self.last_constraints) + '\n')
                f.write(self.last_task['domain'])
            with open(os.path.join(path, 'problem_' + prefix + '.pddl'), "w") as f:
                f.write(self.last_task['problem'])
        except BaseException as e:
            print('Failed to write PDDL domain/problem to disk: ' + str(e))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
 
    parser.add_argument("--grammar-file", help="Grammar to parse", required=True)
    parser.add_argument("--number-of-plans", help="The overall number of pipelines", type=int, default=10)
    parser.add_argument("--constraints", help="Constraints to guide search", nargs='+')

    parser.add_argument("--domain", help="PDDL domain file")
    parser.add_argument("--problem", help="PDDL problem file")

    args = parser.parse_args()

    obj = Grammar2Lale(grammar_file=args.grammar_file, domain_file=args.domain, problem_file=args.problem)
    pipelines = obj.get_plans(num_pipelines=args.number_of_plans, constraints=args.constraints)
    print("Obtained " + str(len(pipelines)) + " pipelines:")
    print([p['pipeline'] for p in pipelines])
