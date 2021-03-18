#! /usr/bin/env python

from __future__ import print_function

import argparse
import logging

from lark import Lark
import json
import sys
import os

class HTN(object): 
    def __init__(self, grammar_file):
        self.lark_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'autoAI-grammar.lark')
        self.grammar_file = grammar_file
        self.actionID = 0
        self.parsed_grammar = self.parse_grammar()
        self.mapping = self.generate_action_mapping()
        self.inverted_mapping = None
        
        self.all_action_names = self.generate_action_names()
        self.all_actions_in_htn = self.generate_actions()

        self.task_names = set()
        self.rules_mapping = {}
        self.allMethods = self.get_all_methods()


    def parse_grammar(self):
        lark_parser = Lark.open(self.lark_file, parser='lalr')    
        with open(self.grammar_file, "r") as grammar_text_file:
            grammar_text = grammar_text_file.read()
            return lark_parser.parse(grammar_text)    

    def get_next_action_name(self):
        name = "primitiveTask" + str(self.actionID)
        self.actionID += 1
        return name

    def generate_action(self, action_name):
        return "(:action   " + action_name + "\n" + \
               "    :parameters ()" + "\n" + \
               "    :task (" + action_name  + ")" + "\n" + \
               "    :precondition ()" + "\n" + \
               "    :effect (and (" + action_name + "))" + "\n" + \
               ")"


    def clean_up_grammar_names(self, name):
        alternative = { "&" : "XX_in_parallel_with_XX",
        "," : "XX_comma_XX", 
        "(":"XX_open_br_XX",
        ")":"XX_close_br_XX",
        ">>":"XX_flows_into_XX",
        "=":"XX_equals_XX"
        }
        if name in alternative:
            return alternative[name]

        return name.replace("'", "YY_quote_YY").replace("(","YY_open_br_YY").replace(")","YY_close_br_YY")

    def generate_action_mapping(self):
        generated_mapping_actions = {}
        found_actions = self.parsed_grammar.find_data('action')
        for found_action in found_actions:
            found_action = found_action.children[0]
            if found_action not in generated_mapping_actions.keys(): #first time we see this
                generated_mapping_actions[found_action] = self.get_next_action_name()
                # Overwriting the generated name
                generated_mapping_actions[found_action] = self.clean_up_grammar_names(found_action)

        generated_mapping_actions[""] = "emptyString"

        return generated_mapping_actions

    def generate_actions(self):
        ret = "\n"
        for action_name in self.mapping.values():
            ret += self.generate_action(action_name) + "\n"
        return ret

    def generate_action_names(self):
        action_names = list(self.mapping.values())
        return "(" + ' '.join(action_names).replace(" ", ") (") + ")"

    def get_subtask(self, found_subtasks):

        found_subtasks_actions = found_subtasks.find_data("actions")
        found_subtasks_task = found_subtasks.find_data("task")
        found_subtasks_pluscase = found_subtasks.find_data("pluscase")

        sub_tasks = ""
        
        #    I assume it is one or the other and just add to found_subtasks
        for found_plus_case in found_subtasks_pluscase:
            #print(found_plus_case.pretty())
            for found_case in found_plus_case.children:
                #print(found_case.pretty())
                pluscase, tasks = self.get_subtask(found_case)
                sub_tasks += tasks
            
            ## Michael: check logic - returns after the first iteration of the external for
            return 1, sub_tasks
        
        for found_tasks in found_subtasks_task :
            found_task = found_tasks.children[0]
            sub_tasks += " " + "(" + found_task + ")"

        for the_actions in found_subtasks_actions :
            #print(the_actions.pretty())
            found_actions = the_actions.children
            for found_action in found_actions:
                found_action = found_action.children[0]
                if found_action in self.mapping.keys():
                    sub_tasks += " " + "(" + self.mapping[found_action] + ")"
                else:
                    logging.debug("Did not find string for action " + found_action)

        return 0, sub_tasks

    def get_HTN_method(self, name, task, subtasks):
        return "(:method " + name + "\n" + \
               "  :parameters () " + "\n" + \
               "  :task (" + task + ")" + "\n" + \
               "  :precondition () " + "\n" + \
               "  :tasks (" + subtasks +")" + "\n" + \
               ")" + "\n"



    def get_all_methods(self):

        methods_in_htn = ""
        all_rules = self.parsed_grammar.find_data('rule_')
        for rule in all_rules:
            leftside = rule.children[0].find_data('task')
            for found_task in leftside:
                # Check that this is executed only once
                found_task = found_task.children[0]
                self.task_names.add(found_task)
                #self.rules_mapping[found_task] = []
            
            task_name = found_task
            self.rules_mapping[task_name] = []
            
            rightside = rule.children[1].find_data('alternative')
            i = 0
            for alternative in rightside:
                i+=1
                method_name = task_name + "_case_" + str(i)
                #print("method +" + str(i))
                subtasks = ""
                plus_one_case_tasks = ""
                new_task_to_add = ""
                #print("alternative\n" + alternative.pretty())
                for found_subtasks in alternative.children:
                    (pluscase, tasks) = self.get_subtask(found_subtasks)
                    if pluscase == 0:
                        subtasks += tasks
                    else:
                        plus_one_case_tasks = tasks
                        new_task_to_add = task_name + "Plus"
                        subtasks += "(" + new_task_to_add + ")"
                        self.task_names.add(new_task_to_add)
            
                methods_in_htn += "\n" + self.get_HTN_method(method_name, task_name, subtasks) 
                self.rules_mapping[task_name].append(method_name)
                if(plus_one_case_tasks !=""):
                    base_task_name = task_name + "Base"
                    method_name_empty = task_name + "BaseEmpty"
                    method_name_right_recursive = task_name + "BaseRightRecursive"
                    
                    plus_one_case_tasks += "(" + base_task_name + ")"
                    methods_in_htn += "\n" + self.get_HTN_method(new_task_to_add, new_task_to_add, plus_one_case_tasks) 
                    self.rules_mapping[task_name].append(new_task_to_add)
                    methods_in_htn += "\n" + self.get_HTN_method(method_name_empty, base_task_name, "(emptyString)") 
                    self.rules_mapping[task_name].append(method_name_empty)
                    methods_in_htn += "\n" + self.get_HTN_method(method_name_right_recursive, base_task_name, plus_one_case_tasks)  
                    self.rules_mapping[task_name].append(method_name_right_recursive)
                    self.task_names.add(base_task_name)
                    
        return methods_in_htn

    def generate_predicates(self):
        return "(:predicates %s)" % self.all_action_names

    def generate_tasks(self):
        task_predicates = ["(" + name + ")" for name in self.task_names]
        return "(:tasks %s \n %s)" % (self.all_action_names, " ".join(task_predicates))

    def generate_methods(self):
        return self.allMethods

    def generate_domain(self):
        return "(define (domain datascience)" + "\n" + \
               "  (:requirements :strips)" + "\n" + \
               "  (:types OBJ)" + "\n" + \
               "  " + self.generate_predicates() + "\n" + \
               "  " + self.generate_tasks() + "\n" + \
               "  " + self.generate_methods() + "\n" + \
               "  " + self.generate_actions() + "\n" + \
               ")"

    def generate_problem(self):
        return """(define
        (problem datascience_grammar_to_htn)
        (:domain datascience)
        (:objects 
        )
        (:init
        )
        (:goal (and ))
    )"""

    def invert_mapping(self):
        if self.inverted_mapping is None:
            self.inverted_mapping = {}
            for k, v in self.mapping.items():
                if v in self.inverted_mapping:
                    logging.debug('Mapping has double value: ' + v)
                else:
                    self.inverted_mapping[v.lower()] = k
        return self.inverted_mapping




