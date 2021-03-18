#! /usr/bin/env python

import sys, os, glob
import re
import argparse

import json

_PLAN_INFO_REGEX = re.compile(r"; cost = (\d+) \((unit cost|general cost)\)\n")

class Plan(object):
    def __init__(self, path):
        self._path = path
        self._plan_cost = None
        self._actions = []
        with open(path) as input_file:
            line = None
            for line in input_file:
                if line.strip().startswith(";"):
                    continue
                l = line.strip()
                self._actions.append(l[l.find("(")+1:l.find(")")])
            # line is now the last line
            match = _PLAN_INFO_REGEX.match(line)
            if match:
                self._plan_cost = int(match.group(1)) 

    def __repr__(self):
        return "Plan with cost %s:\n%s" % (self._plan_cost, "\n".join(self._actions))
    def __eq__(self, other):
        if isinstance(other, Plan):
            return ((self._plan_cost == other._plan_cost) and (self._actions == other._actions))
        else:
            return False
    def __ne__(self, other):
        return (not self.__eq__(other))
    def __hash__(self):
        return hash(self.__repr__())
    def toJSON(self):
        return { 'cost' : self._plan_cost, 'actions' : self._actions }


class PlanManager(object):
    def __init__(self, plan_prefix, path):
        self._plan_prefix = plan_prefix
        self._path = path
        self._plans = [Plan(f) for f in glob.glob(os.path.join(path, "%s.*" % self._plan_prefix))]

    def toJSON(self):
        return { 'plans' : [plan.toJSON() for plan in self._plans] }

def main(folder, json):
    with open(json, 'w') as outfile:
        pm = PlanManager("sas_plan", folder)
        json.dump(pm.toJSON(), outfile, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--folder", help="plans folder", required=True)
    parser.add_argument("--json", help="json file name", default="result.json")
    args = parser.parse_args()
    main(args.folder, args.json)
