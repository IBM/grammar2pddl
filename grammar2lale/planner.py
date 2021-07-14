#! /usr/bin/env python

import logging
import shutil
import json
import os, sys
from read_plans import PlanManager

# This is the command to check for to determine if a local planner exists
PLANNER_LOCAL_COMMAND="kstar"
# If this variable is defined in the environment, we will use a planner service
PLANNER_URL_ENV_VAR="PLANNER_URL"

def make_call(command, local_folder, enable_output=False):
    if (sys.version_info > (3, 0)):
        import subprocess
    else:
        import subprocess32 as subprocess

    print("Running " + ' '.join(command))

    if enable_output:
        subprocess.check_call(["/bin/bash", "-c", ' '.join(command)], cwd=local_folder)
    else:
        FNULL = open(os.devnull, 'w')
        subprocess.check_call(["/bin/bash", "-c", ' '.join(command)], cwd=local_folder, stdout=FNULL, stderr=subprocess.STDOUT)


def run_planner(task):
    # First, check if local version exists, and run it
    if PLANNER_URL_ENV_VAR in os.environ:
        print('Running planner at ' + os.environ[PLANNER_URL_ENV_VAR])
        return run_planner_service(task, os.environ[PLANNER_URL_ENV_VAR])
    elif local_planner_exists():
        return run_planner_local(task)
    else: raise RuntimeError('Cannot find planner service or local planner');


def local_planner_exists():
    return os.system(PLANNER_LOCAL_COMMAND + " >& /dev/null") != 127


def run_planner_local(task):
    import time

    start = time.time()

    # creating names for local run 
    import uuid
    local_folder_name = os.path.join("/", "tmp", str(uuid.uuid4()).lower())
    if not os.path.exists(local_folder_name):
        os.makedirs(local_folder_name)

    domain_file = os.path.join(local_folder_name, 'domain.pddl')
    problem_file = os.path.join(local_folder_name, 'problem.pddl')
    

    # Writing domain and problem to disk
    with open(domain_file, "w") as d:
        d.write(task['domain'])
    with open(problem_file, "w") as d:
        d.write(task['problem'])

    if os.path.exists(domain_file):
        print("Created domain file in %s" % domain_file )
    if os.path.exists(problem_file):
        print("Created problem file in %s" % problem_file )

    k = str(task['numplans'])
    callstring = get_callstring_local(local_folder_name, domain_file, problem_file, k)

    try:
        make_call(callstring, local_folder=local_folder_name, enable_output=True)
    except:
        return time.time() - start, None

    result = get_result_local(local_folder_name)
    return time.time() - start, result


def get_callstring_local(local_folder_name, domain_file, problem_file, k):
# This is the format for the local command
# PLANNER_LOCAL_COMMAND_FORMAT='kstar {domain} {problem} --search "kstar(blind(),k={k},json_file_to_dump={output}"'
    return ['kstar', domain_file, problem_file, '--search', '"kstar(blind(),k=' + str(k) + ',json_file_to_dump=result.json)"']


def get_result_local(local_folder_name):
    outfile = os.path.join(local_folder_name, 'result.json')
    # reading from existing json file
    with open(outfile, "r") as d:
        return json.load(d)        


def run_planner_service(task, post_url):
    import time
    import requests
    import urllib3
    from requests.packages.urllib3.exceptions import InsecureRequestWarning

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    start = time.time()
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

    resp = requests.post(post_url, json=task, verify=False)

    resp.raise_for_status()
    if resp.status_code != 200:
        logging.error(resp.json())
        return None, None

    return time.time() - start, resp.json()


def plan_to_pipeline(plan, mapping):
    action_strings = []
    for action in plan['actions']:
        # skip all htn_... actions and __forgo_ actions
        if not action.startswith('htn_') and not action.startswith('__forgo_'):
            action_name = action.strip().split()[0].lower()
            if action_name in mapping:
                action_strings.append(mapping[action_name])
            else:
                action_strings.append(action_name)
#               logging.debug("Did not find string for action " + action_name)
    return ' '.join(action_strings)


def plan_actions(plan):
    return [action.strip().split()[0].lower() for action in plan['actions'] if not action.startswith('htn_') and not action.startswith('__forgo_')]


def plan_tokens(plan, mapping):
    return [mapping[act] for act in plan_actions(plan)]

