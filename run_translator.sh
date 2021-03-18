#!/bin/bash

if [[ $# -lt 2 ]]; then
    echo "Run as: ./run_translator.sh HTN_DOMAIN_PATH HTN_PROBLEM_PATH"
    exit 1
fi

tmp_dir=$(mktemp -d -t htn2pddl-XXXXXXXXXX)

DOMAIN_FILE=$(basename $1)
PROBLEM_FILE=$(basename $2)

cp $1 $tmp_dir/
cp $2 $tmp_dir/

docker run -v $tmp_dir:/files -e HTN_DOMAIN=/files/$DOMAIN_FILE -e HTN_PROBLEM=/files/$PROBLEM_FILE htn2pddl

cp $tmp_dir/${DOMAIN_FILE}.pddl $(dirname $1)/
cp $tmp_dir/${PROBLEM_FILE}.pddl $(dirname $2)/

rm -rf $tmp_dir