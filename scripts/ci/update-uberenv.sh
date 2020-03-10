#!/bin/bash

if [[ ! ${1} ]]
then
    echo "ERROR: expecting reference for uberenv repo" >&2
else
    uberenv_ref="${1}"
fi

uberenv_file="scripts/uberenv/uberenv.py"
uberenv_master="https://raw.githubusercontent.com/LLNL/uberenv/${uberenv_ref}/uberenv.py"

curl --fail --output ${uberenv_file} ${uberenv_master}
