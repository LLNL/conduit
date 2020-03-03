#!/bin/bash

uberenv_file="scripts/uberenv/uberenv.py"
uberenv_master="https://raw.githubusercontent.com/LLNL/uberenv/master/uberenv.py"

curl --fail --output ${uberenv_file} ${uberenv_master}
