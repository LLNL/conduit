# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

import sys
import json
import os

from optparse import OptionParser

def parse_args():
    "Parses args from command line"
    parser = OptionParser()
    parser.add_option("--args",
                      dest="cmake_args",
                      default=None,
                      help="cmake arguments")
    parser.add_option("--host-config",
                      dest="host_config",
                      default=None,
                      help="path to host config file")
    opts, extras = parser.parse_args()
    # we want a dict b/c
    opts = vars(opts)
    return opts, extras


def write_vscode_settings(settings):
    settings_file = os.path.abspath(".vscode/settings.json")
    print("[creating: {0}]".format(settings_file))
    print("[contents]")
    print(json.dumps(settings, indent=2))
    open(settings_file, "w").write(json.dumps(settings, indent=2))

def gen_vscode_settings(opts):
    settings_comm_file = os.path.abspath(".vscode/settings_common.json")
    res = {}
    if os.path.isfile(settings_comm_file):
        res = json.load(open(settings_comm_file))
    else:
        print("[warning: {0} not found]".format(settings_comm_file)) 

    if not opts["cmake_args"] is None:
        res["cmake.configureArgs"] = [opts["cmake_args"]]
    if not opts["host_config"] is None:
        res["cmake.cacheInit"] = [opts["host_config"]]

    return res

def main():
    opts, extras = parse_args()
    settings = gen_vscode_settings(opts)
    write_vscode_settings(settings)

if __name__ == "__main__":
    main()
