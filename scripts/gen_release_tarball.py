# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

import sys
import subprocess
import json
from optparse import OptionParser

def parse_args():
    "Parses args from command line"
    parser = OptionParser()
    parser.add_option("--version",
                      dest="ver",
                      default=None,
                      help="version string")
    opts, extras = parser.parse_args()
    # we want a dict b/c
    opts = vars(opts)
    return opts


def main():
    opts = parse_args()
    print(json.dumps(opts,indent=2))
    cmd = "{0} scripts/git_archive_all.py --prefix conduit-v{1} conduit-v{1}-src-with-blt.tar.gz".format(sys.executable,opts["ver"])
    print("[sexe: {0}]".format(cmd))
    subprocess.call(cmd,shell=True)


if __name__ == "__main__":
    main()
