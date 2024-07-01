# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
"""
file: gen_relnotes_variants_from_changelog.py
description:
 Release helper -- converts change log entries into text for:
  - the github release entry
  - the sphinx docs release entry
  - the release entry for the llnl open source news github url

usage:
  python gen_relnotes_variants_from_changelog.py Unreleased
  python gen_relnotes_variants_from_changelog.py 0.5.1

note:
 assumes CHANGELOG.md is at ".."
"""

import os
import sys
import datetime

def proc_changelog_rel_id_line(l):
    active_rel = l.split()[1]
    if active_rel.startswith("["):
        active_rel =active_rel[1:-1]
    return active_rel

def timestamp(t=None,sep="_"):
    """ Creates a timestamp that can easily be included in a filename. """
    if t is None:
        t = datetime.datetime.now()
    #sargs = (t.year,t.month,t.day,t.hour,t.minute,t.second)
    #sbase = "".join(["%04d",sep,"%02d",sep,"%02d",sep,"%02d",sep,"%02d",sep,"%02d"])
    sargs = (t.year,t.month,t.day)
    sbase = "".join(["%04d",sep,"%02d",sep,"%02d"])
    return  sbase % sargs

def conduit_blurb():
    return "[Conduit](https://github.com/LLNL/conduit) provides an intuitive model for describing hierarchical scientific data in C++, C, Fortran, and Python. It is used for data coupling between packages in-core, serialization, and I/O tasks."

def release_date(release_id,src):
    for l in src.split("\n"):
        if l.startswith("## "):
            sub_open = False
            active_rel = proc_changelog_rel_id_line(l)
            if active_rel == release_id:
                # grab the date, it will be the last token
                return l.split()[-1]
    return "Unknown Date"

def gen_llnl_news_entry(release_id,src):
    txt  = "---\n"
    txt += 'title: "Conduit {0} Released"\n'.format(release_id)
    txt += 'categories: release\n'
    txt += "---\n\n"
    txt += conduit_blurb()
    txt += "\n"
    txt += "\n\n"
    txt += "Learn more:\n"
    txt += "- [Release notes](https://github.com/LLNL/conduit/releases/tag/v{0})\n".format(release_id)
    txt += "- [Documentation](hhttp://llnl-conduit.readthedocs.io/)\n"
    txt += "- [GitHub repo](https://github.com/LLNL/conduit)\n"
    return txt

def gen_sphinx_entry(release_id,src):
    txt  = "v{0}\n".format(release_id)
    txt += "---------------------------------\n\n"
    txt += "* Released {0}\n".format(release_date(release_id,src))
    txt += "* `Source Tarball <https://github.com/LLNL/conduit/releases/download/v{0}/conduit-v{0}-src-with-blt.tar.gz>`__\n\n".format(release_id)
    txt += "Highlights\n"
    txt += "++++++++++++++++++++++++++++++++++++\n\n"
    txt += "(Extracted from Conduit's :download:`Changelog <../../../CHANGELOG.md>`)\n\n"
    sub_open = False
    active_rel = ""
    for l in src.split("\n"):
        if l.startswith("## "):
            sub_open = False
            active_rel = proc_changelog_rel_id_line(l)
            #print(active_rel)
            #print("BEGIN RELEASE {0}",l)
        elif l.startswith("### ") and active_rel == release_id:
            sub_open = True
            sub_title = l[3:].strip()
            txt += "\n"+ sub_title +"\n"
            txt += "~" * len(sub_title) + "\n\n"
        elif l.startswith("#### ") and active_rel == release_id:
            sub_open = True
            sub_title = l[4:].strip()
            txt += "\n* **" + sub_title +"**\n\n"
        elif sub_open and active_rel == release_id and l.startswith("-"):
            txt += " * " + sphinx_translate_ticks(l[1:].strip())+"\n"
    return txt

def sphinx_translate_ticks(l):
    # todo, we may need to do a more robust job of this ..
    return l.replace("`","``")

def gen_github_entry(release_id,src):
    txt  = "# {0} Release Highlights\n\n".format(release_id)
    txt += "(adapted from Conduit's [Changelog](https://github.com/LLNL/conduit/blob/develop/CHANGELOG.md) )\n\n"
    txt += "Released {0}\n\n".format(release_date(release_id,src))
    sub_open = False
    active_rel = ""
    for l in src.split("\n"):
        if l.startswith("## "):
            sub_open = False
            active_rel = proc_changelog_rel_id_line(l)
            #print(active_rel)
            #print("BEGIN RELEASE {0}",l)
        elif l.startswith("### ") and active_rel == release_id:
            sub_open = True
            txt += l +"\n"
            print("BEGIN SUB {0}",l)
        elif sub_open and active_rel == release_id:
            txt += l +"\n"
    return txt

def main():
    release_id = "Unreleased"
    if len(sys.argv) > 1:
        release_id = sys.argv[1]
    src = open("../CHANGELOG.md").read()
    print("----GITHUB_ENTRY----BEG----")
    print(gen_github_entry(release_id,src))
    print("----GITHUB_ENTRY----END----")
    print("")
    print("----SPHINX_ENTRY----BEG----")
    print(gen_sphinx_entry(release_id,src))
    print("----SPHINX_ENTRY----END----")
    print("")
    print("----LLNLNEWS_ENTRY----BEG----")
    ntxt = gen_llnl_news_entry(release_id,src)
    print(ntxt)
    print("----LLNLNEWS_ENTRY----END----")
    nfile = timestamp(sep="-") + "-conduit.md"
    print("[saving llnl news entry to: {0}]".format(nfile))
    open(nfile,"w").write(ntxt)
    

if __name__ == "__main__":
    main()

