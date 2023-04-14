# Copyright 2013-2022 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import glob
import json
import os
import platform
import re
import subprocess
import sys
from shutil import copy
from typing import Dict, List

import llnl.util.tty as tty
from llnl.util.filesystem import (
    is_nonsymlink_exe_with_shebang,
    path_contains_subdirectory,
)
from llnl.util.lang import dedupe, match_predicate

from spack.build_environment import dso_suffix, stat_suffix
from spack.package import *
from spack.util.environment import is_system_path
from spack.util.prefix import Prefix

is_windows = sys.platform == "win32"


class Python(Package):
    """The Python programming language."""

    homepage = "https://www.python.org/"
    url = "https://www.python.org/ftp/python/3.8.0/Python-3.8.0.tgz"
    list_url = "https://www.python.org/ftp/python/"
    list_depth = 1

    maintainers = ["adamjstewart", "skosukhin", "scheibelp", "pradyunsg"]

    phases = ["configure", "build", "install"]

    #: phase
    install_targets = ["install"]
    build_targets = []  # type: List[str]

    version("3.11.2", sha256="2411c74bda5bbcfcddaf4531f66d1adc73f247f529aee981b029513aefdbf849")
    version("3.11.1", sha256="baed518e26b337d4d8105679caf68c5c32630d702614fc174e98cb95c46bdfa4")
    version("3.11.0", sha256="64424e96e2457abbac899b90f9530985b51eef2905951febd935f0e73414caeb")
    version("3.10.10", sha256="fba64559dde21ebdc953e4565e731573bb61159de8e4d4cedee70fb1196f610d")
    version("3.10.9", sha256="4ccd7e46c8898f4c7862910a1703aa0e63525913a519abb2f55e26220a914d88")
    version("3.10.8", sha256="f400c3fb394b8bef1292f6dc1292c5fadc3533039a5bc0c3e885f3e16738029a")
    version("3.10.7", sha256="1b2e4e2df697c52d36731666979e648beeda5941d0f95740aafbf4163e5cc126")
    version("3.10.6", sha256="848cb06a5caa85da5c45bd7a9221bb821e33fc2bdcba088c127c58fad44e6343")
    version("3.10.5", sha256="18f57182a2de3b0be76dfc39fdcfd28156bb6dd23e5f08696f7492e9e3d0bf2d")
    version("3.10.4", sha256="f3bcc65b1d5f1dc78675c746c98fcee823c038168fc629c5935b044d0911ad28")
    version("3.10.3", sha256="5a3b029bad70ba2a019ebff08a65060a8b9b542ffc1a83c697f1449ecca9813b")
    version("3.10.2", sha256="3c0ede893011319f9b0a56b44953a3d52c7abf9657c23fb4bc9ced93b86e9c97")
    version("3.10.1", sha256="b76117670e7c5064344b9c138e141a377e686b9063f3a8a620ff674fa8ec90d3")
    version("3.10.0", sha256="c4e0cbad57c90690cb813fb4663ef670b4d0f587d8171e2c42bd4c9245bd2758")
    version(
        "3.9.13",
        sha256="829b0d26072a44689a6b0810f5b4a3933ee2a0b8a4bfc99d7c5893ffd4f97c44",
        preferred=True,
    )
    version("3.9.12", sha256="70e08462ebf265012bd2be88a63d2149d880c73e53f1712b7bbbe93750560ae8")
    version("3.9.11", sha256="3442400072f582ac2f0df30895558f08883b416c8c7877ea55d40d00d8a93112")
    version("3.9.10", sha256="1aa9c0702edbae8f6a2c95f70a49da8420aaa76b7889d3419c186bfc8c0e571e")
    version("3.9.9", sha256="2cc7b67c1f3f66c571acc42479cdf691d8ed6b47bee12c9b68430413a17a44ea")
    version("3.9.8", sha256="7447fb8bb270942d620dd24faa7814b1383b61fa99029a240025fd81c1db8283")
    version("3.9.7", sha256="a838d3f9360d157040142b715db34f0218e535333696a5569dc6f854604eb9d1")
    version("3.9.6", sha256="d0a35182e19e416fc8eae25a3dcd4d02d4997333e4ad1f2eee6010aadc3fe866")
    version("3.9.5", sha256="e0fbd5b6e1ee242524430dee3c91baf4cbbaba4a72dd1674b90fda87b713c7ab")
    version("3.9.4", sha256="66c4de16daa74a825cf9da9ddae1fe020b72c3854b73b1762011cc33f9e4592f")
    version("3.9.3", sha256="3afeb61a45b5a2e6f1c0f621bd8cf925a4ff406099fdb3d8c97b993a5f43d048")
    version("3.9.2", sha256="7899e8a6f7946748830d66739f2d8f2b30214dad956e56b9ba216b3de5581519")
    version("3.9.1", sha256="29cb91ba038346da0bd9ab84a0a55a845d872c341a4da6879f462e94c741f117")
    version("3.9.0", sha256="df796b2dc8ef085edae2597a41c1c0a63625ebd92487adaef2fed22b567873e8")
    version("3.8.13", sha256="903b92d76354366b1d9c4434d0c81643345cef87c1600adfa36095d7b00eede4")
    version("3.8.12", sha256="316aa33f3b7707d041e73f246efedb297a70898c4b91f127f66dc8d80c596f1a")
    version("3.8.11", sha256="b77464ea80cec14581b86aeb7fb2ff02830e0abc7bcdc752b7b4bdfcd8f3e393")
    version("3.8.10", sha256="b37ac74d2cbad2590e7cd0dd2b3826c29afe89a734090a87bf8c03c45066cb65")
    version("3.8.9", sha256="9779ec1df000bf86914cdd40860b88da56c1e61db59d37784beca14a259ac9e9")
    version("3.8.8", sha256="76c0763f048e4f9b861d24da76b7dd5c7a3ba7ec086f40caedeea359263276f7")
    version("3.8.7", sha256="20e5a04262f0af2eb9c19240d7ec368f385788bba2d8dfba7e74b20bab4d2bac")
    version("3.8.6", sha256="313562ee9986dc369cd678011bdfd9800ef62fbf7b1496228a18f86b36428c21")
    version("3.8.5", sha256="015115023c382eb6ab83d512762fe3c5502fa0c6c52ffebc4831c4e1a06ffc49")
    version("3.8.4", sha256="32c4d9817ef11793da4d0d95b3191c4db81d2e45544614e8449255ca9ae3cc18")
    version("3.8.3", sha256="6af6d4d2e010f9655518d0fc6738c7ff7069f10a4d2fbd55509e467f092a8b90")
    version("3.8.2", sha256="e634a7a74776c2b89516b2e013dda1728c89c8149b9863b8cea21946daf9d561")
    version("3.8.1", sha256="c7cfa39a43b994621b245e029769e9126caa2a93571cee2e743b213cceac35fb")
    version("3.8.0", sha256="f1069ad3cae8e7ec467aa98a6565a62a48ef196cb8f1455a245a08db5e1792df")
    version("3.7.13", sha256="e405417f50984bc5870c7e7a9f9aeb93e9d270f5ac67f667a0cd3a09439682b5")
    version("3.7.12", sha256="33b4daaf831be19219659466d12645f87ecec6eb21d4d9f9711018a7b66cce46")
    version("3.7.11", sha256="b4fba32182e16485d0a6022ba83c9251e6a1c14676ec243a9a07d3722cd4661a")
    version("3.7.10", sha256="c9649ad84dc3a434c8637df6963100b2e5608697f9ba56d82e3809e4148e0975")
    version("3.7.9", sha256="39b018bc7d8a165e59aa827d9ae45c45901739b0bbb13721e4f973f3521c166a")
    version("3.7.8", sha256="0e25835614dc221e3ecea5831b38fa90788b5389b99b675a751414c858789ab0")
    version("3.7.7", sha256="8c8be91cd2648a1a0c251f04ea0bb4c2a5570feb9c45eaaa2241c785585b475a")
    version("3.7.6", sha256="aeee681c235ad336af116f08ab6563361a0c81c537072c1b309d6e4050aa2114")
    version("3.7.5", sha256="8ecc681ea0600bbfb366f2b173f727b205bb825d93d2f0b286bc4e58d37693da")
    version("3.7.4", sha256="d63e63e14e6d29e17490abbe6f7d17afb3db182dbd801229f14e55f4157c4ba3")
    version("3.7.3", sha256="d62e3015f2f89c970ac52343976b406694931742fbde2fed8d1ce8ebb4e1f8ff")
    version("3.7.2", sha256="f09d83c773b9cc72421abba2c317e4e6e05d919f9bcf34468e192b6a6c8e328d")
    version("3.7.1", sha256="36c1b81ac29d0f8341f727ef40864d99d8206897be96be73dc34d4739c9c9f06")
    version("3.7.0", sha256="85bb9feb6863e04fb1700b018d9d42d1caac178559ffa453d7e6a436e259fd0d")
    version(
        "3.6.15",
        sha256="54570b7e339e2cfd72b29c7e2fdb47c0b7b18b7412e61de5b463fc087c13b043",
        deprecated=True,
    )
    version(
        "3.6.14",
        sha256="70064897bc434d6eae8bcc3e5678f282b5ea776d60e695da548a1219ccfd27a5",
        deprecated=True,
    )
    version(
        "3.6.13",
        sha256="614950d3d54f6e78dac651b49c64cfe2ceefea5af3aff3371a9e4b27a53b2669",
        deprecated=True,
    )
    version(
        "3.6.12",
        sha256="12dddbe52385a0f702fb8071e12dcc6b3cb2dde07cd8db3ed60e90d90ab78693",
        deprecated=True,
    )
    version(
        "3.6.11",
        sha256="96621902f89746fffc22f39749c07da7c2917b232e72352e6837d41850f7b90c",
        deprecated=True,
    )
    version(
        "3.6.10",
        sha256="7034dd7cba98d4f94c74f9edd7345bac71c8814c41672c64d9044fa2f96f334d",
        deprecated=True,
    )
    version(
        "3.6.9",
        sha256="47fc92a1dcb946b9ed0abc311d3767b7215c54e655b17fd1d3f9b538195525aa",
        deprecated=True,
    )
    version(
        "3.6.8",
        sha256="7f5b1f08b3b0a595387ef6c64c85b1b13b38abef0dd871835ee923262e4f32f0",
        deprecated=True,
    )
    version(
        "3.6.7",
        sha256="b7c36f7ed8f7143b2c46153b7332db2227669f583ea0cce753facf549d1a4239",
        deprecated=True,
    )
    version(
        "3.6.6",
        sha256="7d56dadf6c7d92a238702389e80cfe66fbfae73e584189ed6f89c75bbf3eda58",
        deprecated=True,
    )
    version(
        "3.6.5",
        sha256="53a3e17d77cd15c5230192b6a8c1e031c07cd9f34a2f089a731c6f6bd343d5c6",
        deprecated=True,
    )
    version(
        "3.6.4",
        sha256="7dc453e1a93c083388eb1a23a256862407f8234a96dc4fae0fc7682020227486",
        deprecated=True,
    )
    version(
        "3.6.3",
        sha256="ab6193af1921b30f587b302fe385268510e80187ca83ca82d2bfe7ab544c6f91",
        deprecated=True,
    )
    version(
        "3.6.2",
        sha256="7919489310a5f17f7acbab64d731e46dca0702874840dadce8bd4b2b3b8e7a82",
        deprecated=True,
    )
    version(
        "3.6.1",
        sha256="aa50b0143df7c89ce91be020fe41382613a817354b33acdc6641b44f8ced3828",
        deprecated=True,
    )
    version(
        "3.6.0",
        sha256="aa472515800d25a3739833f76ca3735d9f4b2fe77c3cb21f69275e0cce30cb2b",
        deprecated=True,
    )
    version(
        "3.5.10",
        sha256="3496a0daf51913718a6f10e3eda51fa43634cb6151cb096f312d48bdbeff7d3a",
        deprecated=True,
    )
    version(
        "3.5.9",
        sha256="67a1d4fc6e4540d6a092cadc488e533afa961b3c9becc74dc3d6b55cb56e0cc1",
        deprecated=True,
    )
    version(
        "3.5.8",
        sha256="18c88dfd260147bc7247e6356010e5d4916dfbfc480f6434917f88e61228177a",
        deprecated=True,
    )
    version(
        "3.5.7",
        sha256="542d94920a2a06a471a73b51614805ad65366af98145b0369bc374cf248b521b",
        deprecated=True,
    )
    version(
        "3.5.6",
        sha256="30d2ff093988e74283e1abfee823292c6b59590796b9827e95ba4940b27d26f8",
        deprecated=True,
    )
    version(
        "3.5.5",
        sha256="2f988db33913dcef17552fd1447b41afb89dbc26e3cdfc068ea6c62013a3a2a5",
        deprecated=True,
    )
    version(
        "3.5.4",
        sha256="6ed87a8b6c758cc3299a8b433e8a9a9122054ad5bc8aad43299cff3a53d8ca44",
        deprecated=True,
    )
    version(
        "3.5.3",
        sha256="d8890b84d773cd7059e597dbefa510340de8336ec9b9e9032bf030f19291565a",
        deprecated=True,
    )
    version(
        "3.5.2",
        sha256="1524b840e42cf3b909e8f8df67c1724012c7dc7f9d076d4feef2d3eff031e8a0",
        deprecated=True,
    )
    version(
        "3.5.1",
        sha256="687e067d9f391da645423c7eda8205bae9d35edc0c76ef5218dcbe4cc770d0d7",
        deprecated=True,
    )
    version(
        "3.5.0",
        sha256="584e3d5a02692ca52fce505e68ecd77248a6f2c99adf9db144a39087336b0fe0",
        deprecated=True,
    )
    version(
        "3.4.10",
        sha256="217757699249ab432571b381386d441e12b433100ab5f908051fcb7cced2539d",
        deprecated=True,
    )
    version(
        "3.4.3",
        sha256="8b743f56e9e50bf0923b9e9c45dd927c071d7aa56cd46569d8818add8cf01147",
        deprecated=True,
    )
    version(
        "3.3.6",
        sha256="0a58ad1f1def4ecc90b18b0c410a3a0e1a48cf7692c75d1f83d0af080e5d2034",
        deprecated=True,
    )
    version(
        "3.2.6",
        sha256="fc1e41296e29d476f696303acae293ae7a2310f0f9d0d637905e722a3f16163e",
        deprecated=True,
    )
    version(
        "3.1.5",
        sha256="d12dae6d06f52ef6bf1271db4d5b4d14b5dd39813e324314e72b648ef1bc0103",
        deprecated=True,
    )
    version(
        "2.7.18",
        sha256="da3080e3b488f648a3d7a4560ddee895284c3380b11d6de75edb986526b9a814",
        deprecated=True,
    )
    version(
        "2.7.17",
        sha256="f22059d09cdf9625e0a7284d24a13062044f5bf59d93a7f3382190dfa94cecde",
        deprecated=True,
    )
    version(
        "2.7.16",
        sha256="01da813a3600876f03f46db11cc5c408175e99f03af2ba942ef324389a83bad5",
        deprecated=True,
    )
    version(
        "2.7.15",
        sha256="18617d1f15a380a919d517630a9cd85ce17ea602f9bbdc58ddc672df4b0239db",
        deprecated=True,
    )
    version(
        "2.7.14",
        sha256="304c9b202ea6fbd0a4a8e0ad3733715fbd4749f2204a9173a58ec53c32ea73e8",
        deprecated=True,
    )
    version(
        "2.7.13",
        sha256="a4f05a0720ce0fd92626f0278b6b433eee9a6173ddf2bced7957dfb599a5ece1",
        deprecated=True,
    )
    version(
        "2.7.12",
        sha256="3cb522d17463dfa69a155ab18cffa399b358c966c0363d6c8b5b3bf1384da4b6",
        deprecated=True,
    )
    version(
        "2.7.11",
        sha256="82929b96fd6afc8da838b149107078c02fa1744b7e60999a8babbc0d3fa86fc6",
        deprecated=True,
    )
    version(
        "2.7.10",
        sha256="eda8ce6eec03e74991abb5384170e7c65fcd7522e409b8e83d7e6372add0f12a",
        deprecated=True,
    )
    version(
        "2.7.9",
        sha256="c8bba33e66ac3201dabdc556f0ea7cfe6ac11946ec32d357c4c6f9b018c12c5b",
        deprecated=True,
    )
    version(
        "2.7.8",
        sha256="74d70b914da4487aa1d97222b29e9554d042f825f26cb2b93abd20fdda56b557",
        deprecated=True,
    )

    extendable = True

    # Variants to avoid cyclical dependencies for concretizer
    variant("libxml2", default=True, description="Use a gettext library build with libxml2")

    variant(
        "debug", default=False, description="debug build with extra checks (this is high overhead)"
    )

    # --enable-shared is known to cause problems for some users on macOS
    # This is a problem for Python 2.7 only, not Python3
    # See http://bugs.python.org/issue29846
    variant("shared", default=True, description="Enable shared libraries")
    # From https://docs.python.org/2/c-api/unicode.html: Python's default
    # builds use a 16-bit type for Py_UNICODE and store Unicode values
    # internally as UCS2. It is also possible to build a UCS4 version of Python
    # (most recent Linux distributions come with UCS4 builds of Python).  These
    # builds then use a 32-bit type for Py_UNICODE and store Unicode data
    # internally as UCS4. Note that UCS2 and UCS4 Python builds are not binary
    # compatible.
    variant("ucs4", default=False, description="Enable UCS4 (wide) unicode strings")
    variant("pic", default=True, description="Produce position-independent code (for shared libs)")
    variant(
        "optimizations",
        default=False,
        description="Enable expensive build-time optimizations, if available",
    )
    # See https://legacy.python.org/dev/peps/pep-0394/
    variant(
        "pythoncmd",
        default=not is_windows,
        description="Symlink 'python3' executable to 'python' " "(not PEP 394 compliant)",
    )

    # Optional Python modules
    variant("readline", default=not is_windows, description="Build readline module")
    variant("ssl", default=True, description="Build ssl module")
    variant("sqlite3", default=True, description="Build sqlite3 module")
    variant("dbm", default=True, description="Build dbm module")
    variant("nis", default=False, description="Build nis module")
    variant("zlib", default=True, description="Build zlib module")
    variant("bz2", default=True, description="Build bz2 module")
    variant("lzma", default=True, description="Build lzma module", when="@3.3:")
    variant("pyexpat", default=True, description="Build pyexpat module")
    variant("ctypes", default=True, description="Build ctypes module")
    variant("tkinter", default=False, description="Build tkinter module")
    variant("uuid", default=True, description="Build uuid module")
    variant("tix", default=False, description="Build Tix module")

    if not is_windows:
        depends_on("pkgconfig@0.9.0:", type="build")
        depends_on("gettext +libxml2", when="+libxml2")
        depends_on("gettext ~libxml2", when="~libxml2")

        # Optional dependencies
        # See detect_modules() in setup.py for details
        depends_on("readline", when="+readline")
        depends_on("ncurses", when="+readline")
        depends_on("openssl", when="+ssl")
        # https://raw.githubusercontent.com/python/cpython/84471935ed2f62b8c5758fd544c7d37076fe0fa5/Misc/NEWS
        # https://docs.python.org/3.5/whatsnew/changelog.html#python-3-5-4rc1
        depends_on("openssl@:1.0.2z", when="@:2.7.13,3.0.0:3.5.2+ssl")
        depends_on(
            "openssl@1.0.2:", when="@3.7:+ssl"
        )  # https://docs.python.org/3/whatsnew/3.7.html#build-changes
        depends_on(
            "openssl@1.1.1:", when="@3.10:+ssl"
        )  # https://docs.python.org/3.10/whatsnew/3.10.html#build-changes
        depends_on("sqlite@3.0.8:", when="@:3.9+sqlite3")
        depends_on(
            "sqlite@3.7.15:", when="@3.10:+sqlite3"
        )  # https://docs.python.org/3.10/whatsnew/3.10.html#build-changes
        depends_on("gdbm", when="+dbm")  # alternatively ndbm or berkeley-db
        depends_on("libnsl", when="+nis")
        depends_on("zlib@1.1.3:", when="+zlib")
        depends_on("bzip2", when="+bz2")
        depends_on("xz", when="@3.3:+lzma")
        depends_on("expat", when="+pyexpat")
        depends_on("libffi", when="+ctypes")
        depends_on("tk", when="+tkinter")
        depends_on("tcl", when="+tkinter")
        depends_on("uuid", when="+uuid")
        depends_on("tix", when="+tix")

    # Python needs to be patched to build extensions w/ mixed C/C++ code:
    # https://github.com/NixOS/nixpkgs/pull/19585/files
    # https://bugs.python.org/issue1222585
    #
    # NOTE: This patch puts Spack's default Python installation out of
    # sync with standard Python installs. If you're using such an
    # installation as an external and encountering build issues with mixed
    # C/C++ modules, consider installing a Spack-managed Python with
    # this patch instead. For more information, see:
    # https://github.com/spack/spack/pull/16856
    patch("python-2.7.8-distutils-C++.patch", when="@2.7.8:2.7.16")
    patch("python-2.7.17+-distutils-C++.patch", when="@2.7.17:2.7.18")
    patch("python-2.7.17+-distutils-C++-fixup.patch", when="@2.7.17:2.7.18")
    patch("python-3.6.8-distutils-C++.patch", when="@3.6.8,3.7.2")
    patch("python-3.7.3-distutils-C++.patch", when="@3.7.3")
    patch("python-3.7.4+-distutils-C++.patch", when="@3.7.4:")
    patch("python-3.7.4+-distutils-C++-testsuite.patch", when="@3.7.4:3.10.6")
    patch("cpython-windows-externals.patch", when="@:3.9.6 platform=windows")
    patch("tkinter.patch", when="@:2.8,3.3:3.7 platform=darwin")
    # Patch the setup script to deny that tcl/x11 exists rather than allowing
    # autodetection of (possibly broken) system components
    patch("tkinter-3.8.patch", when="@3.8:3.9 ~tkinter")
    patch("tkinter-3.10.patch", when="@3.10: ~tkinter")

    # Ensure that distutils chooses correct compiler option for RPATH on cray:
    patch("cray-rpath-2.3.patch", when="@2.3:3.0.1 platform=cray")
    patch("cray-rpath-3.1.patch", when="@3.1:3  platform=cray")

    # Ensure that distutils chooses correct compiler option for RPATH on fj:
    patch("fj-rpath-2.3.patch", when="@2.3:3.0.1 %fj")
    patch("fj-rpath-3.1.patch", when="@3.1:3.9.7,3.10.0  %fj")
    patch("fj-rpath-3.9.patch", when="@3.9.8:3.9,3.10.1:  %fj")

    # Fixes an alignment problem with more aggressive optimization in gcc8
    # https://github.com/python/cpython/commit/0b91f8a668201fc58fa732b8acc496caedfdbae0
    patch("gcc-8-2.7.14.patch", when="@2.7.14 %gcc@8:")

    # Fixes build with the Intel compilers
    # https://github.com/python/cpython/pull/16717
    patch("intel-3.6.7.patch", when="@3.6.7:3.6.8,3.7.1:3.7.5 %intel")

    # CPython tries to build an Objective-C file with GCC's C frontend
    # https://github.com/spack/spack/pull/16222
    # https://github.com/python/cpython/pull/13306
    conflicts(
        "%gcc platform=darwin",
        msg="CPython does not compile with GCC on macOS yet, use clang. "
        "See: https://github.com/python/cpython/pull/13306",
    )
    # For more information refer to this bug report:
    # https://bugs.python.org/issue29712
    conflicts(
        "@:2.8 +shared",
        when="+optimizations",
        msg="+optimizations is incompatible with +shared in python@2.X",
    )
    conflicts("+tix", when="~tkinter", msg="python+tix requires python+tix+tkinter")
    conflicts("%nvhpc")
    conflicts(
        "@:2.7",
        when="platform=darwin target=aarch64:",
        msg="Python 2.7 is too old for Apple Silicon",
    )

    # Used to cache various attributes that are expensive to compute
    _config_vars = {}  # type: Dict[str, Dict[str, str]]

    # An in-source build with --enable-optimizations fails for python@3.X
    build_directory = "spack-build"

    executables = [r"^python[\d.]*[mw]?$"]

    @classmethod
    def determine_version(cls, exe):
        # Newer versions of Python support `--version`,
        # but older versions only support `-V`
        # Python 2 sends to STDERR, while Python 3 sends to STDOUT
        # Output looks like:
        #   Python 3.7.7
        # On pre-production Ubuntu, this is also possible:
        #   Python 3.10.2+
        output = Executable(exe)("-V", output=str, error=str)
        match = re.search(r"Python\s+([A-Za-z0-9_.-]+)", output)
        return match.group(1) if match else None

    @classmethod
    def determine_variants(cls, exes, version_str):
        python = Executable(exes[0])

        variants = ""
        for exe in exes:
            if os.path.basename(exe) == "python":
                variants += "+pythoncmd"
                break
        else:
            variants += "~pythoncmd"

        for module in ["readline", "sqlite3", "dbm", "nis", "zlib", "bz2", "ctypes", "uuid"]:
            try:
                python("-c", "import " + module, error=os.devnull)
                variants += "+" + module
            except ProcessError:
                variants += "~" + module

        # Some variants enable multiple modules
        try:
            python("-c", "import ssl", error=os.devnull)
            python("-c", "import hashlib", error=os.devnull)
            variants += "+ssl"
        except ProcessError:
            variants += "~ssl"

        try:
            python("-c", "import xml.parsers.expat", error=os.devnull)
            python("-c", "import xml.etree.ElementTree", error=os.devnull)
            variants += "+pyexpat"
        except ProcessError:
            variants += "~pyexpat"

        # Some modules are version-dependent
        if Version(version_str) >= Version("3.3"):
            try:
                python("-c", "import lzma", error=os.devnull)
                variants += "+lzma"
            except ProcessError:
                variants += "~lzma"

        if Version(version_str) >= Version("3"):
            try:
                python("-c", "import tkinter", error=os.devnull)
                variants += "+tkinter"
            except ProcessError:
                variants += "~tkinter"

            try:
                python("-c", "import tkinter.tix", error=os.devnull)
                variants += "+tix"
            except ProcessError:
                variants += "~tix"
        else:
            try:
                python("-c", "import Tkinter", error=os.devnull)
                variants += "+tkinter"
            except ProcessError:
                variants += "~tkinter"

            try:
                python("-c", "import Tix", error=os.devnull)
                variants += "+tix"
            except ProcessError:
                variants += "~tix"

        return variants

    def url_for_version(self, version):
        url = "https://www.python.org/ftp/python/{0}/Python-{1}.tgz"
        return url.format(re.split("[a-z]", str(version))[0], version)

    # TODO: Ideally, these patches would be applied as separate '@run_before'
    # functions enabled via '@when', but these two decorators don't work
    # when used together. See: https://github.com/spack/spack/issues/12736
    def patch(self):
        # NOTE: Python's default installation procedure makes it possible for a
        # user's local configurations to change the Spack installation.  In
        # order to prevent this behavior for a full installation, we must
        # modify the installation script so that it ignores user files.
        if self.spec.satisfies("@2.7:2.8,3.4:"):
            ff = FileFilter("Makefile.pre.in")
            ff.filter(
                r"^(.*)setup\.py(.*)((build)|(install))(.*)$", r"\1setup.py\2 --no-user-cfg \3\6"
            )

        # NOTE: Older versions of Python do not support the '--with-openssl'
        # configuration option, so the installation's module setup file needs
        # to be modified directly in order to point to the correct SSL path.
        # See: https://stackoverflow.com/a/5939170
        if self.spec.satisfies("@:3.6+ssl"):
            ff = FileFilter(join_path("Modules", "Setup.dist"))
            ff.filter(r"^#(((SSL=)|(_ssl))(.*))$", r"\1")
            ff.filter(r"^#((.*)(\$\(SSL\))(.*))$", r"\1")
            ff.filter(r"^SSL=(.*)$", r"SSL={0}".format(self.spec["openssl"].prefix))
        # Because Python uses compiler system paths during install, it's
        # possible to pick up a system OpenSSL when building 'python~ssl'.
        # To avoid this scenario, we disable the 'ssl' module with patching.
        elif self.spec.satisfies("@:3.6~ssl"):
            ff = FileFilter("setup.py")
            ff.filter(r"^(\s+(ssl_((incs)|(libs)))\s+=\s+)(.*)$", r"\1 None and \6")
            ff.filter(r"^(\s+(opensslv_h)\s+=\s+)(.*)$", r"\1 None and \3")

    def setup_build_environment(self, env):
        spec = self.spec

        # TODO: The '--no-user-cfg' option for Python installation is only in
        # Python v2.7 and v3.4+ (see https://bugs.python.org/issue1180) and
        # adding support for ignoring user configuration will require
        # significant changes to this package for other Python versions.
        if not spec.satisfies("@2.7:2.8,3.4:"):
            tty.warn(
                (
                    "Python v{0} may not install properly if Python "
                    "user configurations are present."
                ).format(self.version)
            )

        # TODO: Python has incomplete support for Python modules with mixed
        # C/C++ source, and patches are required to enable building for these
        # modules. All Python versions without a viable patch are installed
        # with a warning message about this potentially erroneous behavior.
        if not spec.satisfies("@2.7.8:2.7.18,3.6.8,3.7.2:"):
            tty.warn(
                (
                    'Python v{0} does not have the C++ "distutils" patch; '
                    "errors may occur when installing Python modules w/ "
                    "mixed C/C++ source files."
                ).format(self.version)
            )

        env.unset("PYTHONPATH")
        env.unset("PYTHONHOME")

        # avoid build error on fugaku
        if spec.satisfies("@3.10.0 arch=linux-rhel8-a64fx"):
            if spec.satisfies("%gcc") or spec.satisfies("%fj"):
                env.unset("LC_ALL")

    def flag_handler(self, name, flags):
        # python 3.8 requires -fwrapv when compiled with intel
        if self.spec.satisfies("@3.8: %intel"):
            if name == "cflags":
                flags.append("-fwrapv")

        # Fix for following issues for python with aocc%3.2.0:
        # https://github.com/spack/spack/issues/29115
        # https://github.com/spack/spack/pull/28708
        if self.spec.satisfies("%aocc@3.2.0", strict=True):
            if name == "cflags":
                flags.extend(["-mllvm", "-disable-indvar-simplify=true"])

        # allow flags to be passed through compiler wrapper
        return (flags, None, None)

    @property
    def plat_arch(self):
        """
        String referencing platform architecture
        filtered through Python's Windows build file
        architecture support map

        Note: This function really only makes
        sense to use on Windows, could be overridden to
        cross compile however.
        """

        arch_map = {"AMD64": "x64", "x86": "Win32", "IA64": "Win32", "EM64T": "Win32"}
        arch = platform.machine()
        if arch in arch_map:
            arch = arch_map[arch]
        return arch

    @property
    def win_build_params(self):
        """
        Arguments must be passed to the Python build batch script
        in order to configure it to spec and system.
        A number of these toggle optional MSBuild Projects
        directly corresponding to the python support of the same
        name.
        """
        args = []
        args.append("-p %s" % self.plat_arch)
        if self.spec.satisfies("+debug"):
            args.append("-d")
        if self.spec.satisfies("~ctypes"):
            args.append("--no-ctypes")
        if self.spec.satisfies("~ssl"):
            args.append("--no-ssl")
        if self.spec.satisfies("~tkinter"):
            args.append("--no-tkinter")
        return args

    def win_installer(self, prefix):
        """
        Python on Windows does not export an install target
        so we must handcraft one here. This structure
        directly mimics the install tree of the Python
        Installer on Windows.

        Parameters:
            prefix (str): Install prefix for package
        """
        proj_root = self.stage.source_path
        pcbuild_root = os.path.join(proj_root, "PCbuild")
        build_root = os.path.join(pcbuild_root, platform.machine().lower())
        include_dir = os.path.join(proj_root, "Include")
        copy_tree(include_dir, prefix.include)
        doc_dir = os.path.join(proj_root, "Doc")
        copy_tree(doc_dir, prefix.Doc)
        tools_dir = os.path.join(proj_root, "Tools")
        copy_tree(tools_dir, prefix.Tools)
        lib_dir = os.path.join(proj_root, "Lib")
        copy_tree(lib_dir, prefix.Lib)
        pyconfig = os.path.join(proj_root, "PC", "pyconfig.h")
        copy(pyconfig, prefix.include)
        shared_libraries = []
        shared_libraries.extend(glob.glob("%s\\*.exe" % build_root))
        shared_libraries.extend(glob.glob("%s\\*.dll" % build_root))
        shared_libraries.extend(glob.glob("%s\\*.pyd" % build_root))
        os.makedirs(prefix.DLLs)
        for lib in shared_libraries:
            file_name = os.path.basename(lib)
            if (
                file_name.endswith(".exe")
                or (file_name.endswith(".dll") and "python" in file_name)
                or "vcruntime" in file_name
            ):
                copy(lib, prefix)
            else:
                copy(lib, prefix.DLLs)
        static_libraries = glob.glob("%s\\*.lib")
        for lib in static_libraries:
            copy(lib, prefix.libs)

    def configure_args(self):
        spec = self.spec
        config_args = []
        cflags = []

        # setup.py needs to be able to read the CPPFLAGS and LDFLAGS
        # as it scans for the library and headers to build
        link_deps = spec.dependencies(deptype="link")

        if link_deps:
            # Header files are often included assuming they reside in a
            # subdirectory of prefix.include, e.g. #include <openssl/ssl.h>,
            # which is why we don't use HeaderList here. The header files of
            # libffi reside in prefix.lib but the configure script of Python
            # finds them using pkg-config.
            cppflags = " ".join("-I" + spec[dep.name].prefix.include for dep in link_deps)

            # Currently, the only way to get SpecBuildInterface wrappers of the
            # dependencies (which we need to get their 'libs') is to get them
            # using spec.__getitem__.
            ldflags = " ".join(spec[dep.name].libs.search_flags for dep in link_deps)

            config_args.extend(["CPPFLAGS=" + cppflags, "LDFLAGS=" + ldflags])

        # https://docs.python.org/3/whatsnew/3.7.html#build-changes
        if spec.satisfies("@:3.6"):
            config_args.append("--with-threads")

        if spec.satisfies("@2.7.13:2.8,3.5.3:", strict=True) and "+optimizations" in spec:
            config_args.append("--enable-optimizations")
            config_args.append("--with-lto")
            config_args.append("--with-computed-gotos")

        if spec.satisfies("%gcc platform=darwin"):
            config_args.append("--disable-toolbox-glue")

        if spec.satisfies("%intel", strict=True) and spec.satisfies(
            "@2.7.12:2.8,3.5.2:3.7", strict=True
        ):
            config_args.append("--with-icc={0}".format(spack_cc))

        if "+debug" in spec:
            config_args.append("--with-pydebug")
        else:
            config_args.append("--without-pydebug")

        if "+shared" in spec:
            config_args.append("--enable-shared")
        else:
            config_args.append("--disable-shared")

        if "+ucs4" in spec:
            if spec.satisfies("@:2.7"):
                config_args.append("--enable-unicode=ucs4")
            elif spec.satisfies("@3.0:3.2"):
                config_args.append("--with-wide-unicode")
            elif spec.satisfies("@3.3:"):
                # https://docs.python.org/3.3/whatsnew/3.3.html#functionality
                raise ValueError("+ucs4 variant not compatible with Python 3.3 and beyond")

        if spec.satisfies("@2.7.9:2,3.4:"):
            config_args.append("--without-ensurepip")

        if "+pic" in spec:
            cflags.append(self.compiler.cc_pic_flag)

        if "+ssl" in spec:
            if spec.satisfies("@3.7:"):
                config_args.append("--with-openssl={0}".format(spec["openssl"].prefix))

        if "+dbm" in spec:
            # Default order is ndbm:gdbm:bdb
            config_args.append("--with-dbmliborder=gdbm")
        else:
            config_args.append("--with-dbmliborder=")

        if "+pyexpat" in spec:
            config_args.append("--with-system-expat")
        else:
            config_args.append("--without-system-expat")

        if "+ctypes" in spec:
            config_args.append("--with-system-ffi")
        else:
            config_args.append("--without-system-ffi")

        if "+tkinter" in spec:
            config_args.extend(
                [
                    "--with-tcltk-includes=-I{0} -I{1}".format(
                        spec["tcl"].prefix.include, spec["tk"].prefix.include
                    ),
                    "--with-tcltk-libs={0} {1}".format(
                        spec["tcl"].libs.ld_flags, spec["tk"].libs.ld_flags
                    ),
                ]
            )

        # https://docs.python.org/3.8/library/sqlite3.html#f1
        if spec.satisfies("@3.2: +sqlite3 ^sqlite+dynamic_extensions"):
            config_args.append("--enable-loadable-sqlite-extensions")

        if spec.satisfies("%oneapi"):
            cflags.append("-fp-model=strict")

        if cflags:
            config_args.append("CFLAGS={0}".format(" ".join(cflags)))

        return config_args

    def configure(self, spec, prefix):
        """Runs configure with the arguments specified in
        :meth:`~spack.build_systems.autotools.AutotoolsPackage.configure_args`
        and an appropriately set prefix.
        """
        with working_dir(self.stage.source_path, create=True):
            if is_windows:
                pass
            else:
                options = getattr(self, "configure_flag_args", [])
                options += ["--prefix={0}".format(prefix)]
                options += self.configure_args()
                configure(*options)

    def build(self, spec, prefix):
        """Makes the build targets specified by
        :py:attr:``~.AutotoolsPackage.build_targets``
        """
        # Windows builds use a batch script to drive
        # configure and build in one step
        with working_dir(self.stage.source_path):
            if is_windows:
                pcbuild_root = os.path.join(self.stage.source_path, "PCbuild")
                builder_cmd = os.path.join(pcbuild_root, "build.bat")
                try:
                    subprocess.check_output(  # novermin
                        " ".join([builder_cmd] + self.win_build_params), stderr=subprocess.STDOUT
                    )
                except subprocess.CalledProcessError as e:
                    raise ProcessError(
                        "Process exited with status %d" % e.returncode,
                        long_message=e.output.decode("utf-8"),
                    )
            else:
                # See https://autotools.io/automake/silent.html
                params = ["V=1"]
                params += self.build_targets
                make(*params)

    def install(self, spec, prefix):
        """Makes the install targets specified by
        :py:attr:``~.AutotoolsPackage.install_targets``
        """
        with working_dir(self.stage.source_path):
            if is_windows:
                self.win_installer(prefix)
            else:
                make(*self.install_targets)

    @run_after("install")
    def filter_compilers(self):
        """Run after install to tell the configuration files and Makefiles
        to use the compilers that Spack built the package with.

        If this isn't done, they'll have CC and CXX set to Spack's generic
        cc and c++. We want them to be bound to whatever compiler
        they were built with."""
        if is_windows:
            return
        kwargs = {"ignore_absent": True, "backup": False, "string": True}

        filenames = [self.get_sysconfigdata_name(), self.config_vars["makefile_filename"]]

        filter_file(spack_cc, self.compiler.cc, *filenames, **kwargs)
        if spack_cxx and self.compiler.cxx:
            filter_file(spack_cxx, self.compiler.cxx, *filenames, **kwargs)

    @run_after("install")
    def symlink(self):
        if is_windows:
            return
        spec = self.spec
        prefix = self.prefix

        # TODO:
        # On OpenSuse 13, python uses <prefix>/lib64/python2.7/lib-dynload/*.so
        # instead of <prefix>/lib/python2.7/lib-dynload/*.so. Oddly enough the
        # result is that Python can not find modules like cPickle. A workaround
        # for now is to symlink to `lib`:
        src = os.path.join(prefix.lib64, "python{0}".format(self.version.up_to(2)), "lib-dynload")
        dst = os.path.join(prefix.lib, "python{0}".format(self.version.up_to(2)), "lib-dynload")
        if os.path.isdir(src) and not os.path.isdir(dst):
            mkdirp(dst)
            for f in os.listdir(src):
                os.symlink(os.path.join(src, f), os.path.join(dst, f))

        if spec.satisfies("@3:") and spec.satisfies("+pythoncmd"):
            os.symlink(os.path.join(prefix.bin, "python3"), os.path.join(prefix.bin, "python"))
            os.symlink(
                os.path.join(prefix.bin, "python3-config"),
                os.path.join(prefix.bin, "python-config"),
            )

    @run_after("install")
    def install_python_gdb(self):
        # https://devguide.python.org/gdb/
        src = os.path.join("Tools", "gdb", "libpython.py")
        if os.path.exists(src):
            install(src, self.command.path + "-gdb.py")

    @run_after("install")
    @on_package_attributes(run_tests=True)
    def import_tests(self):
        """Test that basic Python functionality works."""

        spec = self.spec

        with working_dir("spack-test", create=True):
            # Ensure that readline module works
            if "+readline" in spec:
                self.command("-c", "import readline")

            # Ensure that ssl module works
            if "+ssl" in spec:
                self.command("-c", "import ssl")
                self.command("-c", "import hashlib")

            # Ensure that sqlite3 module works
            if "+sqlite3" in spec:
                self.command("-c", "import sqlite3")

            # Ensure that dbm module works
            if "+dbm" in spec:
                self.command("-c", "import dbm")

            # Ensure that nis module works
            if "+nis" in spec:
                self.command("-c", "import nis")

            # Ensure that zlib module works
            if "+zlib" in spec:
                self.command("-c", "import zlib")

            # Ensure that bz2 module works
            if "+bz2" in spec:
                self.command("-c", "import bz2")

            # Ensure that lzma module works
            if spec.satisfies("@3.3:"):
                if "+lzma" in spec:
                    self.command("-c", "import lzma")

            # Ensure that pyexpat module works
            if "+pyexpat" in spec:
                self.command("-c", "import xml.parsers.expat")
                self.command("-c", "import xml.etree.ElementTree")

            # Ensure that ctypes module works
            if "+ctypes" in spec:
                self.command("-c", "import ctypes")

            # Ensure that tkinter module works
            # https://wiki.python.org/moin/TkInter
            if "+tkinter" in spec:
                # Only works if ForwardX11Trusted is enabled, i.e. `ssh -Y`
                if "DISPLAY" in env:
                    if spec.satisfies("@3:"):
                        self.command("-c", "import tkinter; tkinter._test()")
                    else:
                        self.command("-c", "import Tkinter; Tkinter._test()")
                else:
                    if spec.satisfies("@3:"):
                        self.command("-c", "import tkinter")
                    else:
                        self.command("-c", "import Tkinter")

            # Ensure that uuid module works
            if "+uuid" in spec:
                self.command("-c", "import uuid")

            # Ensure that tix module works
            if "+tix" in spec:
                if spec.satisfies("@3:"):
                    self.command("-c", "import tkinter.tix")
                else:
                    self.command("-c", "import Tix")

    # ========================================================================
    # Set up environment to make install easy for python extensions.
    # ========================================================================

    @property
    def command(self):
        """Returns the Python command, which may vary depending
        on the version of Python and how it was installed.

        In general, Python 2 comes with ``python`` and ``python2`` commands,
        while Python 3 only comes with a ``python3`` command. However, some
        package managers will symlink ``python`` to ``python3``, while others
        may contain ``python3.6``, ``python3.5``, and ``python3.4`` in the
        same directory.

        Returns:
            Executable: the Python command
        """
        # We need to be careful here. If the user is using an externally
        # installed python, several different commands could be located
        # in the same directory. Be as specific as possible. Search for:
        #
        # * python3.6
        # * python3
        # * python
        #
        # in that order if using python@3.6.5, for example.
        version = self.spec.version
        for ver in [version.up_to(2), version.up_to(1), ""]:
            if not is_windows:
                path = os.path.join(self.prefix.bin, "python{0}".format(ver))
            else:
                path = os.path.join(self.prefix, "python{0}.exe".format(ver))
            if os.path.exists(path):
                return Executable(path)

        else:
            msg = "Unable to locate {0} command in {1}"
            raise RuntimeError(msg.format(self.name, self.prefix.bin))

    def print_string(self, string):
        """Returns the appropriate print string depending on the
        version of Python.

        Examples:

        * Python 2

          .. code-block:: python

             >>> self.print_string('sys.prefix')
             'print sys.prefix'

        * Python 3

          .. code-block:: python

             >>> self.print_string('sys.prefix')
             'print(sys.prefix)'
        """
        if self.spec.satisfies("@:2"):
            return "print {0}".format(string)
        else:
            return "print({0})".format(string)

    @property
    def config_vars(self):
        """Return a set of variable definitions associated with a Python installation.

        Wrapper around various ``sysconfig`` functions. To see these variables on the
        command line, run:

        .. code-block:: console

           $ python -m sysconfig

        Returns:
            dict: variable definitions
        """
        cmd = """
import json
from sysconfig import (
    get_config_vars,
    get_config_h_filename,
    get_makefile_filename,
    get_paths,
)

config = get_config_vars()
config['config_h_filename'] = get_config_h_filename()
config['makefile_filename'] = get_makefile_filename()
config.update(get_paths())

%s
""" % self.print_string(
            "json.dumps(config)"
        )

        dag_hash = self.spec.dag_hash()
        lib_prefix = "lib" if not is_windows else ""
        if dag_hash not in self._config_vars:
            # Default config vars
            version = self.version.up_to(2)
            if is_windows:
                version = str(version).split(".")[0]
            config = {
                # get_config_vars
                "BINDIR": self.prefix.bin,
                "CC": "cc",
                "CONFINCLUDEPY": self.prefix.include.join("python{}").format(version),
                "CXX": "c++",
                "INCLUDEPY": self.prefix.include.join("python{}").format(version),
                "LIBDEST": self.prefix.lib.join("python{}").format(version),
                "LIBDIR": self.prefix.lib,
                "LIBPL": self.prefix.lib.join("python{0}")
                .join("config-{0}-{1}")
                .format(version, sys.platform),
                "LDLIBRARY": "{}python{}.{}".format(lib_prefix, version, dso_suffix),
                "LIBRARY": "{}python{}.{}".format(lib_prefix, version, stat_suffix),
                "LDSHARED": "cc",
                "LDCXXSHARED": "c++",
                "PYTHONFRAMEWORKPREFIX": "/System/Library/Frameworks",
                "base": self.prefix,
                "installed_base": self.prefix,
                "installed_platbase": self.prefix,
                "platbase": self.prefix,
                "prefix": self.prefix,
                # get_config_h_filename
                "config_h_filename": self.prefix.include.join("python{}")
                .join("pyconfig.h")
                .format(version),
                # get_makefile_filename
                "makefile_filename": self.prefix.lib.join("python{0}")
                .join("config-{0}-{1}")
                .Makefile.format(version, sys.platform),
                # get_paths
                "data": self.prefix,
                "include": self.prefix.include.join("python{}".format(version)),
                "platinclude": self.prefix.include64.join("python{}".format(version)),
                "platlib": self.prefix.lib64.join("python{}".format(version)).join(
                    "site-packages"
                ),
                "platstdlib": self.prefix.lib64.join("python{}".format(version)),
                "purelib": self.prefix.lib.join("python{}".format(version)).join("site-packages"),
                "scripts": self.prefix.bin,
                "stdlib": self.prefix.lib.join("python{}".format(version)),
            }

            try:
                config.update(json.loads(self.command("-c", cmd, output=str)))
            except (ProcessError, RuntimeError):
                pass
            self._config_vars[dag_hash] = config
        return self._config_vars[dag_hash]

    def get_sysconfigdata_name(self):
        """Return the full path name of the sysconfigdata file."""

        libdest = self.config_vars["LIBDEST"]

        filename = "_sysconfigdata.py"
        if self.spec.satisfies("@3.6:"):
            # Python 3.6.0 renamed the sys config file
            cmd = "from sysconfig import _get_sysconfigdata_name; "
            cmd += self.print_string("_get_sysconfigdata_name()")
            filename = self.command("-c", cmd, output=str).strip()
            filename += ".py"

        return join_path(libdest, filename)

    @property
    def home(self):
        """Most of the time, ``PYTHONHOME`` is simply
        ``spec['python'].prefix``. However, if the user is using an
        externally installed python, it may be symlinked. For example,
        Homebrew installs python in ``/usr/local/Cellar/python/2.7.12_2``
        and symlinks it to ``/usr/local``. Users may not know the actual
        installation directory and add ``/usr/local`` to their
        ``packages.yaml`` unknowingly. Query the python executable to
        determine exactly where it is installed.
        """
        return Prefix(self.config_vars["base"])

    def find_library(self, library):
        # Spack installs libraries into lib, except on openSUSE where it installs them
        # into lib64. If the user is using an externally installed package, it may be
        # in either lib or lib64, so we need to ask Python where its LIBDIR is.
        libdir = self.config_vars["LIBDIR"]

        # In Ubuntu 16.04.6 and python 2.7.12 from the system, lib could be in LBPL
        # https://mail.python.org/pipermail/python-dev/2013-April/125733.html
        libpl = self.config_vars["LIBPL"]

        # The system Python installation on macOS and Homebrew installations
        # install libraries into a Frameworks directory
        frameworkprefix = self.config_vars["PYTHONFRAMEWORKPREFIX"]

        # Get the active Xcode environment's Framework location.
        macos_developerdir = os.environ.get("DEVELOPER_DIR")
        if macos_developerdir and os.path.exists(macos_developerdir):
            macos_developerdir = os.path.join(macos_developerdir, "Library", "Frameworks")
        else:
            macos_developerdir = ""

        # Windows libraries are installed directly to BINDIR
        win_bin_dir = self.config_vars["BINDIR"]
        win_root_dir = self.config_vars["prefix"]

        directories = [
            libdir,
            libpl,
            frameworkprefix,
            macos_developerdir,
            win_bin_dir,
            win_root_dir,
        ]

        # The Python shipped with Xcode command line tools isn't in any of these locations
        for subdir in ["lib", "lib64"]:
            directories.append(os.path.join(self.config_vars["base"], subdir))

        directories = dedupe(directories)
        for directory in directories:
            path = os.path.join(directory, library)
            if os.path.exists(path):
                return LibraryList(path)

    @property
    def libs(self):
        py_version = self.version.up_to(2)
        if is_windows:
            py_version = str(py_version).replace(".", "")
        lib_prefix = "lib" if not is_windows else ""
        # The values of LDLIBRARY and LIBRARY aren't reliable. Intel Python uses a
        # static binary but installs shared libraries, so sysconfig reports
        # libpythonX.Y.a but only libpythonX.Y.so exists. So we add our own paths, too.
        shared_libs = [
            self.config_vars["LDLIBRARY"],
            "{}python{}.{}".format(lib_prefix, py_version, dso_suffix),
        ]
        static_libs = [
            self.config_vars["LIBRARY"],
            "{}python{}.{}".format(lib_prefix, py_version, stat_suffix),
        ]

        # The +shared variant isn't reliable, as `spack external find` currently can't
        # detect it. If +shared, prefer the shared libraries, but check for static if
        # those aren't found. Vice versa for ~shared.
        if "+shared" in self.spec:
            candidates = shared_libs + static_libs
        else:
            candidates = static_libs + shared_libs

        candidates = dedupe(candidates)

        for candidate in candidates:
            lib = self.find_library(candidate)
            if lib:
                return lib

        raise spack.error.NoLibrariesError(
            "Unable to find {} libraries with the following names:".format(self.name),
            "\n".join(candidates),
        )

    @property
    def headers(self):
        # Locations where pyconfig.h could be
        # This varies by system, especially on macOS where the command line tools are
        # installed in a very different directory from the system python interpreter.
        py_version = str(self.version.up_to(2))
        candidates = [
            os.path.dirname(self.config_vars["config_h_filename"]),
            self.config_vars["INCLUDEPY"],
            self.config_vars["CONFINCLUDEPY"],
            os.path.join(self.config_vars["base"], "include", py_version),
            os.path.join(self.config_vars["base"], "Headers"),
        ]
        candidates = list(dedupe(candidates))

        for directory in candidates:
            headers = find_headers("pyconfig", directory)
            if headers:
                config_h = headers[0]
                break
        else:
            raise spack.error.NoHeadersError(
                "Unable to locate {} headers in any of these locations:".format(self.name),
                "\n".join(candidates),
            )

        headers.directories = [os.path.dirname(config_h)]
        return headers

    # https://docs.python.org/3/library/sysconfig.html#installation-paths
    # https://discuss.python.org/t/understanding-site-packages-directories/12959
    # https://github.com/pypa/pip/blob/22.1/src/pip/_internal/locations/__init__.py
    # https://github.com/pypa/installer/pull/103

    # NOTE: XCode Python's sysconfing module was incorrectly patched, and hard-codes
    # everything to be installed in /Library/Python. Therefore, we need to use a
    # fallback in the following methods. For more information, see:
    # https://github.com/pypa/pip/blob/22.1/src/pip/_internal/locations/__init__.py#L486

    @property
    def platlib(self):
        """Directory for site-specific, platform-specific files.

        Exact directory depends on platform/OS/Python version. Examples include:

        * ``lib/pythonX.Y/site-packages`` on most POSIX systems
        * ``lib64/pythonX.Y/site-packages`` on RHEL/CentOS/Fedora with system Python
        * ``lib/pythonX/dist-packages`` on Debian/Ubuntu with system Python
        * ``lib/python/site-packages`` on macOS with framework Python
        * ``Lib/site-packages`` on Windows

        Returns:
            str: platform-specific site-packages directory
        """
        prefix = self.config_vars["platbase"] + os.sep
        path = self.config_vars["platlib"]
        if path.startswith(prefix):
            return path.replace(prefix, "")
        return os.path.join("lib64", "python{}".format(self.version.up_to(2)), "site-packages")

    @property
    def purelib(self):
        """Directory for site-specific, non-platform-specific files.

        Exact directory depends on platform/OS/Python version. Examples include:

        * ``lib/pythonX.Y/site-packages`` on most POSIX systems
        * ``lib/pythonX/dist-packages`` on Debian/Ubuntu with system Python
        * ``lib/python/site-packages`` on macOS with framework Python
        * ``Lib/site-packages`` on Windows

        Returns:
            str: platform-independent site-packages directory
        """
        prefix = self.config_vars["base"] + os.sep
        path = self.config_vars["purelib"]
        if path.startswith(prefix):
            return path.replace(prefix, "")
        return os.path.join("lib", "python{}".format(self.version.up_to(2)), "site-packages")

    @property
    def include(self):
        """Directory for non-platform-specific header files.

        Exact directory depends on platform/Python version/ABI flags. Examples include:

        * ``include/pythonX.Y`` on most POSIX systems
        * ``include/pythonX.Yd`` for debug builds
        * ``include/pythonX.Ym`` for malloc builds
        * ``include/pythonX.Yu`` for wide unicode builds
        * ``include`` on macOS with framework Python
        * ``Include`` on Windows

        Returns:
            str: platform-independent header file directory
        """
        prefix = self.config_vars["installed_base"] + os.sep
        path = self.config_vars["include"]
        if path.startswith(prefix):
            return path.replace(prefix, "")
        return os.path.join("include", "python{}".format(self.version.up_to(2)))

    @property
    def easy_install_file(self):
        return join_path(self.purelib, "easy-install.pth")

    def setup_run_environment(self, env):
        env.prepend_path("CPATH", os.pathsep.join(self.spec["python"].headers.directories))

    def setup_dependent_build_environment(self, env, dependent_spec):
        """Set PYTHONPATH to include the site-packages directory for the
        extension and any other python extensions it depends on.
        """
        # If we set PYTHONHOME, we must also ensure that the corresponding
        # python is found in the build environment. This to prevent cases
        # where a system provided python is run against the standard libraries
        # of a Spack built python. See issue #7128
        env.set("PYTHONHOME", self.home)

        path = os.path.dirname(self.command.path)
        if not is_system_path(path):
            env.prepend_path("PATH", path)

        # Add installation prefix to PYTHONPATH, needed to run import tests
        prefixes = set()
        if dependent_spec.package.extends(self.spec):
            prefixes.add(dependent_spec.prefix)

        # Add direct build/run/test dependencies to PYTHONPATH,
        # needed to build the package and to run import tests
        for direct_dep in dependent_spec.dependencies(deptype=("build", "run", "test")):
            if direct_dep.package.extends(self.spec):
                prefixes.add(direct_dep.prefix)

                # Add recursive run dependencies of all direct dependencies,
                # needed by direct dependencies at run-time
                for indirect_dep in direct_dep.traverse(deptype="run"):
                    if indirect_dep.package.extends(self.spec):
                        prefixes.add(indirect_dep.prefix)

        for prefix in prefixes:
            # Packages may be installed in platform-specific or platform-independent
            # site-packages directories
            for directory in {self.platlib, self.purelib}:
                env.prepend_path("PYTHONPATH", os.path.join(prefix, directory))

        # We need to make sure that the extensions are compiled and linked with
        # the Spack wrapper. Paths to the executables that are used for these
        # operations are normally taken from the sysconfigdata file, which we
        # modify after the installation (see method filter compilers). The
        # modified file contains paths to the real compilers, not the wrappers.
        # The values in the file, however, can be overridden with environment
        # variables. The first variable, CC (CXX), which is used for
        # compilation, is set by Spack for the dependent package by default.
        # That is not 100% correct because the value for CC (CXX) in the
        # sysconfigdata file often contains additional compiler flags (e.g.
        # -pthread), which we lose by simply setting CC (CXX) to the path to the
        # Spack wrapper. Moreover, the user might try to build an extension with
        # a compiler that is different from the one that was used to build
        # Python itself, which might have unexpected side effects. However, the
        # experience shows that none of the above is a real issue and we will
        # not try to change the default behaviour. Given that, we will simply
        # try to modify LDSHARED (LDCXXSHARED), the second variable, which is
        # used for linking, in a consistent manner.

        for compile_var, link_var in [("CC", "LDSHARED"), ("CXX", "LDCXXSHARED")]:
            # First, we get the values from the sysconfigdata:
            config_compile = self.config_vars[compile_var]
            config_link = self.config_vars[link_var]

            # The dependent environment will have the compilation command set to
            # the following:
            new_compile = join_path(
                spack.paths.build_env_path,
                dependent_spec.package.compiler.link_paths[compile_var.lower()],
            )

            # Normally, the link command starts with the compilation command:
            if config_link.startswith(config_compile):
                new_link = new_compile + config_link[len(config_compile) :]
            else:
                # Otherwise, we try to replace the compiler command if it
                # appears "in the middle" of the link command; to avoid
                # mistaking some substring of a path for the compiler (e.g. to
                # avoid replacing "gcc" in "-L/path/to/gcc/"), we require that
                # the compiler command be surrounded by spaces. Note this may
                # leave "config_link" unchanged if the compilation command does
                # not appear in the link command at all, for example if "ld" is
                # invoked directly (no change would be required in that case
                # because Spack arranges for the Spack ld wrapper to be the
                # first instance of "ld" in PATH).
                new_link = config_link.replace(
                    " {0} ".format(config_compile), " {0} ".format(new_compile)
                )

            # There is logic in the sysconfig module that is sensitive to the
            # fact that LDSHARED is set in the environment, therefore we export
            # the variable only if the new value is different from what we got
            # from the sysconfigdata file:
            if config_link != new_link and not is_windows:
                env.set(link_var, new_link)

    def setup_dependent_run_environment(self, env, dependent_spec):
        """Set PYTHONPATH to include the site-packages directory for the
        extension and any other python extensions it depends on.
        """
        for d in dependent_spec.traverse(deptype=("run"), root=True):
            if d.package.extends(self.spec):
                # Packages may be installed in platform-specific or platform-independent
                # site-packages directories
                for directory in {self.platlib, self.purelib}:
                    env.prepend_path("PYTHONPATH", os.path.join(d.prefix, directory))

    def setup_dependent_package(self, module, dependent_spec):
        """Called before python modules' install() methods."""

        module.python = self.command

        module.python_include = join_path(dependent_spec.prefix, self.include)
        module.python_platlib = join_path(dependent_spec.prefix, self.platlib)
        module.python_purelib = join_path(dependent_spec.prefix, self.purelib)

        # Make the site packages directory for extensions
        if dependent_spec.package.is_extension:
            mkdirp(module.python_platlib)
            mkdirp(module.python_purelib)

    # ========================================================================
    # Handle specifics of activating and deactivating python modules.
    # ========================================================================

    def python_ignore(self, ext_pkg, args):
        """Add some ignore files to activate/deactivate args."""
        ignore_arg = args.get("ignore", lambda f: False)

        # Always ignore easy-install.pth, as it needs to be merged.
        patterns = [r"(site|dist)-packages/easy-install\.pth$"]

        # Ignore pieces of setuptools installed by other packages.
        # Must include directory name or it will remove all site*.py files.
        if ext_pkg.name != "py-setuptools":
            patterns.extend(
                [
                    r"bin/easy_install[^/]*$",
                    r"(site|dist)-packages/setuptools[^/]*\.egg$",
                    r"(site|dist)-packages/setuptools\.pth$",
                    r"(site|dist)-packages/site[^/]*\.pyc?$",
                    r"(site|dist)-packages/__pycache__/site[^/]*\.pyc?$",
                ]
            )
        if ext_pkg.name != "py-pygments":
            patterns.append(r"bin/pygmentize$")
        if ext_pkg.name != "py-numpy":
            patterns.append(r"bin/f2py[0-9.]*$")

        return match_predicate(ignore_arg, patterns)

    def write_easy_install_pth(self, exts, prefix=None):
        if not prefix:
            prefix = self.prefix

        paths = []
        unique_paths = set()

        for ext in sorted(exts.values()):
            easy_pth = join_path(ext.prefix, self.easy_install_file)

            if not os.path.isfile(easy_pth):
                continue

            with open(easy_pth) as f:
                for line in f:
                    line = line.rstrip()

                    # Skip lines matching these criteria
                    if not line:
                        continue
                    if re.search(r"^(import|#)", line):
                        continue
                    if ext.name != "py-setuptools" and re.search(r"setuptools.*egg$", line):
                        continue

                    if line not in unique_paths:
                        unique_paths.add(line)
                        paths.append(line)

        main_pth = join_path(prefix, self.easy_install_file)

        if not paths:
            if os.path.isfile(main_pth):
                os.remove(main_pth)

        else:
            with open(main_pth, "w") as f:
                f.write("import sys; sys.__plen = len(sys.path)\n")
                for path in paths:
                    f.write("{0}\n".format(path))
                f.write(
                    "import sys; new=sys.path[sys.__plen:]; "
                    "del sys.path[sys.__plen:]; "
                    "p=getattr(sys,'__egginsert',0); "
                    "sys.path[p:p]=new; "
                    "sys.__egginsert = p+len(new)\n"
                )

    def activate(self, ext_pkg, view, **args):
        ignore = self.python_ignore(ext_pkg, args)
        args.update(ignore=ignore)

        super(Python, self).activate(ext_pkg, view, **args)

        extensions_layout = view.extensions_layout
        exts = extensions_layout.extension_map(self.spec)
        exts[ext_pkg.name] = ext_pkg.spec

        self.write_easy_install_pth(exts, prefix=view.get_projection_for_spec(self.spec))

    def deactivate(self, ext_pkg, view, **args):
        args.update(ignore=self.python_ignore(ext_pkg, args))

        super(Python, self).deactivate(ext_pkg, view, **args)

        extensions_layout = view.extensions_layout
        exts = extensions_layout.extension_map(self.spec)
        # Make deactivate idempotent
        if ext_pkg.name in exts:
            del exts[ext_pkg.name]
            self.write_easy_install_pth(exts, prefix=view.get_projection_for_spec(self.spec))

    def add_files_to_view(self, view, merge_map, skip_if_exists=True):
        bin_dir = self.spec.prefix.bin if sys.platform != "win32" else self.spec.prefix
        for src, dst in merge_map.items():
            if not path_contains_subdirectory(src, bin_dir):
                view.link(src, dst, spec=self.spec)
            elif not os.path.islink(src):
                copy(src, dst)
                if is_nonsymlink_exe_with_shebang(src):
                    filter_file(
                        self.spec.prefix,
                        os.path.abspath(view.get_projection_for_spec(self.spec)),
                        dst,
                        backup=False,
                    )
            else:
                # orig_link_target = os.path.realpath(src) is insufficient when
                # the spack install tree is located at a symlink or a
                # descendent of a symlink. What we need here is the real
                # relative path from the python prefix to src
                # TODO: generalize this logic in the link_tree object
                #    add a method to resolve a link relative to the link_tree
                #    object root.
                realpath_src = os.path.realpath(src)
                realpath_prefix = os.path.realpath(self.spec.prefix)
                realpath_rel = os.path.relpath(realpath_src, realpath_prefix)
                orig_link_target = os.path.join(self.spec.prefix, realpath_rel)

                new_link_target = os.path.abspath(merge_map[orig_link_target])
                view.link(new_link_target, dst, spec=self.spec)

    def remove_files_from_view(self, view, merge_map):
        bin_dir = self.spec.prefix.bin if not is_windows else self.spec.prefix
        for src, dst in merge_map.items():
            if not path_contains_subdirectory(src, bin_dir):
                view.remove_file(src, dst)
            else:
                os.remove(dst)

    def test(self):
        # do not use self.command because we are also testing the run env
        exe = self.spec["python"].command.name

        # test hello world
        msg = "hello world!"
        reason = "test: running {0}".format(msg)
        options = ["-c", 'print("{0}")'.format(msg)]
        self.run_test(exe, options=options, expected=[msg], installed=True, purpose=reason)

        # checks import works and executable comes from the spec prefix
        reason = "test: checking import and executable"
        print_str = self.print_string("sys.executable")
        options = ["-c", "import sys; {0}".format(print_str)]
        self.run_test(
            exe, options=options, expected=[self.spec.prefix], installed=True, purpose=reason
        )
