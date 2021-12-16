#!/usr/bin/env python

import atexit
import os
import matplotlib
from setuptools import setup, find_packages
from setuptools.command.install import install
import shutil
import sys

if sys.version_info[:2] < (3, 0):
    raise RuntimeError("Python version >= 3.0 required.")

CLASSIFIERS = """\
Development Status :: 1 - Production
Intended Audience :: Science/Research/Prototyping
License :: GNU-CPL v3 and CeCILL v2 licences
Programming Language :: Python :: 3
Topic :: Scientific/Engineering
Operating System :: Unix
"""

MAJOR = 0
MINOR = 0
MICRO = 1
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


def install_mplstyle():
    stylename = "paper.mplstyle"
    stylefile = '%s @ file://localhost/%s/%s' % (stylename,os.getcwd(),stylename)

    mpl_stylelib_dir = os.path.join(matplotlib.get_configdir() ,"stylelib")
    if not os.path.exists(mpl_stylelib_dir):
        os.makedirs(mpl_stylelib_dir)

    print("Installing style into", mpl_stylelib_dir)
    shutil.copy(
        os.path.join(os.path.dirname(__file__), stylefile),
        os.path.join(mpl_stylelib_dir, stylefile))


class PostInstallMoveFile(install):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        atexit.register(install_mplstyle)


def setup_package():

    # minimal requirement on Python version
    python_minversion = '3.8'
    req_py = '>={}'.format(python_minversion)
    # install requirement for local packages
    loc_pkgs = ["progress"]
    req_loc = []
    for pkg in loc_pkgs:
        req_loc += ['%s @ file://localhost/%s/%s/' % (pkg,os.getcwd(),pkg)]

    metadata = dict(
        name = 'scpmel',
        version = VERSION,
        description = 'SCPMel : A Pytorch-based package providing tools for building 1D PINN models',
        license = 'GNU GPL v3, CeCILLv2',
        maintainer = 'Laurent Gilquin',
        maintainer_email = 'lgilquin@free.fr',
        url = 'https://github.com/gilquin',
        classifiers = [_f for _f in CLASSIFIERS.split('\n') if _f],
        platforms=["Linux"],
        install_requires = req_loc,
        packages = find_packages(),
        python_requires = req_py,
        cmdclass={'install': PostInstallMoveFile,}
    )

    setup(**metadata)

if __name__ == '__main__':
    setup_package()