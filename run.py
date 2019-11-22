# -*- coding: utf-8 -*-
# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

"""A complete pipeline of processes for features extraction and training/evaluating state-of-the-art classifiers for
polygon classification.

Command line::

    Usage:
      run.py [options]
      run.py (-h | --help)
      run.py --version

    Options:
        -h --help                   show this screen.
        --version                   show version.
        --customparams              [default: False].

"""

import os
from docopt import docopt

from src.core import StrategyEvaluator
from src.helpers import getRelativePathtoWorking
import src.config as config


def main(args):
    # UTF8Writer = codecs.getwriter('utf8')
    # sys.stdout = UTF8Writer(sys.stdout)

    if os.path.isfile(getRelativePathtoWorking(config.dataset)) and os.path.isfile(getRelativePathtoWorking(config.dian)):
        seval = StrategyEvaluator()
        if args['--customparams']:
            seval.exec_classifiers()
        else:
            seval.hyperparamTuning()
    else:
        print("Input files in config are not found!!!\n")


if __name__ == "__main__":
    docopt_args = docopt(__doc__, version='LGM-PolygonClassification 0.2.1')
    main(docopt_args)
