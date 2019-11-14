# -*- coding: utf-8 -*-
# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

"""A complete pipeline of processes for features extraction and training/evaluating state-of-the-art classifiers for
toponym interlinking.

Command line::

    Usage:
      run.py [options]
      run.py (-h | --help)
      run.py --version

    Options:
        -h --help                   show this screen.
        --version                   show version.
        --dtrain <fpath>            relative path to the train dataset. If this is null, the assigned
                                    value to `train_dataset` parameter in config.py is used instead.
        --dtest <fpath>             relative path to the test dataset. If this is null, the assigned
                                    value to `test_dataset` parameter in config.py is used instead.

    Arguments:
        encoding_type               global
                                    latin

"""

import os, sys
import codecs
# from docopt import docopt

from src.core import StrategyEvaluator
from src.helpers import getRelativePathtoWorking
import src.config as config


def main():
    UTF8Writer = codecs.getwriter('utf8')
    # sys.stdout = UTF8Writer(sys.stdout)

    if os.path.isfile(getRelativePathtoWorking(config.dataset)) and os.path.isfile(getRelativePathtoWorking(config.dian)):
        seval = StrategyEvaluator()
        seval.hyperparamTuning()
    else:
        print("Input files in config are not found!!!\n")


if __name__ == "__main__":
    # arguments = docopt(__doc__, version='LGM-PolygonClassification 0.2.0')
    # main(arguments)
    main()
