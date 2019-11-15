# coding= utf-8
# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

import os
import re
import __main__


def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def getBasePath():
    # return os.path.abspath(os.path.dirname(__main__.__file__))
    return os.path.dirname(os.path.realpath(__main__.__file__))


def getRelativePathtoWorking(ds):
    return os.path.join(getBasePath(), ds)


class StaticValues:
    featureColumns = [

    ]
