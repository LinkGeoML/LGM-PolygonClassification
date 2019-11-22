# coding= utf-8
# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

import os
import __main__


def getBasePath():
    # return os.path.abspath(os.path.dirname(__main__.__file__))
    return os.path.dirname(os.path.realpath(__main__.__file__))


def getRelativePathtoWorking(ds):
    return os.path.join(getBasePath(), ds)


class StaticValues:
    featureColumns = [

    ]
