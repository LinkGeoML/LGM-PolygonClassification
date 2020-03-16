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
    featureCols = [
        'Area_poly_1',
        'Area_poly_2',
        'Percent_Cover_poly_1',
        'Percent_Cover_poly_2',
        'Perimeter_poly_1',
        'Perimeter_poly_2',
        'Corner_no_poly_1',
        'Corner_no_poly_2',
        'Avg_edge_len_per_corner_poly_1',
        'Avg_edge_len_per_corner_poly_2',
        'Variance_of_edge_len_per_corner_poly_1',
        'Variance_of_edge_len_per_corner_poly_2',
    ]

    extra_featureCols = [
        'Area_Convex_hull_poly_1',
        'Area_Convex_hull_poly_2',
        'Percent_Cover_Convex_Hull_poly_1',
        'Percent_Cover_Convex_Hull_poly_2',
        'Poly_Centroids_dist',
    ]
