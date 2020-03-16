# -*- coding: utf-8 -*-
# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

import numpy as np
from polygon_classification import config
from shapely.geometry import LineString, Point
from shapely.wkt import loads
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class Features:
    """This class builds features regarding polygon properties.

    See Also
    --------
    :func:`compute_features`: Details on the implemented features.
    """

    def __init__(self):
        pass

    def build(self, X):
        """Build features and return them as an ndarray of floats.

        Parameters
        ----------
        X: array-like or sparse matrix, shape = [n_samples, n_features]
            The train/test input samples.

        Returns
        -------
        fX: ndarray
            The computed features to use as input to ML classifiers.
        """
        fX = np.asarray(list(map(self.compute, X['pst_geom'], X['dian_geom'])), dtype=float)
        # print('Before normalization: ', np.amin(fX, axis=0), np.amax(fX, axis=0))
        fX = MinMaxScaler().fit_transform(fX)
        # fX = StandardScaler().fit_transform(fX)
        # fX = RobustScaler().fit_transform(fX)
        # print('After normalization: ', np.amin(fX, axis=0), np.amax(fX, axis=0))

        if np.any(np.isnan(fX)): print(np.where(np.isnan(fX)))
        if not np.any(np.isfinite(fX)): print('infinite')

        return fX

    def compute(self, poly1, poly2):
        """
        This method builds an ndarray of the following features:

        * *core*: basic geometric attributes, i.e.,
            #. area of each polygon,
            #. percentage of coverage/intersection area per polygon,
            #. perimeter of each polygon,
            #. number of corners of each polygon,
            #. average edges' length per corner of each polygon,
            #. variance of edges' length per corner of each polygon,
        * *extra*: these features are computed only when parameter :py:attr:`~src.config.MLConf.extra_features` is set
           to True value. In such case, the following additional features are calculated:
            #. area of of each polygon convex hull,
            #. percentage of coverage/intersection of convex hull area per polygon,
            #. distance of centroids of polygons

        Parameters
        ----------
        poly1, poly2: str
            Input geometric objects, i.e., shapely Polygons.

        Returns
        -------
        :obj:`list`
            It returns a list (vector) of features.
        """
        f = []

        geom1 = loads(poly1)
        geom2 = loads(poly2)

        # convex hull
        convex1 = geom1.convex_hull
        convex2 = geom2.convex_hull

        # area
        area1 = geom1.area
        area2 = geom2.area
        convex_area1 = convex1.area
        convex_area2 = convex2.area

        # % coverage
        intersect = geom1.intersection(geom2).area
        cover1 = intersect / area1
        cover2 = intersect / area2
        convex_intersect = convex1.intersection(convex2).area
        convex_cover1 = convex_intersect / convex_area1
        convex_cover2 = convex_intersect / convex_area2

        # polygon length
        l1 = geom1.length
        l2 = geom2.length

        coords1 = list(zip(*geom1.exterior.coords.xy))
        coords2 = list(zip(*geom2.exterior.coords.xy))

        # no of coords
        no_coords1 = len(coords1) - 1
        no_coords2 = len(coords2) - 1

        # calculate the length of each side of the poly
        poly1_lengths = [LineString((coords1[i], coords1[i + 1])).length for i in range(len(coords1) - 1)]
        poly2_lengths = [LineString((coords2[i], coords2[i + 1])).length for i in range(len(coords2) - 1)]

        # avg length per edge
        # avg1 = l1 / no_coords1
        # avg2 = l2 / no_coords2
        avg1 = np.mean(poly1_lengths)
        avg2 = np.mean(poly2_lengths)

        # std on edge lengths
        # std1 = np.std(poly1_lengths)
        # std2 = np.std(poly2_lengths)
        var1 = np.var(poly1_lengths)
        var2 = np.var(poly2_lengths)

        # centroid dist
        centroid1 = geom1.centroid.coords
        centroid2 = geom2.centroid.coords
        centroid_dist = Point(centroid1).distance(Point(centroid2))

        f = [
            area1, area2, cover1, cover2, l1, l2,
            no_coords1, no_coords2, avg1, avg2, var1, var2,
        ]

        if config.MLConf.extra_features: f += [convex_area1, convex_area2, convex_cover1, convex_cover2, centroid_dist]

        return f
