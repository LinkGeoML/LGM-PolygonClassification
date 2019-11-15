# -*- coding: utf-8 -*-
# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

import numpy as np
from src import config
from shapely.geometry import shape, LineString, Point
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class Features:
    """
    This class loads the dataset, frequent terms and builds features that are used as input to supported classification
    groups:

    * *basic*: similarity features based on basic similarity measures.
    * *basic_sorted*: similarity features based on sorted version of the basic similarity measures used in *basic* group.
    * *lgm*: similarity features based on variations of LGM-Sim similarity measures.

    See Also
    --------
    :func:`compute_features`: Details on the metrics each classification group implements.
    """
    max_freq_terms = 200

    fields = [
        "s1",
        "s2",
        "status",
        "gid1",
        "gid2",
        "alphabet1",
        "alphabet2",
        "alpha2_cc1",
        "alpha2_cc2",
    ]

    dtypes = {
        's1': str, 's2': str,
        'status': str,
        'gid1': np.int32, 'gid2': np.int32,
        'alphabet1': str, 'alphabet2': str,
        'alpha2_cc1': str, 'alpha2_cc2': str
    }

    d = {
        'TRUE': True,
        'FALSE': False
    }

    def __init__(self):
        pass

    def build(self, X):
        """Build features depending on the assignment of parameter :py:attr:`~src.config.MLConf.classification_method`
        and return values (fX, y) as ndarray of floats.

        Returns
        -------
        fX: ndarray
            The computed features that will be used as input to ML classifiers.
        y: ndarray
            Binary labels {True, False} to train the classifiers.
        """
        fX = np.asarray(list(map(self.compute_features, X.geometry, X['dian_geom'])), dtype=float)
        fX = MinMaxScaler().fit_transform(fX)
        # fX = StandardScaler().fit_transform(fX)
        # fX = RobustScaler().fit_transform(fX)

        if np.any(np.isnan(fX)): print(np.where(np.isnan(fX)))
        if not np.any(np.isfinite(fX)): print('infinite')

        return fX

    def compute_features(self, poly1, poly2):
        """
        Depending on the group assigned to parameter :py:attr:`~src.config.MLConf.classification_method`,
        this method builds an ndarray of the following groups of features:

        * *basic*: various similarity measures, i.e.,
          :func:`~src.sim_measures.damerau_levenshtein`,
          :func:`~src.sim_measures.jaro`,
          :func:`~src.sim_measures.jaro_winkler` and the reversed one,
          :func:`~src.sim_measures.sorted_winkler`,
          :func:`~src.sim_measures.cosine`,
          :func:`~src.sim_measures.jaccard`,
          :func:`~src.sim_measures.strike_a_match`,
          :func:`~src.sim_measures.monge_elkan`,
          :func:`~src.sim_measures.soft_jaccard`,
          :func:`~src.sim_measures.davies`,
          :func:`~src.sim_measures.lgm_jaro_winkler` and the reversed one,
          :func:`~src.sim_measures.skipgrams`.
        * *basic_sorted*: sorted versions of similarity measures utilized in *basic* group, except for the
          :func:`~src.sim_measures.sorted_winkler`.
        * *lgm*: LGM-Sim variations that integrate, as internal, the similarity measures utilized in *basic* group,
          except for the :func:`~src.sim_measures.sorted_winkler`.

        Parameters
        ----------
        s1, s2: str
            Input toponyms.
        sorted: bool, optional
            Value of True indicate to build features for groups *basic* and *basic_sorted*, value of False only for *basic* group.
        lgm_sims: bool, optional
            Values of True or False indicate whether to build or not features for group *lgm*.

        Returns
        -------
        :obj:`list`
            It returns a list (vector) of features.
        """
        f = []

        geom1 = poly1
        geom2 = shape(poly2)

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
