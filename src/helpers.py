# coding= utf-8
# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

import os
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
        "Damerau-Levenshtein",
        "Jaro",
        "Jaro-Winkler",
        "Jaro-Winkler reversed",
        "Sorted Jaro-Winkler",
        # "Permuted Jaro-Winkler",
        "Cosine N-grams",
        "Jaccard N-grams",
        "Dice bigrams",
        "Jaccard skipgrams",
        "Monge-Elkan",
        "Soft-Jaccard",
        "Davis and De Salles",
        "Damerau-Levenshtein Sorted",
        "Jaro Sorted",
        "Jaro-Winkler Sorted",
        "Jaro-Winkler reversed Sorted",
        # "Sorted Jaro-Winkler Sorted",
        # "Permuted Jaro-Winkler Sorted",
        "Cosine N-grams Sorted",
        "Jaccard N-grams Sorted",
        "Dice bigrams Sorted",
        "Jaccard skipgrams Sorted",
        "Monge-Elkan Sorted",
        "Soft-Jaccard Sorted",
        "Davis and De Salles Sorted",
        "LinkGeoML Jaro-Winkler",
        "LinkGeoML Jaro-Winkler reversed",
        # "LSimilarity",
        "LSimilarity_wavg",
        # "LSimilarity_davies",
        # "LSimilarity_skipgram",
        # "LSimilarity_soft_jaccard",
        # "LSimilarity_strike_a_match",
        # "LSimilarity_cosine",
        # "LSimilarity_monge_elkan",
        # "LSimilarity_jaro_winkler",
        # "LSimilarity_jaro",
        # "LSimilarity_jaro_winkler_reversed",
        "LSimilarity_davies_wavg",
        "LSimilarity_skipgram_wavg",
        "LSimilarity_soft_jaccard_wavg",
        "LSimilarity_strike_a_match_wavg",
        "LSimilarity_cosine_wavg",
        "LSimilarity_jaccard_wavg",
        "LSimilarity_monge_elkan_wavg",
        "LSimilarity_jaro_winkler_wavg",
        "LSimilarity_jaro_wavg",
        "LSimilarity_jaro_winkler_reversed_wavg",
        "LSimilarity_l_jaro_winkler_wavg",
        "LSimilarity_l_jaro_winkler_reversed_wavg",
        # "LSimilarity_baseScore",
        # "LSimilarity_mismatchScore",
        # "LSimilarity_specialScore",
        "Avg LSimilarity_baseScore",
        "Avg LSimilarity_mismatchScore",
        "Avg LSimilarity_specialScore",
        # non metric features
        # "contains_str1",
        # "contains_str2",
        # "WordsNo_str1",
        # "WordsNo_str2",
        # "dashed_str1",
        # "dashed_str2",
        # "hasFreqTerm_str1",
        # "hasFreqTerm_str2",
        # "posOfHigherSim_str1_start",
        # "posOfHigherSim_str1_middle",
        # "posOfHigherSim_str1_end",
        # "posOfHigherSim_str2_start",
        # "posOfHigherSim_str2_middle",
        # "posOfHigherSim_str2_end",
    ]
