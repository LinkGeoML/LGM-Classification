# -*- coding: utf-8 -*-

"""Feature extraction and traditional classifiers for toponym matching.

Usage:
  feml.py [options]
  feml.py (-h | --help)
  feml.py --version

Options:
  -h --help                 show this screen.
  --version                 show version.
  -c <classifier_method>    various supported classifiers. [default: rf].
  -d <dataset-name>         The relative path to the directory of the script being run of the dataset to use for
                            experiments. [default: dataset-string-similarity.txt]
  --permuted                Use permuted Jaro-Winkler metrics. Default is False.
  --stemming                Perform stemming. Default is False.
  --sort                    Sort alphanumerically.
  --ev <evaluator_type>     Type of experiments to conduct. [default: SotAMetrics]
  --print                   Print only computed variables. Default is False.
  --accuracyresults         Store predicted results (TRUE/FALSE) in file. Default is False

Arguments:
  classifier_method:        'rf' (default)
                            'et'
                            'svm'
                            'xgboost'
  evaluator_type            'SotAMetrics' (default)
                            'SotAML'
                            'customFEML'
                            'DLearninng'

"""

import os, sys
import csv
import time
from collections import Counter
import re
from abc import ABCMeta, abstractmethod
import itertools
import math
import json
from operator import is_not
from functools import partial

# import configparser
from docopt import docopt
from nltk import SnowballStemmer, wordpunct_tokenize
from nltk.corpus import stopwords
from langdetect import detect, lang_detect_exception
import pycountry
from kitchen.text.converters import getwriter

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import preprocessing
from xgboost import XGBClassifier

import helpers
from external.datasetcreator import damerau_levenshtein, jaccard, jaro, jaro_winkler, monge_elkan, cosine, strike_a_match, \
    soft_jaccard, sorted_winkler, permuted_winkler, skipgram, davies
# from datasetcreator import detect_alphabet, fields


"""
Compute the Damerau-Levenshtein distance between two given
strings (s1 and s2)
https://www.guyrutenberg.com/2008/12/15/damerau-levenshtein-distance-in-python/
"""
# def damerau_levenshtein_distance(s1, s2):
#     d = {}
#     lenstr1 = len(s1)
#     lenstr2 = len(s2)
#     for i in xrange(-1, lenstr1 + 1):
#         d[(i, -1)] = i + 1
#     for j in xrange(-1, lenstr2 + 1):
#         d[(-1, j)] = j + 1
#
#     for i in xrange(lenstr1):
#         for j in xrange(lenstr2):
#             if s1[i] == s2[j]:
#                 cost = 0
#             else:
#                 cost = 1
#             d[(i, j)] = min(
#                 d[(i - 1, j)] + 1,  # deletion
#                 d[(i, j - 1)] + 1,  # insertion
#                 d[(i - 1, j - 1)] + cost,  # substitution
#             )
#             if i and j and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
#                 d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + cost)  # transposition
#
#     return d[lenstr1 - 1, lenstr2 - 1]

def get_langnm(str, lang_detect=False):
    lname = 'english'
    try:
        lname = pycountry.languages.get(alpha_2=detect(str)).name.lower() if lang_detect else 'english'
    except lang_detect_exception.LangDetectException as e:
        print(e)

    return lname

# Clean the string from stopwords, puctuations based on language detections feature
# Returned values #1: non-stopped words, #2: stopped words
def normalize_str(str, sstopwords=None, sorting=False, lang_detect=False):
    lname = get_langnm(str, lang_detect)

    tokens = wordpunct_tokenize(str)
    words = [word.lower() for word in tokens if word.isalpha()]
    stopwords_set = set(stopwords.words(lname)) if sstopwords is None else set(sstopwords)

    filtered_words = sorted_nicely(filter(lambda token: token not in stopwords_set, words)) if sorting else \
        filter(lambda token: token not in stopwords_set, words)
    stopped_words = sorted_nicely(filter(lambda token: token not in filtered_words, words)) if sorting else \
        filter(lambda token: token not in filtered_words, words)

    return filtered_words, stopped_words

def perform_stemming(str, lang_detect=False):
    try:
        lname = get_langnm(str, lang_detect)

        if lname in SnowballStemmer.languages: # See which languages are supported
            stemmer = SnowballStemmer(lname)  # Choose a language
            str = stemmer.stem(str)  # Stem a word
    except KeyError as e:
        pass
        # print("Unicode error for {0}\n".format(e))

    return str

def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

# def enum(*sequential, **named):
#     enums = dict(zip(sequential, range(len(sequential))), **named)
#     reverse = dict((value, key) for key, value in enums.iteritems())
#     enums['reverse_mapping'] = reverse
#     return type('Enum', (), enums)


class StaticValues:
    algorithms = {
        'damerau_levenshtein': damerau_levenshtein,
        'davies': davies,
        'skipgram': skipgram,
        'permuted_winkler': permuted_winkler,
        'sorted_winkler': sorted_winkler,
        'soft_jaccard': soft_jaccard,
        'strike_a_match': strike_a_match,
        'cosine': cosine,
        'monge_elkan': monge_elkan,
        'jaro_winkler': jaro_winkler,
        'jaro': jaro,
        'jaccard': jaccard,
    }

    methods = [["Damerau-Levenshtein", 0.55],
               ["Jaro", 0.75],
               ["Jaro-Winkler", 0.7],
               ["Jaro-Winkler reversed", 0.75],
               ["Sorted Jaro-Winkler", 0.7],
               ["Permuted Jaro-Winkler", 0.7],
               ["Cosine N-grams", 0.4],
               ["Jaccard N-grams", 0.25],
               ["Dice bigrams", 0.5],
               ["Jaccard skipgrams", 0.45],
               ["Monge-Elkan", 0.7],
               ["Soft-Jaccard", 0.6],
               ["Davis and De Salles", 0.65]]


class FEMLFeatures:
    # TODO to_be_removed = "()/.,:!'"  # check the list of chars
    # Returned vals: #1: str1 is subset of str2, #2 str2 is subset of str1
    def contains(self, strA, strB, sorting=False):
        strA, _ = normalize_str(strA, sorting)
        strB, _ = normalize_str(strB, sorting)
        return set(strA).issubset(set(strB)), set(strB).issubset(set(strA))

    def contains_freq_term(self, str, freqTerms=None):
        str, _ = normalize_str(str)
        return True if freqTerms != None and str in freqTerms else False

    def contains_specific_freq_term(self, str):
        pass

    def is_matched(self, str):
        """
        Finds out how balanced an expression is.
        With a string containing only brackets.

        >>> is_matched('[]()()(((([])))')
        False
        >>> is_matched('[](){{{[]}}}')
        True
        """
        opening = tuple('({[')
        closing = tuple(')}]')
        mapping = dict(zip(opening, closing))
        queue = []

        for letter in str:
            if letter in opening:
                queue.append(mapping[letter])
            elif letter in closing:
                if not queue or letter != queue.pop():
                    return False
        return not queue

    def hasEncoding_err(self, str):
        return self.is_matched(str)

    def containsAbbr(self, str):
        abbr = re.search(r"\b[A-Z]([A-Z\.]{1,}|[sr\.]{1,2})\b", str)
        return '-' if abbr is None else abbr.group()

    def containsTermsInParenthesis(self, str):
        tokens = re.split('[{\[(]', str)
        bflag = True if len(tokens) > 1 else False
        return bflag

    def containsDashConnected_words(self, str):
        """
        Hyphenated words are considered to be:
            * a number of word chars
            * followed by any number of: a single hyphen followed by word chars
        """
        is_dashed = re.search(r"\w+(?:-\w+)+", str)
        return False if is_dashed is None else True

    def no_of_words(self, str):
        str, _ = normalize_str(str)
        return len(set(str))

    def freq_ngram_tokens(self, str1, str2):
        pass

    def containsInPos(self, str1, str2):
        fvec_str1 = []
        fvec_str2 = []

        step = math.ceil(len(str1) / 3)
        for idx in xrange(0, len(str1), step):
            if str1[idx:idx + step]:
                sim = damerau_levenshtein(str1[idx:idx + step], str2)
                if sim >= 0.55:
                    fvec_str1.append(1)
                else:
                    fvec_str1.append(0)

        step = math.ceil(len(str2) / 3)
        for idx in xrange(0, len(str2), step):
            if str2[idx:idx + step]:
                sim = damerau_levenshtein(str1, str2[idx:idx + step])
                if sim >= 0.55:
                    fvec_str2.append(1)
                else:
                    fvec_str2.append(0)

        return fvec_str1, fvec_str2

    def fagiSim(self, strA, strB, stop_words):
        # TODO identifyAndExpandAbbr
        # remove punctuations and stopwords, lowercase, sort alphanumerically
        lstrA, _ = normalize_str(strA, sorting=True, sstopwords=stop_words)
        lstrB, _ = normalize_str(strB, sorting=True, sstopwords=stop_words)
        # TODO extractSpecialTerms
        base, mis = self.compareAndSplit_names(lstrA, lstrB)

    def compareAndSplit_names(self, listA, listB):
        mis = {'A': [], 'B': []}
        base = {'A': [],'B': []}

        cur = {'A': 0, 'B': 0}
        while cur['A'] < len(listA) and cur['B'] < len(listB):
            sim = jaro_winkler(listA[cur['A']], listB[cur['B']])
            if sim > 0.5:
                base['A'].append(listA[cur['A']])
                base['B'].append(listA[cur['B']])
                cur['A'] += 1
                cur['B'] += 1
            else:
                if listA[cur['A']] < listB[cur['B']]:
                    mis['B'].append(listB[cur['B']])
                    cur['B'] += 1
                else:
                    mis['A'].append(listB[cur['A']])
                    cur['A'] += 1

        if cur['A'] < len(listA):
            mis['A'].extend(listA[cur['A'] + 1:])
        if cur['B'] < len(listB):
            mis['B'].extend(listB[cur['B'] + 1:])

        return base, mis


class baseMetrics:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, size, accuracyresults=False):
        self.num_true_predicted_true = [0.0] * size
        self.num_true_predicted_false = [0.0] * size
        self.num_false_predicted_true = [0.0] * size
        self.num_false_predicted_false = [0.0] * size
        self.num_true = 0.0
        self.num_false = 0.0

        self.timer = 0.0
        self.timers = [0.0] * size
        self.result = {}
        self.file = None
        self.accuracyresults = accuracyresults
        if self.accuracyresults:
            self.file = open('dataset-accuracyresults-sim-metrics.txt', 'w+')

        self.predictedState = {
            'num_true_predicted_true': self.num_true_predicted_true,
            'num_true_predicted_false': self.num_true_predicted_false,
            'num_false_predicted_true': self.num_false_predicted_true,
            'num_false_predicted_false': self.num_false_predicted_false
        }

    def __del__(self):
        if self.accuracyresults:
            self.file.close()

    def preprocessing(self, row):
        if row['res'] == "TRUE": self.num_true += 1.0
        else: self.num_false += 1.0

    def transform(self, strA, strB, sorting=False, stemming=False):
        a = strA
        b = strB

        # print("{0} - norm: {1}".format(row['s1'], normalize_str(row['s1'])))
        if sorting:
            a = " ".join(sorted_nicely(a.split(" ")))
            b = " ".join(sorted_nicely(b.split(" ")))
        if stemming:
            a = perform_stemming(a)
            b = perform_stemming(b)
        a = a.decode('utf-8')
        b = b.decode('utf-8')
        return a, b

    @abstractmethod
    def evaluate(self, row, sorting=False, stemming=False, permuted=False, freqTerms=None):
        pass

    @abstractmethod
    def print_stats(self):
        pass

    def prediction(self, sim_id, pred_val, real_val):
        result = ""
        var_name = ""
        if real_val == 1.0:
            if pred_val >= StaticValues.methods[sim_id - 1][1]:
                var_name = 'num_true_predicted_true'
                result = "\tTRUE"
            else:
                var_name = 'num_true_predicted_false'
                result = "\tFALSE"
        else:
            if pred_val >= StaticValues.methods[sim_id - 1][1]:
                var_name = 'num_false_predicted_true'
                result = "\tTRUE"
            else:
                var_name = 'num_false_predicted_false'
                result = "\tFALSE"

        return result, var_name


class calcSotAMetrics(baseMetrics):
    def __init__(self, accures):
        super(calcSotAMetrics, self).__init__(len(StaticValues.methods), accures)

    def generic_evaluator(self, idx, algnm, str1, str2, match):
        start_time = time.time()
        sim = StaticValues.algorithms[algnm](str1, str2)
        res, varnm = self.prediction(idx, sim, match)
        self.timers[idx - 1] += (time.time() - start_time)
        self.predictedState[varnm][idx - 1] += 1.0
        return res

    def evaluate(self, row, sorting=False, stemming=False, permuted=False, freqTerms=None):
        tot_res = ""
        real = 1.0 if row['res'] == "TRUE" else 0.0

        row['s1'], row['s2'] = self.transform(row['s1'], row['s2'], sorting=sorting, stemming=stemming)

        tot_res += self.generic_evaluator(1, 'damerau_levenshtein', row['s1'], row['s2'], real)
        tot_res += self.generic_evaluator(8, 'jaccard', row['s1'], row['s2'], real)
        tot_res += self.generic_evaluator(2, 'jaro', row['s1'], row['s2'], real)
        tot_res += self.generic_evaluator(3, 'jaro_winkler', row['s1'], row['s2'], real)
        tot_res += self.generic_evaluator(4, 'jaro_winkler', row['s1'][::-1], row['s2'][::-1], real)
        tot_res += self.generic_evaluator(11, 'monge_elkan', row['s1'], row['s2'], real)
        tot_res += self.generic_evaluator(7, 'cosine', row['s1'], row['s2'], real)
        tot_res += self.generic_evaluator(9, 'strike_a_match', row['s1'], row['s2'], real)
        tot_res += self.generic_evaluator(12, 'soft_jaccard', row['s1'], row['s2'], real)
        tot_res += self.generic_evaluator(5, 'sorted_winkler', row['s1'], row['s2'], real)
        if permuted:
            tot_res += self.generic_evaluator(6, 'permuted_winkler', row['s1'], row['s2'], real)
        tot_res += self.generic_evaluator(10, 'skipgram', row['s1'], row['s2'], real)
        tot_res += self.generic_evaluator(13, 'davies', row['s1'], row['s2'], real)

        if self.accuracyresults:
            if real == 1.0:
                self.file.write("TRUE{0}".format(tot_res + "\n"))
            else:
                self.file.write("FALSE{0}".format(tot_res + "\n"))

    def print_stats(self):
        for idx in range(len(StaticValues.methods)):
            try:
                timer = ( self.timers[idx] / float( int( self.num_true + self.num_false ) ) ) * 50000.0
                acc = ( self.num_true_predicted_true[idx] + self.num_false_predicted_false[idx] ) / \
                      ( self.num_true + self.num_false )
                pre = ( self.num_true_predicted_true[idx] ) / \
                      ( self.num_true_predicted_true[idx] + self.num_false_predicted_true[idx] )
                rec = ( self.num_true_predicted_true[idx] ) / \
                      ( self.num_true_predicted_true[idx] + self.num_true_predicted_false[idx] )
                f1 = 2.0 * ( ( pre * rec ) / ( pre + rec ) )

                print("Metric = Supervised Classifier :" , StaticValues.methods[idx][0])
                print("Accuracy =", acc)
                print("Precision =", pre)
                print("Recall =", rec)
                print("F1 =", f1)
                print("Processing time per 50K records =", timer)
                print("")
                print("| Method\t\t& Accuracy\t& Precision\t& Recall\t& F1-Score\t& Time (50K Pairs)")
                print("||{0}\t& {1}\t& {2}\t& {3}\t& {4}\t& {5}".format(StaticValues.methods[idx][0], acc, pre, rec, f1, timer))
                print("")
                sys.stdout.flush()
            except ZeroDivisionError:
                pass
                # print "{0} is divided by zero\n".format(idx)

        # if results:
        #     return self.result


class calcCustomFEML(baseMetrics):
    names = [# "Nearest Neighbors",
        "Linear SVM", "RBF SVM", # "Gaussian Process",
        "Decision Tree", "Random Forest", "Neural Net", "AdaBoost", "Naive Bayes", "QDA",
        "ExtraTreeClassifier", "XGBOOST"
    ]

    def __init__(self, accures):
        self.X1 = []
        self.Y1 = []
        self.X2 = []
        self.Y2 = []
        self.scores = []
        self.importances = []
        self.classifiers = [
            # KNeighborsClassifier(4, n_jobs=3),
            SVC(kernel="linear", C=1.0, random_state=0),
            SVC(gamma=2, C=1, random_state=0),
            # GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs=3, warm_start=True),
            DecisionTreeClassifier(random_state=0, max_depth=100, max_features='auto'),
            RandomForestClassifier(n_estimators=600, random_state=0, n_jobs=3, max_depth=100),
            MLPClassifier(alpha=1, random_state=0),
            AdaBoostClassifier(DecisionTreeClassifier(max_depth=100), n_estimators=600, random_state=0),
            GaussianNB(),
            QuadraticDiscriminantAnalysis(),
            ExtraTreesClassifier(n_estimators=600, random_state=0, n_jobs=3, max_depth=100),
            XGBClassifier(n_estimators=3000, seed=0, nthread=3),
        ]
        super(calcCustomFEML, self).__init__(len(self.classifiers), accures)

    def evaluate(self, row, sorting=False, stemming=False, permuted=False, freqTerms=False):
        if row['res'] == "TRUE":
            if len(self.Y1) < ((self.num_true + self.num_false) / 2.0): self.Y1.append(1.0)
            else: self.Y2.append(1.0)
        else:
            if len(self.Y1) < ((self.num_true + self.num_false) / 2.0): self.Y1.append(0.0)
            else: self.Y2.append(0.0)

        row['s1'], row['s2'] = self.transform(row['s1'], row['s2'], sorting=sorting, stemming=stemming)

        start_time = time.time()
        sim1 = damerau_levenshtein(row['s1'], row['s2'])
        sim8 = jaccard(row['s1'], row['s2'])
        sim2 = jaro(row['s1'], row['s2'])
        sim3 = jaro_winkler(row['s1'], row['s2'])
        sim4 = jaro_winkler(row['s1'][::-1], row['s2'][::-1])
        sim11 = monge_elkan(row['s1'], row['s2'])
        sim7 = cosine(row['s1'], row['s2'])
        sim9 = strike_a_match(row['s1'], row['s2'])
        sim12 = soft_jaccard(row['s1'], row['s2'])
        sim5 = sorted_winkler(row['s1'], row['s2'])
        if permuted: sim6 = permuted_winkler(row['s1'], row['s2'])
        sim10 = skipgram(row['s1'], row['s2'])
        sim13 = davies(row['s1'], row['s2'])
        self.timer += (time.time() - start_time)
        if permuted:
            if len(self.X1) < ((self.num_true + self.num_false) / 2.0):
                self.X1.append([sim1, sim2, sim3, sim4, sim5, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            else: self.X2.append([sim1, sim2, sim3, sim4, sim5, sim6, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
        else:
            if len(self.X1) < ((self.num_true + self.num_false) / 2.0):
                self.X1.append([sim1, sim2, sim3, sim4, sim5, sim7, sim8, sim9, sim10, sim11, sim12, sim13])
            else: self.X2.append([sim1, sim2, sim3, sim4, sim5, sim7, sim8, sim9, sim10, sim11, sim12, sim13])

    def train_classifiers(self, polynomial=False):
        if polynomial:
            self.X1 = preprocessing.PolynomialFeatures().fit_transform(self.X1)
            self.X2 = preprocessing.PolynomialFeatures().fit_transform(self.X2)

        # iterate over classifiers
        for i, (name, clf) in enumerate(zip(self.names, self.classifiers)):
            scoreL = []
            importances = None
            print("Training {}...".format(name))

            clf.fit(np.array(self.X1), np.array(self.Y1))
            start_time = time.time()
            predictedL = list(clf.predict(np.array(self.X2)))
            self.timers[i] += (time.time() - start_time)
            if hasattr(clf, "feature_importances_"):
                importances = clf.feature_importances_
            elif hasattr(clf, "coef_"):
                importances = clf.coef_.ravel()
            scoreL.append(clf.score(np.array(self.X2), np.array(self.Y2)))

            clf.fit(np.array(self.X2), np.array(self.Y2))
            start_time = time.time()
            predictedL += list(clf.predict(np.array(self.X1)))
            self.timers[i] += (time.time() - start_time)
            if hasattr(clf, "feature_importances_"):
                importances += clf.feature_importances_
            elif hasattr(clf, "coef_"):
                # TODO when coef_ is added to importances that already contains another one, it throws a
                # ValueError: output array is read-only
                importances = clf.coef_.ravel()
            scoreL.append(clf.score(np.array(self.X1), np.array(self.Y1)))

            self.timers[i] += self.timer
            self.importances.append(importances)
            self.scores.append(scoreL)

            print("Matching records...")
            real = self.Y2 + self.Y1
            for pos in range(len(real)):
                if real[pos] == 1.0:
                    if predictedL[pos] == 1.0:
                        self.num_true_predicted_true[i] += 1.0
                        if self.accuracyresults:
                            self.file.write("TRUE\tTRUE\n")
                    else:
                        self.num_true_predicted_false[i] += 1.0
                        if self.accuracyresults:
                            self.file.write("TRUE\tFALSE\n")
                else:
                    if predictedL[pos] == 1.0:
                        self.num_false_predicted_true[i] += 1.0
                        if self.accuracyresults:
                            self.file.write("FALSE\tTRUE\n")
                    else:
                        self.num_false_predicted_false[i] += 1.0
                        if self.accuracyresults:
                            self.file.write("FALSE\tFALSE\n")

            # if hasattr(clf, "decision_function"):
            #     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            # else:
            #     Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    def print_stats(self):
        for idx, (name, clf) in enumerate(zip(self.names, self.classifiers)):
            try:
                timer = ( self.timers[idx] / float( int( self.num_true + self.num_false ) ) ) * 50000.0
                acc = ( self.num_true_predicted_true[idx] + self.num_false_predicted_false[idx] ) / \
                      ( self.num_true + self.num_false )
                pre = ( self.num_true_predicted_true[idx] ) / \
                      ( self.num_true_predicted_true[idx] + self.num_false_predicted_true[idx] )
                rec = ( self.num_true_predicted_true[idx] ) / \
                      ( self.num_true_predicted_true[idx] + self.num_true_predicted_false[idx] )
                f1 = 2.0 * ( ( pre * rec ) / ( pre + rec ) )

                print("Metric = Supervised Classifier :" , name)
                print("Score (X2, X1) = ", self.scores[idx][0], self.scores[idx][1])
                print("Accuracy =", acc)
                print("Precision =", pre)
                print("Recall =", rec)
                print("F1 =", f1)
                print("Processing time per 50K records =", timer)
                print("Number of training instances =", min(len(self.Y1), len(self.Y2)))
                print("")
                print("| Method\t\t& Accuracy\t& Precision\t& Recall\t& F1-Score\t& Time (50K Pairs)")
                print("||{0}\t& {1}\t& {2}\t& {3}\t& {4}\t& {5}".format(name, acc, pre, rec, f1, timer))
                print("")
                sys.stdout.flush()

                try:
                    importances = self.importances[idx] / 2.0
                    indices = np.argsort(importances)[::-1]
                    for f in range(importances.shape[0]):
                        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
                except TypeError:
                    print("The classifier {} does not expose \"coef_\" or \"feature_importances_\" attributes".format(name))

                # if hasattr(clf, "feature_importances_"):
                #         # if results:
                #         #     result[indices[f]] = importances[indices[f]]
                print("")
                sys.stdout.flush()
            except ZeroDivisionError:
                pass

        # if results:
        #     return self.result


class calcDLearning(baseMetrics):
    pass


class calcSotAML(baseMetrics):
    pass


class Evaluator:
    evaluatorType_action = {
        'SotAMetrics': calcSotAMetrics,
        'SotAML': calcSotAML,
        'customFEML': calcCustomFEML,
        'DLearning': calcDLearning,
    }

    def __init__(self, permuted=False, stemming=False, sorting=False, do_printing=False):
        self.permuted = permuted
        self.stemming = stemming
        self.sorting = sorting
        self.only_printing = do_printing

        self.freqTerms = {
            'gram': Counter(),
            '2gram_1': Counter(), '3gram_1': Counter(), '2gram_2': Counter(), '3gram_2': Counter(), '3gram_3': Counter(),
        }
        self.stop_words = []
        self.abbr = {'A': [], 'B': []}
        self.fsorted = None
        self.evalClass = None

    def getTMabsPath(self, str):
        return os.path.join(os.path.abspath('../Toponym-Matching'), 'dataset', str)

    def initialize(self, dataset, evalType='SotAMetrics', accuracyresults=False):
        try:
            self.evalClass = self.evaluatorType_action[evalType](accuracyresults)
        except KeyError:
            print("Unkown method")
            return 1

        # These are the available languages with stopwords from NLTK
        NLTKlanguages = ["dutch", "finnish", "german", "italian", "portuguese", "spanish", "turkish", "danish",
                         "english", "french", "hungarian", "norwegian", "russian", "swedish"]
        for lang in NLTKlanguages:
            self.stop_words.extend(stopwords.words(lang))

        FREElanguages = [
            'zh', 'ja', 'id', 'fa', 'ar', 'bn', 'ro', 'th', 'el', 'hi', 'gl', 'hy', 'ko', 'yo', 'vi',
            'sw', 'so', 'he', 'ha', 'br', 'af', 'ku', 'ms', 'tl', 'ur'
        ]
        if os.path.isfile(os.path.join(os.getcwd(), 'stopwords-iso.json')):
            with open("stopwords-iso.json", "r") as read_file:
                data = json.load(read_file)
                for lang in FREElanguages:
                    self.stop_words.extend(data[lang])

        if self.only_printing:
            self.fsorted = open('sorted.csv', 'w')
            self.fsorted.write("Original_A\tSorted_A\tOriginal_B\tSorted_B\n")

        feml = FEMLFeatures()
        with open(dataset) as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                    delimiter='\t')
            for row in reader:
                self.evalClass.preprocessing(row)

                # Calc frequent terms
                # (str1, str2)
                for str in ['s1', 's2']:
                    fterms, stop_words = normalize_str(row[str], self.stop_words)
                    for term in fterms:
                        self.freqTerms['gram'][term] += 1
                    for ngram in list(itertools.chain.from_iterable(
                        [[fterms[i:i + n] for i in range(len(fterms) - (n - 1))] for n in [2, 3]])):
                        if len(ngram) == 2:
                            self.freqTerms['2gram_1'][ngram[0]] += 1
                            self.freqTerms['2gram_2'][ngram[1]] += 1
                        else:
                            self.freqTerms['3gram_1'][ngram[0]] += 1
                            self.freqTerms['3gram_2'][ngram[1]] += 1
                            self.freqTerms['3gram_3'][ngram[2]] += 1

                # calc the number of abbr that exist
                self.abbr['A'].append(feml.containsAbbr(row['s1']))
                self.abbr['B'].append(feml.containsAbbr(row['s2']))

                if self.only_printing:
                    self.fsorted.write(row['s1'])
                    self.fsorted.write("\t")
                    self.fsorted.write(" ".join(sorted_nicely(row['s1'].split(" "))))
                    self.fsorted.write("\t")
                    self.fsorted.write(row['s2'])
                    self.fsorted.write("\t")
                    self.fsorted.write(" ".join(sorted_nicely(row['s2'].split(" "))))
                    self.fsorted.write("\n")

        if self.only_printing:
            self.fsorted.close()
            self.do_the_printing()

        return 0

    def do_the_printing(self):
        print("Printing 10 most common single freq terms...")
        print("gram: {0}".format(self.freqTerms['gram'].most_common(20)))

        print("Printing 10 most common freq terms in bigrams...")
        print("bi-gram pos 1: {0}".format(self.freqTerms['2gram_1'].most_common(20)))
        print("\t pos 2: {0}".format(self.freqTerms['2gram_2'].most_common(20)))

        print("Printing 10 most common freq terms in trigrams...")
        print("tri-gram pos 1: {0}".format(self.freqTerms['3gram_1'].most_common(20)))
        print("\t pos 2: {0}".format(self.freqTerms['3gram_2'].most_common(20)))
        print("\t pos 3: {0}".format(self.freqTerms['3gram_3'].most_common(20)))

        print("Number of abbr found: {0}".format(len(filter(partial(is_not, '-'), self.abbr['A'])) +
                                                 len(filter(partial(is_not, '-'), self.abbr['B']))))

        with open("freqTerms.csv", "w") as f:
            f.write('gram\t')
            f.write('bigram_pos_1\t')
            f.write('bigram_pos_2\t')
            f.write('trigram_pos_1\t')
            f.write('trigram_pos_2\t')
            f.write('trigram_pos_3')
            f.write('\n')

            sorted_freq_gram_terms = self.freqTerms['gram'].most_common()
            sorted_freq_bigram_terms_pos1 = self.freqTerms['2gram_1'].most_common()
            sorted_freq_bigram_terms_pos2 = self.freqTerms['2gram_2'].most_common()
            sorted_freq_trigram_terms_pos1 = self.freqTerms['3gram_1'].most_common()
            sorted_freq_trigram_terms_pos2 = self.freqTerms['3gram_2'].most_common()
            sorted_freq_trigram_terms_pos3 = self.freqTerms['3gram_3'].most_common()

            min_top = min(
                len(sorted_freq_gram_terms),
                len(sorted_freq_bigram_terms_pos1),
                len(sorted_freq_bigram_terms_pos2),
                len(sorted_freq_trigram_terms_pos1),
                len(sorted_freq_trigram_terms_pos2),
                len(sorted_freq_trigram_terms_pos3),
            )

            for i in range(min_top):
                f.write("{},{}\t".format(sorted_freq_gram_terms[i][0], sorted_freq_gram_terms[i][1]))
                f.write("{},{}\t".format(sorted_freq_bigram_terms_pos1[i][0], sorted_freq_bigram_terms_pos1[i][1]))
                f.write("{},{}\t".format(sorted_freq_bigram_terms_pos2[i][0], sorted_freq_bigram_terms_pos2[i][1]))
                f.write("{},{}\t".format(sorted_freq_trigram_terms_pos1[i][0], sorted_freq_trigram_terms_pos1[i][1]))
                f.write("{},{}\t".format(sorted_freq_trigram_terms_pos2[i][0], sorted_freq_trigram_terms_pos2[i][1]))
                f.write("{},{}\t".format(sorted_freq_trigram_terms_pos3[i][0], sorted_freq_trigram_terms_pos3[i][1]))
                f.write('\n')

        with open("abbr.csv", "w") as f:
            f.write('strA\tstrB\tline_pos\n')
            for i in range(min(len(self.abbr['A']), len(self.abbr['B']))):
                if self.abbr['A'][i] != '-' or self.abbr['B'][i] != '-':
                    f.write("{}\t{}\t{}\n".format(self.abbr['A'][i], self.abbr['B'][i], i))

    def evaluate_metrics(self, dataset='dataset-string-similarity.txt'):
        if self.evalClass is not None:
            print("Reading dataset...")
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset)) as csvfile:
                reader = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "c1", "c2", "a1", "a2", "cc1", "cc2"],
                                        delimiter='\t')

                for row in reader:
                    self.evalClass.evaluate(row, self.sorting, self.stemming, self.permuted, self.freqTerms)
                if hasattr(self.evalClass, "train_classifiers"): self.evalClass.train_classifiers()
                self.evalClass.print_stats()


def main(args):
    UTF8Writer = getwriter('utf8')
    sys.stdout = UTF8Writer(sys.stdout)

    dataset_path = args['-d']

    eval = Evaluator(args['--permuted'], args['--stemming'], args['--sort'], args['--print'])
    full_dataset_path = eval.getTMabsPath(dataset_path)

    if os.path.isfile(full_dataset_path):
        eval.initialize(full_dataset_path, args['--ev'], args['--accuracyresults'])
        if args['--print']:
            sys.exit(0)

        eval.evaluate_metrics(full_dataset_path)
    else: print("No file {0} exists!!!\n".format(full_dataset_path))


if __name__ == "__main__":
    arguments = docopt(__doc__, version='FE-ML 0.1')
    main(arguments)
