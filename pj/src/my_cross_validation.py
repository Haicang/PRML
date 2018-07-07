"""
This file is based on [https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/model_selection/_validation.py#L39]

Because the data is pre-split, and sklearn only support indexes in cross_validation,
so I modified 2 functions:
    cross_validate
    _fit_and_score
and rename them as:
    my_cross_validate
    _my_fit_and_score

The docs for this 2 functions is similar to the raw functions,
except for that I use the pre-split dataset.

"""


import warnings
import numbers
import time

import numpy as np
import scipy.sparse as sp

from sklearn.base import is_classifier, clone
from sklearn.utils import indexable, check_random_state, safe_indexing
from sklearn.utils.deprecation import DeprecationDict
from sklearn.utils.validation import _is_arraylike, _num_samples
from sklearn.utils.metaestimators import _safe_split
from sklearn.externals.joblib import Parallel, delayed, logger
# from sklearn.externals.six.moves import zip
from sklearn.metrics.scorer import check_scoring, _check_multimetric_scoring
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import _score, _aggregate_score_dicts, _index_param_value
from sklearn.preprocessing import LabelEncoder


# __all__ = ['cross_validate', 'cross_val_score', 'cross_val_predict',
#            'permutation_test_score', 'learning_curve', 'validation_curve']


def my_cross_validate(estimator, X, y, groups=None, scoring=None, cv=None,
                    n_jobs=1, verbose=0, fit_params=None,
                    pre_dispatch='2*n_jobs', return_train_score="warn"):
    """
    In this project, data is pre-split,
    and estimator is always a classifier so:
    cv: None (do not use)
    groups: None (do not use)
    X: ((X_train1, X_test1), (X_train2, X_test2), ...)
    y: ((y_train1, y_test1), (y_train2, y_test2), ...)
    """

    # X, y, groups = indexable(X, y, groups)

    # cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorers, _ = _check_multimetric_scoring(estimator, scoring=scoring)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    scores = parallel(
        delayed(_my_fit_and_score)(
            clone(estimator), Xi, yi, scorers, verbose, None,
            fit_params, return_train_score=return_train_score,
            return_times=True)
        for Xi, yi in zip(X, y))

    if return_train_score:
        train_scores, test_scores, fit_times, score_times = zip(*scores)
        train_scores = _aggregate_score_dicts(train_scores)
    else:
        test_scores, fit_times, score_times = zip(*scores)
    test_scores = _aggregate_score_dicts(test_scores)

    # TODO: replace by a dict in 0.21
    ret = DeprecationDict() if return_train_score == 'warn' else {}
    ret['fit_time'] = np.array(fit_times)
    ret['score_time'] = np.array(score_times)

    for name in scorers:
        ret['test_%s' % name] = np.array(test_scores[name])
        if return_train_score:
            key = 'train_%s' % name
            ret[key] = np.array(train_scores[name])
            if return_train_score == 'warn':
                message = (
                    'You are accessing a training score ({!r}), '
                    'which will not be available by default '
                    'any more in 0.21. If you need training scores, '
                    'please set return_train_score=True').format(key)
                # warn on key access
                ret.add_warning(key, message, FutureWarning)

    return ret


def _my_fit_and_score(estimator, X, y, scorer, verbose,
                    parameters, fit_params, return_train_score=False,
                    return_parameters=False, return_n_test_samples=False,
                    return_times=False, error_score='raise'):

    if verbose > 1:
        if parameters is None:
            msg = ''
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v)
                                    for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    fit_params = fit_params if fit_params is not None else {}

    test_scores = {}
    train_scores = {}
    if parameters is not None:
        estimator.set_params(**parameters)

    X_train, X_test = X
    y_train, y_test = y

    start_time = time.time()

    is_multimetric = not callable(scorer)
    n_scorers = len(scorer.keys()) if is_multimetric else 1

    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            if is_multimetric:
                test_scores = dict(zip(scorer.keys(),
                                       [error_score, ] * n_scorers))
                if return_train_score:
                    train_scores = dict(zip(scorer.keys(),
                                            [error_score, ] * n_scorers))
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
            warnings.warn("Classifier fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%r" % (error_score, e), FitFailedWarning)
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)")

    else:
        fit_time = time.time() - start_time
        # _score will return dict if is_multimetric is True
        test_scores = _score(estimator, X_test, y_test, scorer, is_multimetric)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = _score(estimator, X_train, y_train, scorer,
                                  is_multimetric)

    if verbose > 2:
        if is_multimetric:
            for scorer_name, score in test_scores.items():
                msg += ", %s=%s" % (scorer_name, score)
        else:
            msg += ", score=%s" % test_scores
    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = "%s, total=%s" % (msg, logger.short_format_time(total_time))
        print("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))

    ret = [train_scores, test_scores] if return_train_score else [test_scores]

    if return_n_test_samples:
        ret.append(_num_samples(X_test))
    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    return ret
