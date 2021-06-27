import json
import joblib
import numpy as np
from sklearn.metrics import (
    explained_variance_score,
    make_scorer,
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from aem.config import Config
from aem.models import modelmaps
from aem.logger import aemlogger as log


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


regression_metrics = {
    'r2_score': lambda y, py, w:  r2_score(y, py, sample_weight=w),
    'expvar': lambda y, py, w: explained_variance_score(y, py, sample_weight=w),
    'mse': lambda y, py, w: mean_squared_error(y, py, sample_weight=w),
    'mae': lambda y, py, w: mean_absolute_error(y, py, sample_weight=w),
}


def on_step(optim_result):
    score = searchcv.best_score_
    print("best score: %s" % score)
    if score >= 0.98:
        print('Interrupting!')
        return True


def bayesian_optimisation(X_train, y_train, w_train, X_val, y_val, w_val, conf: Config):

    def my_custom_scorer(reg, y, y_pred):
        """learn on train data and predict on test data to ensure total out of sample validation"""
        y_val_pred = reg.predict(X_val)
        r2 = r2_score(y_val, y_val_pred, sample_weight=w_val)
        return r2

    reg = modelmaps[conf.algorithm](** conf.model_params)
    search_space = {k: eval(v) for k, v in conf.opt_params_space.items()}
    searchcv = BayesSearchCV(
        reg,
        search_spaces=search_space,
        ** conf.opt_searchcv_params,
        scoring=my_custom_scorer,
        fit_params={'sample_weight': w_train}
    )
    log.info(f"Optimising params using BayesSearchCV .....")

    searchcv.fit(X_train, y_train)

    joblib.dump(searchcv.best_params_, conf.searchcv_file)
    log.info(f"saved bayes searchcv output in {conf.searchcv_file}")

    opt_model = modelmaps[conf.algorithm](** searchcv.best_params_)
    opt_model.fit(X_train, y_train, sample_weight=w_train)

    train_scores = score_model(opt_model, X_train, y_train, w_train)
    val_scores = score_model(opt_model, X_val, y_val, w_val)

    all_scores = {'train_scores': train_scores, 'val_scores': val_scores}

    score_string = "Optimised model scores:\n"

    # report model performance on screen
    for k, scores in all_scores.items():
        score_string += f"{k}:\n"
        for metric, score in scores.items():
            score_string += "{}\t= {}\n".format(metric, score)
    log.info(score_string)

    # and also save a scores json file on disc
    with open(conf.optimised_model_scores, 'w') as f:
        json.dump(all_scores, f, sort_keys=True, indent=4)


def score_model(trained_model, X, y, w=None):
    scores = {}
    y_pred = trained_model.predict(X)
    for k, m in regression_metrics.items():
        scores[k] = m(y, y_pred, w)
    return scores
