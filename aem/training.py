import numpy as np

from sklearn.metrics import make_scorer, r2_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from aem.config import Config

np.random.seed(6)

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


def my_custom_scorer(reg, X, y, X_val, y_val, w_val):
    """learn on train data and predict on test data to ensure total out of sample validation"""
    y_val_pred = reg.predict(X_val)
    r2 = r2_score(y_val, y_val_pred, sample_weight=w_val)
    return r2


def on_step(optim_result):
    score = searchcv.best_score_
    print("best score: %s" % score)
    if score >= 0.98:
        print('Interrupting!')
        return True


def bayesian_optimisation(X, y, conf: Config):
    reg = conf.algorithm

    searchcv = BayesSearchCV(
        reg,
        search_spaces=conf.opt_space,
        n_iter=48,
        cv=2,  # use 2 when using custom scoring using X_test
        verbose=1000,
        n_points=24,
        n_jobs=12,
        scoring=my_custom_scorer
    )

    searchcv.fit(X, y)


def train_model(X, y, conf: Config):
    model = conf.algorithm(** conf.model_params)
    model.fit(X, y)