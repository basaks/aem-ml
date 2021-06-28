import numpy as np
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from xgboost.sklearn import XGBRegressor
from aem.logger import aemlogger as log


class QuantileGradientBoosting(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        mean_model_params,
        upper_quantile_params,
        lower_quantile_params
    ):
        self.gb = GradientBoostingRegressor(** mean_model_params)
        self.gb_quantile_upper = GradientBoostingRegressor(** upper_quantile_params)
        self.gb_quantile_lower = GradientBoostingRegressor(** lower_quantile_params)
        self.upper_alpha = upper_quantile_params['alpha']
        self.lower_alpha = lower_quantile_params['alpha']

    @staticmethod
    def collect_prediction(regressor, X_test):
        y_pred = regressor.predict(X_test)
        return y_pred

    def fit(self, X, y, **kwargs):
        log.info('Fitting xgb base model')
        self.gb.fit(X, y, **kwargs)
        log.info('Fitting xgb upper quantile model')
        self.gb_quantile_upper.fit(X, y, **kwargs)
        log.info('Fitting xgb lower quantile model')
        self.gb_quantile_lower.fit(X, y, **kwargs)

    def predict(self, X, *args, **kwargs):
        return self.predict_dist(X, *args, **kwargs)[0]

    def predict_dist(self, X, interval=0.95):
        Ey = self.gb.predict(X)

        ql_ = self.collect_prediction(self.gb_quantile_lower, X)
        qu_ = self.collect_prediction(self.gb_quantile_upper, X)
        # divide qu - ql by the normal distribution Z value diff between the quantiles, square for variance
        Vy = ((qu_ - ql_)/(norm.ppf(self.upper_alpha) - norm.ppf(self.lower_alpha))) ** 2

        # to make gbm quantile model consistent with other quantile based models
        ql, qu = norm.interval(interval, loc=Ey, scale=np.sqrt(Vy))

        return Ey, Vy, ql, qu


class QuantileRandomForestRegressor(RandomForestRegressor):
    """
    Implements a "probabilistic" output by looking at the variance of the
    decision tree estimator ouputs.
    """

    def predict_dist(self, X, interval=0.95):
        Ey = self.predict(X)
        Vy = np.zeros_like(Ey)
        for dt in self.estimators_:
            Vy += (dt.predict(X) - Ey)**2

        Vy /= len(self.estimators_)
        # FIXME what if elements of Vy are zero?
        ql, qu = norm.interval(interval, loc=Ey, scale=np.sqrt(Vy))

        return Ey, Vy, ql, qu


modelmaps = {
    'xgboost': XGBRegressor,
    'gradientboost': GradientBoostingRegressor,
    'quantilegb': QuantileGradientBoosting,
    'randomforest': QuantileRandomForestRegressor,
}
