import numpy as np
from functools import partial
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from xgboost.sklearn import XGBRegressor
from aem.logger import aemlogger as log


class XGBQuantileRegressor(XGBRegressor):
    def __init__(self,
                 alpha, delta, thresh, variance,
                 **kwargs
                 ):
        self.alpha = alpha
        self.delta = delta
        self.thresh = thresh
        self.variance = variance

        super(XGBQuantileRegressor, self).__init__(**kwargs)

    def fit(self, X, y, **kwargs):
        objective = partial(XGBQuantileRegressor.quantile_loss, alpha=self.alpha, delta=self.delta,
                            threshold=self.thresh, var=self.variance)
        super().set_params(objective=objective)
        super().fit(X, y)
        return self

    def predict(self, X, **kwargs):
        return super().predict(X)

    def score(self, X, y, **kwargs):
        y_pred = super().predict(X)
        score = self.quantile_score(y, y_pred, self.alpha)
        score = 1. / score
        return score

    @staticmethod
    def quantile_loss(y_true, y_pred, alpha, delta, threshold, var):
        x = y_true - y_pred
        grad = (x < (alpha - 1.0) * delta) * (1.0 - alpha) - \
               ((x >= (alpha - 1.0) * delta) & (x < alpha * delta)) * x / delta - \
               alpha * (x > alpha * delta)
        hess = ((x >= (alpha - 1.0) * delta) & (x < alpha * delta)) / delta

        grad = (np.abs(x) < threshold) * grad - (np.abs(x) >= threshold) * (
            2 * np.random.randint(2, size=len(y_true)) - 1.0) * var
        hess = (np.abs(x) < threshold) * hess + (np.abs(x) >= threshold)
        return grad, hess

    # @staticmethod
    # def original_quantile_loss(y_true, y_pred, alpha, delta):
    #     x = y_true - y_pred
    #     grad = (x < (alpha - 1.0) * delta) * (1.0 - alpha) - (
    #         (x >= (alpha - 1.0) * delta) & (x < alpha * delta)) * x / delta - alpha * (x > alpha * delta)
    #     hess = ((x >= (alpha - 1.0) * delta) & (x < alpha * delta)) / delta
    #     return grad, hess

    @staticmethod
    def quantile_score(y_true, y_pred, alpha):
        score = XGBQuantileRegressor.quantile_cost(x=y_true - y_pred, alpha=alpha)
        score = np.sum(score)
        return score

    @staticmethod
    def quantile_cost(x, alpha):
        return (alpha - 1.0) * x * (x < 0) + alpha * x * (x >= 0)

    @staticmethod
    def get_split_gain(gradient, hessian, l=1):
        split_gain = list()
        for i in range(gradient.shape[0]):
            split_gain.append(np.sum(gradient[:i]) / (np.sum(hessian[:i]) + l) + np.sum(gradient[i:]) / (
                    np.sum(hessian[i:]) + l) - np.sum(gradient) / (np.sum(hessian) + l))

        return np.array(split_gain)


class QuantileXGB(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        mean_model_params,
        upper_quantile_params,
        lower_quantile_params
    ):
        self.mean_model_params = mean_model_params
        self.upper_quantile_params = upper_quantile_params
        self.lower_quantile_params = lower_quantile_params
        self.gb = XGBRegressor(**mean_model_params)
        self.gb_quantile_upper = XGBQuantileRegressor(**upper_quantile_params)
        self.gb_quantile_lower = XGBQuantileRegressor(**lower_quantile_params)
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
        Vy = ((qu_ - ql_) / (norm.ppf(self.upper_alpha) - norm.ppf(self.lower_alpha))) ** 2

        # to make gbm quantile model consistent with other quantile based models
        ql, qu = norm.interval(interval, loc=Ey, scale=np.sqrt(Vy))

        return Ey, Vy, ql, qu


class QuantileGradientBoosting(BaseEstimator, RegressorMixin):
    """
    Bespoke Quantile Gradient Boosting Regression implementation.
    """
    def __init__(
        self,
        mean_model_params,
        upper_quantile_params,
        lower_quantile_params
    ):
        """
        :param mean_model_params:
        :param upper_quantile_params:
        :param lower_quantile_params:
        """
        self.mean_model_params = mean_model_params
        self.upper_quantile_params = upper_quantile_params
        self.lower_quantile_params = lower_quantile_params
        self.gb = GradientBoostingRegressor(**mean_model_params)
        self.gb_quantile_upper = GradientBoostingRegressor(**upper_quantile_params)
        self.gb_quantile_lower = GradientBoostingRegressor(**lower_quantile_params)
        self.upper_alpha = upper_quantile_params['alpha']
        self.lower_alpha = lower_quantile_params['alpha']

    @staticmethod
    def collect_prediction(regressor, X_test):
        y_pred = regressor.predict(X_test)
        return y_pred

    def fit(self, X, y, **kwargs):
        log.info('Fitting gb base model')
        self.gb.fit(X, y, **kwargs)
        log.info('Fitting gb upper quantile model')
        self.gb_quantile_upper.fit(X, y, **kwargs)
        log.info('Fitting gb lower quantile model')
        self.gb_quantile_lower.fit(X, y, **kwargs)

    def predict(self, X, *args, **kwargs):
        return self.predict_dist(X, *args, **kwargs)[0]

    def predict_dist(self, X, interval=0.95):
        Ey = self.gb.predict(X)

        ql_ = self.collect_prediction(self.gb_quantile_lower, X)
        qu_ = self.collect_prediction(self.gb_quantile_upper, X)
        # divide qu - ql by the normal distribution Z value diff between the quantiles, square for variance
        Vy = ((qu_ - ql_) / (norm.ppf(self.upper_alpha) - norm.ppf(self.lower_alpha))) ** 2

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
            Vy += (dt.predict(X) - Ey) ** 2

        Vy /= len(self.estimators_)
        # FIXME what if elements of Vy are zero?
        ql, qu = norm.interval(interval, loc=Ey, scale=np.sqrt(Vy))

        return Ey, Vy, ql, qu


modelmaps = {
    'xgboost': XGBRegressor,
    'gradientboost': GradientBoostingRegressor,
    'quantilegb': QuantileGradientBoosting,
    'randomforest': QuantileRandomForestRegressor,
    'quantilexgb': QuantileXGB,
}
