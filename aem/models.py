from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from xgboost.sklearn import XGBRegressor


class QuantileGradientBoosting(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        alpha=0.95,
        **kwargs
    ):
        if 'loss' in kwargs:
            loss = kwargs.pop('loss')
        else:
            loss = 'ls'

        self.alpha = alpha
        self.gb = GradientBoostingRegressor(loss=loss, **kwargs)
        self.gb_quantile_upper = GradientBoostingRegressor(
            loss='quantile',
            alpha=alpha,
            **kwargs
        )
        self.gb_quantile_lower = GradientBoostingRegressor(
            loss='quantile',
            alpha=1 - alpha,
            **kwargs
        )

    @staticmethod
    def collect_prediction(regressor, X_test):
        y_pred = regressor.predict(X_test)
        return y_pred

    def fit(self, X, y, **kwargs):
        self.gb.fit(X, y, **kwargs)
        self.gb_quantile_upper.fit(X, y, **kwargs)
        self.gb_quantile_lower.fit(X, y, **kwargs)

    def predict(self, X, *args, **kwargs):
        return self.predict_dist(X, *args, **kwargs)[0]

    def predict_dist(self, X, *args, **kwargs):
        Ey = self.gb.predict(X)

        ql = self.collect_prediction(self.gb_quantile_lower, X)
        qu = self.collect_prediction(self.gb_quantile_upper, X)
        # divide qu - ql by the normal distribution Z value diff between the quantiles, square for variance
        Vy = ((qu - ql)/(norm.ppf(self.alpha) - norm.ppf(1-self.alpha))) ** 2

        return Ey, Vy, ql, qu


modelmaps = {
    'xgboost': XGBRegressor,
    'gradientboost': GradientBoostingRegressor,
    'quantilegb': QuantileGradientBoosting,
}
