import numpy as np
from functools import partial
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from xgboost.sklearn import XGBRegressor
from aem.logger import aemlogger as log
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


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
    def __init__(
        self,
        mean_model_params,
        upper_quantile_params,
        lower_quantile_params
    ):
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


def optimizer(args):
    pass


# class TensorFlow(RandomForestRegressor):
#     """
#     Implements a Tensor Flow probability output using regression and probabilistic layers
#     """
#
#     # Simple Linear Regression (TF)
#
#     def negloglik = lambda y, p_y: -p_y.logprob(y)
#
#     def modeltf = tf.keras.Sequential([
#         tf.keras.layer.Dense(1),
#         tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1))
#     ])
#
#     def modeltf.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=negloglik)
#
#     def modeltf.fit(x, y, epochs =500, verbose=False)
#
#     def yhattf = modeltf(x_tst)
#
#     # Known Unknowns TF
#
#     def modelKunk = tfk.Sequential([
#         tf.keras.layers.Dense(1 + 1)
#         tfp.layers.DistributionLambda(
#             lambda t: tfd.Normal(loc=t[..., :1],
#                                  scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]))),
#         ])
#
#     def modelKunk.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=negloglik)
#
#     def modelKunk.fit(x, y, epochs = 500, verbose = False)
#
#     def yhatKunk = modelKunk(x_tst)
#
#     # Unknown Unknowns TF
#
#     def modelUkunk = tf.keras.Sequential([
#         tfp.layers.DenseVariational(1, posterior_mean_field, prior_trainable),
#         tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
#         ])
#
#     def modelUkunk.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), loss=negloglik)
#
#     def modelUkunk.fit(x, y, epochs = 500, verbose = False)
#
#     def yhatsUkunk = [modelUkunk(x_tst) for i in range(100)]
#
#     # Known and Unknown Unknowns TF
#
#     def modelKUunk = tf.keras.Sequential([
#         tfp.layers.DenseVariational(1 + 1, posterior_mean_field, prior_trainable),
#         tfp.layers.DistributionLambda(
#             lambda t: tfd.Normal(loc=t[..., :1],
#                                  scale=1e-3 + tf.math.softplus(0.01 * t[..., 1:]))
#         )
#     ])
#
#     def modelKUunk.compile(optimizer=tf.optimizers.Adam(learning_rate = 0.05), loss=negloglik)
#     def modelKUunk.fit(x, y, epochs=500, verbose=False);
#     def yhatsKUunk = modelKUunk(x_tst) for _ in range(100)]

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import LearningRateScheduler, History, EarlyStopping
from tensorflow.keras import backend as K
# from: https://stackoverflow.com/questions/44132652/keras-how-to-perform-a-prediction-using-kerasregressor
# https://www.kaggle.com/hendraherviawan/regression-with-kerasregressor
from keras.wrappers.scikit_learn import KerasRegressor
epochs = 100
learning_rate = 0.1  # initial learning rate
decay_rate = 0.1
momentum = 0.8

normalizer = preprocessing.Normalization()

def exp_decay(epoch):
    lrate = learning_rate * np.exp(-decay_rate*epoch)
    return lrate

# learning schedule callback
loss_history = History()
lr_rate = LearningRateScheduler(exp_decay)
early_stopping = EarlyStopping(monitor='loss', min_delta=1.0e-6, verbose=1, patience=10)
callbacks_list = [loss_history, lr_rate, early_stopping]

# TODO: Tensorflow or a DNN regression class


class KerasRegressorWrapper(KerasRegressor):

    def score(self, X, y, **kwargs):
        y_pred = self.predict(X)
        return r2_score(y, y_pred, **kwargs)


class TFProbRegression:

    def build_and_compile_model(self, X, y, norm):
        model = tf.keras.Sequential([
            norm,
            layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            layers.Dense(1, activation='linear')
        ])

        model.compile(loss='mean_absolute_error',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=['mean_absolute_error', 'mean_squared_error', r2_score]
                      )

        dnn_model = build_KerasRegressorand_compile_model(normalizer)
        history = dnn_model.fit(
            X, y,
            validation_split=0.2,
            batch_size=200,
            callbacks=callbacks_list,
            verbose=2, epochs=epochs
        )
        return history

    plot_loss(history)

    test_results['dnn_model'] = dnn_model.evaluate(test_features, y_test, verbose=1)
    # print(pd.DataFrame(test_results, index=['Mean absolute error [Ceno Depth]']).T)

    # print(r2_score(y_test, linear_model.predict(test_features)))
    print('r2 score dnn: ', r2_score_sklearn(y_test, dnn_model.predict(test_features)))

    import time
    # pickle.dump(searchcv, open(f"{reg.__class__.__name__}.{int(time.time())}.model", 'wb'))
    str_with_time = f"dnn.{int(time.time())}.model"
    Path('saved_model').mkdir(exist_ok=True)
    model_file_name = Path('saved_model').joinpath(str_with_time)
    dnn_model.save(model_file_name)


modelmaps = {
    'xgboost': XGBRegressor,
    'gradientboost': GradientBoostingRegressor,
    'quantilegb': QuantileGradientBoosting,
    'randomforest': QuantileRandomForestRegressor,
    'quantilexgb': QuantileXGB,
    'tfregression': KerasRegressorWrapper,
}
