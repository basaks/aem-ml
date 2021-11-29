import pandas as pd
from aem import utils
from aem.config import Config
from aem.logger import aemlogger as log


def add_pred_to_data(X: pd.DataFrame, conf: Config, model, oos: bool = False) -> pd.DataFrame:
    model_cols = utils.select_columns_for_model(conf)
    prefix = 'oos_' if oos else ''
    if hasattr(model, 'predict_dist'):
        p, v, ql, qu = model.predict_dist(X[model_cols], interval=conf.quantiles)
        attrs = ['pred', 'variance', 'lower_quantile', 'upper_quantile']
        pred = pd.DataFrame(
            {prefix + a: v for a, v in zip(attrs, [p, v, ql, qu])},
            index=X.index
        )
        log.info("Added prediction, variance and quantiles to output dataframe")
    else:
        p = model.predict(X[model_cols])
        pred = pd.DataFrame({prefix + 'pred': p}, index=X.index)
        log.info("Added prediction to output dataframe")

    X = pd.concat((X, pred), axis=1)

    return X
"""
Defines the addition of prediction data to the dataframe with the configs, models and oos defined
in the aem file. Adds new columns; pred, variance, lower_quantile and upper_quantile to the new
pred dataframe from an index. Logging info is output to the user with the relevant message being
output. The new datafram is the concatted.
"""
