import pandas as pd
from aem import utils
from aem.config import Config
from aem.logger import aemlogger as log


def add_pred_to_data(X: pd.DataFrame, conf: Config, model, oos: bool = False) -> pd.DataFrame:
    model_cols = utils.select_cols_used_in_model(conf)
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
