import pandas as pd
from aem import utils
from aem.config import Config
from aem.logger import aemlogger as log


def add_pred_to_data(X: pd.DataFrame, conf: Config, model) -> pd.DataFrame:
    model_cols = utils.select_columns_for_model(conf)
    if hasattr(model, 'predict_dist'):
        p, v, ql, qu = model.predict_dist(X[model_cols], interval=conf.quantiles)
        X = pd.concat((X, pd.DataFrame({'pred': p, 'variance': v, 'lower_quantile': ql, 'upper_quantile': qu}, index=X.index)),
                      axis=1)
        log.info("Added prediction, variance and quantiles to output dataframe")
    else:
        p = model.predict(X[model_cols])
        X = pd.concat((X, pd.DataFrame({'pred': p}, index=X.index)), axis=1)
        log.info("Added prediction to output dataframe")
    return X
