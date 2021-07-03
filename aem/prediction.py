import pandas as pd
from aem import utils
from aem.config import Config
from aem.logger import aemlogger as log


def add_pred_to_data(X: pd.DataFrame, conf: Config, model) -> None:
    model_cols = utils.select_columns_for_model(conf)
    if hasattr(model, 'predict_dist'):
        X.loc[:, 'pred'], X.loc[:, 'variance'], X.loc[:, 'lower_quantile'], X.loc[:, 'upper_quantile'] = \
            model.predict_dist(X[model_cols], interval=conf.quantiles)
        log.info("Added prediction, variance and quantiles to output dataframe")
    else:
        X.loc[:, 'pred'] = model.predict(X[model_cols])
