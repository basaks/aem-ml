from aem import utils
from aem.logger import aemlogger as log


def add_pred_to_data(X, conf, model):
    model_cols = utils.select_columns_for_model(conf)
    if hasattr(model, 'predict_dist'):
        X['pred'], X['variance'], X['lower_quantile'], X['upper_quantile'] = model.predict_dist(
            X[model_cols], interval=conf.quantiles
        )
        log.info("Added variance and quantiles to output dataframe")
    else:
        X['pred'] = model.predict(X[model_cols])
