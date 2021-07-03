"""Console script for aem."""
import sys
import click
import json
import numpy as np
import geopandas as gpd
from aem import __version__
from aem.config import Config, cluster_line_segment_id, cluster_line_no
from aem import utils
from aem.data import load_data
from aem.prediction import add_pred_to_data
from aem.models import modelmaps
from aem import training
from aem.logger import configure_logging, aemlogger as log
from aem.utils import import_model
from sklearn.model_selection import cross_validate, GroupKFold, train_test_split


@click.group()
@click.version_option(version=__version__)
@click.option("-v", "--verbosity",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
              default="INFO", help="Level of logging")
def main(verbosity: str) -> int:
    """Train a model and use it to make predictions."""
    configure_logging(verbosity)
    return 0


@main.command()
@click.option("-c", "--config", type=click.Path(exists=True), required=True,
              help="The model configuration file")
def learn(config: str) -> None:
    """Train a model specified by a config file."""
    log.info(f"Training Model using config {config}")
    conf = Config(config)
    np.random.seed(conf.numpy_seed)

    X, y, w = load_data(conf)
    model = modelmaps[conf.algorithm](**conf.model_params)
    model_cols = utils.select_columns_for_model(conf)

    log.info(f"Running cross validation of {conf.algorithm} model with kfold {conf.cross_validation_folds}")
    scores = cross_validate(model, X[model_cols], y, groups=X['cluster_line_segment_id'],
                            fit_params={'sample_weight': w}, n_jobs=-1, verbose=1000,
                            cv=GroupKFold(conf.cross_validation_folds),
                            return_train_score=True)

    log.info(f"Finished {conf.algorithm} cross validation")

    log.info(f"Average cross validation score {scores['test_score'].mean()}")

    # report model performance on screen
    score_string = "Model scores: \n"

    for k, v in scores.items():
        if isinstance(v, np.ndarray):
            scores[k] = v.tolist()
        score_string += "{}\t= {}\n".format(k, v)

    log.info(score_string)

    log.info("Fit final model with all training data")
    model.fit(X[model_cols], y, sample_weight=w)

    # and also save a scores json file on disc
    with open(conf.outfile_scores, 'w') as f:
        json.dump(scores, f, sort_keys=True, indent=4)

    utils.export_model(model, conf, learn=True)

    add_pred_to_data(X, conf, model)
    X['target'] = y
    X['weights'] = w
    X.to_csv(conf.train_data, index=False)
    log.info(f"Saved training data and target and prediction at {conf.train_data}")


@main.command()
@click.option("-c", "--config", type=click.Path(exists=True), required=True,
              help="The model configuration file")
def optimise(config: str) -> None:
    """Optimise model parameters using Bayesian regression."""
    conf = Config(config)
    X, y, w = load_data(conf)

    model_cols = utils.select_columns_for_model(conf)
    groups = X[cluster_line_segment_id]
    model = training.bayesian_optimisation(X[model_cols], y, w, groups, conf)

    add_pred_to_data(X, conf, model)
    X['target'] = y
    X['weights'] = w
    X.to_csv(conf.optimisation_data, index=False)

    log.info("Finished optimisation of model parameters!")


@main.command()
@click.option("-c", "--config", type=click.Path(exists=True), required=True,
              help="The model configuration file")
@click.option('--model-type',
              type=click.Choice(['learn', 'optimised'], case_sensitive=False))
def predict(config: str, model_type: str) -> None:
    """Predict using a model saved on disc."""
    conf = Config(config)
    log.info(f"Predicting using trained model file found in location {conf.model_file}")
    log.info(f"Prediction covariates are read from {conf.aem_pred_data}")

    pred_aem_data = gpd.GeoDataFrame.from_file(conf.aem_pred_data, rows=conf.shapefile_rows)

    model_cols = utils.select_columns_for_model(conf)
    X = utils.prepare_aem_data(conf, pred_aem_data)[model_cols]
    learn = model_type == 'learn'
    state_dict = import_model(conf, learn=learn)
    log.info(f"loaded trained model from location {conf.model_file}")
    model = state_dict["model"]
    config = state_dict["config"]

    add_pred_to_data(X, conf, model)
    log.info(f"Finished predicting {conf.algorithm} model")
    X.to_csv(conf.pred_data, index=False)
    log.info(f"Saved training data and target and prediction at {conf.train_data}")


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover

    import IPython; IPython.embed(); import sys; sys.exit()
