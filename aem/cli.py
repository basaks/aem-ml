"""Console script for aem."""
import sys
import click
import json
import numpy as np
import geopandas as gpd
from aem import __version__
from aem.config import Config
from aem import utils
from aem.data import load_data
from aem.prediction import add_pred_to_data
from aem.models import modelmaps
from aem import training
from aem.logger import configure_logging, aemlogger as log
from aem.utils import import_model


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

    X, y, w, X_train, y_train, w_train, X_val, y_val, w_val, X_test, y_test, w_test, X_train_val, y_train_val, \
        w_train_val = load_data(conf)
    model = modelmaps[conf.algorithm](**conf.model_params)
    model_cols = utils.select_columns_for_model(conf)
    log.info(f"Training {conf.algorithm} model")
    model.fit(X_train_val[model_cols], y_train_val, sample_weight=w_train_val)

    log.info(f"Finished training {conf.algorithm} model")

    train_scores = training.score_model(model, X_train_val[model_cols], y_train_val, w_train_val)
    test_scores = training.score_model(model, X_test[model_cols], y_test, w_test)

    all_scores = {'test_scores': test_scores, 'train_scores': train_scores}

    score_string = "Training complete:\n"

    # report model performance on screen
    for k, scores in all_scores.items():
        score_string += f"{k}:\n"
        for metric, score in scores.items():
            score_string += "{}\t= {}\n".format(metric, score)
    log.info(score_string)

    # and also save a scores json file on disc
    with open(conf.outfile_scores, 'w') as f:
        json.dump(all_scores, f, sort_keys=True, indent=4)

    utils.export_model(model, conf, learn=True)

    add_pred_to_data(X, conf, model)
    X['target'] = y
    X['weights'] = w
    # TODO: insert line number of interpretation
    X.to_csv(conf.train_data, index=False)

    log.info(f"Saved training data and target and prediction at {conf.train_data}")


@main.command()
@click.option("-c", "--config", type=click.Path(exists=True), required=True,
              help="The model configuration file")
def optimise(config: str) -> None:
    """Optimise model parameters using Bayesian regression."""
    conf = Config(config)
    X, y, w, X_train, y_train, w_train, X_val, y_val, w_val, X_test, y_test, w_test, X_train_val, y_train_val, \
        w_train_val = load_data(conf)
    model_cols = utils.select_columns_for_model(conf)
    model = training.bayesian_optimisation(X_train[model_cols], y_train, w_train,
                                           X_val[model_cols], y_val, w_val, conf)
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
