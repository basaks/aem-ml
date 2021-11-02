"""Console script for aem."""
import sys
from enum import Enum
import click
import json
import numpy as np
from sklearn.model_selection import cross_validate, GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error, mean_absolute_error
from aem import __version__
from aem.config import Config, cluster_line_segment_id
from aem import utils
from aem.data import load_data, load_covariates
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
    """
    The purpose of "main" is to create a function that runs with the model that is set out
    by the parameters below

    :param verbosity: Write regular expressions which are more readable
    """

    configure_logging(verbosity)
    return 0


regression_metrics = {
    r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
}


@main.command()
@click.option("-c", "--config", type=click.Path(exists=True), required=True,
              help="The model configuration file")
def learn(config: str) -> None:
    """
    Sets the config file to learn from in the AEM Folder and trains the model file as
    specified

    :param config:  Config class instance
    str: a string defined by user
    """

    log.info(f"Training Model using config {config}")
    conf = Config(config)
    np.random.seed(conf.numpy_seed)

    X, y, w = load_data(conf)

    import IPython; IPython.embed(); import sys; sys.exit()
    model = modelmaps[conf.algorithm](**conf.model_params)
    model_cols = utils.select_columns_for_model(conf)
    """
    Imports the IPython module to visualise the running of the set model in the
    previous section to give the user an idea of the output to expect
    """

    if conf.cross_validate:
        log.info(f"Running cross validation of {conf.algorithm} model with kfold {conf.cross_validation_folds}")
        predictions = cross_val_predict(model, X[model_cols], y, groups=X['cluster_line_segment_id'],
                                        fit_params={'sample_weight': w}, n_jobs=-1, verbose=1000,
                                        cv=GroupKFold(conf.cross_validation_folds))
        scores = {v.__name__: v(y_true=y, y_pred=predictions, sample_weight=w) for v in regression_metrics}
        log.info(f"Finished {conf.algorithm} cross validation")
    """
    Runs a cross validation of the algorithm specified in the config stage to ensure
    that the model is valid and accurate for the test set provided

    :param predictions: The predicted y_vals to be run in the cross configs
    :model: model defined by the user
    :X,y: the X and y vals from the dataset to be run in the model

    """

        score_string = "Model scores: \n"
        for k, v in scores.items():
            if isinstance(v, np.ndarray):
                scores[k] = v.tolist()
            score_string += "{}\t= {}\n".format(k, v)
    """
    Reports the performance on the model on the screen for the user to visualise
    """

        log.info(score_string)
        # and also save a scores json file on disc
        with open(conf.outfile_scores, 'w') as f:
            json.dump(scores, f, sort_keys=True, indent=4)
        X['cross_val_pred'] = predictions
    """
    Saves the performance scores for the specified model as a json file on the
    disk of the user
    """


    log.info("Fit final model with all training data")
    model.fit(X[model_cols], y, sample_weight=w)

    utils.export_model(model, conf, model_type='learn')

    X = add_pred_to_data(X, conf, model)
    X['target'] = y
    X['weights'] = w
    X.to_csv(conf.train_data, index=False)
    log.info(f"Saved training data and target and prediction at {conf.train_data}")
    """
    Fits the model specified by the user with the y vals as the X 'target' and the sample
    weights and creates a .csv file with the output for the user to export with the file
    name conf.train_data.
    """


@main.command()
@click.option("-c", "--config", type=click.Path(exists=True), required=True,
              help="The model configuration file")
def optimise(config: str) -> None:

    conf = Config(config)
    X, y, w = load_data(conf)

    groups = X[cluster_line_segment_id]
    model = training.bayesian_optimisation(X, y, w, groups, conf)
    utils.export_model(model, conf, model_type='optimise')

    X = add_pred_to_data(X, conf, model)
    X['target'] = y
    X['weights'] = w
    X.to_csv(conf.optimisation_data, index=False)

    log.info("Finished optimisation of model parameters!")

    """
    Optimises the config class instance from the AEM folder with a bayesian optimisation
    to ensure a more robust dataset is created once the model is specified by the user.
    After this process is completed, creates a csv file called conf.optimisation_data and
    lets the user know the process is finished

    :param config: Config class instance designated by user from the AEM folder
    :return: Optimises model parameters using a Bayesian regression

    X: Predicted data with the clustered line and bayesian optimisation
    y, w: all values loaded from previous code; target and weight
    groups: groups a clustered line segment for X values
    to_csv: function that creates a csv file for the data output from the model


    """


@main.command()
@click.option("-c", "--config", type=click.Path(exists=True), required=True,
              help="The model configuration file")
@click.option('--model-type', required=True,
              type=click.Choice(['learn', 'optimised'], case_sensitive=False))
def validate(config: str, model_type: str) -> None:

    conf = Config(config)
    conf.oos_validation = True
    model, _ = import_model(conf, model_type)

    X, y, w = load_data(conf=conf)
    X['target'] = y
    X['weights'] = w
    X = add_pred_to_data(X, conf, model, oos=True)
    log.info(f"Finished predicting {conf.algorithm} model")
    predictions = X['oos_pred']
    scores = {v.__name__: v(y_true=y, y_pred=predictions, sample_weight=w) for v in regression_metrics}
    log.info(f"Finished {conf.algorithm} oos validation")

    """
    Validates the config class instance from the AEM folder to ensure the model chosen
    is working correctly and logs te information that may be interesting to the user
    to discern te steps of the processes.

    :param config: Config class instance designated by user from AEM folder
    :param model_type: Model type defined by user (XGB, Randomforest, QuantileXGB, Quantileb, GradientBoost)
    :return: Validation of an OOS shapefile utilising the model outlined by model_type

    X: Predicted validated data added to original X_vals from previous sections
    y, w: targets, weights
    log.info: Logs information for user to analyse
    """

    score_string = "Model scores: \n"
    for k, v in scores.items():
        if isinstance(v, np.ndarray):
            scores[k] = v.tolist()
        score_string += "{}\t= {}\n".format(k, v)
    """
    Visualises the model scores for the user so they can draw conclusions from data
    """

    log.info(score_string)
    with open(conf.oos_validation_scores, 'w') as f:
        json.dump(scores, f, sort_keys=True, indent=4)

    X.to_csv(conf.oos_data, index=False)
    log.info(f"Saved oos data and target and oos predictions at {conf.oos_data}")

    """
    Saves the oos validations scores as a json file on the disk and a .csv file for the
    user to export
    """


@main.command()
@click.option("-c", "--config", type=click.Path(exists=True), required=True,
              help="The model configuration file")
@click.option('--model-type', required=True,
              type=click.Choice(['learn', 'optimised'], case_sensitive=False))
def predict(config: str, model_type: str) -> None:
    conf = Config(config)
    model, _ = import_model(conf, model_type)

    pred_aem_data = load_covariates(is_train=False, conf=conf)

    X = utils.prepare_aem_data(conf, pred_aem_data)[utils.select_required_data_cols(conf)]

    X = add_pred_to_data(X, conf, model)
    log.info(f"Finished predicting {conf.algorithm} model")
    X.to_csv(conf.pred_data, index=False)
    log.info(f"Saved training data and target and prediction at {conf.pred_data}")

    """
    Predicts the data input in the config class instance from the AEM folder

    :param config: Config class instance defined in AEM folder
    :param model_type: Model type defined by user (XGB, Randomforest, QuantileXGB, Quantileb, GradientBoost)
    :return: A prediction of the dataset outlined by the user through the model type defined

    pred_aem_data: loads the covariates from the file specifed in the AEM folder specific
    to the config class instance defined by the user
    X: Prediction data added to the X_vals of the datset chosen by the model for the user
    log.info: Logs information for the user to analyse
    X.to_csv: Creates a csv file with the predicted data for the user to export
    """


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
