from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from aem.config import Config
from aem import utils
from aem.logger import aemlogger as log
from aem.utils import create_interp_data, create_train_test_set


def load_data(conf: Config):
    log.info("Reading covariates...")
    log.info("reading interp data...")
    all_interp_training_datasets = [gpd.GeoDataFrame.from_file(i, rows=conf.shapefile_rows) for i in conf.interp_data]
    train_weights = conf.train_data_weights

    # apply the weights due to confidence levels assigned by the interpreter on the interpretation/target values
    # plus the weights due to the datasets themselves
    if conf.weighted_model:
        for a, w in zip(all_interp_training_datasets, train_weights):
            a['weight'] = a[conf.weight_col].map(conf.weight_dict) * w

    # TODO: generate multiple segments from same survey line (2)
    # todo: add weights to target shapefile (3)
    # TODO: different search radius for different targets (3)
    # TODO: geology/polygon impact (4)
    # TODO: Scaling of covariates and targets (5)

    log.info("reading covariates ...")
    original_aem_datasets = [gpd.GeoDataFrame.from_file(i, rows=conf.shapefile_rows) for i in conf.aem_train_data]
    all_interp_training_data = pd.concat(all_interp_training_datasets, axis=0)
    original_aem_data = pd.concat(original_aem_datasets, axis=0)

    # how many lines in interp data
    lines_in_data = np.unique(all_interp_training_data[conf.line_col])

    train_and_val_lines_in_data, test_lines_in_data = train_test_split(lines_in_data, test_size=conf.test_fraction)
    train_lines_in_data, val_lines_in_data = train_test_split(train_and_val_lines_in_data,
                                                              test_size=conf.val_fraction/(1-conf.test_fraction))

    all_lines = utils.create_interp_data(conf, all_interp_training_data, included_lines=list(lines_in_data))

    # import IPython; IPython.embed(); import sys; sys.exit()
    aem_xy_and_other_covs = utils.prepare_aem_data(conf, original_aem_data)[utils.select_required_data_cols(conf)]
    smooth = '_smooth_' if conf.smooth_twod_covariates else '_'
    data_path = f'covariates_targets_2d{smooth}weights.data'
    if not Path(data_path).exists():
        data = utils.convert_to_xy(conf, aem_xy_and_other_covs, all_lines)
        log.info("saving data on disc for future use")
        joblib.dump(data, open(data_path, 'wb'))
    else:
        log.warning("Reusing data from disc!!!")
        data = joblib.load(open(data_path, 'rb'))

    train_data_lines = [create_interp_data(conf, all_interp_training_data, included_lines=i) for i in train_lines_in_data]
    val_data_lines = [create_interp_data(conf, all_interp_training_data, included_lines=i) for i in val_lines_in_data]
    test_data_lines = [create_interp_data(conf, all_interp_training_data, included_lines=i) for i in test_lines_in_data]

    all_data_lines = train_data_lines + val_data_lines + test_data_lines

    X_train, y_train, w_train, _ = create_train_test_set(conf, data, *train_data_lines)

    X_val, y_val, w_val, _ = create_train_test_set(conf, data, *val_data_lines)
    X_test, y_test, w_test, _ = create_train_test_set(conf, data, *test_data_lines)
    X_train_val, y_train_val, w_train_val, _ = create_train_test_set(conf, data, *train_data_lines, *val_data_lines)
    X, y, w, _ = create_train_test_set(conf, data, * all_data_lines)

    return X, y, w, X_train, y_train, w_train, X_val, y_val, w_val, X_test, y_test, w_test, X_train_val, y_train_val, \
           w_train_val
