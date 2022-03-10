from pathlib import Path
import geopandas as gpd
import joblib
from itertools import cycle, islice
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from aem.config import Config, cluster_line_no
from aem import utils
from aem.logger import aemlogger as log


def split_flight_lines_into_multiple_segments(aem_data: pd.DataFrame, is_train: bool, conf: Config) -> pd.DataFrame:
    """
    Accepts aem covariates with 'POINT_X', 'POINT_Y as coordinates and assigns a cluster number to each row of
    covariates/observations. These

    :param is_train: train or predict
    :param aem_data: aem training data
    :param conf: Config instance
    :return: aem_data with line_no added based on
    """
    log.info("Segmenting aem lines using DBSCAN clustering algorithm")
    from matplotlib.colors import ListedColormap

    _X = aem_data.loc[:, utils.twod_coords]
    dbscan = DBSCAN(eps=conf.aem_line_scan_eps, n_jobs=-1, min_samples=10)
    # t0 = time.time()
    dbscan.fit(_X)
    line_no = dbscan.labels_.astype(np.uint16)
    rc_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # list of colours
    colors = np.array(list(islice(cycle(rc_colors), int(max(line_no) + 1))))
    # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])
    # colors = ListedColormap(colors)
    plt.figure(figsize=(16, 10))
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    # lines = np.unique(line_no)
    scatter = plt.scatter(_X.iloc[:, 0], _X.iloc[:, 1], s=10, c=colors[line_no], cmap=colors)
    # plt.legend(handles=scatter.legend_elements()[0], labels=list(np.unique(lines)))
    if is_train:
        if conf.oos_validation:
            fig_file = conf.aem_lines_plot_oos
        else:
            fig_file = conf.aem_lines_plot_train
    else:
        fig_file = conf.aem_lines_plot_pred

    plt.savefig(fig_file)
    log.info(f"Saved segments in {fig_file}")
    aem_data[cluster_line_no] = line_no

    aem_data = aem_data.groupby(cluster_line_no).apply(utils.add_delta, conf=conf)
    log.info(f"Found {len(np.unique(line_no))} groups")
    log.info("Finished segmentation")
    return aem_data


def load_data(conf: Config):
    """
    Loads covariates specified in the config file
    :param conf: Config class instance
    """
    original_aem_data = load_covariates(is_train=True, conf=conf)
    if conf.oos_validation:
        all_interp_training_datasets = [gpd.GeoDataFrame.from_file(i, rows=conf.shapefile_rows) for i in
                                        conf.oos_interp_data]
    else:
        all_interp_training_datasets = [gpd.GeoDataFrame.from_file(i, rows=conf.shapefile_rows) for i in
                                        conf.interp_data]

    log.info("reading interp data...")

    train_weights = conf.train_data_weights

    # apply the weights due to confidence levels assigned by the interpreter on the interpretation/target values
    # plus the weights due to the datasets themselves
    if conf.weighted_model:
        for a, w in zip(all_interp_training_datasets, train_weights):
            if conf.weight_col not in a.columns:
                a[conf.weight_col] = 1  # this takes care of the drillhole files
            a['weight'] = a[conf.weight_col].map(conf.weight_dict) * w

    all_interp_training_data = pd.concat(all_interp_training_datasets, axis=0, ignore_index=True)
    # how many lines in interp data
    interp_data = utils.create_interp_data(conf, all_interp_training_data)

    aem_xy_and_other_covs = utils.prepare_aem_data(conf, original_aem_data)[utils.select_required_data_cols(conf)]
    smooth = '_smooth_' if conf.smooth_twod_covariates else '_'
    data = utils.convert_to_xy(conf, aem_xy_and_other_covs, interp_data)
    # data_path = f'covariates_targets_2d{smooth}weights.data'
    # if (not Path(data_path).exists()) or conf.oos_validation:
    #     data = utils.convert_to_xy(conf, aem_xy_and_other_covs, interp_data)
    #     log.info("saving data on disc for future use")
    #     if not conf.oos_validation:  # only during training
    #         joblib.dump(data, open(data_path, 'wb'))
    # else:
    #     log.warning("Reusing data from disc!!!")
    #     data = joblib.load(open(data_path, 'rb'))

    X = data['covariates']
    y = data['targets']
    if conf.weighted_model:
        w = data['weights']
    else:
        w = np.ones_like(y)
    return X, y, w


def load_covariates(is_train: bool, conf: Config):
    if conf.oos_validation:
        aem_files = conf.oos_validation_data
    else:
        # either train or prediction
        aem_files = conf.aem_train_data

    log.info(f"Processing covariates from {aem_files}....")
    # TODO: Scaling of covariates and targets (5) - similar performance with xgboost without scaling (2)
    # TODO: different search radius for different targets (3)
    # TODO: geology/polygon impact (4)
    # TODO: True probabilistic models (gaussian process/GPs, tensorflow/pytorch probability model classes)
    # TODO: move segmenting flight line after interpretation point intersection/interpolation
    original_aem_datasets = [gpd.GeoDataFrame.from_file(i, rows=conf.shapefile_rows) for i in aem_files]
    aem_data = pd.concat(original_aem_datasets, axis=0)
    aem_data = split_flight_lines_into_multiple_segments(aem_data, is_train, conf)
    return aem_data
