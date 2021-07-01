from pathlib import Path
import geopandas as gpd
import joblib
from itertools import cycle, islice
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from aem.config import Config
from aem import utils
from aem.logger import aemlogger as log


def split_flight_lines_into_multiple_lines(aem_data: pd.DataFrame, conf: Config) -> pd.DataFrame:
    """
    :param aem_data: aem training data
    :param conf: Config instance
    :return: aem_data with line_no added based on
    """
    from matplotlib.colors import ListedColormap

    X = aem_data[utils.twod_coords]
    dbscan = DBSCAN(eps=conf.aem_line_dbscan_eps, n_jobs=-1, min_samples=10)
    # t0 = time.time()
    dbscan.fit(X)
    line_no = dbscan.labels_.astype(np.uint16)
    rc_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # list of colours
    colors = np.array(list(islice(cycle(rc_colors), int(max(line_no) + 1))))
    # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])
    # colors = ListedColormap(colors)
    plt.figure(figsize=(16, 10))
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    lines = np.unique(line_no)
    scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], s=10, c=colors[line_no], cmap=colors)
    # plt.legend(handles=scatter.legend_elements()[0], labels=list(np.unique(lines)))

    plt.savefig(conf.aem_lines_plot)

    aem_data['cluster_line_no'] = line_no

    aem_data = aem_data.groupby('cluster_line_no').apply(utils.add_delta, conf=conf)
    return aem_data


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
    # TODO: different search radius for different targets (3)
    # TODO: geology/polygon impact (4)
    # TODO: Scaling of covariates and targets (5)
    # TODO: explore rectangular filter (6)

    log.info("reading covariates ...")
    original_aem_datasets = [gpd.GeoDataFrame.from_file(i, rows=conf.shapefile_rows) for i in conf.aem_train_data]
    all_interp_training_data = pd.concat(all_interp_training_datasets, axis=0)
    original_aem_data = pd.concat(original_aem_datasets, axis=0)

    original_aem_data = split_flight_lines_into_multiple_lines(original_aem_data, conf)

    # how many lines in interp data
    lines_in_data = np.unique(all_interp_training_data[conf.line_col])

    all_lines = utils.create_interp_data(conf, all_interp_training_data, included_lines=list(lines_in_data))

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

    X = data['covariates']
    y = data['targets']
    if conf.weighted_model:
        w = data['weights']
    else:
        w = np.ones_like(y)
    return X, y, w
