import joblib
from pathlib import Path
from typing import Tuple, Optional, List, Union

import numpy as np
from scipy.signal import medfilt2d
import pandas as pd
from sklearn.neighbors import KDTree
from aem.config import twod_coords, threed_coords, Config, additional_cols_for_tracking
from aem.logger import aemlogger as log

# distance within which an interpretation point is considered to contribute to target values
radius = 500
dis_tol = 100  # meters, distance tolerance used`


def prepare_aem_data(conf: Config, aem_data: pd.DataFrame):
    """
    :param conf:
    :param in_scope_aem_data:
    :param interp_data: dataframe with
    :param include_thickness:
    :param include_conductivity_derivatives:
    :return:
    """
    aem_data.sort_values(by=['POINT_Y'], ascending=False, inplace=True)
    if conf.smooth_twod_covariates:
        for line in np.unique(aem_data.cluster_line_no):
            row_loc = aem_data.index[aem_data.cluster_line_no == line]
            aem_line_data = aem_data[aem_data.cluster_line_no == line][conf.conductivity_cols]
            aem_data.loc[row_loc, conf.conductivity_cols] = apply_twod_median_filter(conf, aem_line_data)

    aem_data.loc[:, conf.thickness_cols] = aem_data[conf.thickness_cols].cumsum(axis=1)
    conductivity_copy = aem_data[conf.conductivity_cols].copy()
    conductivity_copy.columns = conf.conductivity_derivatives_cols
    conductivity_diff = conductivity_copy.diff(axis=1, periods=-1)
    conductivity_diff.fillna(axis=1, method='ffill', inplace=True)
    aem_data.loc[:, conf.conductivity_derivatives_cols] = conductivity_diff
    return aem_data


def apply_twod_median_filter(conf: Config, aem_conductivities: pd.DataFrame):
    kernel_size = conf.smooth_covariates_kernel_size
    log.info(f"smooth conductivity data using scipy.signal.medfilt2d using kernel size {kernel_size}")
    aem_data = aem_conductivities.to_numpy()
    aem_data = medfilt2d(aem_data, kernel_size=kernel_size)
    df = pd.DataFrame(aem_data, columns=aem_conductivities.columns, index=aem_conductivities.index)
    return df


def select_required_data_cols(conf: Config):
    cols = select_columns_for_model(conf)[:]
    return cols + twod_coords + additional_cols_for_tracking


def select_columns_for_model(conf: Config):
    cols = conf.conductivity_cols[:]
    if conf.include_aem_covariates:
        cols += conf.aem_covariate_cols
    if conf.include_conductivity_derivatives:
        cols += conf.conductivity_derivatives_cols
    if conf.include_thickness:
        cols += conf.thickness_cols

    return cols


def extent_of_data(data: pd.DataFrame) -> Tuple[float, float, float, float]:
    x_min, x_max = min(data['POINT_X']), max(data['POINT_X'])
    y_min, y_max = min(data['POINT_Y']), max(data['POINT_Y'])
    return x_max, x_min, y_max, y_min


def weighted_target(line_required: pd.DataFrame, tree: KDTree, x: np.ndarray, weighted_model):
    ind, dist = tree.query_radius(x, r=radius, return_distance=True)
    ind, dist = ind[0], dist[0]
    if len(dist):
        dist += 1e-6  # add just in case of we have a zero distance
        df = line_required.iloc[ind]
        weighted_depth = np.sum(df.Z_coor * (1 / dist) ** 2) / np.sum((1 / dist) ** 2)
        if weighted_model:
            weighted_weight = np.sum(df.weight * (1 / dist) ** 2) / np.sum((1 / dist) ** 2)
        else:
            weighted_weight = None

        return weighted_depth, weighted_weight
    else:
        return None, None


def convert_to_xy(conf: Config, aem_data, interp_data):
    log.info("convert to xy and target values...")
    weighted_model = conf.weighted_model

    selected = []
    tree = KDTree(interp_data[twod_coords])
    target_depths = []
    target_weights = []
    for xy in aem_data.iterrows():
        i, covariates_including_xy_ = xy
        x_y = covariates_including_xy_[twod_coords].values.reshape(1, -1)
        y, w = weighted_target(interp_data, tree, x_y, weighted_model)
        if y is not None:
            if weighted_model:
                if w is not None:
                    selected.append(covariates_including_xy_)  # in 2d conductivities are already in xy
                    target_depths.append(y)
                    target_weights.append(w)
            else:
                selected.append(covariates_including_xy_)  # in 2d conductivities are already in xy
                target_depths.append(y)
                target_weights.append(1.0)
    X = pd.DataFrame(selected)
    y = pd.Series(target_depths, name='target', index=X.index)
    w = pd.Series(target_weights, name='weight', index=X.index)

    return {'covariates': X, 'targets': y, 'weights': w}


def create_interp_data(conf: Config, input_interp_data, included_lines):
    weighted_model = conf.weighted_model
    if not isinstance(included_lines, list):
        included_lines = [included_lines]
    line = input_interp_data[(input_interp_data['Type'] != 'WITHIN_Cenozoic')
                             & (input_interp_data['Type'] != 'BASE_Mesozoic_TOP_Paleozoic')
                             & (input_interp_data[conf.line_col].isin(included_lines))]
    line = line.rename(columns={'DEPTH': 'Z_coor'})
    if weighted_model:
        line_required = line[threed_coords + ['weight']]
    else:
        line_required = line[threed_coords]
    return line_required


def add_delta(line: pd.DataFrame, conf: Config, origin=None):
    """
    :param line:
    :param origin: origin of flight line, if not provided assumed ot be the at the lowest y value
    :return:
    """
    line = line.sort_values(by='POINT_Y', ascending=True)
    line_cols = list(line.columns)
    line['POINT_X_diff'] = line['POINT_X'].diff()
    line['POINT_Y_diff'] = line['POINT_Y'].diff()
    line['delta'] = np.sqrt(line.POINT_X_diff ** 2 + line.POINT_Y_diff ** 2)
    line['delta'] = line['delta'].fillna(value=0.0)
    if origin is not None:
        line['delta'].iat[0] = np.sqrt(
            (line.POINT_X.iat[0] - origin[0]) ** 2 +
            (line.POINT_Y.iat[0] - origin[1]) ** 2
        )
    line['d'] = line['delta'].cumsum()
    line = line.sort_values(by=['d'], ascending=True)  # must sort by distance from origin of flight line
    cluster_id = str(np.unique(line.cluster_line_no)[0]) + '_'
    arrs = np.array_split(range(line.shape[0]), line.shape[0] // conf.aem_line_splits)
    for i, a in enumerate(arrs):
        arrs[i] = np.ones_like(a) * i
    arr = np.concatenate(arrs).ravel().astype(str)
    line['cluster_line_segment_id'] = pd.Series([cluster_id + b for b in arr], index=line.index)

    return line[line_cols + ['d', 'cluster_line_segment_id']]


def plot_2d_section(X: pd.DataFrame,
                    cluster_line_no,
                    conf: Config,
                    col_names: Optional[Union[str, List[str]]] = None,
                    log_conductivity=False,
                    slope=False,
                    flip_column=False, v_min=0.3, v_max=0.8,
                    topographic_drape=True):
    if col_names is None:
        col_names = []
    if isinstance(col_names, str):
        col_names = [col_names]

    from scipy.signal import savgol_filter
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize, SymLogNorm, PowerNorm
    # from matplotlib.colors import Colormap
    X = X[X.cluster_line_no == cluster_line_no]
    origin = (X.POINT_X.iat[0], X.POINT_Y.iat[0])
    if slope:
        Z = X[conf.conductivity_derivatives_cols]
        Z = Z - np.min(np.min((Z))) + 1.0e-10
    else:
        Z = X[conf.conductivity_cols]

    if log_conductivity:
        Z = np.log10(Z)

    h = X[conf.thickness_cols]
    h = h.mul(-1)
    elevation = X['elevation'] if topographic_drape else 0
    elevation_stack = np.repeat(np.atleast_2d(elevation).T, h.shape[1], axis=1)
    h = h.add(elevation_stack)
    dd = X.d
    ddd = np.atleast_2d(dd).T
    d = np.repeat(ddd, h.shape[1], axis=1)
    fig, ax = plt.subplots(figsize=(40, 4))
    cmap = plt.get_cmap('turbo')

    if slope:
        norm = LogNorm(vmin=v_min, vmax=v_max)
    else:
        norm = Normalize(vmin=v_min, vmax=v_max)

    im = ax.pcolormesh(d, h, Z, norm=norm, cmap=cmap, linewidth=1, rasterized=True, shading='auto')
    fig.colorbar(im, ax=ax)
    # ax.set_ylim([-200, 0])
    axs = ax.twinx()
    if 'cross_val_pred' in X.columns:  # training/cross-val
        y_pred = X['cross_val_pred']
    elif 'oos_pred' in X.columns:
        y_pred = X['oos_pred']
    else:  # prediction
        y_pred = X['pred']
    if 'target' in X.columns:
        target = X['target']
        ax.plot(X.d, elevation - target, label='interpretation', linewidth=2, color='k')

    pred = savgol_filter(y_pred, 11, 3)  # window size 51, polynomial order 3
    ax.plot(X.d, elevation - pred, label='prediction', linewidth=2, color='c')

    for c in col_names:
        axs.plot(X.d, -X[c] if flip_column else X[c], label=c, linewidth=2, color='red')

    ax.set_xlabel('distance along aem line (m)')
    ax.set_ylabel('depth (m)')
    if slope:
        plt.title("d(Conductivity) vs depth")
    else:
        plt.title("Conductivity vs depth")

    ax.legend()
    axs.legend()
    # plt.show()
    plt.savefig(str(cluster_line_no) + ".jpg")


def plot_conductivity(X: pd.DataFrame,
                      conf: Config,
                      cluster_line_no,
                      slope=False, flip_column=False, v_min=0.3, v_max=0.8,
                      log_conductivity=False
                      ):
    from scipy.signal import savgol_filter
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize, SymLogNorm, PowerNorm
    from matplotlib.colors import Colormap
    X = X[X.cluster_line_no == cluster_line_no]
    if slope:
        Z = X[conf.conductivity_derivatives_cols]
        Z = Z - np.min(np.min((Z))) + 1.0e-10
    else:
        Z = X[conf.conductivity_cols]

    if log_conductivity:
        Z = np.log10(Z)

    h = X[conf.thickness_cols]
    dd = X.d
    ddd = np.atleast_2d(dd).T
    d = np.repeat(ddd, h.shape[1], axis=1)
    fig, ax = plt.subplots(figsize=(40, 4))
    cmap = plt.get_cmap('turbo')

    if slope:
        norm = LogNorm(vmin=v_min, vmax=v_max)
    else:
        norm = Normalize(vmin=v_min, vmax=v_max)

    im = ax.pcolormesh(d, -h, Z, norm=norm, cmap=cmap, linewidth=1, rasterized=True, shading='auto')
    fig.colorbar(im, ax=ax)

    ax.set_xlabel('distance along aem line (m)')
    ax.set_ylabel('depth (m)')
    if slope:
        plt.title("d(Conductivity) vs depth")
    else:
        plt.title("Conductivity vs depth")

    ax.legend()
    plt.show()


def export_model(model, conf: Config, model_type: str = 'learn'):
    if model_type not in {'learn', 'optimise'}:
        raise AttributeError("Model type must be one of 'learn' or 'optimise'")
    learned_model = model_type == 'learn'  # as opposed to optimised_model
    model_file = conf.model_file if learned_model else conf.optimised_model_file
    state_dict = {"model": model, "config": conf}
    with open(model_file, 'wb') as f:
        joblib.dump(state_dict, f)
        log.info(f"Wrote model on disc {model_file}")


def import_model(conf: Config, model_type: str = 'learn'):
    learned_model = model_type == 'learn'  # as opposed to optimised_model
    model_file = conf.model_file if learned_model else conf.optimised_model_file
    if not model_file.exists():
        raise FileExistsError(f"Model file {model_file.as_posix()} does not exist. Train or optimise model first!!")
    with open(model_file, 'rb') as f:
        state_dict = joblib.load(f)
        log.info(f"loaded trained model from location {conf.model_file}")
    model, conf = state_dict["model"], state_dict['config']
    return model, conf


def plot_cond_mesh(X, conf):
    return
#
# def plot_feature_importance(X, y, optimised_model: BayesSearchCV):
#     xgb_model = XGBRegressor(**optimised_model.best_params_)
#     xgb_model.fit(X, y)
#     non_zero_indices = xgb_model.feature_importances_ >= 0.001
#     non_zero_cols = X_all.columns[non_zero_indices]
#     non_zero_importances = xgb_model.feature_importances_[non_zero_indices]
#     sorted_non_zero_indices = non_zero_importances.argsort()
#     plt.barh(non_zero_cols[sorted_non_zero_indices], non_zero_importances[sorted_non_zero_indices])
#     plt.xlabel("Xgboost Feature Importance")
