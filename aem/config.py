import yaml
from pathlib import Path
import geopandas as gpd

twod_coords = ['POINT_X', 'POINT_Y']
threed_coords = twod_coords + ['Z_coor']
cluster_line_no = 'cluster_line_no'
cluster_line_segment_id = 'cluster_line_segment_id'
additional_cols_for_tracking = ['fiducial', 'uniqueid', 'flight', 'line', cluster_line_no, 'd',
                                cluster_line_segment_id]


class Config:
    """Class representing the global configuration of the aem scripts

    This class is *mostly* read-only, but it does also contain the Transform
    objects which have state.

    Parameters
    ----------
    yaml_file : string
        The path to the yaml config file.
    """

    def __init__(self, yaml_file):
        with open(yaml_file, 'r') as f:
            s = yaml.safe_load(f)

        self.name = Path(yaml_file).stem

        # output dir
        self.output_dir = s['output']['directory']
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)

        # data
        self.aem_folder = s['data']['aem_folder']
        self.interp_data = [Path(self.aem_folder).joinpath(p) for p in s['data']['train_data']['targets']]
        self.train_data_weights = s['data']['train_data']['weights']

        self.target_class_indicator_col = s['data']['train_data']['target_class_indicator_col'] if \
            'target_class_indicator_col' in s['data']['train_data'] else None

        self.aem_train_data = [Path(self.aem_folder).joinpath(p) for p in s['data']['train_data']['aem_train_data']]
        self.aem_pred_data = [Path(self.aem_folder).joinpath(p) for p in s['data']['apply_model']]
        self.shapefile_rows = s['data']['rows']
        self.aem_line_scan_eps = s['data']['aem_line_scan_radius']
        self.aem_line_splits = s['data']['aem_line_splits']
        self.cutoff_radius = s['data']['cutoff_radius']
        # oos_validation
        self.oos_validation = False
        self.oos_validation_data = [Path(self.aem_folder).joinpath(p)
                                    for p in s['data']['oos_validation']['aem_validation_data']]
        self.oos_interp_data = [Path(self.aem_folder).joinpath(p)
                                for p in s['data']['oos_validation']['targets']]

        # np randomisation
        self.numpy_seed = s['learning']['numpy_seed']

        # training
        self.algorithm = s['learning']['algorithm']
        self.model_params = s['learning']['params']
        if 'cross_validation' in s['learning']:
            self.cross_validation_folds = s['learning']['cross_validation']['kfold']
            self.cross_validate = True
        else:
            self.cross_validate = False
        self.include_aem_covariates = s['learning']['include_aem_covariates']
        self.include_thickness = s['learning']['include_thickness']
        self.include_conductivity_derivatives = s['learning']['include_conductivity_derivatives']
        self.smooth_twod_covariates = s['learning']['smooth_twod_covariates']
        self.smooth_covariates_kernel_size = eval(s['learning']['smooth_covariates_kernel_size'])

        # model parameter optimisation
        self.hpopt = False
        self.skopt = False
        if 'optimisation' in s['learning']:
            if 'searchcv_params' in s['learning']['optimisation']:
                self.opt_searchcv_params = s['learning']['optimisation']['searchcv_params']
                self.opt_params_space = s['learning']['optimisation']['params_space']
                self.skopt = True
            if 'hyperopt_params' in s['learning']['optimisation']:
                self.hyperopt_params = s['learning']['optimisation']['hyperopt_params']
                self.hp_params_space = s['learning']['optimisation']['hp_params_space']
                self.hpopt = True

            if self.skopt and self.hpopt:
                raise ConfigException("Only one of searchcv_params or hyperopt_params can be specified")

        # weighted model params
        if 'weighted_model' in s['learning']:
            self.weighted_model = True
            if 'weights_map' in s['learning']:
                self.weights_map = s['learning']['weights_map']
            else:
                self.weights_map = False
            self.weight_col = s['data']['weight_col']
        else:
            self.weighted_model = False

        # data description
        self.target_col = s['data']['target_col']
        self.target_type_col = s['data']['target_type_col']
        self.included_target_type_categories = s['data']['included_target_type_categories']
        self.conductivity_columns_prefix = s['data']['conductivity_columns_prefix']
        self.thickness_columns_prefix = s['data']['thickness_columns_prefix']
        self.aem_covariate_cols = s['data']['aem_covariate_cols']

        original_aem_data = gpd.GeoDataFrame.from_file(self.aem_train_data[0].as_posix(), rows=1)

        conductivity_cols = [c for c in original_aem_data.columns if c.startswith(self.conductivity_columns_prefix)]
        d_conductivities = ['d_' + c for c in conductivity_cols]
        conductivity_and_derivatives_cols = conductivity_cols + d_conductivities
        thickness_cols = [t for t in original_aem_data.columns if t.startswith(self.thickness_columns_prefix)]

        self.thickness_cols = thickness_cols
        self.conductivity_cols = conductivity_cols
        self.conductivity_derivatives_cols = d_conductivities
        self.conductivity_and_derivatives_cols = conductivity_and_derivatives_cols

        # co-ordination
        self.model_file = Path(self.output_dir).joinpath(self.name + ".model")
        self.optimised_model_params = Path(self.output_dir).joinpath(self.name + "_searchcv_params.json")
        self.optimised_model_file = Path(self.output_dir).joinpath(self.name + "_searchcv.model")
        self.outfile_scores = Path(self.output_dir).joinpath(self.name + "_scores.json")
        self.oos_validation_scores = Path(self.output_dir).joinpath(self.name + "_oos_validation_scores.json")
        self.optimised_model_scores = Path(self.output_dir).joinpath(self.name + "_searchcv_scores.json")

        # outputs
        self.train_data = Path(self.output_dir).joinpath(self.name + "_train.csv")
        self.optimisation_data = Path(self.output_dir).joinpath(self.name + "_optimisation.csv")
        self.optimisation_output_hpopt = Path(self.output_dir).joinpath(self.name + '_optimisation_hpopt.csv')
        self.pred_data = [Path(self.output_dir).joinpath(self.name + f"_pred_{p.stem}.csv")
                          for p in self.aem_pred_data]
        self.oos_data = Path(self.output_dir).joinpath(self.name + "_oos.csv")
        self.quantiles = s['output']['pred']['quantiles']
        self.aem_lines_plot_train = Path(self.output_dir).joinpath('aem_survey_lines_train.jpg')
        self.aem_lines_plot_oos = Path(self.output_dir).joinpath('aem_survey_lines_oos.jpg')
        self.aem_lines_plot_pred = Path(self.output_dir).joinpath('aem_survey_lines_pred.jpg')
        self.train_scatter_plot = Path(self.output_dir).joinpath('train_scatter.jpg')

        # test train val split
        self.train_fraction = s['data']['test_train_split']['train']
        self.test_fraction = s['data']['test_train_split']['test']
        self.val_fraction = s['data']['test_train_split']['val']


class ConfigException(Exception):
    pass
