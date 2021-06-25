import yaml
from pathlib import Path
import geopandas as gpd


# column containing H, M, L categories corresponding to confidence levels of interpretation
confidence_indicator_col = 'BoundConf'

twod_coords = ['POINT_X', 'POINT_Y']
threed_coords = twod_coords + ['Z_coor']


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

        # data
        self.aem_folder = s['data']['aem_folder']
        self.interp_data = Path(self.aem_folder).joinpath(s['data']['interp_data'])
        self.aem_data = Path(self.aem_folder).joinpath(s['data']['aem_data'])

        # training
        self.algorithm = s['learning']['algorithm']
        self.model_params = s['learning']['params']

        # model parameter optimisation
        self.opt_space = s['learning']['optimisation']

        # weighted model params
        if 'weighted_model' in s['learning']:
            self.weighted_model = True
            self.weight_dict = s['learning']['weighted_model']['weights']
            self.weight_col = s['learning']['weighted_model']['weight_col']
        else:
            self.weighted_model = False

        # data description
        self.line_col = s['data']['line_col']
        self.conductivity_columns_starts_with = s['data']['conductivity_columns_starts_with']
        self.thickness_columns_starts_with = s['data']['thickness_columns_starts_with']
        self.aem_covariate_cols = s['data']['aem_covariate_cols']

        # all_interp_data = gpd.GeoDataFrame.from_file(self.interp_data, rows=1)
        original_aem_data = gpd.GeoDataFrame.from_file(self.aem_data, rows=1)

        conductivity_cols = [c for c in original_aem_data.columns if c.startswith(self.conductivity_columns_starts_with)]
        d_conductivities = ['d_' + c for c in conductivity_cols]
        conductivity_and_derivatives_cols = conductivity_cols + d_conductivities
        thickness_cols = [t for t in original_aem_data.columns if t.startswith(self.thickness_columns_starts_with)]

        self.thickness_cols = thickness_cols
        self.conductivity_cols = conductivity_cols
        self.conductivity_derivatives_cols = d_conductivities
        self.conductivity_and_derivatives_cols = conductivity_and_derivatives_cols

        # co-ordination
        self.output_dir = s['output']['directory']
        self.outfile_state = Path(self.output_dir).joinpath(self.name + ".model")
        self.outfile_scores = Path(self.output_dir).joinpath(self.name + "_scores.json")

