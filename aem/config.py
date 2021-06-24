import yaml
from pathlib import Path


# column containing H, M, L categories corresponding to confidence levels of interpretation
confidence_indicator_col = 'BoundConf'

conductivity_columns_starts_with = 'cond'
thickness_columns_starts_with = 'thick'

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
            self.weight_dict = s['learning']['weighted_model']['weights']
            self.weight_col = s['learning']['weighted_model']['weight_col']

        # data description
        self.line_col = s['data']['line_col']


