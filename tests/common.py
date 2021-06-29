from pathlib import Path
from subprocess import run
from aem.aem import AEMPATH


TESTS = AEMPATH.joinpath('tests')
CONFIGS = AEMPATH.joinpath('configs')

all_configs = list(CONFIGS.glob('*.yaml'))


def sub_process_run(cmd, *args, **kwargs):
    return run(cmd, *args, shell=True, check=True, **kwargs)
