import pytest
from tests.common import sub_process_run


def test_all_configs_work(demo_config):

    for process in ['learn', 'predict']:
        sub_process_run(f"aem {process} --config {demo_config}")
