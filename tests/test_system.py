from tests.common import sub_process_run


def test_all_configs_work(demo_config):

    for process in ['learn', 'predict']:
        cmd = f"aem {process} --config {demo_config}"
        if process == 'predict':
            cmd += f" --model-type=learn"
        sub_process_run(cmd)
