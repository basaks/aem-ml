import pytest
from tests.common import all_configs


@pytest.fixture(params=all_configs)
def demo_config(request):
    return request.param
