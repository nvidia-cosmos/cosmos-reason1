from cosmos_reason1_utils.script import init_script
import pytest


@pytest.mark.parametrize("verbose", [True, False])
def test_init_script(verbose: bool):
    init_script(verbose=verbose)
