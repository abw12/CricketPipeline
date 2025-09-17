import pytest
import sys
import os

current_dir = os.path.abspath('')
# Add parent directory to Python path to allow module imports
sys.path.append(os.path.abspath(os.path.join(current_dir,'..','..')))

from etl.common import get_spark

@pytest.fixture(scope="session")
def spark():
    s = get_spark("tests")
    yield s
    s.stop()