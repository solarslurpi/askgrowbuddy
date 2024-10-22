import json
import logging
import pytest

from src.logging_config import setup_logging
from src.soil_test_analyst import SoilTestAnalyst


setup_logging()
logger = logging.getLogger(__name__)

@pytest.fixture
def soil_property_name():
    return "ph"

@pytest.fixture
def m3_pdf_path():
    return r'C:\Users\happy\Documents\Projects\askgrowbuddy\Margaret Johnson-Soil-20240911-179093.pdf'

@pytest.fixture
def sp_pdf_path():
    return r'C:\Users\happy\Documents\Projects\askgrowbuddy\Margaret Johnson-Saturated Paste-20240911-179093.pdf'

def test_generate_response(soil_property_name, m3_pdf_path, sp_pdf_path):
    analyst = SoilTestAnalyst(m3_pdf_path, sp_pdf_path)
    response = analyst.generate_response(soil_property_name)
    # Pretty print the response as JSON
    print(json.dumps(response, indent=2, sort_keys=True))
    assert response is not None