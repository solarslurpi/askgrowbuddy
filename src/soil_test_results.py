'''Read in Logan Labs soil test results PDF files and extract the data into a dictionary.'''
# Create a Pydantic model to represent the mehlic-3 values.

from pydantic import BaseModel, Field, ValidationError
from typing import Optional
import PyPDF2
import re
import logging

from base_processor import BaseProcessor

class LoganLabsMehlich3Values(BaseModel):
    """Pydantic model representing Logan Labs Mehlich-3 soil test values."""

    # Sample information
    sample_id: str = Field(..., description="Sample ID")
    lab_number: str = Field(..., description="Lab Number")
    date: str = Field(..., description="Test Date")

    # Basic soil properties
    ph: float = Field(..., description="Soil pH")
    buffer_ph: Optional[float] = Field(None, description="Buffer pH")
    cec: float = Field(..., description="CEC (meq/100g)")
    percent_base_saturation: float = Field(..., description="% Base Saturation")
    organic_matter: float = Field(..., description="% Organic Matter")

    # Nutrients (ppm)
    phosphorus: float = Field(..., alias="P", description="Phosphorus (P) in ppm")
    calcium: float = Field(..., alias="Ca", description="Calcium (Ca) in ppm")
    magnesium: float = Field(..., alias="Mg", description="Magnesium (Mg) in ppm")
    potassium: float = Field(..., alias="K", description="Potassium (K) in ppm")
    sodium: float = Field(..., alias="Na", description="Sodium (Na) in ppm")
    sulfur: float = Field(..., alias="S", description="Sulfur (S) in ppm")
    boron: float = Field(..., alias="B", description="Boron (B) in ppm")
    iron: float = Field(..., alias="Fe", description="Iron (Fe) in ppm")
    manganese: float = Field(..., alias="Mn", description="Manganese (Mn) in ppm")
    copper: float = Field(..., alias="Cu", description="Copper (Cu) in ppm")
    zinc: float = Field(..., alias="Zn", description="Zinc (Zn) in ppm")
    aluminum: float = Field(..., alias="Al", description="Aluminum (Al) in ppm")

    # Base saturations (%)
    ca_base_saturation: float = Field(..., description="Calcium Base Saturation (%)")
    mg_base_saturation: float = Field(..., description="Magnesium Base Saturation (%)")
    k_base_saturation: float = Field(..., description="Potassium Base Saturation (%)")
    na_base_saturation: float = Field(..., description="Sodium Base Saturation (%)")
    other_bases: float = Field(..., description="Other Bases (%)")
    h_base_saturation: float = Field(..., description="Hydrogen Base Saturation (%)")

    # Element ratios
    ca_mg_ratio: float = Field(..., description="Ca/Mg Ratio")
    ca_k_ratio: float = Field(..., description="Ca/K Ratio")
    mg_k_ratio: float = Field(..., description="Mg/K Ratio")

    class Config:
        allow_population_by_field_name = True

def parse_logan_labs_m3_pdf(pdf_path: str) -> LoganLabsMehlich3Values:
    """
    Reads a Logan Labs Mehlich-3 PDF report and returns a populated LoganLabsMehlich3Values instance.

    :param pdf_path: Path to the PDF file
    :return: LoganLabsMehlich3Values instance
    """
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

    logger.debug(f"Extracted text: {text}")

    # Extract all numeric values
    numeric_values = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    numeric_values = [float(value) for value in numeric_values]

    logger.debug(f"Extracted numeric values: {numeric_values}")

    # Map numeric values to fields
    values = {
        'sample_id': "Unknown",
        'lab_number': str(int(numeric_values[0])) if len(numeric_values) > 0 else "Unknown",
        'date': "Unknown",
        'cec': numeric_values[1] if len(numeric_values) > 1 else None,
        'ph': numeric_values[2] if len(numeric_values) > 2 else None,
        'organic_matter': numeric_values[3] if len(numeric_values) > 3 else None,
        'S': numeric_values[4] if len(numeric_values) > 4 else None,
        'P': numeric_values[5] if len(numeric_values) > 5 else None,
        'Ca': numeric_values[6] if len(numeric_values) > 6 else None,
        'Mg': numeric_values[7] if len(numeric_values) > 7 else None,
        'K': numeric_values[8] if len(numeric_values) > 8 else None,
        'Na': numeric_values[9] if len(numeric_values) > 9 else None,
        'ca_base_saturation': numeric_values[10] if len(numeric_values) > 10 else None,
        'mg_base_saturation': numeric_values[11] if len(numeric_values) > 11 else None,
        'k_base_saturation': numeric_values[12] if len(numeric_values) > 12 else None,
        'na_base_saturation': numeric_values[13] if len(numeric_values) > 13 else None,
        'other_bases': numeric_values[14] if len(numeric_values) > 14 else None,
        'h_base_saturation': numeric_values[15] if len(numeric_values) > 15 else None,
        'B': numeric_values[16] if len(numeric_values) > 16 else None,
        'Fe': numeric_values[17] if len(numeric_values) > 17 else None,
        'Mn': numeric_values[18] if len(numeric_values) > 18 else None,
        'Cu': numeric_values[19] if len(numeric_values) > 19 else None,
        'Zn': numeric_values[20] if len(numeric_values) > 20 else None,
        'Al': numeric_values[21] if len(numeric_values) > 21 else None,
        'ca_mg_ratio': numeric_values[22] if len(numeric_values) > 22 else None,
        'ca_k_ratio': numeric_values[23] if len(numeric_values) > 23 else None,
        'mg_k_ratio': numeric_values[24] if len(numeric_values) > 24 else None,
    }

    # Calculate percent_base_saturation
    base_saturations = [values[f] for f in ['ca_base_saturation', 'mg_base_saturation', 'k_base_saturation', 'na_base_saturation', 'other_bases'] if values[f] is not None]
    values['percent_base_saturation'] = sum(base_saturations) if base_saturations else None

    logger.debug(f"Mapped values: {values}")

    try:
        return LoganLabsMehlich3Values(**values)
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise

# Example usage:
try:
    result = parse_logan_labs_m3_pdf(r'C:\Users\happy\Documents\Projects\askgrowbuddy\data\Margaret Johnson-Soil-20240911-179093.pdf')
    print(result.model_dump_json(indent=4))
except Exception as e:
    print(f"An error occurred: {e}")
