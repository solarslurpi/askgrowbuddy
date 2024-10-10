import logging

from pydantic import BaseModel, Field, ConfigDict
from datetime import date
from typing import Optional, Dict, Any

from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
# The aliases come from the names on the pdfs.  They are used to map the values to the correct fields.
class BaseSoilReport(BaseModel):
    sample_location: Optional[str] = None
    sample_id: Optional[str] = None
    lab_number: Optional[str] = None
    sample_date: date

    model_config = ConfigDict(populate_by_name=True)

class M3Report(BaseSoilReport):
    ph: float = Field(..., alias="pH of Soil Sample")
    organic_matter: str = Field(..., alias="Organic Matter, Percent")
    tec: float = Field(..., alias="Total Exchange Capacity (M. E.)")
    calcium_lbs: float = Field(..., alias="Desired Value\nCALCIUM:\nValue Found\nlbs / acre\nDeficit")
    magnesium_lbs: float = Field(..., alias="Desired Value\nMAGNESIUM:\nValue Found\nlbs / acre\nDeficit")
    potassium_lbs: float = Field(..., alias="Desired Value\nPOTASSIUM:\nlbs / acre Value Found\nDeficit")
    sodium_lbs: float = Field(..., alias="SODIUM: lbs / acre")
    Phosphate_lbs: float = Field(..., alias="Mehlich III Phosphorous: as (P O )\n2 5\nlbs / acre")
    sulfur_ppm: float = Field(..., alias="SULFUR: p.p.m.")
    calcium_pct: float = Field(..., alias="Calcium (60 to 70%)")
    magnesium_pct: float = Field(..., alias="Magnesium (10 to 20%)")
    potassium_pct: float = Field(..., alias="Potassium (2 to 5%)")
    sodium_pct: float = Field(..., alias="Sodium (.5 to 3%)")
    other_bases_pct: float = Field(..., alias="Other Bases (Variable)")
    exchangeable_hydrogen_pct: float = Field(..., alias="Exchangable Hydrogen (10 to 15%)")
    boron_ppm: float = Field(..., alias="Boron (p.p.m.)")
    iron_ppm: float = Field(..., alias="Iron (p.p.m.)")
    manganese_ppm: float = Field(..., alias="Manganese (p.p.m.)")
    copper_ppm: float = Field(..., alias="Copper (p.p.m.)")
    zinc_ppm: float = Field(..., alias="Zinc (p.p.m.)")
    aluminum_ppm: float = Field(..., alias="Aluminum (p.p.m.)")
    ammonium_ppm: float = Field(..., alias="Ammonium (p.p.m.)")
    nitrate_ppm: float = Field(..., alias="Nitrate (p.p.m.)")
    media_density: float = Field(..., alias="Media Density g/cm3")

    # Allow the field names to be used as keys for the values.
    model_config = ConfigDict(populate_by_name=True)

# Saturated Paste Report Model
class SPReport(BaseSoilReport):
    ph: float = Field(..., alias="pH")
    soluble_salts_ppm: float = Field(..., alias="Soluble Salts ppm")
    chloride_ppm: float = Field(..., alias="Chloride (Cl) ppm")
    bicarbonate_ppm: float = Field(..., alias="Bicarbonate (HCO3) ppm")
    sulfur_ppm: float = Field(..., alias="SULFUR")
    phosphorus_ppm: float = Field(..., alias="PHOSPHORUS")
    calcium_ppm: float = Field(..., alias="CALCIUM")
    magnesium_ppm: float = Field(..., alias="MAGNESIUM")
    potassium_ppm: float = Field(..., alias="POTASSIUM:")
    sodium_ppm: float = Field(..., alias="SODIUM")
    calcium_pct: float = Field(..., alias="Calcium")
    magnesium_pct: float = Field(..., alias="Magnesium")
    potassium_pct: float = Field(..., alias="Potassium")
    sodium_pct: float = Field(..., alias="Sodium")
    boron_ppm: float = Field(..., alias="Boron (p.p.m.)")
    iron_ppm: float = Field(..., alias="Iron (p.p.m.)")
    manganese_ppm: float = Field(..., alias="Manganese (p.p.m.)")
    copper_ppm: float = Field(..., alias="Copper (p.p.m.)")
    zinc_ppm: float = Field(..., alias="Zinc (p.p.m.)")
    aluminum_ppm: float = Field(..., alias="Aluminum (p.p.m.)")

    model_config = ConfigDict(populate_by_name=True)

    @classmethod
    def convert_float_strs_to_float(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts string representations of numbers in the dictionary to floats.
        Handles special cases like strings starting with '>'.

        Args:
            data (Dict[str, Any]): The dictionary with string values to convert.

        Returns:
            Dict[str, Any]: The dictionary with converted float values.
        """
        converted_data = {}
        for key, value in data.items():
            if isinstance(value, str) and key != 'sample_date':
                try:
                    # Handle cases like '>20'
                    if value.startswith(('>', '<')):
                        converted_value = float(value[1:])
                    else:
                        converted_value = float(value.replace(',', ''))
                    converted_data[key] = converted_value
                except ValueError:
                    logging.warning(f"Unable to convert field '{key}' with value '{value}' to float.")
                    converted_data[key] = value  # Keep original value if conversion fails
            else:
                converted_data[key] = value  # Keep original value if not a string
        return converted_data


# Readings to exclude
readings_to_exclude = set([
    "Sample Location",
    "Sample ID",
    "Lab Number",
    "Sample Depth in inches",
    "Water Used",
])