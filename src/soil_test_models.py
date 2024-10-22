import logging
from typing import Optional

from datetime import date

from pydantic import BaseModel, Field, ConfigDict

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
    organic_matter: float = Field(..., alias="Organic Matter, Percent")
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
    calcium_meq: float = Field(None, alias="CALCIUM meq/l")
    magnesium_meq: float = Field(None, alias="MAGNESIUM meq/l")
    potassium_meq: float = Field(None, alias="POTASSIUM meq/l")
    sodium_meq: float = Field(None, alias="SODIUM meq/l")
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

class SoilReportSingleton:
    _instance = None

    def __init__(self):
        self.m3_report: Optional[M3Report] = None
        self.sp_report: Optional[SPReport] = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.m3_report = None
            cls._instance.sp_report = None
        return cls._instance

    def set_m3_report(self, report: M3Report):
        self.m3_report = report

    def set_sp_report(self, report: SPReport):
        self.sp_report = report

    def get_m3_report(self) -> Optional[M3Report]:
        return self.m3_report

    def get_sp_report(self) -> Optional[SPReport]:
        return self.sp_report

    def set_reports(self, m3_report, sp_report):
        self.m3_report = m3_report
        self.sp_report = sp_report


# Readings to exclude
readings_to_exclude = set([
    "Sample Location",
    "Sample ID",
    "Lab Number",
    "Sample Depth in inches",
    "Water Used",
])

# Create a global instance of the singleton
soil_report_instance = SoilReportSingleton()

# Add this line at the end of the file
__all__ = ['soil_report_instance']