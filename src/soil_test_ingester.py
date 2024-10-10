import pdfplumber
from pydantic import BaseModel, Field
from typing import Optional

class M3(BaseModel):
    Sample_Location: Optional[str] = Field(None, alias="Sample Location")
    TEC: Optional[float] = Field(None, alias="Total Exchange Capacity (M. E.)")
    pH: Optional[float] = Field(None, alias="pH of Soil Sample")
    Organic_: Optional[float] = Field(None, alias="Organic Matter, Percent")
    Sulfur_ppm: Optional[float] = Field(None, alias="SULFUR (ppm)")
    Bicarbonate_ppm: Optional[float] = Field(None, alias="Bicarbonate (HCO3) ppm")
    Phosphorous_lbs: Optional[float] = Field(None, alias="Mehlich III Phosphorous: as (P O )\n2 5\nlbs / acre")
    Calcium_lbs: Optional[float] = Field(None, alias="Desired Value\nCALCIUM:\nValue Found\nlbs / acre\nDeficit")
    Magnesium_lbs: Optional[float] = Field(None, alias="Desired Value\nMAGNESIUM:\nValue Found\nlbs / acre\nDeficit")
    Potassium_lbs: Optional[float] = Field(None, alias="Desired Value\nPOTASSIUM:\nlbs / acre Value Found\nDeficit")
    Sodium_lbs: Optional[float] = Field(None, alias="SODIUM: lbs / acre")
    Calcium_: Optional[float] = Field(None, alias="Calcium (60 to 70%)")
    Magnesium_: Optional[float] = Field(None, alias="Magnesium (10 to 20%)")
    Potassium_: Optional[float] = Field(None, alias="Potassium (2 to 5%)")
    Sodium_: Optional[float] = Field(None, alias="Sodium (.5 to 3%)")
    Soluble_Salts_ppm: Optional[float] = Field(None, alias="Soluble Salts ppm")
    Chloride_ppm: Optional[float] = Field(None, alias="Chloride (Cl) ppm")
    Other_bases: Optional[float] = Field(None, alias="Other Bases (Variable)")
    Exchangable_Hydrogen: Optional[float] = Field(None, alias="Exchangable Hydrogen (10 to 15%)")
    Boron_ppm: Optional[float] = Field(None, alias="Boron (p.p.m.)")
    Iron_ppm: Optional[float] = Field(None, alias="Iron (p.p.m.)")
    Manganese_ppm: Optional[float] = Field(None, alias="Manganese (p.p.m.)")
    Copper_ppm: Optional[float] = Field(None, alias="Copper (p.p.m.)")
    Zinc_ppm: Optional[float] = Field(None, alias="Zinc (p.p.m.)")
    Aluminum_ppm: Optional[float] = Field(None, alias="Aluminum (p.p.m.)")
    Ammonium_ppm: Optional[float] = Field(None, alias="Ammonium (p.p.m.)")
    Nitrate_ppm: Optional[float] = Field(None, alias="Nitrate (p.p.m.)")
    Media_Density: Optional[float] = Field(None, alias="Media Density g/cm3")
    Phosphorous_ppm: Optional[float] = Field(None, alias="PHOSPHORUS (ppm)")
    Calcium_ppm: Optional[float] = Field(None, alias="CALCIUM (ppm)")
    Calcium_meq_L: Optional[float] = Field(None, alias="CALCIUM (meq/l)")
    Magnesium_ppm: Optional[float] = Field(None, alias="MAGNESIUM (ppm)")
    Magnesium_meq_L: Optional[float] = Field(None, alias="MAGNESIUM (meq/l)")
    Potassium_ppm: Optional[float] = Field(None, alias="POTASSIUM: (ppm)")
    Potassium_meq_L: Optional[float] = Field(None, alias="POTASSIUM: (meq/l)")
    Sodium_ppm: Optional[float] = Field(None, alias="SODIUM (ppm)")
    Sodium_meq_L: Optional[float] = Field(None, alias="SODIUM (meq/l)")

    class Config:
        allow_population_by_field_name = True

class M3Processor(BaseProcessor):
    def __init__(self) -> None:
        super().__init__()
        self._measurement_name = "M3"

    def process_pdf(self, pdf_file: str) -> None:
        with pdfplumber.open(pdf_file) as pdf:
            table = pdf.pages[0].extract_table()

        # Parse the table
        field_names, values = self._parse_header(table, 7, 2)
        bottom_field_names, bottom_values = self._parse_bottom(table, 2, 7)

        field_names.extend(bottom_field_names)
        values.extend(bottom_values)

        # Create a dictionary from field names and values
        data_dict = dict(zip(field_names, values))

        # Create an M3 instance
        m3_data = M3(**data_dict)

        # Convert M3 instance to a dictionary
        m3_dict = m3_data.dict(by_alias=True)

        # Create a DataFrame
        df = pd.DataFrame([m3_dict])

        # Store the results
        self._store_markdown_file("M3", df, pdf_file)

    @property
    def measurement_name(self):
        return self._measurement_name