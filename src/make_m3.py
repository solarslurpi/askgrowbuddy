# from base_processor import BaseProcessor
from soil_test_models import M3Report
import pdfplumber
from datetime import datetime

class M3Processor:
    def __init__(self) -> None:
        super().__init__()
        self._measurement_name = "M3"

    def process_pdf(self, pdf_file: str) -> M3Report:
        with pdfplumber.open(pdf_file) as pdf:
            table = pdf.pages[0].extract_table()

        data = {}
        # Get the names and aliases of the fields in the M3Report model.
        field_aliases = {field.alias: name for name, field in M3Report.__fields__.items()}
        # Process the headers.
        # Iterate over the table rows and map the field names to the actual field names in the model.
        for row in table:
            field_name = row[0].strip()
            value = row[1].strip()

            if field_name in field_aliases:
                mapped_field_name = field_aliases[field_name]
                data[mapped_field_name] = float(value.strip('%')) / 100 if '%' in value else float(value)

        m3_report = M3Report(
            sample_location="Sample Location",  # Replace with actual value
            sample_id="Sample ID",  # Replace with actual value
            lab_number="Lab Number",  # Replace with actual value
            sample_date=datetime.strptime("2023-10-01", "%Y-%m-%d"),  # Replace with actual value
            **data
        )

        return m3_report

    @property
    def measurement_name(self):
        return self._measurement_name

m3_processor = M3Processor()
m3_processor.process_pdf(r"C:\Users\happy\Documents\Projects\askgrowbuddy\data\Margaret Johnson-Soil-20240911-179093.pdf")