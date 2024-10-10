import pdfplumber

from src.base_processor import BaseProcessor
from src.soil_test_models import SPReport





class SPProcessor(BaseProcessor):
    def __init__(self) -> None:
        super().__init__()

    def process_pdf(self, pdf_file: str) -> SPReport:
        with pdfplumber.open(pdf_file) as pdf:
            table = pdf.pages[0].extract_table()
        field_aliases = {field.alias: name for name, field in SPReport.__fields__.items()}
        sp_report_dict = self._parse_header(table,field_aliases,8,3)
        sp_report_dict.update(self._parse_midsection(table,field_aliases))
        sp_report_dict.update(self._parse_bottom(table,field_aliases,3,18))
        sp_report_dict['sample_date'] = self._extract_date_from_pdf(pdf_file)
        # Convert string values to floats where applicable
        sp_report_dict = SPReport.convert_float_strs_to_float(sp_report_dict)
        sp_report = SPReport(**sp_report_dict)
        return sp_report


    def _parse_midsection(self, table: list[list[str]], field_aliases: dict) -> dict:
        part_of_soil_test_dict = {}
        for row in table[8:18]:
            # Check if there's a new field name in the current row. When no row[1], it is the meq/L measurement.
            if row[1]:  # If row[1] is not empty, update the current field name
                field_name = row[1]
            # Assign the value from the fourth column, if available
            value = row[3] if row[3] else None
            if field_name and value is not None:
                mapped_field_name = field_aliases[field_name]
                part_of_soil_test_dict[mapped_field_name] = value
        return part_of_soil_test_dict



# sp = SPProcessor()
# sp_report = sp.process_pdf(r"C:\Users\happy\Documents\Projects\askgrowbuddy\data\Margaret Johnson-Saturated Paste-20240911-179093.pdf")
# print(sp_report.model_dump_json(indent=4))
