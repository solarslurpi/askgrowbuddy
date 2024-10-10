import pdfplumber
from src.base_processor import BaseProcessor
from src.soil_test_models import M3Report


class M3Processor(BaseProcessor):
    def __init__(self) -> None:
        super().__init__()

    def process_pdf(self, pdf_file: str) -> M3Report:
        with pdfplumber.open(pdf_file) as pdf:
            table = pdf.pages[0].extract_table()
        field_aliases = {field.alias: name for name, field in M3Report.__fields__.items()}
        m3_report_dict = self._parse_header(table,field_aliases,7,2)
        m3_report_dict.update(self._parse_bottom(table,field_aliases,2,7))
        m3_report_dict['sample_date'] = self._extract_date_from_pdf(pdf_file)
        m3_report = M3Report(**m3_report_dict)
        return m3_report


# m3 = M3Processor()
# m3_report = m3.process_pdf(r"C:\Users\happy\Documents\Projects\askgrowbuddy\data\Margaret Johnson-Soil-20240911-179093.pdf")
# print(m3_report.model_dump_json(indent=4))
