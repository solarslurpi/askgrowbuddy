import pdfplumber
from src.base_processor import BaseProcessor
from src.soil_test_models import M3Report, SPReport

class M3Processor(BaseProcessor):
    def process_pdf(self, pdf_file: str) -> M3Report:
        with pdfplumber.open(pdf_file) as pdf:
            table = pdf.pages[0].extract_table()

        # Parsing the header section
        m3_report_dict = self._parse_section(
            table,
            report_model=M3Report,
            row_range=(0, 7),
            value_col=2,
            field_name_col=0
        )

        # Parsing the bottom section
        m3_report_dict.update(self._parse_section(
            table,
            report_model=M3Report,
            row_range=(7, 28),
            value_col=2,
            field_name_col=1
        ))

        m3_report_dict['sample_date'] = self._extract_date_from_pdf(pdf_file)
        return M3Report(**m3_report_dict)

class SPProcessor(BaseProcessor):
    def process_pdf(self, pdf_file: str) -> SPReport:
        with pdfplumber.open(pdf_file) as pdf:
            table = pdf.pages[0].extract_table()

        # Parsing the header section
        sp_report_dict = self._parse_section(
            table,
            report_model=SPReport,
            row_range=(0, 8),
            value_col=3,
            field_name_col=0
        )

        # Parsing the midsection and bottom section
        sp_report_dict.update(self._parse_section(
            table,
            report_model=SPReport,
            row_range=(8, 28),
            value_col=3,
            field_name_col=1
        ))
        sp_report_dict['sample_date'] = self._extract_date_from_pdf(pdf_file)
        return SPReport(**sp_report_dict)

# m3 = M3Processor()
# m3_report = m3.process_pdf(r"C:\Users\happy\Documents\Projects\askgrowbuddy\data\Margaret Johnson-Soil-20240911-179093.pdf")
# print(m3_report.model_dump_json(indent=4))

# sp = SPProcessor()
# sp_report = sp.process_pdf(r"C:\Users\happy\Documents\Projects\askgrowbuddy\data\Margaret Johnson-Saturated Paste-20240911-179093.pdf")
# print(sp_report.model_dump_json(indent=4))
