import re
import logging
from typing import Any
from abc import ABC
from datetime import datetime
import pdfplumber
from src.soil_test_models import readings_to_exclude

logger = logging.getLogger(__name__)

class BaseProcessor(ABC):
    def __init__(self) -> None:
        pass

    def _parse_section(self, table: list[list[str | None]], report_model, row_range: tuple[int, int], value_col: int, field_name_col: int = 1) -> dict[str, Any]:
        '''
        Parses a section from the table list and extracts the data for a given report model. The section parsed depends on the row_range. The table list was built using the pdfplumber library extracting results from either a mehlic-3 or saturated paste soil test report.
        Args:
            table (list[list[str]]): The rows from the the PDF soil test report.  It was built using the pdfplumber library.
            report_model (BaseModel): This is either the Pydantic model for the M3 report or the SP report.
            row_range (tuple[int, int]): The range of rows in the table listto parse.
            value_col (int): The column index containing the values.
            field_name_col (int, optional): The column index containing the field names. Defaults to 1.

        Returns:
            dict[str, Any]: The parsed data.
        '''
        # The field_aliases dictionary maps the names (e.g.:'pH of Soil Sample' to the property name (e.g.: 'ph'))
        field_aliases = {self._normalize_field_name(field.alias): name for name, field in report_model.__fields__.items()}
        parsed_data = {}
        current_field_name = None
        row_index = row_range[0]
        while row_index <row_range[1]:
            row = table[row_index]
            if len(row) > field_name_col and row[field_name_col]:
                current_field_name = self._normalize_field_name(row[field_name_col].strip())

            value = row[value_col].strip() if len(row) > value_col and row[value_col] else None

            # Skip excluded fields or empty field names
            if not current_field_name or current_field_name in readings_to_exclude or value is None:
                row_index += 1
                continue

            # Only check for meq value if the field name matches a soluble cation
            if current_field_name in ["CALCIUM", "MAGNESIUM", "POTASSIUM:", "SODIUM"]:
                # Store ppm value
                if current_field_name in field_aliases:
                    mapped_field_name = field_aliases[current_field_name]
                    parsed_data[mapped_field_name] = value

                # Check next row for meq value\
                row_index += 1
                if row_index  < row_range[1]:
                    next_row = table[row_index ]
                    if len(next_row) > value_col and 'meq' in next_row[value_col-1].lower():
                        meq_value = next_row[value_col].strip()
                        meq_field_name = f"{current_field_name.lower()}_meq"
                        if meq_field_name in report_model.__fields__:
                            parsed_data[meq_field_name] = meq_value
            else:
                if current_field_name in field_aliases:
                    mapped_field_name = field_aliases[current_field_name]
                    parsed_data[mapped_field_name] = value
            row_index += 1
        parsed_data = self._convert_float_strs_to_float(parsed_data)
        return parsed_data

    def _normalize_field_name(self, field_name: str) -> str:
        if not field_name:
            return ""
        return re.sub(r'\s+', ' ', field_name).strip()

    def _extract_date_from_pdf(self, pdf_file) -> str:
        # Extract date from the PDF text
        with pdfplumber.open(pdf_file) as pdf:
            text = pdf.pages[0].extract_text()
        date_match = re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", text)

        if date_match:
            date_str = date_match.group(0)
            # Parse the date string into a datetime object
            date_obj = datetime.strptime(date_str, "%m/%d/%Y")
            # Convert the datetime object into a date string (YYYY-MM-DD)
            iso_date_str = date_obj.strftime("%Y-%m-%d")
        else:
            # Return the current date in YYYY-MM-DD format
            iso_date_str = datetime.now().strftime("%Y-%m-%d")

        # Ensure only the date string is returned without any time component
        return iso_date_str

    def _convert_float_strs_to_float(self,data: dict[str, Any]) -> dict[str, Any]:
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
