import logging
from src.logging_config  import setup_logging
from src.soil_test_analyst import SoilTestAnalyst

setup_logging()
logger = logging.getLogger(__name__)

def test_load_reports():
    '''Test loading in the pdf files sent over email from Logan Labs.  It assumes the availability of a mehlic-3 and saturated paste soil test report.  Disecting the PDF is a bit brittle and will need to be updated when the format changes.  When the test fails it is likely due to a change in the format of the PDF reports.'''
    m3_pdf = r'G:\My Drive\Audios_To_Knowledge\knowledge\AskGrowBuddy\AskGrowBuddy\soil test results\2024-09-11\Margaret Johnson-Soil-20240911-179093.pdf'
    sp_pdf = r'G:\My Drive\Audios_To_Knowledge\knowledge\AskGrowBuddy\AskGrowBuddy\soil test results\2024-09-11\Margaret Johnson-Saturated Paste-20240911-179093.pdf'
    m3_report, sp_report = SoilTestAnalyst.load_reports(m3_pdf_filename=m3_pdf, sp_pdf_filename=sp_pdf)
    assert m3_report is not None
    assert sp_report is not None
    assert m3_report.ph == 6.9

        # Dump m3_report
    m3_dump = m3_report.model_dump_json(indent=2)
    logger.info("M3 Report Dump:")
    logger.info(m3_dump)

    # Dump sp_report
    sp_dump = sp_report.model_dump_json(indent=2)
    logger.info("SP Report Dump:")
    logger.info(sp_dump)
