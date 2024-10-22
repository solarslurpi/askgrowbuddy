
from typing import Tuple
import gradio as gr
from src.soil_test_processors import M3Processor, SPProcessor
from src.soil_test_analyst import SoilTestAnalyst

def process_upload(mehlic_pdf:str=None, paste_pdf:str=None, require_upload:bool=False, main_column:gr.Column=False) -> Tuple[gr.State, gr.State]:
    if require_upload: # Usually this is true. It could be false for ui testing.
        # Retrieve the reports
        try:
            m3_report = M3Processor().process_pdf(mehlic_pdf)
            sp_report = SPProcessor().process_pdf(paste_pdf)
            m3_report_state = gr.State(m3_report)
            sp_report_state = gr.State(sp_report)
            # Show the rest of the UI.
            main_column.update(visible=True)
            return m3_report_state, sp_report_state
        except Exception as e:
            raise f"Error: {str(e)}"

    return gr.State(None), gr.State(None)

def chat_function(user_input: str, chat_history: list[Tuple[str, str]],  current_soil_property: str) -> Tuple[list[Tuple[str, str]], str]:
    analyst = SoilTestAnalyst()
    bot_message = analyst.generate_response(current_soil_property )
    # bot_message = """Let's start our analysis by looking at the pH results. The pH value is 6.9, which is right at the sweet spot for Cannabis cultivation. That's a great start!""""
    chat_history.append((user_input, bot_message))
    return chat_history, ""  # Return empty string to clear input
