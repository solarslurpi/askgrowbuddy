'''Soil Test Analyzer UI'''

import gradio as gr
from src.staui_helpers import chat_function
from src.soil_test_analyst import SoilTestAnalyst
from src.soil_test_models import soil_report_instance
with gr.Blocks() as demo:
    gr.Markdown("Soil Test Analyzer")
    current_soil_property = gr.State("ph")
#  WE've got this UI working, so commenting out for faster development.
    # Checkbox to control upload requirement
    # require_upload_checkbox = gr.Checkbox(label="Require PDF Uploads Before Proceeding", value=True)

    # # PDF Upload Section
    with gr.Row():
        # mehlic_upload = gr.File(label="Upload Mehlic-3 Soil Test PDF", file_types=[".pdf"])
        mehlic_upload = r'C:\Users\happy\Documents\Projects\askgrowbuddy\Margaret Johnson-Soil-20240911-179093.pdf'
        paste_upload = r'C:\Users\happy\Documents\Projects\askgrowbuddy\Margaret Johnson-Saturated Paste-20240911-179093.pdf'
        m3_report, sp_report = SoilTestAnalyst.load_reports(mehlic_upload, paste_upload)
        soil_report_instance.set_reports(m3_report, sp_report)
        # Chat Section
        # paste_upload = gr.File(label="Upload Saturated Paste Soil Test PDF", file_types=[".pdf"])
        upload_btn = gr.Button("Upload and Analyze")
        # Chat Section
    with gr.Column(visible=True) as main_column:
        gr.Markdown("Current Soil Property: " + str(current_soil_property.value))
        chatbot = gr.Chatbot(label="AI Chat")
        user_input = gr.Textbox(label="Your Question")
        follow_up_btn = gr.Button("Follow Up Question")
        next_property_btn = gr.Button("Next Property")

    upload_btn.click(
        fn=chat_function,
        inputs=[user_input, chatbot,current_soil_property],
        outputs=[chatbot, user_input]
    )
    # upload_btn.click(
    #     fn=process_upload,
    #     inputs=[mehlic_upload, paste_upload, require_upload_checkbox, main_column],
    #     outputs=[m3_report_state, sp_report_state, main_column]
    # ).then(
    #     fn=chat_function,
    #     inputs=[user_input, chatbot, m3_report_state, sp_report_state],
    #     outputs=[chatbot, user_input]
    # )

demo.launch()
