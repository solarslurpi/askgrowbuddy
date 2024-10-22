import gradio as gr

# Function to handle PDF file uploads and extract initial analysis
def handle_pdf_upload(mehlic_pdf, paste_pdf, require_upload):
    if require_upload:
        if mehlic_pdf is None or paste_pdf is None:
            return "Please upload both Mehlic-3 and Saturated Paste soil test PDFs."
        # Placeholder for file analysis logic
        return (
            f"Files received: Mehlic-3 ({mehlic_pdf.name}) and Saturated Paste ({paste_pdf.name}). Analysis in progress.",
        )
    else:
        return "PDF Uploads are not required, proceeding to chat and feedback."

# Function to simulate chat response
def chatbot_response(user_input, history):
    history = history or []
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": f"AI: {user_input}"})
    return history

# Function for handling accordion button click for additional analysis
def handle_accordion_click(soil_input):
    return f"**Soil Analysis Results:** Nutrient levels for {soil_input} analyzed successfully."

# Placeholder function for saving feedback
def save_feedback(response_quality, feedback):
    return f"Feedback saved with quality rating: {response_quality}/10"

# Gradio interface setup
with gr.Blocks() as demo:
    # Checkbox to control upload requirement
    require_upload_checkbox = gr.Checkbox(label="Require PDF Uploads Before Proceeding", value=True)

    # PDF Upload Section
    with gr.Row():
        mehlic_upload = gr.File(label="Upload Mehlic-3 Soil Test PDF", file_types=[".pdf"])
        paste_upload = gr.File(label="Upload Saturated Paste Soil Test PDF", file_types=[".pdf"])
        upload_btn = gr.Button("Upload and Analyze")
        initial_analysis_output = gr.Markdown()

        # Link upload button to handle file uploads
        upload_btn.click(fn=handle_pdf_upload, inputs=[mehlic_upload, paste_upload, require_upload_checkbox], outputs=initial_analysis_output)

    # Chat and Feedback UI Elements (initially hidden)
    with gr.Column(visible=False) as main_column:
        chatbot = gr.Chatbot(label="AI Chat", type="messages")
        user_input = gr.Textbox(label="Your Question")
        send_btn = gr.Button("Send")

        # Link chat input with chatbot response
        send_btn.click(fn=chatbot_response, inputs=[user_input, chatbot], outputs=chatbot)

        # Accordion Section for Detailed Feedback
        with gr.Accordion("Feedback", open=False):
            # First Row: LLM and Token Info in Markdown
            llm_info = gr.Markdown("LLM: Llama3.2 | Number of Prompt Tokens: 3082 | Number of Answer Tokens: 346")

            # Response Quality and Feedback
            response_quality = gr.Slider(minimum=0, maximum=10, label="Response Quality", value=5)
            feedback = gr.Textbox(label="Feedback", lines=4)
            save_feedback_btn = gr.Button("Save Feedback")

            # Link save feedback button to function
            save_feedback_btn.click(fn=save_feedback, inputs=[response_quality, feedback], outputs=initial_analysis_output)

            # Retriever Prompt and Context
            retriever_prompt = gr.Textbox(label="Retriever Prompt", lines=4)
            prompt_intro = gr.Textbox(label="Prompt Intro", lines=2)

            context = gr.Markdown(value="3 nodes retrieved")
            with gr.Row():
                node_text = gr.Textbox(label="Node Text", lines=2)
                node_metadata = gr.Textbox(label="Metadata")
                node_score = gr.Textbox(label="Score")
                navigation = gr.Row([gr.Button("Previous"), gr.Button("Next")])

            # Prompt Outro
            prompt_outro = gr.Textbox(label="Prompt Outro", lines=4)

            # LLM Selection and Similarity Parameters on the Same Row
            with gr.Row():
                llm_selector = gr.Dropdown(choices=["Llama3.2", "Mistral"], label="Choose LLM")
                similarity_index = gr.Number(label="Similarity Top-K Index")
                similarity_rank = gr.Number(label="Similarity Top-K Rank")

            # Redo Button
            redo_btn = gr.Button("Redo")

    # Make the main_column visible after PDF upload
    upload_btn.click(fn=lambda: gr.update(visible=True), inputs=[], outputs=main_column)

demo.launch()
