import gradio as gr

# Example response from LLM
example_response = """
The optimal soil pH for the selected range is 6.5 - 6.8. For growing tomatoes, pH as suggested is necessary based on the information provided in the test. The range indicates that soil should be kept neutral or slightly acidic to ensure optimal conditions for the tomato crop's nutrient uptake and health. Factors like pH will need to be adjusted with additional soil enhancers (such as lime to raise pH and sulfur to lower it) depending on specific soil readings to create optimal growing conditions for tomatoes planted in a field environment.
"""

# Gradio UI
def main():
    with gr.Blocks() as app:
        # Section to open results based on search criteria
        with gr.Row():
            soil_characteristic = gr.Dropdown(label="Soil Characteristic", choices=["pH", "Calcium", "Soluble Salts", "Nitrogen"], value="pH")
            start_date = gr.Textbox(label="Start Date (YYYY-MM-DD)", placeholder="2024-01-01")
            end_date = gr.Textbox(label="End Date (YYYY-MM-DD)", placeholder="2024-12-31")
            llm_used = gr.Dropdown(label="LLM Used", choices=["llama-3.2", "gpt-3.5", "astral"], value="llama-3.2")
            keywords = gr.Textbox(label="Additional Keywords", placeholder="Enter any tags or keywords...")
            search_button = gr.Button("Search")

        # Placeholder for search results
        search_results = gr.Dropdown(label="Select a Result", choices=["Result 1", "Result 2", "Result 3"], interactive=True)

        # Button to load the selected result
        load_result_button = gr.Button("Load Selected Result")

        # Tabs for the user to choose between Response and Prompt sections
        with gr.Tabs():
            # First Tab: Response Tab
            with gr.Tab(label="Response"):
                response_section = gr.Markdown(value=f"### Response\n{example_response}")
                llm_info = gr.Markdown(value="**LLM**: llama-3.2 | **Prompt Token Count**: 305 | **Response Token Count**: 234")

                # Feedback Section
                feedback_slider = gr.Slider(minimum=0, maximum=10, label="Response Quality", value=5, interactive=True)
                feedback_text = gr.Textbox(label="Comment", lines=3, placeholder="Provide your feedback here...")
                save_feedback_button = gr.Button("Save Feedback")
                feedback_status = gr.Markdown(visible=False)

                def save_feedback(quality, comment):
                    return f'### Feedback Saved Successfully\n**Quality**: {quality} | **Comment**: {comment}'

                save_feedback_button.click(save_feedback, [feedback_slider, feedback_text], feedback_status)

    app.launch()

if __name__ == "__main__":
    main()
