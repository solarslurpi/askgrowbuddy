import gradio as gr

# Example response from LLM
example_response = """
The optimal soil pH for the selected range is 6.5 - 6.8. For growing tomatoes, pH as suggested is necessary based on the information provided in the test. The range indicates that soil should be kept neutral or slightly acidic to ensure optimal conditions for the tomato crop's nutrient uptake and health. Factors like pH will need to be adjusted with additional soil enhancers (such as lime to raise pH and sulfur to lower it) depending on specific soil readings to create optimal growing conditions for tomatoes planted in a field environment.
"""

example_nodes = [
    {"text": "soils with a pH below 7 (acidic soils)", "metadata": {"topic": "calcium", "context": "soil chemistry"}, "score": 0.37},
    {"text": "soils with a pH above 7 (alkaline soils)", "metadata": {"topic": "potassium", "context": "nutrient uptake"}, "score": 0.52},
    {"text": "optimal pH range for most plants is 6.0 to 7.0", "metadata": {"topic": "general", "context": "plant growth"}, "score": 0.44}
]

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

            # Second Tab: Prompt Tab
            with gr.Tab(label="Prompt"):
                prompt_intro = gr.Textbox(label="Prompt Intro", value="Using only the knowledge provided in the context:", interactive=True)
                context_nodes = gr.Markdown(value=f"**Context**: {len(example_nodes)} nodes")

                # Node Navigation Section
                with gr.Row():
                    prev_node_button = gr.Button("Previous Node")
                    next_node_button = gr.Button("Next Node")

                with gr.Accordion(label="Node Information", open=True):
                    node_text = gr.Markdown(value=f"**Node Text**: {example_nodes[0]['text']}")
                    node_metadata = gr.Markdown(value=f"**Node Metadata**: {example_nodes[0]['metadata']}")
                    node_score = gr.Markdown(value=f"**Score**: {example_nodes[0]['score']}")

                def update_node_display(index):
                    index = int(index) % len(example_nodes)
                    node = example_nodes[index]
                    return f"**Node Text**: {node['text']}", f"**Node Metadata**: {node['metadata']}", f"**Score**: {node['score']}", index

                prev_node_button.click(lambda idx: update_node_display(idx - 1), [context_nodes], [node_text, node_metadata, node_score])
                next_node_button.click(lambda idx: update_node_display(idx + 1), [context_nodes], [node_text, node_metadata, node_score])

                prompt_outro = gr.Textbox(label="Prompt Outro", value="Provide the Cannabis grower advice on whether the pH value of 6.9 is within this optimal range [6.8, 6.9] for growing Cannabis. If it is not, provide advice on how to adjust the value using only the information provided in the context. If the context does not contain the information needed, respond with 'I do not have enough information to provide advice.'", interactive=True)

                llm_choice = gr.Dropdown(label="Choose LLM", choices=["mistral", "llama-3.2", "gpt-3.5"], value="mistral", interactive=True)
                similarity_top_k_index = gr.Number(label="similarity_top_k index", value=5, interactive=True)
                similarity_top_k_rank = gr.Number(label="similarity_top_k rank", value=3, interactive=True)
                redo_button = gr.Button("Redo")

    app.launch()

if __name__ == "__main__":
    main()
