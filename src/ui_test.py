import gradio as gr
import random

# Mock data and functions for illustration
def generate_results(prompt, model):
    response = f"Response from {model} for prompt: {prompt}"
    nodes_with_scores = [
        {
            "text": f"Node {i} with a detailed explanation that might be longer than fifty characters, providing more context.",
            "score": round(random.uniform(0, 1), 2),
            "metadata": {"key1": f"Value {i}a", "key2": f"Value {i}b"}
        } for i in range(3)
    ]
    return response, nodes_with_scores

def save_feedback(result, rating, feedback):
    # Here you would save the feedback to a database or file
    return f"Feedback saved: Rating {rating}, Comment: {feedback}"

def rerun(prompt, model):
    return generate_results(prompt, model)

# Initial prompts and model list
initial_prompts = [
    "Describe the effects of nitrogen deficiency in plants.",
    "What is the optimal pH range for soil used in tomato farming?",
    "How does calcium impact soil structure?"
]
initial_model = "llama-3.2"

# Gradio UI
def main():
    with gr.Blocks() as app:
        # Response and Model Information
        with gr.Accordion(label="Prompt", open=True):
            prompt_before_context = gr.Textbox(label="Prompt Before Context", lines=3, value="")
            prompt_display = gr.Textbox(label="Main Context Prompt", value=initial_prompts[0], lines=5, interactive=False)
            prompt_after_context = gr.Textbox(label="Prompt After Context", lines=3, value="")
            with gr.Row():
                model_display = gr.Textbox(value=f"Model Used: {initial_model}", label="Model Used", interactive=False)
                prompt_token_count = gr.Textbox(value="0", label="Prompt Token Count", interactive=False)
                response_token_count = gr.Textbox(value="0", label="Response Token Count", interactive=False)

        # Response Output Section
        with gr.Accordion(label="Response Information", open=True):
            response_output = gr.Textbox(label="Response", lines=3, interactive=False)

        # Initial generation of results
        response, nodes = generate_results(initial_prompts[0], initial_model)
        response_output.value = response

    app.launch()

if __name__ == "__main__":
    main()