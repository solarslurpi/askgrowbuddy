import gradio as gr
import traceback
import logging
import os

from rich import print

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_node_viewer(nodes):
    def format_metadata(metadata):
        formatted = "Metadata:\n"
        for key, value in metadata.items():
            if key == 'path':
                value = os.path.basename(value)
            formatted += f"- {key}: {value}\n"
        return formatted

    def show_node(index):
        try:
            logger.debug(f"Showing node at index {index}")
            node = nodes[index]
            formatted_metadata = format_metadata(node.metadata)
            score = node.score if hasattr(node, 'score') else "N/A"
            return_values = (
                node.get_content(),
                formatted_metadata,
                f"Score: {score}",
                f"Node {index + 1} of {len(nodes)}",
                index
            )

            # Print out the return values
            print("Return values from show_node:")
            print(f"Content: {return_values[0][:10]}...")
            print(f"Metadata: {return_values[1][:10]}...")
            print(f"Score: {return_values[2]}")
            print(f"Index display: {return_values[3]}")
            print(f"Index: {return_values[4]}")
            return return_values

        except Exception as e:
            logger.error(f"Error displaying node: {str(e)}")
            logger.error(traceback.format_exc())
            return ("Error displaying node", "Error", "Error", f"Error at index {index}", index)

    def navigate(direction, current_index):
        new_index = (int(current_index) + direction) % len(nodes)
        logger.info(f"Navigating to node {new_index}")
        return show_node(new_index)

    def delete_node(current_index):
        logger.info(f"Deleting node at index {current_index}")
        del nodes[current_index]
        new_index = min(current_index, len(nodes) - 1)
        return show_node(new_index) + (f"{len(nodes)} nodes remaining",)

    with gr.Blocks() as iface:
        current_index = gr.State(0)

        with gr.Row():
            prev_btn = gr.Button("Previous")
            next_btn = gr.Button("Next")
            delete_btn = gr.Button("Delete Current Node")
        index_display = gr.Markdown()
        node_text = gr.Textbox(label="Node Content", lines=10)
        metadata = gr.Markdown()
        score = gr.Markdown()
        nodes_remaining = gr.Markdown()

        prev_btn.click(navigate, inputs=[gr.Number(-1), current_index],
                       outputs=[node_text, metadata, score, index_display, current_index])
        next_btn.click(navigate, inputs=[gr.Number(1), current_index],
                       outputs=[node_text, metadata, score, index_display, current_index])
        delete_btn.click(delete_node, inputs=[current_index],
                         outputs=[node_text, metadata, score, index_display, current_index, nodes_remaining])

        # Initialize with the first node
        iface.load(show_node, inputs=[gr.Number(0)],
                   outputs=[node_text, metadata, score, index_display, current_index])

    return iface

def launch_node_viewer(nodes):
    logger.info("Launching node viewer")
    iface = create_node_viewer(nodes)
    iface.launch(inline=True)