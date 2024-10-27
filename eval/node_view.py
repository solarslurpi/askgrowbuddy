import gradio as gr
import traceback
import logging
import os

from rich import print

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def print_node_scores(nodes):
    """
    Print scores for all nodes, sorted from highest to lowest.

    Args:
    nodes (list): List of node objects.
    """
    # Create a list of tuples (index, score) for nodes with scores
    scored_nodes = []
    for index, node in enumerate(nodes):
        if hasattr(node, 'score') and node.score is not None:
            scored_nodes.append((index, node.score))
        else:
            logger.warning(f"Node at index {index} has no score attribute or score is None.")

    # Sort the list based on scores in descending order
    scored_nodes.sort(key=lambda x: x[1], reverse=True)

    # Print the sorted scores
    print("Node scores from highest to lowest:")
    for index, score in scored_nodes:
        logger.info(f"Node {index}: Score = {score}")

    # Print summary
    print(f"Total nodes: {len(nodes)}")
    print(f"Nodes with scores: {len(scored_nodes)}")
    print(f"Nodes without scores: {len(nodes) - len(scored_nodes)}")

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
            return return_values

        except Exception as e:
            logger.error(f"Error displaying node: {str(e)}")
            logger.error(traceback.format_exc())
            return ("Error displaying node", "Error", "Error", f"Error at index {index}", index)

    def navigate(direction, current_index):
        new_index = (int(current_index) + direction) % len(nodes)
        logger.debug(f"Navigating to node {new_index}")
        return show_node(new_index)

    def delete_node(current_index):
        logger.debug(f"Deleting node at index {current_index}")
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