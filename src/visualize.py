import logging
import numpy as np
from typing import Any
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from llama_index.core.schema import NodeWithScore, TextNode
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from umap import UMAP
import gradio as gr
import plotly.graph_objs as go
import plotly.io as pio


from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class Visualize:
    def __init__(self, collection_name:str = 'soil_test_knowledge',         embedding_function:SentenceTransformerEmbeddingFunction=None):
        try:
            if embedding_function is None:
                embedding_function = SentenceTransformerEmbeddingFunction(model_name='multi-qa-mpnet-base-cos-v1')
            # Initialize the ChromaDB client
            chroma_client = chromadb.PersistentClient(path='vectorstore')
            self.collection = chroma_client.get_collection(name=collection_name, embedding_function=embedding_function)
            self.embedding_function = embedding_function
        except Exception as e:
            logger.error(f"Error: Could not find collection '{collection_name}'. {e}")
            raise e

    def _calculate_perplexity(self, n_samples):
        return min(30, n_samples - 1)
    def _prepare_embeddings(self, query: str, nodeswithscore: list[NodeWithScore]):
        query_embedding = self.embedding_function([query])[0]
        result_dicts = self.collection.get(include=['embeddings', 'documents', 'metadatas'])
        all_embeddings = np.array(result_dicts['embeddings'])

        query_embedding = np.array(query_embedding).reshape(1, -1)
        combined_embeddings = np.vstack([query_embedding, all_embeddings])

        id_to_index = {str(i): i for i in range(1, len(combined_embeddings))}
        included_indices = [id_to_index.get(node.node.id_, -1) for node in nodeswithscore]
        included_indices = [i for i in included_indices if i != -1]

        n_samples = combined_embeddings.shape[0]
        perplexity = self._calculate_perplexity(n_samples)

        return combined_embeddings, included_indices, n_samples, perplexity

    def plot_2d_scatter(self, query:str, nodeswithscore:list[NodeWithScore]):
        combined_embeddings, included_indices, n_samples, perplexity = self._prepare_embeddings(query, nodeswithscore)

        # Use t-SNE to reduce dimensionality to 2
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        reduced_embeddings = tsne.fit_transform(combined_embeddings)

        # Create the 2D scatter plot
        plt.figure(figsize=(12, 8))

        # Plot all document splits
        plt.scatter(reduced_embeddings[1:, 0], reduced_embeddings[1:, 1],
                    c='lightgray', marker='o', s=50, label='Unused Doc Splits')

        # Plot the included document splits
        # Create a mapping of node IDs to their index in reduced_embeddings
        id_to_index = {str(i): i for i in range(1, len(reduced_embeddings))}  # Start from 1 as 0 is the query

        # Get the indices for included nodes, skipping any that aren't in the mapping
        included_indices = [id_to_index.get(node.node.id_, -1) for node in nodeswithscore]
        included_indices = [i for i in included_indices if i != -1]

        plt.scatter(reduced_embeddings[1:, 0][included_indices],
                    reduced_embeddings[1:, 1][included_indices],
                    c='green', marker='o', s=100, label='Included Doc Splits')

        # Plot the question point
        plt.scatter(reduced_embeddings[0, 0], reduced_embeddings[0, 1],
                    c='red', marker='*', s=200, label='Question')

        # Add labels to the included document splits
        for i, idx in enumerate(included_indices):
            plt.annotate(f'Doc {i+1}', (reduced_embeddings[1:, 0][idx], reduced_embeddings[1:, 1][idx]))

        # Set labels and title
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('2D Visualization of Question and Document Chunks')

        # Add legend
        plt.legend()

        # Show the plot
        plt.show()

    def _reduce_dimensions_umap(self, embeddings: np.ndarray, n_components: int = 3) -> np.ndarray:
        reducer = UMAP(n_components=n_components, random_state=42)
        return reducer.fit_transform(embeddings)

    def _create_3d_scatter(self, reduced_embeddings, included_indices, n_samples):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot all document splits
        ax.scatter(reduced_embeddings[1:, 0], reduced_embeddings[1:, 1], reduced_embeddings[1:, 2],
                   c='lightgray', marker='o', s=50, label='Unused Doc Splits')

        # Plot the included document splits
        ax.scatter(reduced_embeddings[included_indices, 0],
                   reduced_embeddings[included_indices, 1],
                   reduced_embeddings[included_indices, 2],
                   c='green', marker='o', s=100, label='Included Doc Splits')

        # Plot the question point
        ax.scatter(reduced_embeddings[0, 0], reduced_embeddings[0, 1], reduced_embeddings[0, 2],
                   c='red', marker='*', s=200, label='Question')

        return fig, ax

    def _add_labels_and_legend(self, ax, reduced_embeddings, included_indices, n_samples):
        # Add labels to the included document splits
        for i, idx in enumerate(included_indices):
            ax.text(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], reduced_embeddings[idx, 2],
                    f'Doc {i+1}', fontsize=9)

        ax.set_xlabel('UMAP Component 1')
        ax.set_ylabel('UMAP Component 2')
        ax.set_zlabel('UMAP Component 3')
        ax.set_title(f'3D Visualization of Question and {n_samples-1} Document Chunks using UMAP')
        ax.legend()

    def plot_3d_umap(self, query: str, nodeswithscore: list[NodeWithScore]):
        combined_embeddings, included_indices, n_samples, _ = self._prepare_embeddings(query, nodeswithscore)

        reduced_embeddings = self._reduce_dimensions_umap(combined_embeddings)
        fig, ax = self._create_3d_scatter(reduced_embeddings, included_indices, n_samples)
        self._add_labels_and_legend(ax, reduced_embeddings, included_indices, n_samples)

        plt.show()

    def _create_hover_text(self, documents: list[str]) -> list[str]:
        return ['Question'] + [f"Doc {i}: {doc[:50]}..." for i, doc in enumerate(documents, 1)]

    def _create_plotly_traces(self, reduced_embeddings, hover_text, included_indices, nodeswithscore):
        x, y, z = reduced_embeddings.T

        unused_trace = go.Scatter3d(
            x=x[1:], y=y[1:], z=z[1:],
            mode='markers',
            marker=dict(size=5, color='red', symbol='x'),
            text=hover_text[1:],
            hoverinfo='text',
            name='Unused Doc Splits'
        )

        included_trace = go.Scatter3d(
            x=x[included_indices], y=y[included_indices], z=z[included_indices],
            mode='markers',
            marker=dict(size=8, color='green', symbol='circle'),
            text=[f"ID: {node.node.id_}<br>Score: {node.score:.4f}<br>Text: {node.node.text[:200]}..." for node in nodeswithscore],
            hoverinfo='text',
            name='Included Doc Splits'
        )

        question_trace = go.Scatter3d(
            x=[x[0]], y=[y[0]], z=[z[0]],
            mode='markers',
            marker=dict(size=10, color='blue', symbol='square'),
            text=[hover_text[0]],
            hoverinfo='text',
            name='Question'
        )

        return [unused_trace, included_trace, question_trace]

    def _create_plotly_layout(self) -> go.Layout:
        return go.Layout(
            title='3D Visualization of Question and Document Chunks using UMAP',
            scene=dict(
                xaxis=dict(title='UMAP Component 1'),
                yaxis=dict(title='UMAP Component 2'),
                zaxis=dict(title='UMAP Component 3'),
                aspectmode='cube'
            ),
            margin=dict(r=0, b=0, l=0, t=40)
        )

    def plot_3d_plotly(self):
        def generate_plot(query: str, top_k: int) -> go.Figure:
            try:
                nodeswithscore = self._get_relevant_nodes(query, top_k)
                combined_embeddings, included_indices, n_samples, _ = self._prepare_embeddings(query, nodeswithscore)
                reduced_embeddings = self._reduce_dimensions_umap(combined_embeddings)

                all_docs = self.collection.get(include=['documents'])
                hover_text = self._create_hover_text(all_docs['documents'])

                traces = self._create_plotly_traces(reduced_embeddings, hover_text, included_indices, nodeswithscore)  # Add nodeswithscore here
                layout = self._create_plotly_layout()

                fig = go.Figure(data=traces, layout=layout)
                fig.update_layout(
                    autosize=True,
                    height=800
                )
                return fig
            except Exception as e:
                print(f"Error generating plot: {str(e)}")
                return go.Figure()  # Return an empty figure in case of error

        with gr.Blocks(theme=gr.themes.Base()) as iface:
            gr.Markdown("# 3D Visualization of Document Embeddings")
            gr.Markdown("Enter a query to visualize related documents in 3D space.")

            with gr.Column():
                query_input = gr.Textbox(lines=2, placeholder="Enter your query here...")
                top_k_input = gr.Number(label="Top K Similar Documents", value=10, precision=0)
                submit_button = gr.Button("Submit")
                plot_output = gr.Plot(label="3D Visualization", container=True)

            submit_button.click(
                fn=generate_plot,
                inputs=[query_input, top_k_input],
                outputs=plot_output
            )

        return iface

    def _get_relevant_nodes(self, query: str, top_k: int = 3) -> list[NodeWithScore]:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        nodes_with_score = []
        for i in range(len(results['documents'][0])):
            text_node = TextNode(
                text=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                id_=results['ids'][0][i]
            )
            node_with_score = NodeWithScore(
                node=text_node,
                score=results['distances'][0][i]
            )
            nodes_with_score.append(node_with_score)

        return nodes_with_score

# Usage
visualize = Visualize()  # Initialize with your parameters
iface = visualize.plot_3d_plotly()
iface.launch()
