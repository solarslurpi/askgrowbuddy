import logging
import numpy as np
import chromadb
from chromadb.api import Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from umap import UMAP

from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class Visualize:
    def __init__(self, collection_name:str = 'soil_test_knowledge'):
        try:
            # Initialize the ChromaDB client
            chroma_client = chromadb.PersistentClient(path='vectorstore')
            self.collection = chroma_client.get_collection(name=collection_name)
        except Exception as e:
            logger.error(f"Error: Could not find collection '{collection_name}'. {e}")
            raise e

    def plot_2d_scatter(self, question, results, embed_model_name='multi-qa-mpnet-base-cos-v1'):
        # Get embedding for the question
        embedding_function = SentenceTransformerEmbeddingFunction(model_name=embed_model_name)
        question_embedding = embedding_function.embed_query(question)
        # Get all document embeddings at once
        results = self.collection.get(include=['embeddings', 'documents', 'metadatas'])
        all_embeddings = np.array(results['embeddings'])

        print("Shape of all_embeddings:", all_embeddings.shape)
        print("Shape of question_embedding:", np.array(question_embedding).shape)

        # Ensure question_embedding is 2D
        question_embedding = np.array(question_embedding).reshape(1, -1)

        # Combine all embeddings
        combined_embeddings = np.vstack([question_embedding, all_embeddings])

        # Calculate appropriate perplexity
        n_samples = combined_embeddings.shape[0]
        perplexity = min(30, n_samples - 1)  # Default is 30, but it must be less than n_samples

        # Use t-SNE to reduce dimensionality to 2
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        reduced_embeddings = tsne.fit_transform(combined_embeddings)

        # Create the 2D scatter plot
        plt.figure(figsize=(12, 8))

        # Plot all document splits
        plt.scatter(reduced_embeddings[1:, 0], reduced_embeddings[1:, 1],
                    c='lightgray', marker='o', s=50, label='Unused Doc Splits')

        # Plot the included document splits
        included_indices = []
        for doc in results:
            try:
                # Try to find the index based on the document content
                idx = results['documents'].index(doc.page_content)
                included_indices.append(idx)
            except ValueError:
                print(f"Warning: Could not find matching document for: {doc.page_content[:50]}...")

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

    def plot_3d_umap(self, question, results):
        # Get embedding for the question
        embedding_function = OllamaEmbeddings(model='all-minilm')
        question_embedding = embedding_function.embed_query(question)

        # Get all document embeddings at once
        all_docs = self.vectorstore._collection.get(include=['embeddings', 'documents', 'metadatas'])
        all_embeddings = np.array(all_docs['embeddings'])

        print("Original shape of question_embedding:", np.array(question_embedding).shape)

        # Only reshape if it's 1D
        if len(np.array(question_embedding).shape) == 1:
            question_embedding = np.array(question_embedding).reshape(1, -1)
            print("Reshaped question_embedding to:", question_embedding.shape)
        else:
            question_embedding = np.array(question_embedding)
            print("No reshaping needed for question_embedding")

        print("Final shape of question_embedding:", question_embedding.shape)

        # Combine all embeddings
        combined_embeddings = np.vstack([question_embedding, all_embeddings])

        # Use UMAP to reduce dimensionality to 3
        reducer = UMAP(n_components=3, random_state=42)
        reduced_embeddings = reducer.fit_transform(combined_embeddings)

        # Create the 3D scatter plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot all document splits
        ax.scatter(reduced_embeddings[1:, 0], reduced_embeddings[1:, 1], reduced_embeddings[1:, 2],
                   c='red', marker='x', s=50, label='Unused Doc Splits')

        # Plot the included document splits
        included_indices = []
        for doc in results:
            try:
                # Try to find the index based on the document content
                idx = all_docs['documents'].index(doc.page_content)
                included_indices.append(idx)
            except ValueError:
                print(f"Warning: Could not find matching document for: {doc.page_content[:50]}...")

        ax.scatter(reduced_embeddings[1:, 0][included_indices],
                   reduced_embeddings[1:, 1][included_indices],
                   reduced_embeddings[1:, 2][included_indices],
                   c='green', marker='o', s=100, label='Included Doc Splits')

        # Plot the question point
        ax.scatter(reduced_embeddings[0, 0], reduced_embeddings[0, 1], reduced_embeddings[0, 2],
                   c='blue', marker='s', s=200, label='Question')

        # Add labels to the included document splits
        for i, idx in enumerate(included_indices):
            ax.text(reduced_embeddings[1:, 0][idx], reduced_embeddings[1:, 1][idx], reduced_embeddings[1:, 2][idx],
                    f'Doc {i+1}', fontsize=9)

        # Set labels and title
        ax.set_xlabel('UMAP Component 1')
        ax.set_ylabel('UMAP Component 2')
        ax.set_zlabel('UMAP Component 3')
        ax.set_title('3D Visualization of Question and Document Chunks using UMAP')

        # Add legend
        ax.legend()

        # Show the plot
        plt.show()

    def plot_3d_plotly(self, question, results):
        embedding_function = OllamaEmbeddings(model='all-minilm')
        question_embedding = embedding_function.embed_query(question)

        all_docs = self.vectorstore._collection.get(include=['embeddings', 'documents', 'metadatas'])
        all_embeddings = np.array(all_docs['embeddings'])

        question_embedding = np.array(question_embedding).reshape(1, -1)
        combined_embeddings = np.vstack([question_embedding, all_embeddings])

        # Use UMAP for dimensionality reduction
        reducer = UMAP(n_components=3, random_state=42)
        reduced_embeddings = reducer.fit_transform(combined_embeddings)

        # Prepare data for plotting
        x, y, z = reduced_embeddings.T

        # Create hover text for all points
        hover_text = ['Question'] + [f"Doc {i}: {doc[:50]}..." for i, doc in enumerate(all_docs['documents'], 1)]

        # Create the scatter plot for unused documents
        unused_trace = go.Scatter3d(
            x=x[1:], y=y[1:], z=z[1:],
            mode='markers',
            marker=dict(size=5, color='red', symbol='x'),
            text=hover_text[1:],
            hoverinfo='text',
            name='Unused Doc Splits'
        )

        # Create the scatter plot for included documents
        included_indices = [all_docs['documents'].index(doc.page_content) + 1 for doc in results]
        included_trace = go.Scatter3d(
            x=x[included_indices], y=y[included_indices], z=z[included_indices],
            mode='markers',
            marker=dict(size=8, color='green', symbol='circle'),
            text=[hover_text[i] for i in included_indices],
            hoverinfo='text',
            name='Included Doc Splits'
        )

        # Create the scatter plot for the question
        question_trace = go.Scatter3d(
            x=[x[0]], y=[y[0]], z=[z[0]],
            mode='markers',
            marker=dict(size=10, color='blue', symbol='square'),
            text=[hover_text[0]],
            hoverinfo='text',
            name='Question'
        )

        # Combine all traces
        data = [unused_trace, included_trace, question_trace]

        # Set up the layout
        layout = go.Layout(
            title='3D Visualization of Question and Document Chunks using UMAP',
            scene=dict(
                xaxis=dict(
                    showgrid=True, gridcolor='lightgrey', gridwidth=0.5,
                    zeroline=True, zerolinecolor='darkgrey', zerolinewidth=1,
                    title='UMAP Component 1'
                ),
                yaxis=dict(
                    showgrid=True, gridcolor='lightgrey', gridwidth=0.5,
                    zeroline=True, zerolinecolor='darkgrey', zerolinewidth=1,
                    title='UMAP Component 2'
                ),
                zaxis=dict(
                    showgrid=True, gridcolor='lightgrey', gridwidth=0.5,
                    zeroline=True, zerolinecolor='darkgrey', zerolinewidth=1,
                    title='UMAP Component 3'
                ),
                aspectmode='cube'  # This ensures equal scaling on all axes
            ),
            margin=dict(r=0, b=0, l=0, t=40)
        )

        # Create the figure and show it
        fig = go.Figure(data=data, layout=layout)
        fig.show()

    def plot_2d_umap(self, question, results):
        embedding_function = OllamaEmbeddings(model='all-minilm')
        question_embedding = embedding_function.embed_query(question)

        all_docs = self.vectorstore._collection.get(include=['embeddings', 'documents', 'metadatas'])
        all_embeddings = np.array(all_docs['embeddings'])

        question_embedding = np.array(question_embedding).reshape(1, -1)
        combined_embeddings = np.vstack([question_embedding, all_embeddings])

        # Use UMAP to reduce dimensionality to 2
        reducer = UMAP(n_components=2, random_state=0, transform_seed=0)
        reduced_embeddings = reducer.fit_transform(combined_embeddings)

        # Create the 2D scatter plot
        plt.figure(figsize=(12, 8))

        # Plot all document splits
        plt.scatter(reduced_embeddings[1:, 0], reduced_embeddings[1:, 1],
                    c='gray', s=10, label='Unused Doc Splits')

        # Plot the included document splits
        included_indices = []
        for doc in results:
            try:
                idx = all_docs['documents'].index(doc.page_content)
                included_indices.append(idx)
            except ValueError:
                print(f"Warning: Could not find matching document for: {doc.page_content[:50]}...")

        plt.scatter(reduced_embeddings[1:, 0][included_indices],
                    reduced_embeddings[1:, 1][included_indices],
                    c='green', s=100, facecolors='none', edgecolors='g', label='Included Doc Splits')

        # Plot the question point
        plt.scatter(reduced_embeddings[0, 0], reduced_embeddings[0, 1],
                    c='red', marker='X', s=150, label='Question')

        # Set labels and title
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.title(f'2D UMAP Visualization: {question}')

        # Add legend
        plt.legend()

        # Set aspect ratio to equal for better visualization
        plt.gca().set_aspect('equal', 'datalim')

        # Remove axis ticks
        plt.axis('off')

        # Show the plot
        plt.show()
