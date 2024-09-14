import gradio as gr
from src.ingest_service import IngestService
from src.doc_stats import DocStats
from src.documents import documents

# Initialize the ingest service
ingest_service = IngestService()

def ingest_data(directory, use_documents_py, chunk_size, chunk_overlap, collection_name, embedding_model):
    try:
        # Load documents
        if use_documents_py:
            docs = ingest_service.load_docs(documents)
        else:
             docs = ingest_service.load_docs(directory)

        # Chunk documents
        chunks = ingest_service.chunk_text(docs, chunk_size, chunk_overlap)

        # Store documents
        ingest_service.store_docs_in_chroma(chunks, collection_name, embedding_model)

        # Get statistics
        stats = DocStats.get_summary_stats(chunks)

        # Create status message
        status_message = f"STATUS: Embedding document 4 of {stats['total_documents']} documents..."

        return status_message, stats
    except Exception as e:
        error_message = f"Error: {str(e)}"
        return error_message, {}

def delete_collection(collection_name):
    '''Wrap within try/except to display error (if occurs)'''
    try:
        ingest_service.client.delete_collection(collection_name)
        return "Collection deleted successfully"
    except Exception as e:
        return f"Error: {str(e)}"
# Create the UI layout
with gr.Blocks() as ingest_interface:
    gr.Markdown("# Ingest")
    with gr.Row():
        with gr.Column():
                gr.Markdown("**Load Data**")
                directory_input = gr.Textbox(label='Directory:')
                use_documents_py = gr.Checkbox(label='Use documents.py as the file with the list of documents')
        with gr.Column():
                gr.Markdown("**Chunk Data**")
                chunk_size_input = gr.Number(label='chunk size:', value=500)
                chunk_overlap_input = gr.Number(label='chunk overlap:', value=50)
        with gr.Column():
                gr.Markdown("**Embed Data**")
                collection_name_input = gr.Textbox(label='collection name:', value='chunk-500-overlap-50-model-all-minilm')
                embedding_model_input = gr.Dropdown(label='embedding model:', choices=["all-minilm", "nomic-embed-text"], value="all-minilm")
        with gr.Column():
                gr.Markdown("**Delete Collection**")
                collection_to_delete = gr.Textbox(label='collection name:', value='chunk-500-overlap-50-model-all-minilm')
                delete_collection_button = gr.Button('Delete Collection')

    ingest_button = gr.Button('Ingest')
    output_message = gr.Textbox(label='Status:')
    output_stats = gr.Textbox(label='Stats:', lines=10)

    ingest_button.click(
        ingest_data,
        inputs=[directory_input, use_documents_py, chunk_size_input, chunk_overlap_input, collection_name_input, embedding_model_input],
        outputs=[output_message, output_stats]
    )
    delete_collection_button.click(
        delete_collection,
        inputs=[collection_to_delete],
        outputs=[output_message]
    )

ingest_interface.launch()