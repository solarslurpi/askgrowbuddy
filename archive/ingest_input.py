import gradio as gr
import logging
from src.ingest_service import IngestService
from src.doc_stats import DocStats
from src.documents import documents
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

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
        chunks_stats = DocStats.get_summary_stats(chunks)
        # Convert list properties to strings. These are being attached to the chromadb collection and lists are not allowed.
        chunks_stats = {key: (str(value) if isinstance(value, list) else value) for key, value in chunks_stats.items()}
        source = "documents.py" if use_documents_py else directory
        collection_metadata = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_method": "MarkdownTextSplitter",
            "embed_model": embedding_model,
            "source": source
            }
        # Merge the dictionaries
        merged_metadata = {**chunks_stats, **collection_metadata}
        # Store documents
        ingest_service.create_collection(chunks, collection_name, merged_metadata)

        # Get statistics
        stats = DocStats.get_summary_stats(chunks)
        # Create a readable message
        source = "documents.py" if use_documents_py else directory
        status = f"Source: {source}\n"
        status += "\n".join(f"{k}: {v}" for k, v in stats.items())
        # Create status message
        status_message = f"Done processing {stats['total_documents']} documents..."

        # Get updated list of collections
        updated_collections = ingest_service.collections_name_and_metadata()

        return status_message, status, updated_collections
    except Exception as e:
        error_message = f"Error: {str(e)}"
        logger.error(f"Error ingesting data: {error_message}", exc_info=True)
        return error_message, {}, [], None

def refresh_collection_info(collection_name):
    # Get the metadata
    collections = ingest_service.collections_name_and_metadata()
    info = f"No collection found with name: {collection_name}"
    for name, metadata in collections:
        if name == collection_name:
            info = f"Collection Name: {name}\n"
            info += "\nMetadata:\n" + "\n".join(f"{k}: {v}" for k, v in metadata.items())
            return info

    return info

def refresh_collections_dropdown(collections_info, value_name):
    # Extract collection names from the collections_info
    metadata = None
    name = None
    if value_name:
         # Get metadata
        for n, m in collections_info:
            if n == value_name:
                metadata = m
                name = value_name
                break
    else:
        name, metadata = collections_info[0] if collections_info else (None, None)
    names = [name for name, _ in collections_info]
    dropdown_update = None
    info = None

    if name:
        # Create a new instance of the dropdown with the updated list of collections
                # Debugging: Print updated collections and default name
        logger.debug(f"Updated collections: {names}")
        logger.debug(f"Default name: {name}")
        dropdown_update = gr.Dropdown(choices=names, value=name)
        # Create the collection info
        info = f"Collection Name: {name}\n"
        info += "\nMetadata:\n" + "\n".join(f"{k}: {v}" for k, v in metadata.items())

    return dropdown_update, info

def delete_collection(collection_name):
    try:
        # We have the collection_name, so we can delete the collection
        ingest_service.client.delete_collection(collection_name)
        updated_collections = ingest_service.collections_name_and_metadata()
        dropdown_update, info = refresh_collections_dropdown(updated_collections, None)

        return dropdown_update, info

    except Exception as e:
        logger.error(f"Error deleting collection: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"


# Create the UI layout
with gr.Blocks() as demo:
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
            gr.Markdown("**Manage Collections**")
            collections_info = ingest_service.collections_name_and_metadata()
            collection_names = [name for name, _ in collections_info]
            default_collection = collection_names[0] if collection_names else None
            collections_dropdown = gr.Dropdown(
                label="Select a collection",
                choices=collection_names,
                value=default_collection
            )
            collection_info_textbox = gr.Textbox(label="Collection Info", lines=10, interactive=False)
            delete_button = gr.Button("Delete Selected Collection", visible=bool(default_collection))

            # Display initial metadata if a default collection exists
            if default_collection:
                # Second element in the tuple is the metadata
                info = refresh_collection_info(default_collection)
                collection_info_textbox.value = info
                delete_button.visible = True

    ingest_button = gr.Button('Ingest')
    output_message = gr.Textbox(label='Status:')
    output_stats = gr.Textbox(label='Stats:', lines=10)
    collections_info_state = gr.State([])

    ingest_button.click(
        ingest_data,
        inputs=[directory_input, use_documents_py, chunk_size_input, chunk_overlap_input, collection_name_input, embedding_model_input],
        outputs=[output_message, output_stats, collections_info_state]
    ).then(
        refresh_collections_dropdown,
        inputs=[collections_info_state, collection_name_input],
        outputs=[collections_dropdown, collection_info_textbox]
    )
    # inputs=[collections_dropdown] means that the function will be called with the current value of collections_dropdown
    delete_button.click(
        delete_collection,
        inputs=[collections_dropdown],
        outputs=[collections_dropdown, collection_info_textbox]
    )
    # When the collections dropdown changes, update the collection info textbox
    collections_dropdown.change(
        refresh_collection_info,
        inputs=[collections_dropdown],
        outputs=[collection_info_textbox]
    )
demo.launch()