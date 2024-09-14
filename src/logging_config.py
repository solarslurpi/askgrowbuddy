# src/logging_config.py
import logging

def setup_logging():
    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d',
        handlers=[
            logging.StreamHandler()
        ]
    )

    # Set the logging levels for specific loggers
    logging.getLogger('langchain').setLevel(logging.INFO)
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('gradio').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('query').setLevel(logging.DEBUG)
    logging.getLogger('ingest_input').setLevel(logging.DEBUG)
    logging.getLogger('ingest_service').setLevel(logging.DEBUG)

    # You can add more specific logger configurations here if needed
    # For example:
    # logging.getLogger('ingest').setLevel(logging.DEBUG)
    # logging.getLogger('obsidian_rag').setLevel(logging.DEBUG)
    # logging.getLogger('retrieval').setLevel(logging.DEBUG)
