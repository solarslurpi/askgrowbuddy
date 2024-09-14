import logging
import pytest

from langchain.schema import Document
from src.documents import documents
from src.ingest_service import IngestService
from src.doc_stats import DocStats
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

@pytest.fixture
def docs_list():
    return documents

@pytest.fixture
def ingest_service():
    return IngestService()

@pytest.fixture
def directory():
    return r"C:\Users\happy\Documents\Projects\explore_obsidian_rag\TEST"

@pytest.fixture
def docs(ingest_service, directory):
    return ingest_service.load_docs(directory)

@pytest.fixture
def chunk_size():
    return 500

@pytest.fixture
def chunk_overlap():
    return 50

@pytest.fixture
def embedding_model():
    return "all-minilm"

@pytest.fixture
def chunk_method():
    return "MarkdownTextSplitter"

@pytest.fixture
def collection_metadata(chunk_size, chunk_overlap, chunk_method, embedding_model):
    collection_metadata = {
    "chunk_size": chunk_size,
    "chunk_overlap": chunk_overlap,
    "chunk_method": chunk_method,
    "embed_model": embedding_model
    }
    return collection_metadata

def test_loading_list(ingest_service, docs_list):
    docs = ingest_service.load_docs(docs_list)
    assert len(docs) > 0
    assert all([isinstance(doc, Document) for doc in docs])
    stats_dict = DocStats.get_summary_stats(docs)

    # Print out key/value pairs in stats_dict
    logger.info("\nDocument Statistics:")
    for key, value in stats_dict.items():
        logger.info(f"{key}: {value}")

def test_loading_directory(ingest_service, directory):
    docs = ingest_service.load_docs(directory)
    assert len(docs) > 0
    assert all([isinstance(doc, Document) for doc in docs])
    stats_dict = DocStats.get_summary_stats(docs)

    # Print out key/value pairs in stats_dict
    logger.info("\nDocument Statistics:")
    for key, value in stats_dict.items():
        logger.info(f"{key}: {value}")

def test_chunking_text(ingest_service, docs):
    chunked_docs = ingest_service.chunk_text(docs)
    assert len(chunked_docs) > 0
    assert all([isinstance(doc, Document) for doc in chunked_docs])
    stats_dict = DocStats.get_summary_stats(chunked_docs)

    # Print out key/value pairs in stats_dict
    logger.info("\nDocument Statistics:")
    for key, value in stats_dict.items():
        logger.info(f"{key}: {value}")

def test_create_collection(ingest_service, docs, collection_metadata):
    ingest_service.create_collection(docs, collection="test_collection", collection_metadata=collection_metadata)
    # Write assertions
    n_docs = ingest_service.collection_count("test_collection")
    assert n_docs > 0, "No documents stored in collection."
    print(ingest_service.collection_count("test_collection"))

def test_generate_embedding(ingest_service):
    text = "This is a test sentence."
    embedding = ingest_service.generate_embedding(text, e_model="all-minilm")
    assert len(embedding) > 0
    assert isinstance(embedding, list)
    assert isinstance(embedding[0], float)
    logger.info(f"Embedding: {embedding}")

def test_list_collections(ingest_service):
    collections = ingest_service.collections_name_and_metadata()
    # Look at the metadata for each collection
    for name, metadata in collections:
        # print out the collection name
        logger.info(f"Collection name: {name}")
        # print out each field in collection.metadata
        for key, value in metadata.items():
            logger.info(f"{key}: {value}")

    assert len(collections) > 0
    assert isinstance(collections, list)
    logger.info(f"Collections: {collections}")

def test_delete_collection(ingest_service):
    ingest_service.delete_collection("test_collection")
    collections = ingest_service.collections_name_and_metadata()
    assert "test_collection" not in collections
    logger.info(f"Collections: {collections}")
