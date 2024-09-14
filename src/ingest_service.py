import logging
import os

import chromadb
import ollama
from langchain.document_loaders import ObsidianLoader
from langchain.schema import Document
from langchain.text_splitter import MarkdownTextSplitter


from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class IngestService:
    """
    A service for ingesting, processing, and storing text data within an Obsidian vault for efficient retrieval.

    This class handles the following operations:
    1. Loading documents from a directory or file (e.g., documents.py) using Langchain's ObsidianLoader.
    2. Chunking the loaded texts using Langchain's MarkdownTextSplitter.
    3. Vectorizing the text chunks using Ollama embeddings.
    4. Storing the vectorized embeddings in a ChromaDB instance.

    The class provides methods for each step of the process, allowing for flexible usage
    and easy integration with other components of a document processing pipeline.
    """
    def __init__(self,persisitent_directory:str="vectorstore"):
        self.client = chromadb.PersistentClient(path=persisitent_directory)
        self.persistent_directory = persisitent_directory
        logger.debug(f"Persistent directory for storing vectorized documents: {self.persistent_directory}")

    def load_docs(self, dir_or_list:str|list) -> list[Document]:
        '''Uses Langchain's Obsidian Loader to load in the text found within the directory (and sub directories) or a list of strings in Document objects.'''
        logger.debug(f"Loading documents from: {dir_or_list}")
        docs = None
        if not dir_or_list:
            raise ValueError("A directory with text files or a list of strings must be passed in.")
        if isinstance(dir_or_list,list):
            docs = [Document(page_content=doc) for doc in dir_or_list]
        else:
            if not os.path.exists(dir_or_list):
                raise ValueError(f"Directory does not exist: {dir_or_list}")
            loader: ObsidianLoader = ObsidianLoader(path=dir_or_list)
            docs: list[Document] = loader.load()
        return docs

    def chunk_text(self, docs: list[Document], chunk_size=500, chunk_overlap=50) -> list[Document]:
        '''Use Langchain's markdown splitter to split the documents at an settable chunk_size and chunk_overlap, while maintaining splits that respect the headers.'''
        logger.debug(f"Chunking text with chunk size: {chunk_size} and overlap: {chunk_overlap}")
        text_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_splits: list[Document] = text_splitter.split_documents(docs)

        if not all_splits:
            raise ValueError("No text chunks generated.")
        return all_splits

    def generate_embedding(self, text:str, e_model:str='all-minilm'):
        '''Generate an embedding for a given text using the specified embedding model.'''
        logger.debug(f"Generating embedding for text with model: {e_model}")
        response = ollama.embeddings(model=e_model, prompt=text)
        return response['embedding']


    def create_collection(self, docs:list[Document]=None, collection: str = None,collection_metadata:dict=None) -> None:
        '''The documents (both metadata and text) and embeddings are stored in a chroma db collection along with the embedding function. The embedding model is passed in so that different embedding models. If the collection already exists, it is deleted and recreated.'''
        logger.debug(f"Storing documents in ChromaDB collection: {collection}")
        if not docs:
            raise ValueError("No documents to store.")
        if not collection:
            raise ValueError("No collection name provided.")
        # First delete the collection if it already exists
        if any(name == collection for name, _ in self.collections_name_and_metadata()):
            self.delete_collection(collection)
        # Create a new collection
        e_model = collection_metadata.get("e_model", "all-minilm")
        chunk_size = collection_metadata.get("chunk_size", -1)
        chunk_overlap = collection_metadata.get("chunk_overlap", -1)
        text_splitter = collection_metadata.get("text_splitter", "unknown")
        collection = self.client.create_collection(name=collection, metadata={"embedding_model": e_model, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "text_splitter": text_splitter})

        # store each document
        for i, d in enumerate(docs):
            response = ollama.embeddings(model=e_model, prompt=d.page_content)
            embedding = response["embedding"]
            # Build the parameters dictionary
            params = {
                "ids": [str(i)],
                "embeddings": [embedding],
                "documents": [d.page_content]
            }
            # Conditionally add metadatas if it is not None and not empty
            if d.metadata:
                params["metadatas"] = [d.metadata]
            collection.add(**params)

    def collection_count(self,collection:str) -> int:
        '''Returns the number of documents in the collection.'''

        collection = self.client.get_collection(name=collection)
        return collection.count()

    def add_documents(self, collection_name: str, docs: list[str], e_model: str = 'all-minilm'):
        collection = self.client.get_collection(name=collection)

        # Get the current count of documents in the collection
        current_count = collection.count()

        # Extract page content from each document
        doc_texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        # Generate embeddings for each document
        embeddings = [self.generate_embedding(doc, e_model) for doc in doc_texts]


        # Add new documents to the collection
        collection.add(
            documents=doc_texts,
            ids=[f"doc_{i}" for i in range(current_count, current_count + len(docs))],
            metadatas=metadatas,
            embeddings=embeddings
        )

        print(f"Added {len(docs)} documents to collection '{collection_name}'. Total documents: {collection.count()}")

    def delete_collection(self, collection_name: str):
        try:
            self.client.delete_collection(name=collection_name)
            message = f"Collection '{collection_name}' has been deleted."
        except ValueError as e:
            logger.error(f"Error deleting collection '{collection_name}",exc_info=True)
            raise e
        logger.debug(message)
        return message

    def collections_name_and_metadata(self) -> list[str]:
        '''Returns a list of tuples where each tuple contains a collection name and its corresponding metadata.'''
        collections = self.client.list_collections()
        collection_names = [collection.name for collection in collections]
        collection_metadatas = [collection.metadata for collection in collections]
        return list(zip(collection_names, collection_metadatas))